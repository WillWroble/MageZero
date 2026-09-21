# dataset.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import List, Tuple
from pathlib import Path  # Use the modern pathlib for handling file paths
import sys
from typing import Set
from scipy.sparse import coo_matrix
import h5py


GLOBAL_MAX = 2000000

H5_RECIDX_DTYPE = np.dtype([("file", "i4"), ("row", "i8")])

class H5Indexed(Dataset):
    """
    Preloads all HDF5 shards into RAM for fast random access.
    Expects per-file datasets:
      /indices  (int32, [nnz])
      /offsets  (int64, [N+1], offsets[0]=0, offsets[-1]=nnz)
      /row      (float32, [N, A+4])  -> [policy(A), resultLabel, stateScore, isPlayer(0/1), actionType]
    Returns (per sample):
      (indices:int64[r], policy:float32[A], value:float32[1], is_player:float32[1], action_type:int64[1])
    """

    def __init__(self, dir_path: str, ignore: set[int] | None = None, vocab=None):
        """ignore: feature ids to drop (full-table models).
        vocab: a FeatureVocab (dense-vocab models); ids are mapped to embedding rows and ids
        outside the vocab are dropped. Use one or the other."""
        if ignore and vocab is not None:
            raise ValueError("pass either ignore or vocab, not both")
        p = Path(dir_path)
        h5_paths = sorted(list(p.glob("*.h5")) + list(p.glob("*.hdf5")))
        self.files = [str(pp) for pp in h5_paths]

        if not self.files:
            self.N = 0
            self.A = 0
            self.idxptr_t = torch.zeros(1, dtype=torch.long)
            self.indices_t = torch.empty(0, dtype=torch.int32)
            self.row_t = torch.empty(0, 0, dtype=torch.float32)
            return

        idxptr = [0]
        indices_chunks = []
        row_chunks = []
        nnz_cum = 0
        N_total = 0
        A_ref = None

        for path in self.files:
            with h5py.File(path, "r") as f:
                off = f["/offsets"][...].astype(np.int64, copy=False)  # [N+1]
                idx = f["/indices"][...].astype(np.int32, copy=False)  # [nnz]
                row = f["/row"][...].astype(np.float32, copy=False)  # [N, A+4]
                N = int(off.shape[0] - 1);
                nnz = int(off[-1])
                A_local = int(row.shape[1] - 4)
                if A_ref is None:
                    A_ref = A_local
                else:
                    assert A_local == A_ref, "Inconsistent A across shards"

                indices_chunks.append(idx)
                row_chunks.append(row)
                if N > 0: idxptr.extend((off[1:] + nnz_cum).tolist())
                nnz_cum += nnz
                N_total += N

        self.N = N_total;
        self.A = A_ref if A_ref is not None else 0
        idxptr_np = np.asarray(idxptr, dtype=np.int64)  # [N+1]
        indices_np = (np.concatenate(indices_chunks) if indices_chunks
                      else np.empty(0, dtype=np.int32))  # [nnz]
        row_np = (np.concatenate(row_chunks, axis=0) if row_chunks
                  else np.empty((0, self.A + 4), dtype=np.float32))  # [N, A+4]

        # --- EAGER IGNORE (optional, once) ---
        if ignore:
            ign = np.fromiter(ignore, dtype=np.int32)
            keep_all = ~np.isin(indices_np, ign, assume_unique=False)
            new_idxptr = np.empty_like(idxptr_np)
            new_idxptr[0] = 0
            write_pos = 0
            for i in range(self.N):
                a = idxptr_np[i]
                b = idxptr_np[i + 1]
                if b > a:
                    m = keep_all[a:b]
                    L = int(m.sum())
                    if L:
                        # compact kept indices forward (single pass)
                        src = indices_np[a:b][m]
                        indices_np[write_pos:write_pos + L] = src
                    new_idxptr[i + 1] = write_pos + L
                    write_pos += L
                else:
                    new_idxptr[i + 1] = write_pos
            indices_np = indices_np[:write_pos]
            idxptr_np = new_idxptr

        # the full-table embedding has one row per hash bin, so raw ids fold into that range;
        # dense-vocab rows are produced below and must not be touched by it
        if vocab is None:
            np.mod(indices_np, GLOBAL_MAX, out=indices_np)

        # --- DENSE VOCAB (optional): feature id -> embedding row, unknown ids dropped ---
        if vocab is not None:
            rows = vocab.lookup(indices_np)
            keep = rows >= 0
            kept_before = np.concatenate([[0], np.cumsum(keep)])
            idxptr_np = kept_before[idxptr_np].astype(np.int64)
            indices_np = rows[keep].astype(np.int32)

        # store as tensors; __getitem__ uses zero-copy views
        self.idxptr_t = torch.from_numpy(idxptr_np)  # int64 [N+1]
        self.indices_t = torch.from_numpy(indices_np)  # int32 [nnz]
        self.row_t = torch.from_numpy(row_np)  # float32 [N,A+4]


    def __len__(self) -> int:
        return int(self.N)

    def __getitem__(self, k: int):
        a = int(self.idxptr_t[k].item())
        b = int(self.idxptr_t[k + 1].item())

        sv_idx_t = self.indices_t.narrow(0, a, b - a)  # int32 view
        row_k = self.row_t[k]  # float32 [A+4] view

        A = self.A
        policy_t = row_k.narrow(0, 0, A)  # float32 [A]
        value_t = row_k[A + 0].unsqueeze(0)  # float32 [1]
        isP_t = (row_k[A + 2] > 0.5).float().unsqueeze(0)  # float32 [1]
        aType_t = row_k[A + 3].to(torch.long).unsqueeze(0)  # int64 [1]

        return sv_idx_t, policy_t, value_t, isP_t, aType_t

def collate_batch(batch):
    n = len(batch)
    lens = [b[0].numel() for b in batch]
    total = int(sum(lens))

    idxs = torch.empty(total, dtype=torch.int32)  # keep int32 for now
    offsets = torch.empty(n, dtype=torch.long)

    p = 0
    for i, (ix, _, _, _, _) in enumerate(batch):
        L = ix.numel()
        if L: idxs[p:p+L].copy_(ix)              # bulk copy
        offsets[i] = p
        p += L

    # single conversion for EmbeddingBag (ids are already rows, or raw ids clamped in H5Indexed)
    idxs = idxs.to(torch.long)

    policies     = torch.stack([b[1] for b in batch], 0)
    values       = torch.stack([b[2] for b in batch], 0)
    is_players   = torch.stack([b[3] for b in batch], 0)
    action_types = torch.stack([b[4] for b in batch], 0)

    return idxs, offsets, policies, values, is_players, action_types

def create_redundancy_ignore_list(ds, k=10) -> Set[int]:
    """
    constructs a data derived ignore list which includes perfectly redundant features, and features that occur less than k times.
    """
    # make sparse matrix
    rows_list, cols_list = [], []

    num_samples = 0

    for (idxs, _, _, _, _) in ds:
        idxs_np = idxs.cpu().numpy().astype(np.int32, copy=False)
        rows_list.append(np.full(idxs_np.shape, num_samples, dtype=np.int32))
        cols_list.append(idxs_np)
        num_samples += 1

    rows = np.concatenate(rows_list)
    cols = np.concatenate(cols_list)

    data = np.ones_like(cols, dtype=np.uint8)

    x_csc = coo_matrix(
        (data, (rows, cols)),
        shape=(num_samples, GLOBAL_MAX),
        dtype=np.uint8
    ).tocsc()

    # group identical columns
    groups = {}
    indptr = x_csc.indptr
    idxs = x_csc.indices

    ignore = set()

    for j in range(GLOBAL_MAX):
        start, end = indptr[j], indptr[j + 1]
        key = tuple(idxs[start:end])  # () means unseen/empty
        if len(key) <= k:
            ignore.add(j)
        groups.setdefault(key, []).append(j)

    for key, js in groups.items():
        if key == () or len(js) == 1:
            continue
        js.sort()
        ignore.update(js[1:])

    kept = GLOBAL_MAX - len(ignore)
    print(f"A total of {len(ignore)} feature indices will be ignored.")
    print(f"{kept} feature indices were kept.")

    return ignore

def filter_one_hots(dataset):
    """
    Return a torch.utils.data.Subset that excludes samples whose policy label is one-hot.
    """
    keep_indices = []
    n = len(dataset)


    for i in range(n):
        _, policy, _, _, _ = dataset[i]  # (indices, policy, value)
        # policy is a FloatTensor of shape [A]
        # treat near-zeros as zero via eps
        nonzero = (policy > 0).sum().item()
        is_one_hot = (nonzero == 1)
        if not is_one_hot:
            keep_indices.append(i)


    removed = n - len(keep_indices)
    print(f"[one hot filter] scanned {n} samples "
          f" kept {len(keep_indices)} (removed {removed} one-hot policies).")

    return Subset(dataset, keep_indices)

def filter_opponent_states(dataset, targets_max):
    """
    Return a torch.utils.data.Subset that excludes samples from opponent (Player B)'s perspective. and targeting samples that include opponent targets
    """
    keep_states = []
    n = len(dataset)


    for i in range(n):
        _, policy, _, is_player, d_type = dataset[i]  # (indices, policy, value, isPlayer, decision type)
        if not ((not is_player) and d_type == 0) : #keep anything but opponent priorities
            keep_states.append(i)


    removed = n - len(keep_states)
    print(f"[opponent filter] scanned {n} samples "
          f" kept {len(keep_states)} (removed {removed} opponent states).")

    return Subset(dataset, keep_states)


if __name__ == "__main__":

    # Define the directory where you save your game data files.
    data_directory = "data/MTGA_MonoU/ver1/testing"

    try:
        # Load dataset from the specified folder
        full_dataset = H5Indexed(data_directory)

        winning = 0
        num_samples_to_process = 0

        dl = DataLoader(full_dataset, batch_size=1, shuffle=False, collate_fn=collate_batch)

        for i, (indices_batch, offsets_batch, policies_batch, values_batch, players_batch, action_types_batch) in enumerate(dl):
            num_samples_to_process += 1

            current_sample_indices = indices_batch
            indices_to_print = current_sample_indices.tolist()

            if len(indices_to_print) > 100:
                sb = " ".join(map(str, indices_to_print[:100])) + " ..."
            else:
                sb = " ".join(map(str, indices_to_print))
            if not indices_to_print:
                sb = "[No active features]"

            action = policies_batch[0]
            av = action.tolist()

            label_tensor = values_batch[0]
            lbl = label_tensor.item()

            player = players_batch[0]

            action_type = action_types_batch[0]

            print(f"State: {sb}, Action: {av}, Result: {lbl}, isPlayer: {player}, ActionType: {action_type}")

            if lbl > 0:
                winning += 1
            if i >= 100000:
                break

        print(f"\nDataset size: {len(full_dataset)}\n")

        # Calculate the number of unique feature indices across the entire dataset
        if len(full_dataset) > 0:
            all_feature_indices = set()
            # We iterate through the dataset directly to access each sample's indices
            for sample_idx in range(len(full_dataset)):
                indices, _, _, _, _ = full_dataset[sample_idx]  # Unpack the sample tuple
                all_feature_indices.update(indices.tolist())
            print(f"Total unique feature indices in dataset: {len(all_feature_indices)}")

        if num_samples_to_process > 0:
            print(f"Winning rate (over {num_samples_to_process} samples): {winning / num_samples_to_process:.3f}")
        else:
            print("No samples were processed.")


    except (FileNotFoundError, ValueError, IOError, IndexError) as e:
        print(f"An error occurred: {e}", file=sys.stderr)

