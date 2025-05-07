# dataset.py

import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LabeledStateDataset(Dataset):
    """
    Loads labeled state data from a binary file written by your Java extractor.
    File format:
      - 4 × 4-byte big-endian ints: n (records), S (state dim), wordsPerState, A (action dim)
      - wordsPerState × 8-byte big-endian unsigned longs per record: packed state bits
      - A × 8-byte big-endian floats per record: action distribution
      - 8-byte big-endian float per record: result label
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            header = f.read(16)  # now 4 ints in header
            if len(header) < 16:
                raise IOError(f"File too small for header: {path}")
            n, S, words_per_state, A = struct.unpack(">iiii", header)

            # preallocate
            states_arr  = np.zeros((n, S), dtype=np.uint8)
            actions_arr = np.empty((n, A), dtype=np.float32)
            labels_arr  = np.empty((n,), dtype=np.float32)

            for i in range(n):
                # 1) read packed state bits
                packed_bytes = f.read(words_per_state * 8)
                if len(packed_bytes) != words_per_state * 8:
                    raise IOError(f"Bad state read at record {i}")
                longs = struct.unpack(f">{words_per_state}Q", packed_bytes)
                for bit in range(S):
                    word_idx, bit_idx = divmod(bit, 64)
                    if (longs[word_idx] >> bit_idx) & 1:
                        states_arr[i, bit] = 1

                # 2) read action-distribution (A doubles)
                action_bytes = f.read(A * 8)
                if len(action_bytes) != A * 8:
                    raise IOError(f"Bad action-vector read at record {i}")
                actions_arr[i, :] = struct.unpack(f">{A}d", action_bytes)

                # 3) read result label
                buf = f.read(8)
                if len(buf) != 8:
                    raise IOError(f"Bad label read at record {i}")
                (lbl,) = struct.unpack(">d", buf)
                labels_arr[i] = lbl

        # convert to tensors
        self.states  = torch.from_numpy(states_arr.astype(np.float32))
        self.actions = torch.from_numpy(actions_arr)    # float32 [n,A]
        self.labels  = torch.from_numpy(labels_arr)     # float32 [n]

    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, idx):
        # returns (state_vector, action_distribution, scalar_label)
        return self.states[idx], self.actions[idx], self.labels[idx]


if __name__ == "__main__":
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/UWTempo/ver4/training.bin"
    ds = LabeledStateDataset(path)
    print(f"Dataset size: {len(ds)}")

    wins = 0
    total = len(ds)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    for i, (state, action, label) in enumerate(dl):
        # only first 100 bits
        sb = "".join(str(int(x.item())) for x in state[0, :100])
        av = action[0].tolist()  # full A-length action vector
        lbl = label.item()
        print(f"State: {sb}, Action: {av}, Result: {lbl}")
        if lbl > 0:
            wins += 1
        if i >= 999:
            break

    print(f"Winrate: {wins / total:.3f}")
