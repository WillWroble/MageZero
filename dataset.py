# dataset.py

import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LabeledStateDataset(Dataset):
    """
    Loads labeled state data from a binary file written by your Java extractor.
    File format:
      - 3 × 4-byte big-endian ints: n (records), S (state dim), wordsPerState
      - wordsPerState × 8-byte big-endian unsigned longs per record: packed state bits
      - 4-byte big-endian int per record: action index
      - 8-byte big-endian float per record: result label
    """

    def __init__(self, path):
        with open(path, "rb") as f:
            header = f.read(12)
            if len(header) < 12:
                raise IOError(f"File too small for header: {path}")
            n, S, words_per_state = struct.unpack(">iii", header)

            # preallocate
            states_arr  = np.zeros((n, S), dtype=np.uint8)
            actions_arr = np.empty((n,), dtype=np.int64)
            labels_arr  = np.empty((n,), dtype=np.float32)

            for i in range(n):
                # read packed state
                packed_bytes = f.read(words_per_state * 8)
                if len(packed_bytes) != words_per_state * 8:
                    raise IOError(f"Bad state read at record {i}")
                longs = struct.unpack(f">{words_per_state}Q", packed_bytes)
                # expand bits
                for bit in range(S):
                    word_idx, bit_idx = divmod(bit, 64)
                    if (longs[word_idx] >> bit_idx) & 1:
                        states_arr[i, bit] = 1

                # read action index
                buf = f.read(4)
                if len(buf) != 4:
                    raise IOError(f"Bad action read at record {i}")
                (ai,) = struct.unpack(">i", buf)
                actions_arr[i] = ai

                # read label
                buf = f.read(8)
                if len(buf) != 8:
                    raise IOError(f"Bad label read at record {i}")
                (dbl,) = struct.unpack(">d", buf)
                labels_arr[i] = dbl

        # to tensors
        self.states  = torch.from_numpy(states_arr.astype(np.float32))
        self.actions = torch.from_numpy(actions_arr)         # int64
        self.labels  = torch.from_numpy(labels_arr)

    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.labels[idx]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    import sys

    path = sys.argv[1] if len(sys.argv) > 1 else "data/UWTempo/ver3/training.bin"
    ds = LabeledStateDataset(path)
    print(f"Dataset size: {len(ds)}")

    wins = 0
    total = len(ds)
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    for i, (state, action, label) in enumerate(dl):
        # only first 100 bits
        sb = "".join(str(int(x.item())) for x in state[0, :100])
        ai = int(action.item())
        lbl = label.item()
        print(f"State: {sb}, Action: {ai}, Result: {lbl}")
        if lbl > 0:
            wins += 1
        if i >= 999:
            break

    print(f"Winrate: {wins / total:.3f}")
