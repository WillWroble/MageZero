# dataset.py
import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

class LabeledStateDataset(Dataset):
    """
    Loads labeled state data from a binary file written by your Java extractor.
    File format (per record):
      - 3 × 4-byte big-endian ints: n (records), S (state dim), A (action dim)
      - S bytes per record: state vector (uint8 0/1)
      - A bytes per record: action vector (uint8 0/1)
      - 8-byte big-endian float per record: result label (float64 in [-1,1])
    """
    def __init__(self, path):
        with open(path, "rb") as f:
            header = f.read(12)
            if len(header) < 12:
                raise IOError(f"File too small to contain header: {path}")
            n, S, A = struct.unpack(">iii", header)

            # preallocate numpy arrays
            states_arr  = np.empty((n, S), dtype=np.uint8)
            actions_arr = np.empty((n, A), dtype=np.uint8)
            labels_arr  = np.empty((n,),   dtype=np.float32)

            for i in range(n):
                # read state bits
                buf = f.read(S)
                if len(buf) != S:
                    raise IOError(f"Failed to read state bits for record {i}")
                states_arr[i] = np.frombuffer(buf, dtype=np.uint8)

                # read action bits
                buf = f.read(A)
                if len(buf) != A:
                    raise IOError(f"Failed to read action bits for record {i}")
                actions_arr[i] = np.frombuffer(buf, dtype=np.uint8)

                # read the float64 label
                buf = f.read(8)
                if len(buf) != 8:
                    raise IOError(f"Failed to read label for record {i}")
                (dbl_val,) = struct.unpack(">d", buf)
                labels_arr[i] = dbl_val

        # convert to torch tensors
        self.states  = torch.from_numpy(states_arr.astype(np.float32))
        self.actions = torch.from_numpy(actions_arr.astype(np.float32))
        self.labels  = torch.from_numpy(labels_arr)

    def __len__(self):
        return self.states.size(0)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx], self.labels[idx]

if __name__ == "__main__":
    ds = LabeledStateDataset("data/training.bin")
    print(f"Dataset size: {len(ds)}")
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    for i, (state, action, label) in enumerate(dl):
        sb = "".join(str(int(x.item())) for x in state[0, :100])
        ab = "".join(str(int(x.item())) for x in action[0, :100])
        print(f"State: {sb}, Action: {ab}, Result: {label.item()}")
        if i >= 999:
            break
