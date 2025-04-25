# dataset.py
import struct
import numpy as np
import torch
from torch.utils.data import Dataset

import struct
import numpy as np
import torch
from torch.utils.data import Dataset

class LabeledStateDataset(Dataset):
    """
    Loads labeled state data from a binary file written by your Java extractor.
    File format:
      - 3 × 4-byte big-endian ints: n (records), S (state dim), A (action dim)
      - n × (S bytes) for state vectors (uint8 0/1)
      - n × (A bytes) for action vectors (uint8 0/1)
      - n × 1 byte for result label (uint8 0/1)
    """
    def __init__(self, path):
        with open(path, "rb") as f:
            header = f.read(12)
            if len(header) < 12:
                raise IOError(f"File too small to contain header: {path}")
            n, S, A = struct.unpack(">iii", header)  # Java DataOutputStream big-endian
            raw = np.frombuffer(f.read(), dtype=np.uint8)

        rec_size = S + A + 1
        if raw.size != n * rec_size:
            raise ValueError(
                f"Expected {n}×{rec_size}={n*rec_size} payload bytes, got {raw.size}"
            )

        data = raw.reshape(n, rec_size)

        # Raw bit-vectors (0/1) as float tensors
        self.states  = torch.from_numpy(data[:, :S].astype(np.int8)).float()
        self.actions = torch.from_numpy(data[:, S:S+A].astype(np.int8)).float()
        self.labels  = torch.from_numpy(data[:, -1].astype(np.int8)).float()

    def __len__(self):
        return int(self.states.size(0))

    def __getitem__(self, idx):
        # Returns: state-vector (S), action-vector (A), label
        return self.states[idx], self.actions[idx], self.labels[idx]


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    ds = LabeledStateDataset("data/training.bin")
    dl = DataLoader(ds, batch_size=1, shuffle=False)
    for i, (state, action, label) in enumerate(dl):
        s = state.squeeze(0)
        a = action.squeeze(0)
        # convert floats back to 0/1 ints
        sb = "".join(str(int(b.item())) for b in s[:100])
        ab = "".join(str(int(b.item())) for b in a[:100])
        res = "true" if int(label.item()) == 1 else "false"
        print(f"State: {sb}, Action: {ab}, Result: {res}")
        if i >= 999:
            break



