# dataset.py

import struct
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class LabeledStateDataset(Dataset):
    """
    Loads labeled state data from a binary file written by your Java extractor.
    New File header format:
      - 3 × 4-byte big-endian ints: n (records), S (GLOBAL_VOCAB_SIZE), A (action dim)
      - Per record:
        - 4-byte int: num_active_indices
        - num_active_indices × 4-byte ints: active_indices
        - A × 8-byte big-endian floats: action distribution
        - 8-byte big-endian float: result label
    """

    def __init__(self, path):
        self.samples = []
        self.S = 0  # Will store the global vocabulary size (total number of possible features)
        self.A = 0  # Will store the action dimension

        with open(path, "rb") as f:
            # Read the new 12-byte header
            header = f.read(12)
            if len(header) < 12:
                raise IOError(f"File too small for header: {path}")

            # Unpack only the 3 integers we need
            n, self.S, self.A = struct.unpack(">iii", header)

            for i in range(n):
                # 1) Read the number of active indices for this state
                num_indices_bytes = f.read(4)
                if len(num_indices_bytes) < 4:
                    raise IOError(f"Unexpected EOF when reading num_indices at record {i}")
                (num_indices,) = struct.unpack(">i", num_indices_bytes)

                # 2) Read that many indices
                indices_bytes = f.read(num_indices * 4)
                if len(indices_bytes) < num_indices * 4:
                    raise IOError(f"Unexpected EOF when reading indices at record {i}")
                indices = struct.unpack(f">{num_indices}i", indices_bytes)

                # 3) Read action-distribution
                action_bytes = f.read(self.A * 8)
                if len(action_bytes) < self.A * 8:
                    raise IOError(f"Unexpected EOF when reading action-vector at record {i}")
                actions = struct.unpack(f">{self.A}d", action_bytes)

                # 4) Read result label
                label_bytes = f.read(8)
                if len(label_bytes) < 8:
                    raise IOError(f"Unexpected EOF when reading label at record {i}")
                (label,) = struct.unpack(">d", label_bytes)

                # Store the processed sample
                self.samples.append({
                    "indices": torch.LongTensor(indices),
                    "policy": torch.FloatTensor(actions),
                    "value": torch.FloatTensor([label])
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Return the pre-processed sample dictionary
        sample = self.samples[idx]
        # The collate_fn will handle these individual pieces
        return sample["indices"], sample["policy"], sample["value"]


# This collate_fn is the same one we designed previously.
# It is now ESSENTIAL for batching your variable-length data.
def collate_batch(batch):
    indices_list, policy_list, value_list = [], [], []
    for (_indices, _policy, _value) in batch:
        indices_list.append(_indices)
        policy_list.append(_policy)
        value_list.append(_value)

    offsets = [0] + [len(indices) for indices in indices_list]
    offsets = torch.cumsum(torch.LongTensor(offsets), dim=0)[:-1]

    indices = torch.cat(indices_list)
    policies = torch.stack(policy_list)
    values = torch.stack(value_list)

    return indices, offsets, policies, values


if __name__ == "__main__":
    import sys
    import torch  # Make sure torch is imported if not already at the top
    from torch.utils.data import DataLoader  # Make sure DataLoader is imported

    # Assuming LabeledStateDataset and collate_batch are defined above in the same file

    path = sys.argv[1] if len(sys.argv) > 1 else "data/UWTempo2/ver1/training.bin"  # Ensure this path is correct
    ds = LabeledStateDataset(path)
    print(f"Dataset size: {len(ds)}")
    print(f"Global Vocabulary Size (S) from header: {ds.S}")
    print(f"Action Dimension (A) from header: {ds.A}")

    wins = 0
    num_samples_to_process = 0

    # The DataLoader MUST be given the custom collate function
    # Using batch_size=1 for the test loop to easily inspect one sample at a time
    dl = DataLoader(ds, batch_size=1, shuffle=False, collate_fn=collate_batch)

    for i, (indices_batch, offsets_batch, policies_batch, values_batch) in enumerate(dl):
        num_samples_to_process += 1

        # Get the active global indices for the current (first and only) sample in the batch.
        # For batch_size=1, offsets_batch[0] is 0.
        # The indices for this single sample are the entire indices_batch.
        current_sample_indices = indices_batch

        # --- MODIFIED PRINTING LOGIC FOR STATE ---
        # Print the first 100 active global indices, separated by a space.
        # If fewer than 100 indices, print all of them.
        indices_to_print = current_sample_indices.tolist()  # Convert tensor to list
        if len(indices_to_print) > 100:
            sb = " ".join(map(str, indices_to_print[:100])) + " ..."  # Indicate if truncated
        else:
            sb = " ".join(map(str, indices_to_print))
        if not indices_to_print:  # Handle case with no active features
            sb = "[No active features]"
        # --- END OF MODIFIED PRINTING LOGIC FOR STATE ---

        action = policies_batch[0]  # Get the policy for the first (and only) sample
        av = action.tolist()  # full A-length action vector

        label_tensor = values_batch[0]  # Get the value for the first (and only) sample
        lbl = label_tensor.item()

        print(f"State: {sb}, Action: {av}, Result: {lbl}")

        if lbl > 0:
            wins += 1
        if i >= 999:  # Stop after 1000 samples as in your original code
            break

    if num_samples_to_process > 0:
        print(f"Winrate (over {num_samples_to_process} samples): {wins / num_samples_to_process:.3f}")
    else:
        print("No samples processed.")