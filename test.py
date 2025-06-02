import torch
from torch import nn  # optim is not strictly needed for testing if not optimizing
from torch.utils.data import DataLoader

# If Net is also in dataset.py or a separate model.py, adjust import accordingly.
import train  # Or from train import Net, ACTIONS_MAX (if ACTIONS_MAX is defined there)
from dataset import LabeledStateDataset, collate_batch  # CRITICAL: Import collate_batch


def test():

    ds = LabeledStateDataset("data/UWTempo2/ver1/training.bin")

    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=4, collate_fn=collate_batch)

    # 3. Model instantiation uses global vocab size (ds.S) and action size
    model = train.Net(train.GLOBAL_MAX, train.ACTIONS_MAX).cuda()  # should be GLOBAL_VOCAB_SIZE

    checkpoint_path = "models/ckpt_20.pt"  # Make sure this is the correct checkpoint
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location="cuda"))
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"ERROR: Checkpoint file not found at {checkpoint_path}. Testing with an uninitialized model.")
    except Exception as e:
        print(f"ERROR: Could not load checkpoint. {e}. Testing with an uninitialized model.")

    model.eval()  # Set model to evaluation mode

    ce = nn.CrossEntropyLoss()
    mse = nn.MSELoss()

    # Metrics accumulators
    correct_policy_preds, total_policy_samples = 0, 0
    total_policy_loss, total_value_loss = 0.0, 0.0

    with torch.no_grad():  # Essential for testing to disable gradient calculations
        # 4. Update the loop to unpack all parts returned by collate_batch
        for batch_indices, batch_offsets, batch_policy_labels, batch_value_labels in dl:
            # 5. Move new input tensors to CUDA
            batch_indices = batch_indices.cuda()
            batch_offsets = batch_offsets.cuda()
            batch_policy_labels = batch_policy_labels.cuda()
            batch_value_labels = batch_value_labels.cuda()

            # 6. Model call uses indices and offsets
            policy_logits, value_pred = model(batch_indices, batch_offsets)

            # Assuming batch_policy_labels are MCTS visit counts/probabilities (distributions).
            policy_target_indices = torch.argmax(batch_policy_labels, dim=1)
            lp = ce(policy_logits, policy_target_indices)

            # Ensure batch_value_labels has the same shape as value_pred ([batch_size])
            lv = mse(value_pred, batch_value_labels.squeeze(-1))
            total_policy_loss += lp.item() * batch_policy_labels.size(0)  # Weighted by actual batch size if last batch is smaller
            total_value_loss += lv.item() * batch_policy_labels.size(0)  # Weighted by actual batch size

            # Accuracy calculation
            predicted_actions = torch.argmax(policy_logits, dim=1)
            # policy_target_indices are already the true action indices
            correct_policy_preds += (predicted_actions == policy_target_indices).sum().item()
            total_policy_samples += batch_policy_labels.size(0)  # Count actual samples in the batch
    avg_policy_loss = total_policy_loss / total_policy_samples
    avg_value_loss = total_value_loss / total_policy_samples  # Denominator should be total_policy_samples or len(ds)

    print(f"Test policy_loss={avg_policy_loss:.3f}  value_loss={avg_value_loss:.3f}")
    if total_policy_samples > 0:
        print(f"Test policy_accuracy={correct_policy_preds / total_policy_samples:.3f}")
    else:
        print("No samples in test set to calculate accuracy.")


if __name__ == "__main__":
    test()