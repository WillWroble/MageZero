import torch
import torch.nn.functional as F
from pyroaring import BitMap
from torch import nn  # optim is not strictly needed for testing if not optimizing
from torch.utils.data import DataLoader

from dataset import H5Indexed, collate_batch, filter_opponent_states
from model import NetTransformer, load_model, GLOBAL_MAX, ACTIONS_MAX, PRIORITY_A_MAX, PRIORITY_B_MAX, TARGETS_MAX, BINARY_MAX, ActionType, lambda_pA, lambda_pB, lambda_t, lambda_b, normalize_policy_labels

SHOW_CONFUSION_MATRIX = True

mse = nn.MSELoss()
kld = nn.KLDivLoss(reduction='batchmean')

def populate_matrix(matrix, actual, predicted):
    for true_action, pred_action in zip(actual.cpu(), predicted.cpu()):
        matrix[true_action, pred_action] += 1

def print_matrix(matrix):
    print("--- Policy Confusion Matrix (True \\ Predicted) ---")
    matrix_size = matrix.shape[0]

    if matrix_size == 2:
        # Print header for predicted actions
        header = "  True   |" + "".join([f"{j: >8}" for j in range(matrix_size)])
        print(header)
        print("-" * len(header))

        # Print each row for true actions
        for r in range(matrix_size):
            row_str = f"{r: >8} |"
            row_str += "".join([f"{matrix[r, c].item(): >8}" for c in range(matrix_size)])
            print(row_str)
    else:
        # Print header for predicted actions
        indices = [i for i in range(matrix_size) if matrix[i].sum() > 0]
        header = "True |" + "".join([f"{j: >4}" for j in indices])
        print(header)
        print("-" * len(header))

        # Print each row for true actions
        for r in indices:
            row_str = f"{r: >4} |"
            row_str += "".join([f"{matrix[r, c].item(): >4}" for c in indices])
            print(row_str)

    print("-" * 60)
def correct_from_matrix(matrix) -> int:
    return int(matrix.diag().sum().item())


def total_from_matrix(matrix) -> int:
    return int(matrix.sum().item())

def validate(model, dl):
    # Metrics accumulators
    total_decision_examples = 0
    total_combined_loss, total_pA_loss, total_pB_loss, total_t_loss, total_b_loss, total_v_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # Initialize as floats
    pA_matrix = torch.zeros(PRIORITY_A_MAX, PRIORITY_A_MAX, dtype=torch.long)
    pB_matrix = torch.zeros(PRIORITY_B_MAX, PRIORITY_B_MAX, dtype=torch.long)
    t_matrix = torch.zeros(TARGETS_MAX, TARGETS_MAX, dtype=torch.long)
    b_matrix = torch.zeros(BINARY_MAX, BINARY_MAX, dtype=torch.long)
    #model = train.Net(train.GLOBAL_MAX, train.ACTIONS_MAX).cuda()
    model.eval()
    with torch.no_grad():
        for batch_indices, batch_offsets, batch_policy_labels, batch_value_labels, is_players, action_types in dl:
            # Move new input tensors to CUDA
            batch_indices = batch_indices.cuda()
            batch_offsets = batch_offsets.cuda()
            batch_policy_labels = batch_policy_labels.cuda()
            batch_value_labels = batch_value_labels.cuda()
            is_players = is_players.cuda().squeeze(-1).to(torch.bool)
            action_types = action_types.cuda().squeeze(-1).to(torch.long)

            # Model call uses indices and offsets
            priority_logits, opponent_priority_logits, target_logits, binary_logits, value_pred = model(batch_indices,
                                                                                                        batch_offsets)

            nonzero = (batch_policy_labels > 0).sum(dim=1)  # [B]
            decision_mask = nonzero > 0  # [B] states where more than one action is available
            priority_mask = (action_types == ActionType.PRIORITY.value) & is_players & decision_mask
            opponent_priority_mask = (action_types == ActionType.PRIORITY.value) & (~is_players) & decision_mask
            target_mask = (action_types == ActionType.CHOOSE_TARGET.value) & decision_mask
            binary_mask = (action_types == ActionType.CHOOSE_USE.value) & decision_mask


            total_decision_examples += decision_mask.sum().item()

            # priority A
            if priority_mask.any():
                log_probs_d = F.log_softmax(priority_logits[priority_mask][:, :PRIORITY_A_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[priority_mask][:, :PRIORITY_A_MAX])
                lpA = kld(log_probs_d, tgt)*lambda_pA
                s = log_probs_d.size(0)
                total_pA_loss += lpA.item() * s
                populate_matrix(pA_matrix, torch.argmax(tgt, dim=1), torch.argmax(log_probs_d, dim=1))
            else:
                lpA = torch.zeros((), device=value_pred.device)

            # priority B (this uses virtual visits)
            if opponent_priority_mask.any():
                log_probs_d = F.log_softmax(opponent_priority_logits[opponent_priority_mask][:, :PRIORITY_B_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[opponent_priority_mask][:, :PRIORITY_B_MAX])
                lpB = kld(log_probs_d, tgt)*lambda_pB
                s = log_probs_d.size(0)
                total_pB_loss += lpB.item() * s
                populate_matrix(pB_matrix, torch.argmax(tgt, dim=1), torch.argmax(log_probs_d, dim=1))
            else:
                lpB = torch.zeros((), device=value_pred.device)

            # targets (shared between both players)
            if target_mask.any():
                log_probs_d = F.log_softmax(target_logits[target_mask][:, :TARGETS_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[target_mask][:, :TARGETS_MAX])
                lt = kld(log_probs_d, tgt)*lambda_t
                s = log_probs_d.size(0)
                total_t_loss += lt.item() * s
                populate_matrix(t_matrix, torch.argmax(tgt, dim=1), torch.argmax(log_probs_d, dim=1))
            else:
                lt = torch.zeros((), device=value_pred.device)

            # binary (choose to use) decisions
            if binary_mask.any():
                log_probs_d = F.log_softmax(binary_logits[binary_mask][:, :BINARY_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[binary_mask][:, :BINARY_MAX])
                lb = kld(log_probs_d, tgt)*lambda_b
                s = log_probs_d.size(0)
                total_b_loss += lb.item() * s
                populate_matrix(b_matrix, torch.argmax(tgt, dim=1), torch.argmax(log_probs_d, dim=1))
            else:
                lb = torch.zeros((), device=value_pred.device)

            lv = mse(value_pred, batch_value_labels.squeeze(-1))

            total_combined_loss += (lpA + lpB + lt + lb + lv).item()

            total_v_loss += lv.item()


        total_pA_examples, total_pB_examples, total_t_examples, total_b_examples = total_from_matrix(pA_matrix), total_from_matrix(pB_matrix), total_from_matrix(t_matrix), total_from_matrix(b_matrix)
        correct_pA, correct_pB, correct_t, correct_b = correct_from_matrix(pA_matrix), correct_from_matrix(pB_matrix), correct_from_matrix(t_matrix), correct_from_matrix(b_matrix)


        avg_pA_loss = (total_pA_loss / max(total_pA_examples, 1))
        avg_pB_loss = (total_pB_loss / max(total_pB_examples, 1))
        avg_t_loss = (total_t_loss / max(total_t_examples, 1))
        avg_b_loss = (total_b_loss / max(total_b_examples, 1))
        avg_v_loss = total_v_loss / len(dl)

        avg_combined_loss = total_combined_loss / len(dl)

        print(f"Validation loss:  priority_A_loss={avg_pA_loss:.3f}  priority_B_loss={avg_pB_loss:.3f} choose_target_loss={avg_t_loss:.3f} choose_use_loss={avg_b_loss:.3f} value_loss={avg_v_loss:.3f} avg_total_loss={avg_combined_loss:.3f} decision_states={total_decision_examples}")


        if total_pA_examples > 0:
            print(f"Test priority_A_accuracy={correct_pA / total_pA_examples:.3f}")
            if SHOW_CONFUSION_MATRIX:
                print_matrix(pA_matrix)
        else:
            print("No priority A samples in test set to calculate accuracy.")
        if total_pB_examples > 0:
            print(f"Test priority_B_accuracy={correct_pB / total_pB_examples:.3f}")
            if SHOW_CONFUSION_MATRIX:
                print_matrix(pB_matrix)
        else:
            print("No priority B samples in test set to calculate accuracy.")
        if total_t_examples > 0:
            print(f"Test choose_target_accuracy={correct_t / total_t_examples:.3f}")
            if SHOW_CONFUSION_MATRIX:
                print_matrix(t_matrix)
        else:
            print("No target samples in test set to calculate accuracy.")
        if total_b_examples > 0:
            print(f"Test choose_use_accuracy={correct_b / total_b_examples:.3f}")
            if SHOW_CONFUSION_MATRIX:
                print_matrix(b_matrix)
        else:
            print("No choose_use samples in test set to calculate accuracy.")

        return avg_combined_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--opponent-head", action="store_true")
    args = parser.parse_args()

    ignore_path = f"models/{args.deck}/ver{args.version}/ignore.roar"
    with open(ignore_path, "rb") as f:
        ignore = BitMap.deserialize(f.read())
    print(f"ignore list size: {len(ignore)}")

    ds = H5Indexed(f"data/{args.deck}/ver{args.version}/testing", set(ignore))
    if not args.opponent_head:
        ds = filter_opponent_states(ds, TARGETS_MAX)

    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=0,
                    collate_fn=collate_batch, pin_memory=True, persistent_workers=False)

    model = NetTransformer(GLOBAL_MAX, ACTIONS_MAX).cuda()
    model.eval()

    checkpoint_path = f"models/{args.deck}/ver{args.version}/model.pt.gz"
    try:
        checkpoint = load_model(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from {checkpoint_path}")
    except FileNotFoundError:
        print(f"ERROR: Checkpoint not found at {checkpoint_path}. Testing with uninitialized model.")
    except Exception as e:
        print(f"ERROR: Could not load checkpoint: {e}. Testing with uninitialized model.")

    validate(model, dl)