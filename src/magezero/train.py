from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import gzip
import shutil

import test
from model import NetTransformer, Net, load_model, GLOBAL_MAX, ACTIONS_MAX, PRIORITY_A_MAX, PRIORITY_B_MAX, TARGETS_MAX, BINARY_MAX, ActionType, lambda_pA, lambda_pB, lambda_t, lambda_b, normalize_policy_labels
from dataset import H5Indexed, collate_batch,  create_redundancy_ignore_list, filter_opponent_states
from pyroaring import BitMap

#add training data under: data/{deck name}/ver{your version num}/training/{your data}.hdf5



def train(
        deck: str,
        version: int,
        epochs: int,
        use_checkpoint: bool = False,
        make_ignore_list: bool = True,
        train_opponent_head: bool = True,
):
    os.makedirs(f"models/{deck}/ver{version}", exist_ok=True)
    ds_raw = H5Indexed(f"data/{deck}/ver{version}/training")


    #ignore handling
    print("Generating ignore list from dataset to use for model")
    ignore_list = create_redundancy_ignore_list(ds_raw)

    # model and data loaders
    model = NetTransformer(GLOBAL_MAX, ACTIONS_MAX).cuda()

    # optional start point
    if use_checkpoint:
        checkpoint_path = f"models/{deck}/ver{version}/model.pt.gz"
        try:
            #checkpoint = torch.load(checkpoint_path, map_location="cuda")
            checkpoint = load_model(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            with open(f"models/{deck}/ver{version}/ignore.roar", "rb") as f:
                ignore_list2 = BitMap.deserialize(f.read())
                ignore_list.intersection_update(ignore_list2)
                #ignore_list = ignore_list2
            print(f"intersected with previous ignore list: {len(ignore_list2)} for final ignore list: {len(ignore_list)} leaving {GLOBAL_MAX-len(ignore_list)} features")
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        except FileNotFoundError:
            print(f"INFO: Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
        except Exception as e:
            print(f"ERROR: Could not load checkpoint. {e}. Starting from scratch.")

    if not make_ignore_list: ignore_list = []
    print("Saving ignore list to ignore.roar")

    ignore = BitMap(ignore_list)  # iterable of ints
    with open(f"models/{deck}/ver{version}/ignore.roar", "wb") as f:
        f.write(ignore.serialize())

    #data sets with redundant filter
    ds = H5Indexed(f"data/{deck}/ver{version}/training", ignore_list)
    test_ds = H5Indexed(f"data/{deck}/ver{version}/testing", ignore_list)

    #if round-robin filter out opponent states AFTER making the ignore list
    if not train_opponent_head:
        ds = filter_opponent_states(ds,TARGETS_MAX)
        test_ds = filter_opponent_states(test_ds,TARGETS_MAX)



    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=0, collate_fn=collate_batch,
                    pin_memory=True, persistent_workers=False)

    dl_test = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=0, collate_fn=collate_batch,
                    pin_memory=True, persistent_workers=False)

    test.SHOW_CONFUSION_MATRIX = False

    #optimizers
    #opt_sparse = optim.SparseAdam(model.embedding_bag.parameters(), lr=1e-4)
    opt_sparse = optim.SparseAdam(model.embedding.parameters(), lr=1e-4)
    dense_params = [p for n, p in model.named_parameters()
                    if "embedding" not in n or "transformer" in n]
    opt_dense = optim.Adam(dense_params, lr=5e-4)

    mse = nn.MSELoss()
    kld = nn.KLDivLoss(reduction='batchmean')

    #stats
    best_val_loss = float('inf')

    #main training loop
    for epoch in range(1, epochs+1):
        total_pA_loss, total_pB_loss, total_t_loss, total_b_loss, total_v_loss, total_l1_sparse_loss, total_l1_dense_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        total_decision_examples, total_pA_examples, total_pB_examples, total_t_examples, total_b_examples = 0,0,0,0,0
        model.train()

        for batch_indices, batch_offsets, batch_policy_labels, batch_value_labels, is_players, action_types in dl:
            # Move new input tensors to CUDA
            batch_indices = batch_indices.cuda()
            batch_offsets = batch_offsets.cuda()
            batch_policy_labels = batch_policy_labels.cuda()
            batch_value_labels = batch_value_labels.cuda()
            is_players = is_players.cuda().squeeze(-1).to(torch.bool)
            action_types = action_types.cuda().squeeze(-1).to(torch.long)

            # Model call uses indices and offsets
            priority_logits, opponent_priority_logits, target_logits, binary_logits ,value_pred = model(batch_indices, batch_offsets)



            nonzero = (batch_policy_labels > 0).sum(dim=1)  # [B]
            decision_mask = nonzero > 0  # [B] states where at least one action is available
            priority_mask = (action_types==ActionType.PRIORITY.value) & is_players & decision_mask
            opponent_priority_mask = (action_types==ActionType.PRIORITY.value) & (~is_players) & decision_mask
            target_mask = (action_types==ActionType.CHOOSE_TARGET.value) & decision_mask
            binary_mask = (action_types==ActionType.CHOOSE_USE.value) & decision_mask


            total_decision_examples += decision_mask.sum().item()

            #priority A
            if priority_mask.any():
                log_probs_d = F.log_softmax(priority_logits[priority_mask][:,:PRIORITY_A_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[priority_mask][:,:PRIORITY_A_MAX])
                lpA = kld(log_probs_d, tgt)*lambda_pA
                s = log_probs_d.size(0)
                total_pA_loss += lpA.item() * s
                total_pA_examples += s
            else:
                lpA = torch.zeros((), device=value_pred.device)

            #priority B (this uses virtual visits)
            if opponent_priority_mask.any():
                log_probs_d = F.log_softmax(opponent_priority_logits[opponent_priority_mask][:,:PRIORITY_B_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[opponent_priority_mask][:,:PRIORITY_B_MAX])
                lpB = kld(log_probs_d, tgt)*lambda_pB
                s = log_probs_d.size(0)
                total_pB_loss += lpB.item() * s
                total_pB_examples += s
            else:
                lpB = torch.zeros((), device=value_pred.device)

            #targets (shared between both players)
            if target_mask.any():
                log_probs_d = F.log_softmax(target_logits[target_mask][:,:TARGETS_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[target_mask][:,:TARGETS_MAX])
                lt = kld(log_probs_d, tgt)*lambda_t
                s = log_probs_d.size(0)
                total_t_loss += lt.item() * s
                total_t_examples += s
            else:
                lt = torch.zeros((), device=value_pred.device)

            # binary (choose to use) decisions
            if binary_mask.any():
                log_probs_d = F.log_softmax(binary_logits[binary_mask][:,:BINARY_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[binary_mask][:,:BINARY_MAX])
                lb = kld(log_probs_d, tgt)*lambda_b
                s = log_probs_d.size(0)
                total_b_loss += lb.item() * s
                total_b_examples += s
            else:
                lb = torch.zeros((), device=value_pred.device)

            lv = mse(value_pred, batch_value_labels.squeeze(-1))
            l1_dense = 0
            for param in dense_params:
                l1_dense += torch.sum(torch.abs(param))
            l1_dense *= 1e-5

            total_l1_dense_loss += l1_dense.item()
            #total_l1_sparse_loss += model.l1_penalty.item()
            loss = lpA + lpB + lt + lb + lv #+ model.l1_penalty #+ l1_dense
            opt_sparse.zero_grad()
            opt_dense.zero_grad()
            loss.backward()
            opt_sparse.step()
            opt_dense.step()


            total_v_loss += lv.item()

        avg_pA_loss = (total_pA_loss / max(total_pA_examples, 1))
        avg_pB_loss = (total_pB_loss / max(total_pB_examples, 1))
        avg_t_loss = (total_t_loss / max(total_t_examples, 1))
        avg_b_loss = (total_b_loss / max(total_b_examples, 1))
        avg_v_loss = total_v_loss / len(dl)
        avg_l1_dense_loss = total_l1_dense_loss / len(dl)
        avg_l1_sparse_loss = total_l1_sparse_loss / len(dl)
        print(f"Epoch {epoch}  priority_A_loss={avg_pA_loss:.3f}  priority_B_loss={avg_pB_loss:.3f} choose_target_loss={avg_t_loss:.3f} choose_use_loss={avg_b_loss:.3f} value_loss={avg_v_loss:.3f} "
              f"l1_dense={avg_l1_dense_loss} l1_sparse={avg_l1_sparse_loss} decision_states={total_decision_examples}")
        #run current model on testing set (if there is one)
        if len(test_ds)>0:
            val_loss = test.validate(model, dl_test)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                checkpoint_save_path = f"models/{deck}/ver{version}/best.pt.gz"
                temp_path = checkpoint_save_path.replace('.gz', '.tmp')

                # Save uncompressed
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_sparse_state_dict': opt_sparse.state_dict(),
                    'optimizer_dense_state_dict': opt_dense.state_dict(),
                    'avg_p_loss': avg_pA_loss,
                    'avg_v_loss': avg_v_loss,
                }, temp_path)

                # Stream-compress in chunks (constant memory)
                with open(temp_path, 'rb') as f_in:
                    with gzip.open(checkpoint_save_path, 'wb', compresslevel=1) as f_out:
                        shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)  # 16MB chunks

                os.remove(temp_path)

        #TODO: make validation based checkpoint schedule
        checkpoint_save_path = f"models/{deck}/ver{version}/model.pt.gz"
        temp_path = checkpoint_save_path.replace('.gz', '.tmp')

        # Save uncompressed
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_sparse_state_dict': opt_sparse.state_dict(),
            'optimizer_dense_state_dict': opt_dense.state_dict(),
            'avg_p_loss': avg_pA_loss,
            'avg_v_loss': avg_v_loss,
        }, temp_path)

        # Stream-compress in chunks (constant memory)
        with open(temp_path, 'rb') as f_in:
            with gzip.open(checkpoint_save_path, 'wb', compresslevel=1) as f_out:
                shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)  # 16MB chunks

        os.remove(temp_path)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    train(args.deck, args.version, args.epochs, args.checkpoint)