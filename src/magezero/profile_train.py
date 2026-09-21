from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import os

import test
from model import NetTransformer, Net, load_model, GLOBAL_MAX, ACTIONS_MAX, PRIORITY_A_MAX, PRIORITY_B_MAX, TARGETS_MAX, BINARY_MAX, ActionType, lambda_pA, lambda_pB, lambda_t, lambda_b, normalize_policy_labels
from dataset import H5Indexed, collate_batch,  create_redundancy_ignore_list, filter_opponent_states
from pyroaring import BitMap
from torch.profiler import profile, ProfilerActivity, schedule
from datetime import datetime



#add training data under: data/{deck name}/ver{your version num}/training/{your data}.hdf5



def profile_train(
        deck: str,
        version: int,
        steps: int,
        use_checkpoint: bool = False,
        train_opponent_head: bool = False,
):
    os.makedirs(f"models/{deck}/ver{version}", exist_ok=True)
    os.makedirs("./traces", exist_ok=True)
    ds_raw = H5Indexed(f"data/{deck}/ver{version}/training")


    #ignore handling
    print("Generating ignore list from dataset to use for model")
    ignore_list = create_redundancy_ignore_list(ds_raw)

    # model and data loaders
    model = NetTransformer().cuda()

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


    #data sets with redundant filter
    ds = H5Indexed(f"data/{deck}/ver{version}/training", ignore_list, fold_bins=GLOBAL_MAX)

    #if round-robin filter out opponent states AFTER making the ignore list
    if not train_opponent_head:
        ds = filter_opponent_states(ds,TARGETS_MAX)



    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0, collate_fn=collate_batch,
                    pin_memory=True, persistent_workers=False)


    test.SHOW_CONFUSION_MATRIX = False

    #optimizers
    #opt_sparse = optim.SparseAdam(model.embedding.parameters(), lr=1e-4)
    #dense_params = [p for n, p in model.named_parameters()
    #                if "embedding" not in n or "transformer" in n]
    #opt_dense = optim.Adam(dense_params, lr=5e-4)
    opt_dense = optim.Adam(model.parameters(), lr=1e-4)


    mse = nn.MSELoss()
    kld = nn.KLDivLoss(reduction='batchmean')
    scaler = torch.amp.GradScaler()

    #single epoch loop
    i = 0
    with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=2, warmup=2, active=6, repeat=1),
            on_trace_ready=lambda p: p.export_chrome_trace(f"./traces/{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.json"),
            record_shapes=True,
            with_stack=True
    ) as prof:
        for batch_indices, batch_offsets, batch_policy_labels, batch_value_labels, is_players, action_types in dl:
            # Move new input tensors to CUDA
            batch_indices = batch_indices.cuda()
            batch_offsets = batch_offsets.cuda()
            batch_policy_labels = batch_policy_labels.cuda()
            batch_value_labels = batch_value_labels.cuda()
            is_players = is_players.cuda().squeeze(-1).to(torch.bool)
            action_types = action_types.cuda().squeeze(-1).to(torch.long)

            # Model call uses indices and offsets
            with torch.amp.autocast('cuda'):
                priority_logits, opponent_priority_logits, target_logits, binary_logits ,value_pred = model(batch_indices, batch_offsets)

                nonzero = (batch_policy_labels > 0).sum(dim=1)  # [B]
                decision_mask = nonzero > 0  # [B] states where at least one action is available
                priority_mask = (action_types==ActionType.PRIORITY.value) & is_players & decision_mask
                opponent_priority_mask = (action_types==ActionType.PRIORITY.value) & (~is_players) & decision_mask
                target_mask = (action_types==ActionType.CHOOSE_TARGET.value) & decision_mask
                binary_mask = (action_types==ActionType.CHOOSE_USE.value) & decision_mask


                #priority A
                log_probs_d = F.log_softmax(priority_logits[priority_mask][:,:PRIORITY_A_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[priority_mask][:,:PRIORITY_A_MAX])
                lpA = torch.nan_to_num(kld(log_probs_d, tgt)*lambda_pA)


                #priority B
                log_probs_d = F.log_softmax(opponent_priority_logits[opponent_priority_mask][:,:PRIORITY_B_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[opponent_priority_mask][:,:PRIORITY_B_MAX])
                lpB = torch.nan_to_num(kld(log_probs_d, tgt)*lambda_pB)


                #targets (shared between both players)
                log_probs_d = F.log_softmax(target_logits[target_mask][:,:TARGETS_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[target_mask][:,:TARGETS_MAX])
                lt = torch.nan_to_num(kld(log_probs_d, tgt)*lambda_t)

                # binary (choose to use) decisions
                log_probs_d = F.log_softmax(binary_logits[binary_mask][:,:BINARY_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[binary_mask][:,:BINARY_MAX])
                lb = torch.nan_to_num(kld(log_probs_d, tgt)*lambda_b)


                lv = mse(value_pred, batch_value_labels.squeeze(-1))

                loss = lpA + lpB + lt + lb + lv
            #opt_sparse.zero_grad()
            opt_dense.zero_grad()
            #loss.backward()
            #opt_sparse.step()
            #opt_dense.step()
            scaler.scale(loss).backward()
            scaler.step(opt_dense)
            scaler.update()
            prof.step()
            i+=1
            if i > steps:
                break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, default=1)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--checkpoint", action="store_true")
    args = parser.parse_args()
    profile_train(args.deck, args.version, args.steps, args.checkpoint)