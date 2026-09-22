from enum import Enum

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import DataLoader
import os
import gzip
import shutil

import test
from model import NetTransformer, load_model, GLOBAL_MAX, PRIORITY_A_MAX, PRIORITY_B_MAX, TARGETS_MAX, BINARY_MAX, ActionType, lambda_pA, lambda_pB, lambda_t, lambda_b, normalize_policy_labels
from dataset import H5Indexed, collate_batch, filter_opponent_states
from vocab import FeatureVocab, initial_rows, kept_feature_ids

#add training data under: data/{deck name}/ver{your version num}/training/{your data}.hdf5



def train(
        deck: str,
        version: int,
        epochs: int,
        steps: int = 15000,
        use_checkpoint: bool = False,
        train_opponent_head: bool = False,
):
    os.makedirs(f"models/{deck}/ver{version}", exist_ok=True)
    vocab, model = prepare_dense_vocab(deck, version, use_checkpoint)
    ds = H5Indexed(f"data/{deck}/ver{version}/training", vocab=vocab)
    test_ds = H5Indexed(f"data/{deck}/ver{version}/testing", vocab=vocab)


    #if round-robin filter out opponent states AFTER making the ignore list
    if not train_opponent_head:
        ds = filter_opponent_states(ds,TARGETS_MAX)
        test_ds = filter_opponent_states(test_ds,TARGETS_MAX)

    train_loop(deck, version, epochs, steps, model, ds, test_ds, vocab)


def prepare_dense_vocab(deck: str, version: int, use_checkpoint: bool):
    """Feature vocab = previous checkpoint's vocab (rows unchanged) + newly kept ids appended.
    The embedding table has one row per vocab entry."""
    print("Building feature vocab from dataset (ignore-list rule over observed ids)")
    ds_raw = H5Indexed(f"data/{deck}/ver{version}/training")
    kept = kept_feature_ids(ds_raw.indices_t.numpy(), ds_raw.idxptr_t.numpy())
    vocab, state = FeatureVocab(feature_hash_bins=GLOBAL_MAX), None
    if use_checkpoint:
        checkpoint_path = f"models/{deck}/ver{version}/model.pt.gz"
        try:
            checkpoint = load_model(checkpoint_path)
            if "feature_vocab" not in checkpoint:
                raise ValueError(f"{checkpoint_path} has no feature vocab (full-table model). ")
            vocab = FeatureVocab.from_state_dict(checkpoint["feature_vocab"])
            vocab.require_encoding(GLOBAL_MAX)
            state = checkpoint["model_state_dict"]
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        except FileNotFoundError:
            print(f"INFO: Checkpoint file not found at {checkpoint_path}. Starting from scratch.")
    prev_rows = len(vocab)
    added = vocab.extend(kept)
    model = NetTransformer(num_embeddings=max(prev_rows, 1) if state is not None else len(vocab))
    dim = model.embedding.embedding_dim
    if state is not None:
        model.load_state_dict(state)
        # rows the appended features would have started from in any run
        model.resize_embedding(len(vocab), initial_rows(vocab.ids[prev_rows:], dim))
    else:
        with torch.no_grad():
            model.embedding.weight.copy_(torch.from_numpy(initial_rows(vocab.ids, dim)))
    print(f"feature vocab: {len(kept)} kept ids in this dataset, {prev_rows} rows from checkpoint, "
          f"{added} added -> {len(vocab)} embedding rows")
    return vocab, model.cuda()



def train_loop(deck, version, epochs, steps, model, ds, test_ds, vocab):


    dl = DataLoader(ds, batch_size=512, shuffle=True, num_workers=0, collate_fn=collate_batch,
                    pin_memory=True, persistent_workers=False)

    dl_test = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0, collate_fn=collate_batch,
                    pin_memory=True, persistent_workers=False)

    test.SHOW_CONFUSION_MATRIX = False

    #optimizers
    opt_dense = optim.Adam(model.parameters(), lr=1e-4)

    mse = nn.MSELoss()
    kld = nn.KLDivLoss(reduction='batchmean')
    scaler = torch.amp.GradScaler()


    #stats
    best_val_loss = float('inf')

    #main training loop
    for epoch in range(1, epochs+1):
        total_pA_loss, total_pB_loss, total_t_loss, total_b_loss, total_v_loss, total_l1_sparse_loss, total_l1_dense_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        total_decision_examples, total_pA_examples, total_pB_examples, total_t_examples, total_b_examples = 0,0,0,0,0
        model.train()
        step = 0

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


                total_decision_examples += decision_mask.sum().item()

                #priority A
                log_probs_d = F.log_softmax(priority_logits[priority_mask][:,:PRIORITY_A_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[priority_mask][:,:PRIORITY_A_MAX])
                lpA = torch.nan_to_num(kld(log_probs_d, tgt)*lambda_pA)
                s = log_probs_d.size(0)
                total_pA_loss += lpA.item() * s
                total_pA_examples += s


                #priority B
                log_probs_d = F.log_softmax(opponent_priority_logits[opponent_priority_mask][:,:PRIORITY_B_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[opponent_priority_mask][:,:PRIORITY_B_MAX])
                lpB = torch.nan_to_num(kld(log_probs_d, tgt)*lambda_pB)
                s = log_probs_d.size(0)
                total_pB_loss += lpB.item() * s
                total_pB_examples += s

                #targets (shared between both players)
                log_probs_d = F.log_softmax(target_logits[target_mask][:,:TARGETS_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[target_mask][:,:TARGETS_MAX])
                lt = torch.nan_to_num(kld(log_probs_d, tgt)*lambda_t)
                s = log_probs_d.size(0)
                total_t_loss += lt.item() * s
                total_t_examples += s


                # binary (choose to use) decisions
                log_probs_d = F.log_softmax(binary_logits[binary_mask][:,:BINARY_MAX], dim=1)
                tgt = normalize_policy_labels(batch_policy_labels[binary_mask][:,:BINARY_MAX])
                lb = torch.nan_to_num(kld(log_probs_d, tgt)*lambda_b)
                s = log_probs_d.size(0)
                total_b_loss += lb.item() * s
                total_b_examples += s


                lv = mse(value_pred, batch_value_labels.squeeze(-1))

                loss = lpA + lpB + lt + lb + lv

            opt_dense.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt_dense)
            scaler.update()


            total_v_loss += lv.item()
            step += 1
            if step > steps:
                break


        avg_pA_loss = (total_pA_loss / max(total_pA_examples, 1))
        avg_pB_loss = (total_pB_loss / max(total_pB_examples, 1))
        avg_t_loss = (total_t_loss / max(total_t_examples, 1))
        avg_b_loss = (total_b_loss / max(total_b_examples, 1))
        avg_v_loss = total_v_loss / min(len(dl),steps)
        avg_l1_dense_loss = total_l1_dense_loss / min(len(dl), steps)
        avg_l1_sparse_loss = total_l1_sparse_loss / min(len(dl), steps)
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
                    'optimizer_dense_state_dict': opt_dense.state_dict(),
                    'avg_p_loss': avg_pA_loss,
                    'avg_v_loss': avg_v_loss,
                    'feature_vocab': vocab.state_dict(),
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
            'optimizer_dense_state_dict': opt_dense.state_dict(),
            'avg_p_loss': avg_pA_loss,
            'avg_v_loss': avg_v_loss,
            'feature_vocab': vocab.state_dict(),
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
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--checkpoint", action="store_true")

    args = parser.parse_args()
    train(args.deck, args.version, args.epochs, args.steps, args.checkpoint)