import json
import os
import time
import gzip
import shutil

import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader

import test
from model import (build_model, NodeType, load_checkpoint, losses, accumulate, averages, decision_states,
                   AMP, DEVICE, GLOBAL_MAX)
from dataset import H5Graphs, collate_batch, read_split, BATCH_SIZE
from vocab import FeatureVocab, feature_id, initial_rows, kept_feature_ids

# seat names: Faeries is always PlayerA in the data, so they identify the deck (a shortcut for the value head)
PLAYER_NAMES = [feature_id("PlayerA"), feature_id("PlayerB")]




def train(
        out: str,
        epochs: int,
        steps: int = 15000,
        use_checkpoint: bool = False,
        arch: dict = None,
        hide_opponent_hand: bool = False,
        lr: float = 1e-4,
        drop_player_names: bool = False,
        td: float = None,
):
    train_shards, val_shards = read_split(f"{out}/split.txt")
    ds = H5Graphs(train_shards, hide_opponent_hand=hide_opponent_hand, td=td)
    if len(ds) == 0:
        found = (f"{len(ds.files)} .hdf5/.h5 files, none with readable states (see warnings above)" if ds.files
                 else "no .hdf5/.h5 files")
        raise SystemExit(f"no training data in {out}/split.txt: {found}")
    vocab, edge_vocab, model = prepare_vocab(out, use_checkpoint, ds, arch or {}, drop_player_names)
    ds.apply_vocab(vocab, edge_vocab)
    val_sets = {name: H5Graphs(shards, vocab, edge_vocab, hide_opponent_hand, td) for name, shards in val_shards.items()}

    train_loop(out, epochs, steps, model, ds, val_sets, vocab, edge_vocab, hide_opponent_hand, lr)


def prepare_vocab(out: str, use_checkpoint: bool, ds: H5Graphs, arch: dict, drop_player_names: bool = False):
    """Leaf and edge label vocabs = the previous checkpoint's (rows unchanged) + ids newly kept in
    this dataset, appended. Each embedding table has one row per vocab entry."""
    print("Building vocabs from dataset (ids seen in more than 10 states)")
    leaf = ds.node_type == NodeType.LEAF
    kept_leaves = kept_feature_ids(ds.ids, ds.offsets, leaf)
    if drop_player_names:  # leaves outside the vocab are dropped with their edges, here and wherever the checkpoint runs
        kept_leaves = kept_leaves[~np.isin(kept_leaves, PLAYER_NAMES)]
    kept_labels = kept_feature_ids(ds.edge_label, ds.edge_offsets)

    model = build_model(**arch)
    vocab, edge_vocab = FeatureVocab(feature_hash_bins=GLOBAL_MAX), FeatureVocab(feature_hash_bins=GLOBAL_MAX)
    if use_checkpoint:
        checkpoint_path = f"{out}/model.pt.gz"
        try:
            model, vocab, edge_vocab = load_checkpoint(checkpoint_path)
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        except FileNotFoundError:
            print(f"INFO: Checkpoint file not found at {checkpoint_path}. Starting from scratch.")

    prev_rows, prev_labels = len(vocab), len(edge_vocab)
    vocab.extend(kept_leaves)
    edge_vocab.extend(kept_labels)
    # rows the appended ids would have started from in any run
    d_model = model.arch["d_model"]
    model.resize_embedding(len(vocab), initial_rows(vocab.ids[prev_rows:], d_model))
    model.resize_embedding(len(edge_vocab) + 1, initial_rows(edge_vocab.ids[prev_labels:], d_model), "edge_embedding")
    print(f"leaf vocab: {len(kept_leaves)} kept ids in this dataset, {prev_rows} rows from checkpoint -> {len(vocab)} rows")
    print(f"edge vocab: {len(kept_labels)} kept labels in this dataset, {prev_labels} rows from checkpoint -> {len(edge_vocab)} rows")
    return vocab, edge_vocab, model.to(DEVICE)


def save_checkpoint(path, epoch, model, opt, vocab, edge_vocab, avg, hide_opponent_hand):
    temp_path = path.replace('.gz', '.tmp')

    # Save uncompressed
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_dense_state_dict': opt.state_dict(),
        'avg_p_loss': avg["priority"],
        'avg_v_loss': avg["value"],
        'feature_vocab': vocab.state_dict(),
        'edge_vocab': edge_vocab.state_dict(),
        'arch': model.arch,
        'hide_opponent_hand': hide_opponent_hand,  # trained on states without it; test.py tests the same way
    }, temp_path)

    # Stream-compress in chunks (constant memory)
    with open(temp_path, 'rb') as f_in:
        with gzip.open(path, 'wb', compresslevel=1) as f_out:
            shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)  # 16MB chunks

    os.remove(temp_path)


def train_loop(out, epochs, steps, model, ds, val_sets, vocab, edge_vocab, hide_opponent_hand, lr=1e-4):
    """val_sets: name -> H5Graphs, each validated after every epoch; the first picks best.pt.gz."""

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_batch,
                    pin_memory=AMP, persistent_workers=False)

    dl_vals = {name: DataLoader(v, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_batch,
                                pin_memory=AMP, persistent_workers=False) for name, v in val_sets.items() if len(v) > 0}

    #test.SHOW_CONFUSION_MATRIX = False

    opt = optim.Adam(model.parameters(), lr=lr)
    legal_only = model.arch.get("policy_domain") == "legal"
    legality_weight = model.arch.get("legality_weight", 0.0)
    scaler = torch.amp.GradScaler(enabled=AMP)

    best_val_loss = float('inf')
    print(f"arch: {model.arch}  parameters: {sum(p.numel() for p in model.parameters()):,}")

    for epoch in range(1, epochs+1):
        stats = {}
        model.train()
        step = 0
        start = time.perf_counter()

        for graphs, targets in dl:
            graphs, targets = graphs.to(DEVICE), targets.to(DEVICE)

            with torch.amp.autocast(DEVICE.type, enabled=AMP):
                heads = losses(model(graphs), graphs, targets, legal_only, legality_weight)
                loss = sum(head[0] for head in heads.values())

            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            accumulate(stats, heads)
            step += 1
            if step >= steps:
                break

        avg = averages(stats)
        print(f"Epoch {epoch}  priority_loss={avg['priority']:.3f}  choose_target_loss={avg['choose_target']:.3f}  "
              f"choose_use_loss={avg['choose_use']:.3f}  value_loss={avg['value']:.3f}  "
              + (f"legality_loss={stats['legality'][0] / max(stats['legality'][1], 1):.3f}  " if "legality" in stats else "")
              + f"decision_states={decision_states(stats)}  steps={step}  time={time.perf_counter() - start:.0f}s")

        for i, (name, dl_val) in enumerate(dl_vals.items()):
            print(f"Testing on {name}")
            val_loss = test.validate(model, dl_val, vocab, edge_vocab)
            if i == 0 and val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(f"{out}/best.pt.gz", epoch, model, opt, vocab, edge_vocab, avg, hide_opponent_hand)

        #TODO: make validation based checkpoint schedule
        save_checkpoint(f"{out}/model.pt.gz", epoch, model, opt, vocab, edge_vocab, avg, hide_opponent_hand)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="model directory: its split.txt (dataset.read_split) names the data; checkpoints and train.log go there")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--steps", type=int, default=15000)
    parser.add_argument("--checkpoint", action="store_true", help="resume from <model>/model.pt.gz")
    parser.add_argument("--arch", type=json.loads, default={}, help='model options as JSON, e.g. \'{"n_passes": 1}\' (NetGraph) or \'{"net": "mlp"}\' (NetMLP)')
    parser.add_argument("--seed", type=int)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--hide-opponent-hand", action="store_true",
                        help="train and test on states without the non-deciding player's hand (H5Graphs.hide_opponent_hand)")
    parser.add_argument("--drop-player-names", action="store_true",
                        help="leave the PlayerA/PlayerB name leaves out of the vocab, so the model never sees them")
    parser.add_argument("--td", type=float, help="train and validate on value labels rebuilt with this TD discount "
                                                 "(dataset.td_labels) instead of the stored ones")

    args = parser.parse_args()
    test.log_to(f"{args.model}/train.log")
    if args.seed is not None:
        torch.manual_seed(args.seed)
    train(args.model, args.epochs, args.steps, args.checkpoint, args.arch, args.hide_opponent_hand, args.lr,
          args.drop_player_names, args.td)
