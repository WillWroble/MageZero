import os
import gzip
import shutil

import torch
from torch import optim
from torch.utils.data import DataLoader

import test
from model import (NetGraph, NodeType, load_checkpoint, losses, accumulate, averages, decision_states,
                   AMP, D_MODEL, DEVICE, GLOBAL_MAX)
from dataset import H5Graphs, collate_batch, states_of, BATCH_SIZE
from vocab import FeatureVocab, initial_rows, kept_feature_ids

#add training data under: data/{deck name}/ver{your version num}/training/{your data}.hdf5



def train(
        deck: str,
        version: int,
        epochs: int,
        steps: int = 15000,
        use_checkpoint: bool = False,
):
    os.makedirs(f"models/{deck}/ver{version}", exist_ok=True)
    data_path = f"data/{deck}/ver{version}/training"
    ds = H5Graphs(data_path)
    if len(ds) == 0:
        found = (f"{len(ds.files)} .hdf5/.h5 files, none with readable states (see warnings above)" if ds.files
                 else "no .hdf5/.h5 files")
        raise SystemExit(f"no training data in {data_path}: {found}")
    vocab, edge_vocab, model = prepare_vocab(deck, version, use_checkpoint, ds)
    ds.apply_vocab(vocab, edge_vocab)
    test_ds = H5Graphs(f"data/{deck}/ver{version}/testing", vocab, edge_vocab)

    train_loop(deck, version, epochs, steps, model, ds, test_ds, vocab, edge_vocab)


def prepare_vocab(deck: str, version: int, use_checkpoint: bool, ds: H5Graphs):
    """Leaf and edge label vocabs = the previous checkpoint's (rows unchanged) + ids newly kept in
    this dataset, appended. Each embedding table has one row per vocab entry."""
    print("Building vocabs from dataset (ids seen in more than 10 states)")
    leaf = ds.node_type == NodeType.LEAF
    kept_leaves = kept_feature_ids(ds.ids[leaf], states_of(ds.offsets)[leaf])
    kept_labels = kept_feature_ids(ds.edge_label, states_of(ds.edge_offsets))

    model = NetGraph()
    vocab, edge_vocab = FeatureVocab(feature_hash_bins=GLOBAL_MAX), FeatureVocab(feature_hash_bins=GLOBAL_MAX)
    if use_checkpoint:
        checkpoint_path = f"models/{deck}/ver{version}/model.pt.gz"
        try:
            model, vocab, edge_vocab = load_checkpoint(checkpoint_path)
            print(f"Successfully loaded checkpoint from {checkpoint_path}")
        except FileNotFoundError:
            print(f"INFO: Checkpoint file not found at {checkpoint_path}. Starting from scratch.")

    prev_rows, prev_labels = len(vocab), len(edge_vocab)
    vocab.extend(kept_leaves)
    edge_vocab.extend(kept_labels)
    # rows the appended ids would have started from in any run
    model.resize_embedding(len(vocab), initial_rows(vocab.ids[prev_rows:], D_MODEL))
    model.resize_embedding(len(edge_vocab) + 1, initial_rows(edge_vocab.ids[prev_labels:], D_MODEL), "edge_embedding")
    print(f"leaf vocab: {len(kept_leaves)} kept ids in this dataset, {prev_rows} rows from checkpoint -> {len(vocab)} rows")
    print(f"edge vocab: {len(kept_labels)} kept labels in this dataset, {prev_labels} rows from checkpoint -> {len(edge_vocab)} rows")
    return vocab, edge_vocab, model.to(DEVICE)


def save_checkpoint(path, epoch, model, opt, vocab, edge_vocab, avg):
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
    }, temp_path)

    # Stream-compress in chunks (constant memory)
    with open(temp_path, 'rb') as f_in:
        with gzip.open(path, 'wb', compresslevel=1) as f_out:
            shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)  # 16MB chunks

    os.remove(temp_path)


def train_loop(deck, version, epochs, steps, model, ds, test_ds, vocab, edge_vocab):

    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, collate_fn=collate_batch,
                    pin_memory=AMP, persistent_workers=False)

    dl_test = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, collate_fn=collate_batch,
                         pin_memory=AMP, persistent_workers=False)

    test.SHOW_CONFUSION_MATRIX = False

    opt = optim.Adam(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(enabled=AMP)

    best_val_loss = float('inf')

    for epoch in range(1, epochs+1):
        stats = {}
        model.train()
        step = 0

        for graphs, targets in dl:
            graphs, targets = graphs.to(DEVICE), targets.to(DEVICE)

            with torch.amp.autocast(DEVICE.type, enabled=AMP):
                heads = losses(model(graphs), graphs, targets)
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
              f"decision_states={decision_states(stats)}")

        #run current model on testing set (if there is one)
        if len(test_ds) > 0:
            val_loss = test.validate(model, dl_test, vocab, edge_vocab)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(f"models/{deck}/ver{version}/best.pt.gz", epoch, model, opt, vocab, edge_vocab, avg)

        #TODO: make validation based checkpoint schedule
        save_checkpoint(f"models/{deck}/ver{version}/model.pt.gz", epoch, model, opt, vocab, edge_vocab, avg)

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
