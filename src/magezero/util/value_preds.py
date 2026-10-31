"""Each model's value predictions on a data set, saved with every state's value label, search root value and game
result (one array per model, named by its last two path components, e.g. v4-legal-gnn):
  PYTHONPATH=. python util/value_preds.py OUT.npz DATA MODEL_DIR ...   (run from src/magezero; DATA: a shard directory)"""
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import H5Graphs, collate_batch, BATCH_SIZE
from model import load_checkpoint, AMP, DEVICE

out_path, data, models = sys.argv[1], sys.argv[2], sys.argv[3:]
results = H5Graphs(data, td=1.0).row[:, 0]  # td 1: each state's label is its game's result
saved = {}
for path in models:
    model, vocab, edge_vocab = load_checkpoint(f"{path}/model.pt.gz")
    model = model.to(DEVICE).eval()
    ds = H5Graphs(data, vocab, edge_vocab)
    preds = []
    with torch.no_grad(), torch.amp.autocast(DEVICE.type, enabled=AMP):
        for g, _ in DataLoader(ds, batch_size=BATCH_SIZE, collate_fn=collate_batch):
            preds.append(model(g.to(DEVICE))[3].float().cpu().numpy())
    pred = saved["-".join(path.rstrip("/").split("/")[-2:])] = np.concatenate(preds)
    print(path, "MSE vs label", float(((pred - ds.row[:, 0]) ** 2).mean()))
np.savez(out_path, label=ds.row[:, 0], root=ds.row[:, 1], result=results, **saved)
