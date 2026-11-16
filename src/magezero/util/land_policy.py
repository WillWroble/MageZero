"""Policy at the last chance for an early land drop: priority states in the decision player's own postcombat main
phase with a land play legal and fewer than 4 own lands (land_pass.early_drops). Per model and shard: the share of
states whose top-scored legal candidate plays a land / is Pass, and the mean policy mass on land plays (softmax over
the legal candidates, as MCTS uses the priors); the search's most-visited move as reference.
  PYTHONPATH=. python util/land_policy.py name=model.pt.gz ... -- tag=shard ..."""
import copy, sys

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import H5Graphs, collate_batch
from model import load_checkpoint, DEVICE
from util.ablate import select
from util.land_pass import early_drops


def states(path):
    """The shard reduced to the last-chance states (raw ids, candidates kept), each one's land / Pass candidates,
    and the search's most-visited candidate."""
    x = [r for r in early_drops(path) if r[2] and r[1] < 4]
    return select(H5Graphs([(path, None)]), [r[0] for r in x], cands=True), [r[3:] for r in x]


def priors(model, ds):
    """Per state, the softmax of the priority scores over its legal candidates."""
    out = []
    with torch.no_grad():
        for g, t in DataLoader(ds, batch_size=64, collate_fn=collate_batch):
            score = model(g.to(DEVICE))[0].float().cpu()[t.cand_node].numpy()
            for k in range(len(t.value)):
                s = score[(t.cand_graph == k).numpy()]; e = np.exp(s - s.max()); out.append(e / e.sum())
    return out


if __name__ == "__main__":
    sep = sys.argv.index("--")
    models = [a.split("=", 1) for a in sys.argv[1:sep]]
    data = {tag: states(path) for tag, path in (a.split("=", 1) for a in sys.argv[sep + 1:])}
    print(f"{'model':36} {'shard':6} {'states':>6}  {'land top':>8}  {'Pass top':>8}  {'land mass':>9}")
    for tag, (_, x) in data.items():
        land = np.mean([l[top] for l, _, top in x]); ps = np.mean([p[top] for _, p, top in x])
        print(f"{'search (most visited)':36} {tag:6} {len(x):6}  {land:8.0%}  {ps:8.0%}")
    for name, path in models:
        try:
            model, vocab, edge_vocab = load_checkpoint(path)
        except Exception as e:
            print(f"{name:36} skipped: {path}: {e}", flush=True); continue
        model.to(DEVICE).eval()
        for tag, (ds, x) in data.items():
            ds = copy.deepcopy(ds)
            if model.arch.get("flat_zones"):
                ds.flatten_zones(model.arch["flat_zones"])
            ds.apply_vocab(vocab, edge_vocab)
            pr = priors(model, ds)
            top = [int(np.argmax(q)) for q in pr]
            print(f"{name:36} {tag:6} {len(x):6}  {np.mean([l[k] for (l, _, _), k in zip(x, top)]):8.0%}  "
                  f"{np.mean([p[k] for (_, p, _), k in zip(x, top)]):8.0%}  "
                  f"{np.mean([q[l].sum() for (l, _, _), q in zip(x, pr)]):9.0%}", flush=True)
