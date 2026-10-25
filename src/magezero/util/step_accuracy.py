"""Target-head top-1 accuracy per game step, against picking a random legal candidate:
  PYTHONPATH=. python util/step_accuracy.py checkpoint test_dir   (run from src/magezero)"""
import sys
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from dataset import H5Graphs, collate_batch
from model import load_checkpoint, graph_index, TARGET_TYPES, ActionType
from vocab import feature_id

STEPS = "UPKEEP DRAW PRECOMBAT_MAIN BEGIN_COMBAT DECLARE_ATTACKERS DECLARE_BLOCKERS COMBAT_DAMAGE END_COMBAT POSTCOMBAT_MAIN END_TURN CLEANUP".split()
torch.set_num_threads(4)
model, vocab, edge_vocab = load_checkpoint(sys.argv[1])
model.eval()
ds = H5Graphs(sys.argv[2], vocab, edge_vocab)
step_ids = {feature_id(s): s for s in STEPS}
st = np.repeat(np.arange(len(ds)), np.diff(ds.offsets))
hit = np.isin(ds.ids, list(step_ids))
step = np.empty(len(ds), dtype=object); step[st[hit]] = [step_ids[i] for i in ds.ids[hit]]
idx = np.flatnonzero(ds.row[:, 3] == ActionType.CHOOSE_TARGET.value)
res = {}
with torch.no_grad():
    for k0 in range(0, len(idx), 128):
        ks = idx[k0:k0 + 128]
        g, t = collate_batch([ds[k] for k in ks])
        _, target, _, _ = model(g)
        ng = graph_index(g.node_offsets).numpy()
        tt = torch.isin(g.node_type, torch.tensor(TARGET_TYPES)).numpy()
        tt[t.cand_node.numpy()] = True
        visits = np.zeros(len(tt)); np.add.at(visits, t.cand_node.numpy(), t.cand_visits.numpy())
        score = target.numpy()
        for b, k in enumerate(ks):
            c = t.cand_graph.numpy() == b
            if t.cand_visits[c].sum() == 0:
                continue
            nodes = np.flatnonzero((ng == b) & tt)
            best = visits[nodes].max()
            top = nodes[np.argmax(score[nodes])]
            cv, cn = t.cand_visits.numpy()[c], t.cand_node.numpy()[c]
            masked = cn[np.argmax(score[cn])]
            r = res.setdefault(step[k], [0, 0, 0.0, 0, 0])
            r[0] += 1; r[1] += visits[top] == best; r[2] += (cv == cv.max()).sum() / len(cv); r[3] += len(cv)
            r[4] += visits[masked] == best
for s in STEPS:
    if s in res and res[s][0] >= 50:
        n, c, rnd, nc, m = res[s]
        print(f"{s:18s} states {n:6d}  acc {c / n:.3f}  legal-only acc {m / n:.3f}  random-candidate {rnd / n:.3f}  "
              f"candidates/state {nc / n:.1f}")
