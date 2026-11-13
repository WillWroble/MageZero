"""Early land drops per shard (decision player's seat): priority states in the decision player's own main phases
with a land play legal and fewer than 3 (and 4) own lands; the share where the search's most-visited move is Pass.
  PYTHONPATH=. python util/land_pass.py name=shard ..."""
import sys
from collections import defaultdict

import h5py
import numpy as np

from dataset import node_types
from model import NodeType as T
from vocab import feature_id as F

DEC, ACT, NAME, LAND, BF = (F(s) for s in ("IsDecisionPlayer", "IsActivePlayer", "name", "LAND", "Battlefield"))
MAIN = {F("PRECOMBAT_MAIN"), F("POSTCOMBAT_MAIN")}
PASS = 679880620


def early_drops(path):
    """(own lands, Pass is the most-visited move) per qualifying state."""
    with h5py.File(path) as f:
        d = {k: f[k][...] for k in ("indices", "offsets", "edge_child", "edge_parent", "edge_label", "edge_offsets",
                                     "policy_node", "policy_offsets", "policy_visits")}
    ids, off = d["indices"], d["offsets"]
    types = node_types(ids)
    out = []
    for s in range(len(off) - 1):
        q0, q1 = d["policy_offsets"][s], d["policy_offsets"][s + 1]
        cand, vis = d["policy_node"][q0:q1], d["policy_visits"][q0:q1]
        if vis.sum() == 0 or len(cand) < 2: continue
        sid, st = ids[off[s]:off[s + 1]], types[off[s]:off[s + 1]]
        if st[cand[0]] != T.ABILITY: continue                     # priority decisions only
        e0, e1 = d["edge_offsets"][s], d["edge_offsets"][s + 1]
        c, p, l = d["edge_child"][e0:e1], d["edge_parent"][e0:e1], d["edge_label"][e0:e1]
        leaf = st[c] == T.LEAF
        kids = defaultdict(set)
        for ci, pi in zip(c[leaf], p[leaf]): kids[pi].add(sid[ci])
        if not MAIN & kids[np.flatnonzero(st == T.ROOT)[0]]: continue
        me = next(x for x in np.flatnonzero(st == T.PLAYER) if DEC in kids[x])
        if ACT not in kids[me]: continue
        parent = dict(zip(c[~leaf], p[~leaf]))
        named = leaf & (l == NAME); name = dict(zip(p[named], sid[c[named]]))
        if not any(LAND in kids.get(parent.get(x), ()) for x in cand): continue
        lands = sum(1 for x in np.flatnonzero(st == T.PERMANENT)
                    if LAND in kids[x] and name.get(parent.get(x)) == BF and parent.get(parent.get(x)) == me)
        out.append((lands, name.get(cand[int(np.argmax(vis))]) == PASS))
    return np.array(out).reshape(-1, 2)


if __name__ == "__main__":
    print(f"{'shard':24} {'<3 lands: pass top (states)':>28} {'<4 lands':>18}")
    for name, path in (a.split("=", 1) for a in sys.argv[1:]):
        x = early_drops(path)
        cells = [f"{x[m, 1].mean():.1%} ({m.sum()})" for m in (x[:, 0] < 3, x[:, 0] < 4)]
        print(f"{name:24} {cells[0]:>28} {cells[1]:>18}", flush=True)
