"""Per shard (decision player's seat): one example game's value curve (search root value, row[:, 1], per state; the
median-length game), and, for own turns with fewer than 4 own lands where a land play was legal, whether the turn ended
with the land drop skipped (CanPlayLand still set at its last state) and the change in root value from the start of that
turn to the start of the next own turn; and the mean value curve of games with and without such a skip. Writes JSON lines to OUT.
  PYTHONPATH=. python util/land_value.py OUT.jsonl name=shard ..."""
import json
import sys
from collections import defaultdict

import h5py
import numpy as np

from dataset import node_types
from model import NodeType as T
from vocab import feature_id as F

DEC, ACT, NAME, LAND, BF, CPL = (F(s) for s in ("IsDecisionPlayer", "IsActivePlayer", "name", "LAND", "Battlefield",
                                               "CanPlayLand"))


def analyse(path):
    with h5py.File(path) as f:
        d = {k: f[k][...] for k in ("indices", "offsets", "edge_child", "edge_parent", "edge_label", "edge_offsets",
                                     "policy_node", "policy_offsets", "row", "game_offsets")}
    ids, off, go, row = d["indices"], d["offsets"], d["game_offsets"], d["row"]
    types = node_types(ids)
    lengths = np.diff(go)
    g = int(np.argsort(lengths)[len(lengths) // 2])
    example = {"won": bool(row[go[g + 1] - 1, 0] > 0), "value": row[go[g]:go[g + 1], 1].round(3).tolist()}
    turns = []   # per game: list of [first state, lands, land offered, skipped]
    for gi in range(len(go) - 1):
        seg, segs = None, []
        for s in range(go[gi], go[gi + 1]):
            n0, n1 = off[s], off[s + 1]
            e0, e1 = d["edge_offsets"][s], d["edge_offsets"][s + 1]
            c, p, l = d["edge_child"][e0:e1], d["edge_parent"][e0:e1], d["edge_label"][e0:e1]
            sid, st = ids[n0:n1], types[n0:n1]
            leaf = st[c] == T.LEAF
            kids = defaultdict(set)
            for ci, pi in zip(c[leaf], p[leaf]):
                kids[pi].add(sid[ci])
            parent = dict(zip(c[~leaf], p[~leaf]))
            named = leaf & (l == NAME)
            name = dict(zip(p[named], sid[c[named]]))
            me = next((x for x in np.flatnonzero(st == T.PLAYER) if DEC in kids[x]), None)
            active = me is not None and ACT in kids[me]
            if not active:
                seg = None
                continue
            if seg is None:
                lands = sum(1 for x in np.flatnonzero(st == T.PERMANENT) if LAND in kids[x]
                            and name.get(parent.get(x)) == BF and parent.get(parent.get(x)) == me)
                seg = [s, lands, False, False]
                segs.append(seg)
            cand = d["policy_node"][d["policy_offsets"][s]:d["policy_offsets"][s + 1]]
            seg[2] |= any(st[x] == T.ABILITY and LAND in kids.get(parent.get(x), ()) for x in cand)
            seg[3] = CPL in kids[me]
        turns.append(segs)
    deltas = {"skipped": [], "played": []}
    for segs in turns:
        for a, b in zip(segs, segs[1:]):
            if a[1] < 4 and a[2]:
                deltas["skipped" if a[3] else "played"].append(float(row[b[0], 1] - row[a[0], 1]))
    # games with at least one skipped early land drop vs none: mean root value by game progress (10 bins)
    curves = {}
    for key in ("skip", "noskip"):
        tot, n = np.zeros(10), np.zeros(10)
        for gi, segs in enumerate(turns):
            if any(a[1] < 4 and a[2] and a[3] for a in segs[:-1]) != (key == "skip"):
                continue
            a, b = go[gi], go[gi + 1]
            k = np.minimum((np.arange(b - a) / (b - a) * 10).astype(int), 9)
            np.add.at(tot, k, row[a:b, 1]); np.add.at(n, k, 1)
        games = sum(1 for segs in turns if any(a[1] < 4 and a[2] and a[3] for a in segs[:-1]) == (key == "skip"))
        curves[key] = {"value": (tot / np.maximum(n, 1)).round(3).tolist(), "games": games}
    return {"example": example, "deltas": deltas, "curves": curves, "games": len(go) - 1}


if __name__ == "__main__":
    with open(sys.argv[1], "w") as out:
        for arg in sys.argv[2:]:
            name, path = arg.split("=", 1)
            out.write(json.dumps({"name": name, **analyse(path)}) + "\n")
            out.flush()
            print(name, flush=True)
