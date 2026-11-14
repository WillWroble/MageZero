"""Opponent-turn land injection test: on P1's Burn mirror shards, the positions on the opponent's turn where the decision
player skipped its land drop with a Mountain in hand (CanPlayLand still set). Two edits per position: "flag" drops
CanPlayLand only; "played" moves the Mountain from hand to battlefield (hand size -1, permanent count +1, flag dropped).
Prints, per model and shard, the share of positions whose value goes up / down (|change| > .005): flag; played over all
positions, in lost games, and with 2 own lands.
  PYTHONPATH=. python util/land_inject.py name=model.pt.gz ... -- shard ..."""
import copy, sys
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import H5Graphs, collate_batch, node_types
from model import load_checkpoint, NodeType as T, DEVICE
from vocab import feature_id as F

DEC, ACT, LAND, BF, MOUNTAIN, TAPPED, HAND, CAN_PLAY_LAND = (F(s) for s in (
    "IsDecisionPlayer", "IsActivePlayer", "LAND", "Battlefield", "Mountain", "Tapped", "Hand", "CanPlayLand"))
PERM_COUNT, HAND_SIZE = 218205224, 502725437  # battlefield zone leaf = permanent count; player leaf = cards in hand


def parts(ds, s):
    n0, n1, e0, e1 = ds.offsets[s], ds.offsets[s + 1], ds.edge_offsets[s], ds.edge_offsets[s + 1]
    return ds.ids[n0:n1], ds.node_type[n0:n1], ds.edge_child[e0:e1], ds.edge_parent[e0:e1], ds.edge_label[e0:e1]


def own_battlefield(ids, st, c, p):
    """(decision player's node, its battlefield zone, kids: node -> leaf ids)."""
    kids = defaultdict(set)
    leaf = st[c] == T.LEAF
    for ci, pi in zip(c[leaf], p[leaf]): kids[pi].add(ids[ci])
    me = next(x for x in np.flatnonzero(st == T.PLAYER) if DEC in kids[x])
    zone = next(ci for ci, pi in zip(c, p) if pi == me and st[ci] == T.ZONE and BF in kids[ci])
    return me, zone, kids


def subtree(c, p, root):
    nodes, frontier = [root], [root]
    while frontier:
        frontier = [ci for ci, pi in zip(c, p) if pi in frontier]; nodes += frontier
    return nodes


def mountain_template(ds):
    """An untapped Mountain permanent from the shard: subtree (ids, values, local edges) and its edge label to the zone."""
    for s in range(ds.N):
        ids, st, c, p, l = parts(ds, s)
        _, zone, kids = own_battlefield(ids, st, c, p)
        for x, lx in ((ci, li) for ci, pi, li in zip(c, p, l) if pi == zone and st[ci] == T.PERMANENT):
            if MOUNTAIN in kids[x] and TAPPED not in kids[x]:
                nodes = subtree(c, p, x); local = {n: i for i, n in enumerate(nodes)}
                edges = np.array([(local[ci], local[pi], li) for ci, pi, li in zip(c, p, l) if pi in local])
                return (ids[nodes], ds.values[ds.offsets[s] + np.array(nodes)], edges), lx
    raise ValueError("no untapped Mountain on the decision player's battlefield")


def rebuild(ds, states, kept):
    """Replaces ds's states with `states` (ids, values, child, parent, label each), keeping rows `kept`."""
    ds.ids, ds.values = (np.concatenate([x[i] for x in states]) for i in (0, 1))
    ds.ids, ds.values = ds.ids.astype(np.int32), ds.values.astype(np.int16)
    ds.edge_child, ds.edge_parent = (np.concatenate([x[i] for x in states]).astype(np.int16) for i in (2, 3))
    ds.edge_label = np.concatenate([x[4] for x in states]).astype(np.int32)
    ds.offsets = np.concatenate([[0], np.cumsum([len(x[0]) for x in states])]).astype(np.int64)
    ds.edge_offsets = np.concatenate([[0], np.cumsum([len(x[2]) for x in states])]).astype(np.int64)
    ds.node_type = node_types(ds.ids)
    ds.row, ds.N = ds.row[kept], len(kept)
    ds.cand_node, ds.cand_visits = np.empty(0, np.int32), np.empty(0, np.int32)
    ds.cand_offsets = np.zeros(len(kept) + 1, np.int64)


def variants(shard):
    """(unchanged, flag dropped, Mountain played) datasets over the skipped-drop positions, raw ids, and per position
    (own lands, game lost)."""
    ds = H5Graphs([(shard, None)])
    lost = H5Graphs([(shard, None)], td=1.0).row[:, 0] < 0
    (t_ids, t_vals, t_edges), zone_label = mountain_template(ds)
    out = {"base": [], "flag": [], "played": []}; kept, lands = [], []
    for s in range(ds.N):
        ids, st, c, p, l = parts(ds, s)
        me, zone, kids = own_battlefield(ids, st, c, p)
        if ACT in kids[me] or CAN_PLAY_LAND not in kids[me]: continue
        hand = next(ci for ci in c[p == me] if st[ci] == T.ZONE and HAND in kids[ci])
        card = next((ci for ci in c[p == hand] if st[ci] == T.CARD and MOUNTAIN in kids[ci]), None)
        if card is None: continue
        vals = ds.values[ds.offsets[s]:ds.offsets[s + 1]]
        out["base"].append((ids, vals, c, p, l))
        flag = next(ci for ci in c[p == me] if ids[ci] == CAN_PLAY_LAND)
        for name, drop in (("flag", {flag}), ("played", {flag, *subtree(c, p, card)})):
            keep = np.array([i not in drop for i in range(len(ids))]); new = np.cumsum(keep) - 1; n = int(keep.sum())
            e = keep[c] & keep[p]; v = vals.copy()
            state = [ids[keep], v[keep], new[c[e]], new[p[e]], l[e]]
            if name == "played":
                v[next(ci for ci in c[p == zone] if ids[ci] == PERM_COUNT)] += 1
                v[next(ci for ci in c[p == me] if ids[ci] == HAND_SIZE)] -= 1
                state = [np.concatenate([ids[keep], t_ids]), np.concatenate([v[keep], t_vals]),
                         np.concatenate([new[c[e]], t_edges[:, 0] + n, [n]]),
                         np.concatenate([new[p[e]], t_edges[:, 1] + n, [new[zone]]]),
                         np.concatenate([l[e], t_edges[:, 2], [zone_label]])]
            out[name].append(state)
        kept.append(s)
        lands.append(sum(1 for ci, pi in zip(c, p) if pi == zone and st[ci] == T.PERMANENT and LAND in kids[ci]))
    sets = {}
    for name, states in out.items():
        sets[name] = copy.copy(ds); rebuild(sets[name], states, np.array(kept))
    return sets, np.array(lands), lost[kept]


def predict(model, ds):
    with torch.no_grad():
        return np.concatenate([model(g.to(DEVICE))[3].float().cpu().numpy()
                               for g, _ in DataLoader(ds, batch_size=64, collate_fn=collate_batch)])


def share(d):
    return f"{np.mean(d > .005):3.0%} / {np.mean(d < -.005):3.0%}"


if __name__ == "__main__":
    sep = sys.argv.index("--")
    models, shards = [a.split("=", 1) for a in sys.argv[1:sep]], sys.argv[sep + 1:]
    data = {sh: variants(sh) for sh in shards}
    print(f"{'model':36} {'shard':8} {'positions':>9}  {'flag':>9}  {'played':>9}  {'lost':>9}  {'2 lands':>9}  (up / down)")
    for name, path in models:
        model, vocab, edge_vocab = load_checkpoint(path); model.to(DEVICE).eval()
        for sh, (sets, lands, lost) in data.items():
            v = {}
            for k, ds in sets.items():
                ds = copy.deepcopy(ds); ds.apply_vocab(vocab, edge_vocab); v[k] = predict(model, ds)
            flag, played = v["flag"] - v["base"], v["played"] - v["base"]
            tag = "".join(ch for ch in sh.split("/")[-1] if ch.isdigit())[:4]
            print(f"{name:36} {tag:8} {len(lands):9}  {share(flag)}  {share(played)}  {share(played[lost])}  "
                  f"{share(played[lands == 2])}", flush=True)
