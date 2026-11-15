"""Does a model use a leaf's information, or the token itself as a sink? The same sample of states for every model:
1. Attention mass in the PLAYER local layers (each pass, mean over heads and PLAYER nodes) per child feature.
2. The same ablations for every model, value MSE against the stored labels:
   - remove a leaf: the mana-pool leaves (constant 0 in these games, so no information) and the Battlefield name;
   - cut every leaf from the land entities (nodes with a LAND leaf) and their abilities: permanents, cards, both at
     once; the entity nodes stay;
   - shuffle a leaf's numeric value across its occurrences: the tokens stay, only their information goes.
3. With --all instead: every leaf id in the sample removed in turn (all its occurrences, from the states that hold it).
Feature names come from the XMage encoder and decks in $MAGE_SRC (WillWroble/mage), if set.
  PYTHONPATH=. python util/ablate.py [--all] OUT N name=model.pt.gz ... -- shard ..."""
import copy, glob, os, re, sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import model as M
from dataset import H5Graphs, collate_batch
from model import load_checkpoint, NodeType as T
from vocab import feature_id as F

MANA = ["WhiteMana", "BlueMana", "BlackMana", "RedMana", "GreenMana", "ColorlessMana"]
SHUFFLE = ["LifeTotal", "HandSize", "BattlefieldSize", "GraveyardSize", "LibrarySize", "ManaValue", "Power", "Toughness"]


def names():
    """feature id -> name: the encoder's string literals and the decks' card names."""
    words = set(MANA + SHUFFLE + ["Battlefield"])
    src = os.environ.get("MAGE_SRC")
    if src:
        enc = glob.glob(f"{src}/**/encoder/StateEncoder.java", recursive=True)
        words |= {w for f in enc for w in re.findall(r'"([^"\n]{1,60})"', open(f).read())}
        for f in glob.glob(f"{src}/Mage/src/main/java/mage/constants/*Type.java"):  # card types and subtypes: enum names
            words |= set(re.findall(r"^\s*([A-Z][A-Z0-9_]*)\(", open(f).read(), re.M))
        for deck in glob.glob(f"{src}/Mage.MageZero/decks/*.txt"):
            words |= {m.group(1) for m in (re.match(r"\s*(?:SB:\s*)?\d+\s+(?:\[[^\]]*\]\s*)?(.+?)\s*$", l)
                                           for l in open(deck)) if m}
    return {F(w): w for w in words}


def sample_states(shards, n):
    """n states drawn with a fixed seed (the same for every model), candidates dropped; raw ids."""
    ds = H5Graphs([(s, None) for s in shards])
    return select(ds, np.sort(np.random.default_rng(0).choice(ds.N, n, replace=False)))


def select(ds, idx):
    """ds reduced to states idx, in place, candidates dropped."""
    nodes = [slice(ds.offsets[s], ds.offsets[s + 1]) for s in idx]
    edges = [slice(ds.edge_offsets[s], ds.edge_offsets[s + 1]) for s in idx]
    ds.ids, ds.values, ds.node_type = (np.concatenate([x[k] for k in nodes]) for x in (ds.ids, ds.values, ds.node_type))
    ds.edge_child, ds.edge_parent, ds.edge_label = (np.concatenate([x[k] for k in edges])
                                                    for x in (ds.edge_child, ds.edge_parent, ds.edge_label))
    ds.offsets = np.concatenate([[0], np.cumsum([k.stop - k.start for k in nodes])]).astype(np.int64)
    ds.edge_offsets = np.concatenate([[0], np.cumsum([k.stop - k.start for k in edges])]).astype(np.int64)
    ds.row, ds.N = ds.row[idx], len(idx)
    ds.cand_node, ds.cand_visits = np.empty(0, np.int32), np.empty(0, np.int32)
    return ds


def drop_leaves(ds, remove):
    """Removes the leaves marked in `remove` (global node mask) and their edges, in place (raw ids)."""
    state = np.repeat(np.arange(ds.N), np.diff(ds.offsets))
    e_state = np.repeat(np.arange(ds.N), np.diff(ds.edge_offsets))
    keep, edges = ~remove, ~remove[ds.edge_child + ds.offsets[e_state]]
    offsets = np.concatenate([[0], np.cumsum(np.bincount(state[keep], minlength=ds.N))])
    local = np.cumsum(keep) - 1 - offsets[state]
    for k in ("edge_child", "edge_parent"):
        setattr(ds, k, local[getattr(ds, k)[edges] + ds.offsets[e_state[edges]]].astype(np.int16))
    ds.edge_label = ds.edge_label[edges]
    ds.edge_offsets = np.concatenate([[0], np.cumsum(np.bincount(e_state[edges], minlength=ds.N))])
    ds.ids, ds.values, ds.node_type, ds.offsets = ds.ids[keep], ds.values[keep], ds.node_type[keep], offsets


def values(model, ds, vocab, edge_vocab):
    ds.apply_vocab(vocab, edge_vocab)
    with torch.no_grad():
        return np.concatenate([model(g.to(M.DEVICE))[3].float().cpu().numpy() for g, _ in DataLoader(ds, batch_size=64, collate_fn=collate_batch)])


def drop_land_leaves(ds, kinds):
    """Cuts every leaf edge into the land entities (nodes of a type in `kinds` with a LAND leaf) and into their
    abilities, in place (raw ids); leaf nodes are shared within a state, so the edges go, not the leaves."""
    e_state = np.repeat(np.arange(ds.N), np.diff(ds.edge_offsets))
    child, parent = ds.edge_child + ds.offsets[e_state], ds.edge_parent + ds.offsets[e_state]
    leaf = ds.node_type[child] == T.LEAF
    lands = np.unique(parent[leaf & (ds.ids[child] == F("LAND"))])
    lands = lands[np.isin(ds.node_type[lands], kinds)]
    entities = np.concatenate([lands, child[np.isin(parent, lands) & (ds.node_type[child] == T.ABILITY)]])
    keep = ~(leaf & np.isin(parent, entities))
    ds.edge_child, ds.edge_parent, ds.edge_label = ds.edge_child[keep], ds.edge_parent[keep], ds.edge_label[keep]
    ds.edge_offsets = np.concatenate([[0], np.cumsum(np.bincount(e_state[keep], minlength=ds.N))])


def mse(model, ds, vocab, edge_vocab):
    return float(np.mean((values(model, ds, vocab, edge_vocab) - ds.row[:, 0]) ** 2))


def sweep(model, shards, n, vocab, edge_vocab):
    """{leaf id: (states holding it, change in value MSE over all n states, mean |value change| where present,
    its parents' node type)}."""
    ds = sample_states(shards, n); base = values(model, copy.deepcopy(ds), vocab, edge_vocab); err = (base - ds.row[:, 0]) ** 2
    state = np.repeat(np.arange(ds.N), np.diff(ds.offsets)); leaf = ds.node_type == T.LEAF
    e_state = np.repeat(np.arange(ds.N), np.diff(ds.edge_offsets))
    child, parent = ds.edge_child + ds.offsets[e_state], ds.edge_parent + ds.offsets[e_state]
    out = {}
    for f in np.unique(ds.ids[leaf]):
        idx = np.unique(state[leaf & (ds.ids == f)])
        sub = select(copy.deepcopy(ds), idx); drop_leaves(sub, (sub.node_type == T.LEAF) & (sub.ids == f))
        v = values(model, sub, vocab, edge_vocab)
        kind = T(int(np.bincount(ds.node_type[parent[ds.ids[child] == f]]).argmax())).name
        out[int(f)] = (len(idx), float(((v - ds.row[idx, 0]) ** 2 - err[idx]).sum() / n), float(np.abs(v - base[idx]).mean()), kind)
    return out


calls = []
_forward = M.LocalLayer.forward


def recording(self, node, child, seg):
    """LocalLayer.forward, also recording each [node, children] token's head-mean attention."""
    S, d = node.shape
    H, dh = self.self_attn.num_heads, self.self_attn.head_dim
    w, b = self.self_attn.in_proj_weight, self.self_attn.in_proj_bias
    q, k = torch.nn.functional.linear(self.norm1(node), w[:2 * d], b[:2 * d]).chunk(2, dim=-1)
    k_child = torch.nn.functional.linear(self.norm1(child), w[d:2 * d], b[d:2 * d])
    s = torch.cat([torch.arange(S, device=seg.device), seg])
    score = (q.reshape(S, H, dh)[s] * torch.cat([k, k_child]).view(-1, H, dh)).sum(-1) / dh ** .5
    calls.append(((score - M.segment_logsumexp(score, s, S)[s]).exp().mean(1), S))
    return _forward(self, node, child, seg)


def player_attention(model, ds, raw, name):
    """{pass: {child feature (a typed child by its type): mean attention mass per PLAYER node}}."""
    M.LocalLayer.forward, acc = recording, {}
    try:
        with torch.no_grad():
            for b, (g, _) in enumerate(DataLoader(ds, batch_size=64, collate_fn=collate_batch)):
                g = g.to(M.DEVICE); calls.clear(); plan = model.plan(g); model(g, plan); types = g.node_type.cpu()
                ids = raw[ds.offsets[64 * b]:ds.offsets[min(64 * b + 64, ds.N)]]
                rec = iter(calls)
                for p, direction in enumerate(model.arch["directions"]):
                    for stage, nodes, c, *_ in (plan.down_stages if direction == "down" else plan.stages):
                        if not len(nodes): continue
                        attn, S = next(rec)
                        if stage != "PLAYER": continue
                        keys = ["self"] * S + [name.get(int(ids[i]), str(int(ids[i]))) if types[i] == T.LEAF
                                               else T(int(types[i])).name for i in c.tolist()]
                        a = acc.setdefault(p, {"#": 0})
                        for k, x in zip(keys, attn.tolist()): a[k] = a.get(k, 0) + x
                        a["#"] += S
    finally:
        M.LocalLayer.forward = _forward
    return {p: {k: x / a["#"] for k, x in a.items() if k != "#"} for p, a in acc.items()}


if __name__ == "__main__":
    torch.set_num_threads(os.cpu_count())
    full = "--all" in sys.argv
    if full: sys.argv.remove("--all")
    sep = sys.argv.index("--")
    out_path, n, models, shards = sys.argv[1], int(sys.argv[2]), [a.split("=", 1) for a in sys.argv[3:sep]], sys.argv[sep + 1:]
    name = names()
    out = open(out_path, "w")
    def say(s): print(s, flush=True); out.write(s + "\n"); out.flush()

    if full:
        for label, path in models:
            model, vocab, edge_vocab = load_checkpoint(path); model.to(M.DEVICE).eval()
            r = sweep(model, shards, n, vocab, edge_vocab)
            say(f"\n## {label}: leaf removed (parent type) -> states holding it, value MSE change over all {n} states, "
                f"mean |value change| where present")
            for f, (k, d, a, kind) in sorted(r.items(), key=lambda x: -x[1][1]):
                say(f"  {name.get(f, f)!s:32} {kind:12} {k:5}  {d:+.4f}  {a:.3f}")
        sys.exit()

    rows = {}
    for label, path in models:
        model, vocab, edge_vocab = load_checkpoint(path); model.to(M.DEVICE).eval()
        say(f"\n## {label}: " + ", ".join(f"{k}={model.arch.get(k)}" for k in
                                         ("policy_domain", "value_token", "hide_decider", "directions", "registers")))
        ds = sample_states(shards, n); raw = ds.ids.copy(); ds.apply_vocab(vocab, edge_vocab)
        for p, a in player_attention(model, ds, raw, name).items():
            say(f"  PLAYER attention, pass {p}: " + "  ".join(f"{k} {x:.2f}" for k, x in sorted(a.items(), key=lambda kx: -kx[1])[:10])
                + f"  | mana-pool leaves {sum(a.get(m, 0) for m in MANA):.2f}")
        r = rows[label] = {"base": mse(model, sample_states(shards, n), vocab, edge_vocab)}
        for f in MANA + ["Battlefield", MANA]:
            ds = sample_states(shards, n); drop_leaves(ds, (ds.node_type == T.LEAF) & np.isin(ds.ids, [F(x) for x in np.atleast_1d(f)]))
            r["-" + (f if isinstance(f, str) else "all mana")] = mse(model, ds, vocab, edge_vocab)
        for col, kinds in (("-land permanent leaves", [T.PERMANENT]), ("-land card leaves", [T.CARD]),
                           ("-all land leaves", [T.PERMANENT, T.CARD])):
            ds = sample_states(shards, n); drop_land_leaves(ds, kinds)
            r[col] = mse(model, ds, vocab, edge_vocab)
        for f in SHUFFLE:
            ds = sample_states(shards, n); m = np.flatnonzero((ds.node_type == T.LEAF) & (ds.ids == F(f)))
            ds.values[m] = ds.values[np.random.default_rng(1).permutation(m)]
            r["~" + f] = mse(model, ds, vocab, edge_vocab)

    cols = list(next(iter(rows.values())))
    say("\n== value MSE: base, then the change from base (-x: leaf x removed, ~x: x's value shuffled across states)")
    say("| model | " + " | ".join(cols) + " |")
    say("|---" * (len(cols) + 1) + "|")
    for label, r in rows.items():
        say(f"| {label} | {r['base']:.3f} | " + " | ".join(f"{r[c] - r['base']:+.3f}" for c in cols[1:]) + " |")
