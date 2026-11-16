"""Does a model use a leaf's information, or the token itself as a sink? The same sample of states for every model:
1. Attention mass in the PLAYER local layers (each pass, mean over heads and PLAYER nodes) per child feature.
2. The same ablations for every model, value MSE against the stored labels:
   - remove a leaf: the mana-pool leaves (constant 0 in these games, so no information) and the Battlefield name;
   - cut every leaf from the land entities (nodes with a LAND leaf) and their abilities: permanents, cards, both at
     once; the entity nodes stay; both at once with BattlefieldSize shuffled too;
   - remove the land permanents outright (nodes, abilities, edges) and lower BattlefieldSize to match;
   - shuffle a leaf's numeric value across its occurrences: the tokens stay, only their information goes.
3. With --all instead: every leaf id in the sample removed in turn (all its occurrences, from the states that hold it).
4. With --entities instead: each card and permanent blanked in turn, one per copy of its state: every leaf edge into
   it and its abilities cut (so e.g. a land's name and subtype can't cover for each other), grouped by card name, zone
   and owner.
5. With --add instead: one permanent added to the decision player's battlefield in every state (BattlefieldSize + 1 each
   time): an untapped basic land ($LAND, default Mountain), a blank permanent (the node alone), or nothing (the count alone).
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
from util.land_inject import BASIC, land_template, own_battlefield, parts, rebuild

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


def select(ds, idx, cands=False):
    """ds reduced to states idx, in place, candidates dropped (kept with cands)."""
    nodes = [slice(ds.offsets[s], ds.offsets[s + 1]) for s in idx]
    edges = [slice(ds.edge_offsets[s], ds.edge_offsets[s + 1]) for s in idx]
    ds.ids, ds.values, ds.node_type = (np.concatenate([x[k] for k in nodes]) for x in (ds.ids, ds.values, ds.node_type))
    ds.edge_child, ds.edge_parent, ds.edge_label = (np.concatenate([x[k] for k in edges])
                                                    for x in (ds.edge_child, ds.edge_parent, ds.edge_label))
    ds.offsets = np.concatenate([[0], np.cumsum([k.stop - k.start for k in nodes])]).astype(np.int64)
    ds.edge_offsets = np.concatenate([[0], np.cumsum([k.stop - k.start for k in edges])]).astype(np.int64)
    ds.row, ds.N = ds.row[idx], len(idx)
    if cands:
        c = [slice(ds.cand_offsets[s], ds.cand_offsets[s + 1]) for s in idx]
        ds.cand_node, ds.cand_visits = (np.concatenate([x[k] for k in c]) for x in (ds.cand_node, ds.cand_visits))
        ds.cand_offsets = np.concatenate([[0], np.cumsum([k.stop - k.start for k in c])]).astype(np.int64)
    else:
        ds.cand_node, ds.cand_visits = np.empty(0, np.int32), np.empty(0, np.int32)
        ds.cand_offsets = np.zeros(len(idx) + 1, np.int64)
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
    if model.arch.get("flat_zones"):
        ds.flatten_zones(model.arch["flat_zones"])
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


def entities_one_at_a_time(ds):
    """Copies of ds's states with one card or permanent blanked in each (every leaf edge into it and its abilities cut),
    and per copy: its state, card name id, zone name id, own (the decision player's)."""
    state = np.repeat(np.arange(ds.N), np.diff(ds.offsets))
    e_state = np.repeat(np.arange(ds.N), np.diff(ds.edge_offsets))
    child, parent = ds.edge_child + ds.offsets[e_state], ds.edge_parent + ds.offsets[e_state]
    leaf = ds.node_type[child] == T.LEAF
    up = dict(zip(child[~leaf], parent[~leaf]))
    named = leaf & (ds.edge_label == F("name"))
    cards = set(ds.ids[child[named & (ds.node_type[parent] == T.CARD)]])        # permanents name their card on another edge
    named |= leaf & np.isin(ds.ids[child], list(cards)) & (ds.node_type[parent] == T.PERMANENT)
    name = dict(zip(parent[named], ds.ids[child[named]]))
    decider = set(parent[leaf & (ds.ids[child] == F("IsDecisionPlayer"))])
    ents = np.flatnonzero(np.isin(ds.node_type, [T.PERMANENT, T.CARD]))
    ents = ents[[ds.node_type[up.get(x, x)] == T.ZONE for x in ents]]           # cards in a zone (not choice options)
    info = [(name.get(x, 0), name.get(up.get(x), 0), up.get(up.get(x)) in decider) for x in ents]
    s, local = state[ents], ents - ds.offsets[state[ents]]
    sub = select(copy.deepcopy(ds), s)
    e_state = np.repeat(np.arange(sub.N), np.diff(sub.edge_offsets))
    child, parent = sub.edge_child + sub.offsets[e_state], sub.edge_parent + sub.offsets[e_state]
    target = np.zeros(len(sub.ids), bool); target[sub.offsets[:-1] + local] = True
    entity = target.copy(); entity[child[target[parent] & (sub.node_type[child] == T.ABILITY)]] = True
    keep = ~((sub.node_type[child] == T.LEAF) & entity[parent])
    sub.edge_child, sub.edge_parent, sub.edge_label = sub.edge_child[keep], sub.edge_parent[keep], sub.edge_label[keep]
    sub.edge_offsets = np.concatenate([[0], np.cumsum(np.bincount(e_state[keep], minlength=sub.N))])
    return sub, s, info


def remove_land_permanents(ds):
    """Deletes the land permanents and their abilities (leaves are shared, so they stay) and lowers each battlefield's
    BattlefieldSize by its lands, in place (raw ids)."""
    state = np.repeat(np.arange(ds.N), np.diff(ds.offsets))
    e_state = np.repeat(np.arange(ds.N), np.diff(ds.edge_offsets))
    child, parent = ds.edge_child + ds.offsets[e_state], ds.edge_parent + ds.offsets[e_state]
    leaf = ds.node_type[child] == T.LEAF
    lands = np.unique(parent[leaf & (ds.ids[child] == F("LAND"))])
    zone = dict(zip(child[~leaf], parent[~leaf]))
    size = {p: c for c, p in zip(child, parent) if ds.ids[c] == F("BattlefieldSize")}
    lands = [x for x in lands if ds.node_type[x] == T.PERMANENT and zone.get(x) in size]   # directly on a battlefield
    gone = np.zeros(len(ds.ids), bool); gone[lands] = True
    gone[child[gone[parent] & (ds.node_type[child] == T.ABILITY)]] = True
    for x in lands: ds.values[size[zone[x]]] -= 1
    keep_e = ~(gone[child] | gone[parent]); keep = ~gone
    offsets = np.concatenate([[0], np.cumsum(np.bincount(state[keep], minlength=ds.N))])
    local = np.cumsum(keep) - 1 - offsets[state]
    ds.edge_child, ds.edge_parent = (local[x[keep_e]].astype(np.int16) for x in (child, parent))
    ds.edge_label = ds.edge_label[keep_e]
    ds.edge_offsets = np.concatenate([[0], np.cumsum(np.bincount(e_state[keep_e], minlength=ds.N))])
    ds.ids, ds.values, ds.node_type, ds.offsets = ds.ids[keep], ds.values[keep], ds.node_type[keep], offsets


def shuffle(ds, f):
    m = np.flatnonzero((ds.node_type == T.LEAF) & (ds.ids == F(f)))
    ds.values[m] = ds.values[np.random.default_rng(1).permutation(m)]


def added(ds, kind):
    """ds with one permanent added to each state's decision-player battlefield and its BattlefieldSize raised by 1:
    kind BASIC (an untapped one from the shard: LAND, default Mountain), "blank" (its node alone) or "count only" (raw ids)."""
    (t_ids, t_vals, t_edges), zone_label = land_template(ds)
    if kind == "blank": t_ids, t_vals, t_edges = t_ids[:1], t_vals[:1], t_edges[:0]
    states = []
    for s in range(ds.N):
        ids, st, c, p, l = parts(ds, s)
        _, zone, _ = own_battlefield(ids, st, c, p)
        v = ds.values[ds.offsets[s]:ds.offsets[s + 1]].copy(); n = len(ids)
        v[next(ci for ci, pi in zip(c, p) if pi == zone and ids[ci] == F("BattlefieldSize"))] += 1
        state = [ids, v, c, p, l]
        if kind != "count only":
            state = [np.concatenate([ids, t_ids]), np.concatenate([v, t_vals]),
                     np.concatenate([c, t_edges[:, 0] + n, [n]]), np.concatenate([p, t_edges[:, 1] + n, [zone]]),
                     np.concatenate([l, t_edges[:, 2], [zone_label]])]
        states.append(state)
    rebuild(ds, states, np.arange(ds.N))
    return ds


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
    full, per_entity, add = "--all" in sys.argv, "--entities" in sys.argv, "--add" in sys.argv
    for flag in ("--all", "--entities", "--add"):
        if flag in sys.argv: sys.argv.remove(flag)
    sep = sys.argv.index("--")
    out_path, n, models, shards = sys.argv[1], int(sys.argv[2]), [a.split("=", 1) for a in sys.argv[3:sep]], sys.argv[sep + 1:]
    name = names()
    out = open(out_path, "w")
    def say(s): print(s, flush=True); out.write(s + "\n"); out.flush()

    if add:
        say(f"one permanent added to the decision player's battlefield, {n} states: mean value change, share up / down")
        for label, path in models:
            model, vocab, edge_vocab = load_checkpoint(path); model.to(M.DEVICE).eval()
            base = values(model, sample_states(shards, n), vocab, edge_vocab)
            for kind in (BASIC, "blank", "count only"):
                d = values(model, added(sample_states(shards, n), kind), vocab, edge_vocab) - base
                say(f"  {label:24} {kind:12} {d.mean():+.4f}  {np.mean(d > .005):4.0%} / {np.mean(d < -.005):4.0%}")
        sys.exit()

    if per_entity:
        say(f"one card or permanent blanked at a time, {n} states: card, zone, owner, removals, mean value change "
            f"(decision player's view), mean |change|, value MSE change per removal")
        for label, path in models:
            model, vocab, edge_vocab = load_checkpoint(path); model.to(M.DEVICE).eval()
            ds = sample_states(shards, n); sub, s, info = entities_one_at_a_time(ds)
            base = values(model, ds, vocab, edge_vocab); v = values(model, sub, vocab, edge_vocab)
            d, y = v - base[s], ds.row[s, 0]
            dmse = (v - y) ** 2 - (base[s] - y) ** 2
            groups = {}
            for i, key in enumerate(info): groups.setdefault(key, []).append(i)
            say(f"\n## {label}")
            for (card, zone, own), idx in sorted(groups.items(), key=lambda g: -np.abs(d[g[1]]).mean()):
                if len(idx) >= 20:
                    say(f"  {name.get(int(card), card)!s:28} {name.get(int(zone), zone)!s:12} {'own' if own else 'opp'}  "
                        f"{len(idx):6}  {d[idx].mean():+.4f}  {np.abs(d[idx]).mean():.4f}  {dmse[idx].mean():+.4f}")
        sys.exit()

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
        ds = sample_states(shards, n)
        if model.arch.get("flat_zones"):
            ds.flatten_zones(model.arch["flat_zones"])
        raw = ds.ids.copy(); ds.apply_vocab(vocab, edge_vocab)
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
        ds = sample_states(shards, n); drop_land_leaves(ds, [T.PERMANENT, T.CARD]); shuffle(ds, "BattlefieldSize")
        r["-all land leaves ~BattlefieldSize"] = mse(model, ds, vocab, edge_vocab)
        ds = sample_states(shards, n); remove_land_permanents(ds)
        r["-land permanents, count adjusted"] = mse(model, ds, vocab, edge_vocab)
        for f in SHUFFLE:
            ds = sample_states(shards, n); shuffle(ds, f)
            r["~" + f] = mse(model, ds, vocab, edge_vocab)

    cols = list(next(iter(rows.values())))
    say("\n== value MSE: base, then the change from base (-x: leaf x removed, ~x: x's value shuffled across states)")
    say("| model | " + " | ".join(cols) + " |")
    say("|---" * (len(cols) + 1) + "|")
    for label, r in rows.items():
        say(f"| {label} | {r['base']:.3f} | " + " | ".join(f"{r[c] - r['base']:+.3f}" for c in cols[1:]) + " |")
