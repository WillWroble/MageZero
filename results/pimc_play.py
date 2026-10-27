"""Faeries' play per shard: when it acts, how it uses counters and flash, search confidence, value calibration."""
import sys, h5py, numpy as np
from collections import Counter
from dataset import node_types
from model import NodeType as T, ActionType
from vocab import feature_id as F

NAMES = {2082092948: "Seer", 1775296276: "Miscreant", 1254598873: "Island", 1758931328: "Elder", 913115100: "Spellstutter",
         10181138: "Snap", 653878615: "Counterspell", 612652355: "Force Spike", 356170266: "Dispel", 1564034202: "Saiba",
         691519731: "Strix loot", 437330601: "Harrier Strix", 235080923: "Ninja", 1211918040: "ninjutsu A",
         758602737: "ninjutsu B", 1370559227: "Of One Mind", 679880620: "Pass"}
COUNTERS = {"Counterspell", "Force Spike", "Dispel"}
NAME, DEC, ACT = F("name"), F("IsDecisionPlayer"), F("IsActivePlayer")

for path in sys.argv[1:]:
    with h5py.File(path) as f:
        d = {k: f[k][...] for k in ("indices", "offsets", "edge_child", "edge_parent", "edge_label", "edge_offsets",
                                     "policy_node", "policy_visits", "policy_offsets", "row", "game_offsets")}
    ids, off, row, go = d["indices"], d["offsets"], d["row"], d["game_offsets"]
    types = node_types(ids)
    games = len(go) - 1
    won = row[go[1:] - 1, 0] > 0
    c = Counter()
    top_share = {True: [], False: []}
    for s in range(len(row)):
        if row[s, 3] != ActionType.PRIORITY.value:
            continue
        p0, p1 = d["policy_offsets"][s], d["policy_offsets"][s + 1]
        cand, visits = d["policy_node"][p0:p1], d["policy_visits"][p0:p1]
        if len(cand) < 2 or visits.sum() == 0:
            continue
        n0, e0, e1 = off[s], d["edge_offsets"][s], d["edge_offsets"][s + 1]
        ch, pa, lb = d["edge_child"][e0:e1], d["edge_parent"][e0:e1], d["edge_label"][e0:e1]
        sid, st = ids[n0:off[s + 1]], types[n0:off[s + 1]]
        name = {int(p): NAMES.get(int(sid[cc]), "other") for cc, p, l in zip(ch, pa, lb) if l == NAME}
        decider = pa[sid[ch] == DEC][0]
        own = bool(np.any((sid[ch] == ACT) & (pa == decider)))
        stack = bool(np.any(st == T.STACK_OBJECT))
        top = name.get(int(cand[np.argmax(visits)]), "other")
        top_share[own].append(visits.max() / visits.sum())
        side = "own" if own else "opp"
        c[f"choices {side}"] += 1
        c[f"acted {side}"] += top != "Pass"
        if top == "Spellstutter":
            c["Spellstutter " + ("own turn" if own else "opp turn, stack " + ("non-empty" if stack else "empty"))] += 1
        if top in COUNTERS:
            c["counterspells"] += 1
        if not own and any(name.get(int(x)) in COUNTERS for x in cand):
            c["counter available opp turn"] += 1
            c["counter used when available"] += top in COUNTERS
        if not own and any(name.get(int(x)) in ("Spellstutter", "Elder", "Strix loot") for x in cand):
            c["flash available opp turn"] += 1
            c["flash used opp turn"] += top in ("Spellstutter", "Elder", "Strix loot")
    # value calibration: the search's root value (row[:,1]) against the game result, by thirds of the game
    pos = np.concatenate([np.linspace(0, 1, n, endpoint=False) for n in np.diff(go)])
    result = np.repeat(np.where(won, 1.0, -1.0), np.diff(go))
    cal = "  ".join(f"{name} {row[m, 1].mean():+.2f} vs {result[m].mean():+.2f}"
                    for name, m in (("early", pos < 1 / 3), ("mid", (pos >= 1 / 3) & (pos < 2 / 3)), ("late", pos >= 2 / 3)))
    g = lambda k: c[k] / games
    print(f"== {path.split('/ver')[1]}  games {games}  WR {won.mean():.1%}")
    print(f"  priority choices per game: own turn {g('choices own'):.1f} (acts {c['acted own'] / c['choices own']:.0%}), "
          f"opp turn {g('choices opp'):.1f} (acts {c['acted opp'] / c['choices opp']:.0%})")
    print(f"  Spellstutter per game: own turn {g('Spellstutter own turn'):.2f}, opp turn onto a spell {g('Spellstutter opp turn, stack non-empty'):.2f}, "
          f"opp turn empty stack {g('Spellstutter opp turn, stack empty'):.2f}")
    print(f"  counterspells per game {g('counterspells'):.2f}; on opp turn with one castable, used {c['counter used when available'] / max(c['counter available opp turn'], 1):.0%} "
          f"({g('counter available opp turn'):.1f} such states/game)")
    print(f"  flash creature/loot on opp turn when castable: {c['flash used opp turn'] / max(c['flash available opp turn'], 1):.0%} ({g('flash available opp turn'):.1f} states/game)")
    print(f"  search: top move's visit share own turn {np.mean(top_share[True]):.2f}, opp turn {np.mean(top_share[False]):.2f}")
    print(f"  root value vs result (mean): {cal}")
