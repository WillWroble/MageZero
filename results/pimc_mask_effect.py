"""The v4 network on the same perfect-info states, with and without the opponent's hand: value and move probabilities."""
import sys, numpy as np, torch
from collections import defaultdict
from dataset import H5Graphs, collate_graphs, states_of
from model import load_checkpoint, NodeType as T, ActionType
from vocab import feature_id as F

NAMES = {913115100: "Spellstutter", 1758931328: "Elder", 691519731: "Strix loot", 653878615: "Counterspell",
         612652355: "Force Spike", 356170266: "Dispel", 679880620: "Pass", 1775296276: "Miscreant", 2082092948: "Seer",
         1254598873: "Island", 1564034202: "Saiba", 10181138: "Snap"}
FLASH, COUNTERS = {"Spellstutter", "Elder", "Strix loot"}, {"Counterspell", "Force Spike", "Dispel"}
NAME, DEC, ACT = F("name"), F("IsDecisionPlayer"), F("IsActivePlayer")

model, vocab, edge_vocab = load_checkpoint(sys.argv[1]); model.eval()
full, masked = (H5Graphs(sys.argv[2], vocab, edge_vocab, h) for h in (False, True))

def info(ds, k):
    n0, e0, e1 = ds.offsets[k], ds.edge_offsets[k], ds.edge_offsets[k + 1]
    ch, pa, lb = ds.edge_child[e0:e1], ds.edge_parent[e0:e1], ds.edge_label[e0:e1]
    ids = ds.ids[n0:ds.offsets[k + 1]]
    name = {int(p): NAMES.get(int(ids[c]), "other") for c, p, l in zip(ch, pa, lb) if l == NAME}
    decider = pa[ids[ch] == DEC][0]
    own = bool(np.any((ids[ch] == ACT) & (pa == decider)))
    cand = ds.cand_node[ds.cand_offsets[k]:ds.cand_offsets[k + 1]]
    return own, [name.get(int(c), "other") for c in cand]

def run(ds, idx):
    g = collate_graphs(*zip(*(ds[k][:6] for k in idx)))
    with torch.no_grad():
        pri, _, _, val = model(g)
    out = []
    for i, k in enumerate(idx):
        cand = ds.cand_node[ds.cand_offsets[k]:ds.cand_offsets[k + 1]] + int(g.node_offsets[i])
        s = pri[cand]
        out.append((float(val[i]), torch.softmax(s, 0).numpy() if len(cand) else None))
    return out

prio = [k for k in range(len(full)) if full.row[k, 3] == ActionType.PRIORITY.value
        and full.cand_offsets[k + 1] - full.cand_offsets[k] >= 2]
rng = np.random.default_rng(0)
sample = sorted(rng.choice(prio, min(len(prio), int(sys.argv[3])), replace=False))
acc = defaultdict(list)
for b in range(0, len(sample), 64):
    idx = sample[b:b + 64]
    for k, (vf, pf), (vm, pm) in zip(idx, run(full, idx), run(masked, idx)):
        own, names = info(full, k)
        assert names == info(masked, k)[1]
        side = "own" if own else "opp"
        visits = full.cand_visits[full.cand_offsets[k]:full.cand_offsets[k + 1]]
        acc["value full"].append(vf); acc["value masked"].append(vm)
        acc[f"pass {side} full"].append(sum(p for p, n in zip(pf, names) if n == "Pass"))
        acc[f"pass {side} masked"].append(sum(p for p, n in zip(pm, names) if n == "Pass"))
        acc[f"top=mcts full"].append(np.argmax(pf) == np.argmax(visits)); acc[f"top=mcts masked"].append(np.argmax(pm) == np.argmax(visits))
        for group, members in (("flash", FLASH), ("counter", COUNTERS)):
            if any(n in members for n in names):
                for v, p in (("full", pf), ("masked", pm)):
                    acc[f"{group} {side} {v}"].append(sum(x for x, n in zip(p, names) if n in members))
        if "Spellstutter" in names:
            for v, p in (("full", pf), ("masked", pm)):
                acc[f"Spellstutter {side} {v}"].append(sum(x for x, n in zip(p, names) if n == "Spellstutter"))
print(f"{len(sample)} Faeries priority states with a choice (perfect-info games), v4 network, full vs opponent hand masked:")
vf, vm = np.array(acc["value full"]), np.array(acc["value masked"])
print(f"  value: full {vf.mean():+.3f}  masked {vm.mean():+.3f}  (masked higher in {np.mean(vm > vf):.0%} of states, mean shift {np.mean(vm - vf):+.3f})")
print(f"  policy top move = MCTS's: full {np.mean(acc['top=mcts full']):.1%}  masked {np.mean(acc['top=mcts masked']):.1%}")
for key in ("pass own", "pass opp", "flash own", "flash opp", "Spellstutter own", "Spellstutter opp", "counter opp"):
    f, m = acc[f"{key} full"], acc[f"{key} masked"]
    if f:
        print(f"  P({key.split()[0]}) on {'own' if 'own' in key else 'opp'} turn when available ({len(f)} states): full {np.mean(f):.3f}  masked {np.mean(m):.3f}")
