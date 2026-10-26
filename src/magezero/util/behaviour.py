"""Faeries behaviour in one shard, from the decision player's seat: Seer vs Miscreant, value labels by board, and
how often each ability is the top move when available. PYTHONPATH=. python util/behaviour.py shard   (from src/magezero)"""
import sys
from collections import Counter, defaultdict
import h5py, numpy as np
from vocab import feature_id
from dataset import node_types
from model import NodeType as T

F = feature_id
ABIL = {2082092948: "Seer", 1775296276: "Miscreant", 1254598873: "Island", 1758931328: "Elder", 913115100: "Spellstutter",
        10181138: "Snap", 653878615: "Counterspell", 612652355: "Force Spike", 356170266: "Dispel", 1564034202: "Saiba",
        691519731: "Strix loot", 437330601: "Harrier Strix", 235080923: "Ninja", 1211918040: "ninjutsu A",
        758602737: "ninjutsu B", 1370559227: "Of One Mind", 679880620: "Pass"}
STEPS = {F(s): s for s in "UPKEEP DRAW PRECOMBAT_MAIN BEGIN_COMBAT DECLARE_ATTACKERS DECLARE_BLOCKERS COMBAT_DAMAGE END_COMBAT POSTCOMBAT_MAIN END_TURN CLEANUP".split()}
NAME = F("name")
DEC, ACT, BF = F("IsDecisionPlayer"), F("IsActivePlayer"), F("Battlefield")
MISC, SEER = F("Faerie Miscreant"), F("Faerie Seer")

with h5py.File(sys.argv[1]) as f:
    d = {k: f[k][...] for k in ("indices", "offsets", "edge_child", "edge_parent", "edge_label", "edge_offsets",
                                 "policy_node", "policy_visits", "policy_offsets", "row", "game_offsets")}
ids, off, row = d["indices"], d["offsets"], d["row"]
types = node_types(ids)
n = len(row)
game = np.repeat(np.arange(len(d["game_offsets"]) - 1), np.diff(d["game_offsets"]))
won = (row[d["game_offsets"][1:] - 1, 0] > 0)[game]

stats = Counter()
both = defaultdict(Counter)            # Seer-vs-Miscreant top choice, by Miscreant already on board
board_labels = defaultdict(list)       # value labels by Faeries' board
avail, chosen = Counter(), Counter()   # ability available / chosen as top
own_turn = defaultdict(Counter)        # chosen ability, own turn vs opponent's
for s in range(n):
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
    players = np.flatnonzero(st == T.PLAYER)
    me = next((x for x in players if DEC in kids[x]), None)
    active = next((x for x in players if ACT in kids[x]), None)
    step = next((STEPS[i] for i in kids[np.flatnonzero(st == T.ROOT)[0]] if i in STEPS), "?")
    # Faeries' battlefield
    mine = [x for x in np.flatnonzero(st == T.PERMANENT)
            if name.get(parent.get(x)) == BF and parent.get(parent.get(x)) == me]
    misc = sum(MISC in kids[x] for x in mine)
    seer = sum(SEER in kids[x] for x in mine)
    if misc >= 2: board_labels["2+ Miscreants"].append((row[s, 0], won[s]))
    elif misc == 1 and seer == 0: board_labels["one Miscreant only"].append((row[s, 0], won[s]))
    elif seer >= 1 and misc == 0: board_labels["Seer only"].append((row[s, 0], won[s]))
    if row[s, 3] != 0:
        continue
    q0, q1 = d["policy_offsets"][s], d["policy_offsets"][s + 1]
    cand, vis = d["policy_node"][q0:q1], d["policy_visits"][q0:q1]
    if len(cand) < 2 or vis.sum() == 0:
        continue
    names = [ABIL.get(int(name.get(x, -1)), "other") for x in cand]
    top = names[int(np.argmax(vis))]
    for a in set(names):
        avail[a] += 1
    chosen[top] += 1
    own_turn[top]["own" if me == active else "opp"] += 1
    own_turn[top][step] += 1
    if "Seer" in names and "Miscreant" in names:
        both[misc > 0][top] += 1

games = len(d["game_offsets"]) - 1
print(f"== {sys.argv[1].split('/training/')[-1]}  games {games}  states/game {n / games:.1f}  "
      f"win {np.mean(row[d['game_offsets'][1:] - 1, 0] > 0):.3f}")
print("Seer vs Miscreant, priority states where both are castable (share of states each is the top move):")
for k, label in ((False, "no Miscreant on Faeries' battlefield"), (True, "a Miscreant already out")):
    t = sum(both[k].values())
    if t:
        print(f"  {label:38s} Seer {both[k]['Seer'] / t:.2f} / Miscreant {both[k]['Miscreant'] / t:.2f}  ({t} states)")
print("Value labels by Faeries' board (correlational):")
for k in ("Seer only", "one Miscreant only", "2+ Miscreants"):
    v = np.array(board_labels[k])
    if len(v):
        print(f"  {k:20s} median {np.median(v[:, 0]):+.3f}  mean {v[:, 0].mean():+.3f}  from won games {v[:, 1].mean():.2f}  ({len(v)} states)")
print("Chosen when available (priority states with a choice), own-turn share of those choices:")
for a in sorted(avail, key=lambda a: -avail[a]):
    if a in ("other",) or avail[a] < 30:
        continue
    o = own_turn[a]
    print(f"  {a:14s} available {avail[a]:6d}  chosen {chosen[a] / avail[a]:.2f}  own turn {o['own'] / max(chosen[a], 1):.2f}")
