"""MCTS root decisions from an XMage console log (MCTSNode.bestChild lines): visit concentration, how often the most
visited action is not the best scored, and Faeries priority choices.  python results/mcts_log.py console.txt"""
import re, sys, collections, numpy as np
FAE = ["Spellstutter Sprite","Humbling Elder","Snap","Faerie Seer","Of One Mind","Faerie Miscreant","Harrier Strix",
       "Counterspell","Moon-Circuit Hacker","Force Spike","Saiba Cryptomancer","Ninja of the Deep Hours","Dispel","Island"]
ACT = re.compile(r"\[([^\[\]]+?) score: (-?[\d.]+) count: (\d+)\]")
stats = collections.defaultdict(list)
avail, chosen = collections.Counter(), collections.Counter()
for line in open(sys.argv[1]):
    if "bestChild" not in line or "actions:" not in line: continue
    acts = ACT.findall(line.split("actions:")[1])
    if len(acts) < 2: continue
    names = [a[0] for a in acts]; s = np.array([float(a[1]) for a in acts]); c = np.array([int(a[2]) for a in acts])
    prio = "Pass" in names
    fae = any(any(f in n for f in FAE) for n in names if n.startswith(("Cast", "Play")))
    seat = "Faeries" if fae else "MonoB"
    kind = "priority" if prio else "other"
    top = c.argmax()
    stats[seat, kind].append((c[top] / c.sum(), s.argmax() != top, c.sum(), s[top]))
    if prio and fae:
        for n in set(names):
            k = n.replace("Cast ", "").replace("Play ", "")
            avail[k] += 1; chosen[k] += names[top] == n
for k, v in sorted(stats.items()):
    v = np.array(v)
    print(k, f"n={len(v)} top visit share={v[:,0].mean():.2f} most-visited != best-scored {v[:,1].mean():.2f} visits={np.median(v[:,2]):.0f} root(top) mean={v[:,3].mean():+.3f}")
print("Faeries priority, chosen when available:")
for k, n in avail.most_common(20):
    print(f"  {k:28s} avail {n:5d} chosen {chosen[k]/n:.2f}")
# Faeries priority decisions where the most-visited action is not the best-scored
pairs, gaps = collections.Counter(), []
for line in open(sys.argv[1]):
    if "bestChild" not in line or "actions:" not in line: continue
    acts = ACT.findall(line.split("actions:")[1])
    names = [a[0] for a in acts]
    if len(acts) < 2 or "Pass" not in names or not any(any(f in n for f in FAE) for n in names if n.startswith(("Cast", "Play"))): continue
    s = np.array([float(a[1]) for a in acts]); c = np.array([int(a[2]) for a in acts])
    t, b = c.argmax(), s.argmax()
    if t != b:
        gaps.append((s[b] - s[t], c[b] / c.sum()))
        pairs[names[t][:30], names[b][:30]] += 1
g = np.array(gaps)
print(f"mismatches {len(g)}: best-scored beats chosen by median {np.median(g[:,0]):.3f}, mean {g[:,0].mean():.3f}; best-scored gets median {np.median(g[:,1]):.2f} of visits")
for (t, b), n in pairs.most_common(12): print(f"  chose {t:30s} over {b:30s} {n}")
