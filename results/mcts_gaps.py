"""How decisive the search is at Faeries priority decisions, from XMage console logs (MCTSNode.bestChild lines):
score gap between the best and second option, spread of option scores, Pass vs the best other option.
  python results/mcts_gaps.py console.txt ..."""
import re, sys, numpy as np
FAE = ["Spellstutter Sprite","Humbling Elder","Snap","Faerie Seer","Of One Mind","Faerie Miscreant","Harrier Strix",
       "Counterspell","Moon-Circuit Hacker","Force Spike","Saiba Cryptomancer","Ninja of the Deep Hours","Dispel","Island"]
ACT = re.compile(r"\[([^\[\]]+?) score: (-?[\d.]+) count: (\d+)\]")
for path in sys.argv[1:]:
    gap12, passgap, spread = [], [], []
    for line in open(path):
        if "bestChild" not in line or "actions:" not in line: continue
        acts = ACT.findall(line.split("actions:")[1]); names = [a[0] for a in acts]
        if len(acts) < 2 or "Pass" not in names or not any(any(f in n for f in FAE) for n in names if n.startswith(("Cast", "Play"))): continue
        s = np.array([float(a[1]) for a in acts]); c = np.array([int(a[2]) for a in acts])
        o = np.sort(s)[::-1]; gap12.append(o[0] - o[1]); spread.append(s.max() - s.min())
        i = names.index("Pass"); passgap.append(s[i] - np.delete(s, i).max())
    g, p, sp = map(np.array, (gap12, passgap, spread))
    print(path.split("/")[-1][:12], f"n={len(g)} best-vs-2nd score gap median {np.median(g):.3f} (<0.02: {(g<0.02).mean():.2f})  "
          f"options spread median {np.median(sp):.3f}  Pass minus best other: median {np.median(p):+.3f}, Pass best {(p>0).mean():.2f}")
