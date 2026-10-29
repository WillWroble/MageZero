"""Value label vs root value histograms by data bucket, from util/value_hist.py output:
  python results/value_hist_plot.py value-hist.json   (writes results/value_histograms.png, prints the table)"""
import json, re, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
DUP = "ver4/training/gen0/PlayerB/Pauper-Faeries_vs_Standard-MonoW.hdf5"  # same file as the PlayerA one
BUCKETS = ["gen0 Pauper", "gen0 Mono", "gen1 reg Pauper", "gen1 reg Mono", "gen1 self-play Pauper",
           "gen1 self-play Mono", "gen1 PIMC (MonoB)", "gen1 policy (MonoB)"]

def bucket(k):
    m = re.search(r"Pauper-Faeries/ver[45]/training/(gen[01])/(Player[AB])/(.*)\.hdf5", k)
    if not m or k.endswith(DUP):
        return None
    gen, seat, name = m.groups()
    pool = "Mono" if "Standard-Mono" in name else "Pauper"
    kind = ("PIMC (MonoB)" if "_pimc" in name else "policy (MonoB)" if "_policy" in name
            else "self-play " + pool if "_selfplay" in name else "reg " + pool)
    return (f"{gen} {pool}" if gen == "gen0" else f"gen1 {kind}"), seat

agg = {}
for k, v in d.items():
    b = bucket(k)
    if b:
        a = agg.setdefault(b, {"label": np.zeros(40), "score": np.zeros(40), "games": 0, "wins": 0})
        a["label"] += v["label"]; a["score"] += v["score"]; a["games"] += v["games"]; a["wins"] += v["wins"]

x = np.linspace(-1, 1, 41); mid = (x[:-1] + x[1:]) / 2
fig, axes = plt.subplots(len(BUCKETS), 2, figsize=(10, 2.0 * len(BUCKETS)), sharex=True)
INK, MUTED = "#1f1f1e", "#8a8983"
print("| bucket | seat | games | seat WR | states | label mean | share |label| > 0.5 | root mean | share |root| > 0.5 |")
print("|---|---|---|---|---|---|---|---|---|")
for r, b in enumerate(BUCKETS):
    for c, seat in enumerate(["PlayerA", "PlayerB"]):
        ax = axes[r, c]
        a = agg.get((b, seat))
        for s in ax.spines.values(): s.set_visible(False)
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        ax.set_yticks([])
        if a is None:
            continue
        n = a["label"].sum()
        ax.bar(mid, a["label"] / n, width=0.046, color="#2a78d6", label="value label (TD 0.92)")
        ax.step(x[:-1], a["score"] / n, where="post", color="#eb6834", lw=2, label="search root value")
        ax.axvline(0, color=MUTED, lw=0.6)
        seatname = "Faeries seat" if seat == "PlayerA" else "opponent seat"
        ax.set_title(f"{b}, {seatname}: {a['games']} games, WR {a['wins'] / a['games']:.0%}", fontsize=9, color=INK, loc="left")
        lm = (a["label"] * mid).sum() / n; sm = (a["score"] * mid).sum() / n
        print(f"| {b} | {seatname} | {a['games']} | {a['wins'] / a['games']:.1%} | {int(n):,} | {lm:+.2f} | "
              f"{a['label'][np.abs(mid) > 0.5].sum() / n:.2f} | {sm:+.2f} | {a['score'][np.abs(mid) > 0.5].sum() / n:.2f} |")
axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
for ax in axes[-1]: ax.set_xlabel("value (seat's perspective)", fontsize=9, color=MUTED)
fig.suptitle("Value labels vs search root values, ver4/ver5 training shards (share of states)", fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("results/value_histograms.png", dpi=110, facecolor="white")
