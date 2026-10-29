"""Value label vs root value histograms by data bucket, from util/value_hist.py output:
  python results/value_hist_plot.py value-hist.json [BARS LINE]   (prints the table)
BARS and LINE are histogram keys (default label, score; label_<td> are labels recomputed with that TD discount).
Writes results/value_histograms.png, or results/value_histograms_<BARS>_<LINE>.png for other keys."""
import json, re, sys
import numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

d = json.load(open(sys.argv[1]))
BARS, LINE = sys.argv[2:4] if len(sys.argv) > 3 else ("label", "score")
NAMES = {"label": "value label (TD 0.92)", "score": "search root value"}
NAMES.update({k: f"value label (TD {k[6:]})" for k in next(iter(d.values())) if k.startswith("label_")})
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
        a = agg.setdefault(b, {BARS: np.zeros(40), LINE: np.zeros(40), "games": 0, "wins": 0})
        a[BARS] += v[BARS]; a[LINE] += v[LINE]; a["games"] += v["games"]; a["wins"] += v["wins"]

x = np.linspace(-1, 1, 41); mid = (x[:-1] + x[1:]) / 2
fig, axes = plt.subplots(len(BUCKETS), 2, figsize=(10, 2.0 * len(BUCKETS)), sharex=True)
INK, MUTED = "#1f1f1e", "#8a8983"
print(f"| bucket | seat | games | seat WR | states | {NAMES[BARS]} mean | beyond ±0.5 | {NAMES[LINE]} mean | beyond ±0.5 |")
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
        n = a[BARS].sum()
        ax.bar(mid, a[BARS] / n, width=0.046, color="#2a78d6", label=NAMES[BARS])
        ax.step(x[:-1], a[LINE] / n, where="post", color="#eb6834", lw=2, label=NAMES[LINE])
        ax.axvline(0, color=MUTED, lw=0.6)
        seatname = "Faeries seat" if seat == "PlayerA" else "opponent seat"
        ax.set_title(f"{b}, {seatname}: {a['games']} games, WR {a['wins'] / a['games']:.0%}", fontsize=9, color=INK, loc="left")
        lm = (a[BARS] * mid).sum() / n; sm = (a[LINE] * mid).sum() / n
        print(f"| {b} | {seatname} | {a['games']} | {a['wins'] / a['games']:.1%} | {int(n):,} | {lm:+.2f} | "
              f"{a[BARS][np.abs(mid) > 0.5].sum() / n:.2f} | {sm:+.2f} | {a[LINE][np.abs(mid) > 0.5].sum() / n:.2f} |")
axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
for ax in axes[-1]: ax.set_xlabel("value (seat's perspective)", fontsize=9, color=MUTED)
fig.suptitle(f"{NAMES[BARS][0].upper() + NAMES[BARS][1:]} vs {NAMES[LINE]}, ver4/ver5 training shards (share of states)", fontsize=11, color=INK, x=0.01, ha="left")
fig.tight_layout()
fig.savefig("results/value_histograms.png" if (BARS, LINE) == ("label", "score")
            else f"results/value_histograms_{BARS}_{LINE}.png", dpi=110, facecolor="white")
