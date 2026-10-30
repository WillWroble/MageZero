"""Value prediction histograms and calibration per model, from util/value_preds.py output:
  python results/value_preds_plot.py OUT.png NAME=FILE.npz ...   (prints the table; one column of panels per file)"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

out, sets = sys.argv[1], [a.split("=", 1) for a in sys.argv[2:]]
data = {name: np.load(path) for name, path in sets}
models = [k for k in next(iter(data.values())) if k not in ("label", "root", "result")]
bins = np.linspace(-1, 1, 41)
mid = (bins[:-1] + bins[1:]) / 2
cal_edges = [-1, -0.5, -0.2, 0.2, 0.5, 1]


def auc(x, y):
    r = np.argsort(np.argsort(x)) + 1
    p = y.sum()
    return (r[y].sum() - p * (p + 1) / 2) / (p * (len(y) - p))


print("| set | model | MSE label | MSE result | AUC result | corr root | std | beyond ±0.5 | calibration: pred bin mean -> result mean |")
print("|---|---|---|---|---|---|---|---|---|")
fig, axes = plt.subplots(len(models), len(sets), figsize=(5 * len(sets), 1.9 * len(models)), sharex=True, squeeze=False)
INK, MUTED = "#1f1f1e", "#8a8983"
for c, (name, d) in enumerate(data.items()):
    label, result, root = d["label"], d["result"], d["root"]
    for r, m in enumerate(models):
        p = d[m]
        b = np.digitize(p, cal_edges[1:-1])
        cal = " ".join(f"{p[b == i].mean():+.2f}->{result[b == i].mean():+.2f}" for i in range(5) if (b == i).any())
        print(f"| {name} | {m} | {((p - label) ** 2).mean():.4f} | {((p - result) ** 2).mean():.3f} | "
              f"{auc(p, result > 0):.3f} | {np.corrcoef(p, root)[0, 1]:.3f} | {p.std():.3f} | {(np.abs(p) > 0.5).mean():.2f} | {cal} |")
        ax = axes[r, c]
        ax.bar(mid, np.histogram(p, bins)[0] / len(p), width=0.046, color="#2a78d6", label="prediction")
        ax.step(bins[:-1], np.histogram(label, bins)[0] / len(p), where="post", color="#eb6834", lw=2, label="label (TD 0.92)")
        ax.set_title(f"{m}, {name}", fontsize=9, color=INK, loc="left")
        ax.set_yticks([])
        ax.tick_params(colors=MUTED, labelsize=8, length=0)
        for s in ax.spines.values():
            s.set_visible(False)
axes[0, 0].legend(fontsize=8, frameon=False, loc="upper left")
fig.tight_layout()
fig.savefig(out, dpi=110, facecolor="white")
