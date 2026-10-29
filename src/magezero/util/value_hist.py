"""Histograms of the value label (resultLabel) and the search's root value (stateScore) per shard, as JSON:
  python util/value_hist.py ROOT OUT.json [TD ...]   (every .h5/.hdf5 under ROOT; 40 bins on [-1, 1])
Each TD also gives the labels recomputed with that discount, as XMage builds them (ParallelDataGenerator)."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

BINS = np.linspace(-1, 1, 41)


def td_labels(score, go, won, td):
    """label[i] = td * label[i + 1] + (1 - td) * score[i] within each game, from the result (+1 / -1) at its end."""
    label = np.empty_like(score)
    for g, (a, b) in enumerate(zip(go[:-1], go[1:])):
        x = 1.0 if won[g] else -1.0
        for i in range(b - 1, a - 1, -1):
            x = td * x + (1 - td) * score[i]
            label[i] = x
    return label


def hist(x):
    return np.histogram(np.clip(x, -1, 1), BINS)[0].tolist()


out = {}
for p in sorted(list(Path(sys.argv[1]).rglob("*.h5")) + list(Path(sys.argv[1]).rglob("*.hdf5"))):
    try:
        with h5py.File(p, "r") as f:
            row, go = f["/row"][:, :2], f["/game_offsets"][...]
    except (OSError, KeyError) as e:
        print("skip", p, e)
        continue
    won = row[go[1:] - 1, 0] > 0
    out[str(p)] = {"games": len(go) - 1, "wins": int(won.sum()), "label": hist(row[:, 0]), "score": hist(row[:, 1])}
    for td in sys.argv[3:]:
        out[str(p)][f"label_{td}"] = hist(td_labels(row[:, 1].astype(np.float64), go, won, float(td)))
    print(p, out[str(p)]["games"])
Path(sys.argv[2]).write_text(json.dumps(out))
