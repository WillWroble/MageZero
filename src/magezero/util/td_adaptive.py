"""Value labels rebuilt with a per-game discount gamma = w ** (1 / N) (N = the game's states in the shard), so the
result has weight w at the first state, next to the fixed TD 0.92. Per shard and setting: calibrated opening value
(2 * win rate - 1), mean label in the first 10% of each game, and the label's mean change per 10% of the game.
  python util/td_adaptive.py OUT.json name=shard ...   (h5py and numpy only)"""
import json
import sys

import h5py
import numpy as np

SETTINGS = {"fixed 0.92": None, "w=0.05": 0.05, "w=0.1": 0.1, "w=0.25": 0.25}


def labels(score, go, won, w):
    out = np.empty_like(score)
    for g, (a, b) in enumerate(zip(go[:-1], go[1:])):
        td = 0.92 if w is None else w ** (1.0 / (b - a))
        x = 1.0 if won[g] else -1.0
        for i in range(b - 1, a - 1, -1):
            x = td * x + (1 - td) * score[i]
            out[i] = x
    return out


def summary(path):
    with h5py.File(path) as f:
        row, go = f["row"][...], f["game_offsets"][...]
    go = go[np.r_[True, np.diff(go) > 0]]
    won = row[go[1:] - 1, 0] > 0
    res = {"games": len(go) - 1, "calibrated": float(2 * won.mean() - 1)}
    for name, w in SETTINGS.items():
        lab = labels(row[:, 1].astype(np.float64), go, won, w)
        bins = np.zeros(10); n = np.zeros(10)
        for a, b in zip(go[:-1], go[1:]):
            k = np.minimum((np.arange(b - a) / (b - a) * 10).astype(int), 9)
            np.add.at(bins, k, lab[a:b]); np.add.at(n, k, 1)
        curve = bins / np.maximum(n, 1)
        res[name] = {"opening": float(curve[0]), "drift": float(np.mean(np.diff(curve))), "curve": curve.round(3).tolist()}
    return res


if __name__ == "__main__":
    out = {}
    for arg in sys.argv[2:]:
        name, path = arg.split("=", 1)
        out[name] = summary(path)
        print(name, flush=True)
    json.dump(out, open(sys.argv[1], "w"))
