"""Histograms of the value label (resultLabel) and the search's root value (stateScore) per shard, as JSON:
  python util/value_hist.py ROOT OUT.json   (every .h5/.hdf5 under ROOT; 40 bins on [-1, 1])"""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

BINS = np.linspace(-1, 1, 41)

out = {}
for p in sorted(list(Path(sys.argv[1]).rglob("*.h5")) + list(Path(sys.argv[1]).rglob("*.hdf5"))):
    try:
        with h5py.File(p, "r") as f:
            row, go = f["/row"][:, :2], f["/game_offsets"][...]
    except (OSError, KeyError) as e:
        print("skip", p, e)
        continue
    out[str(p)] = {"games": len(go) - 1, "wins": int((row[go[1:] - 1, 0] > 0).sum()),
                   "label": np.histogram(np.clip(row[:, 0], -1, 1), BINS)[0].tolist(),
                   "score": np.histogram(np.clip(row[:, 1], -1, 1), BINS)[0].tolist()}
    print(p, out[str(p)]["games"])
Path(sys.argv[2]).write_text(json.dumps(out))
