"""Keep every K-th game (games 0, K, 2K, ...) of every shard under a directory, keeping its layout:
  python thin_games.py src_dir dst_dir K"""
import sys
from pathlib import Path

import h5py

from cut_games import take

src, dst, k = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3])
for path in sorted(list(src.rglob("*.h5")) + list(src.rglob("*.hdf5"))):
    rel = path.relative_to(src)
    if (dst / rel).exists():
        continue
    with h5py.File(path) as f:
        games = len(f["/game_offsets"]) - 1
    (dst / rel).parent.mkdir(parents=True, exist_ok=True)
    take(str(path), str(dst / rel), range(0, games, k))
