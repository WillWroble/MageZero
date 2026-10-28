"""Hold out the last games of every shard under a directory, keeping its layout (shards already split are skipped):
  python split_games.py src_dir train_dir test_dir [test_fraction=0.2] [skip substring ...]"""
import sys
from pathlib import Path

import h5py

from cut_games import cut

src, train, test = (Path(a) for a in sys.argv[1:4])
fraction = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
skip = sys.argv[5:]
for path in sorted(list(src.rglob("*.h5")) + list(src.rglob("*.hdf5"))):
    rel = path.relative_to(src)
    if any(s in path.name for s in skip) or (train / rel).exists() and (test / rel).exists():
        continue
    with h5py.File(path) as f:
        games = len(f["/game_offsets"]) - 1
    held = max(1, round(games * fraction))
    for root, start, end in ((train, 0, games - held), (test, games - held, games)):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        cut(str(path), str(root / rel), start, end)
