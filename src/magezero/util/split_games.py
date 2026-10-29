"""Hold out games of every shard under a directory, keeping its layout (shards already split are skipped):
  python split_games.py src_dir train_dir test_dir [--fraction 0.2 | --every K] [--skip substring ...]
--fraction holds out each shard's last games; --every K holds out every K-th game (games K-1, 2K-1, ...)."""
import argparse
from pathlib import Path

import h5py

from cut_games import take

parser = argparse.ArgumentParser()
parser.add_argument("src", type=Path)
parser.add_argument("train", type=Path)
parser.add_argument("test", type=Path)
parser.add_argument("--fraction", type=float, default=0.2)
parser.add_argument("--every", type=int)
parser.add_argument("--skip", nargs="*", default=[])
args = parser.parse_args()

for path in sorted(list(args.src.rglob("*.h5")) + list(args.src.rglob("*.hdf5"))):
    rel = path.relative_to(args.src)
    if any(s in path.name for s in args.skip) or (args.train / rel).exists() and (args.test / rel).exists():
        continue
    with h5py.File(path) as f:
        games = len(f["/game_offsets"]) - 1
    if args.every:
        held = [i for i in range(games) if i % args.every == args.every - 1]
    else:
        held = list(range(games - max(1, round(games * args.fraction)), games))
    for root, part in ((args.train, sorted(set(range(games)) - set(held))), (args.test, held)):
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        take(str(path), str(root / rel), part)
