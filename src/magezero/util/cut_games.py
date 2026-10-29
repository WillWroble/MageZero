"""Copy games [start, end) of a LabeledStateWriter shard to a new shard: python cut_games.py src dst start end"""
import sys

import h5py
import numpy as np

GROUPS = {"/offsets": ("/indices", "/values"), "/edge_offsets": ("/edge_child", "/edge_parent", "/edge_label"),
          "/policy_offsets": ("/policy_node", "/policy_visits")}


def select_games(f, games) -> dict:
    """The given games (increasing indices) of a shard as a new shard's arrays. f maps the shard's keys to arrays
    or h5py datasets, read only in the selected games' slices."""
    go = f["/game_offsets"][...]
    games = [i for i in games if i < len(go) - 1]
    runs = []  # [first, last + 1) of each run of consecutive games, as row ranges
    for i in games:
        if runs and runs[-1][1] == go[i]:
            runs[-1][1] = int(go[i + 1])
        else:
            runs.append([int(go[i]), int(go[i + 1])])
    out = {"/game_offsets": np.concatenate([[0], np.cumsum(np.diff(go)[games])]).astype(go.dtype),
           "/row": np.concatenate([f["/row"][a:b] for a, b in runs] or [f["/row"][:0]])}
    for off, items in GROUPS.items():
        parts = [f[off][a:b + 1] for a, b in runs]
        out[off] = np.concatenate([[0]] + [np.diff(o) for o in parts]).cumsum().astype(f[off].dtype)
        for k in items:
            out[k] = np.concatenate([f[k][o[0]:o[-1]] for o in parts] or [f[k][:0]])
    return out


def take(src, dst, games):
    """Copy the given games (increasing indices) of shard src to a new shard dst, reading only their slices."""
    with h5py.File(src) as f, h5py.File(dst, "w") as g:
        for k, v in select_games(f, games).items():
            g[k] = v
        print(dst.split("/")[-1], "games", len(g["/game_offsets"]) - 1, "states", int(g["/game_offsets"][-1]))


def cut(src, dst, start, end):
    take(src, dst, range(start, end))


if __name__ == "__main__":
    cut(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
