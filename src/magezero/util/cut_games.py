"""Copy games [start, end) of a LabeledStateWriter shard to a new shard: python cut_games.py src dst start end"""
import sys

import h5py
import numpy as np

GROUPS = {"/offsets": ("/indices", "/values"), "/edge_offsets": ("/edge_child", "/edge_parent", "/edge_label"),
          "/policy_offsets": ("/policy_node", "/policy_visits")}


def take(src, dst, games):
    """Copy the given games (increasing indices) of shard src to a new shard dst."""
    with h5py.File(src) as f, h5py.File(dst, "w") as g:
        go = f["/game_offsets"][...]
        games = np.asarray([i for i in games if i < len(go) - 1], dtype=np.int64)
        rows = np.concatenate([np.arange(go[i], go[i + 1]) for i in games]) if len(games) else np.zeros(0, np.int64)
        g["/game_offsets"] = np.concatenate([[0], np.cumsum(go[games + 1] - go[games])])
        g["/row"] = f["/row"][...][rows]
        for off, items in GROUPS.items():
            o = f[off][...]
            n = o[rows + 1] - o[rows]
            g[off] = np.concatenate([[0], np.cumsum(n)]).astype(o.dtype)
            items_idx = np.concatenate([np.arange(o[r], o[r + 1]) for r in rows]) if len(rows) else np.zeros(0, np.int64)
            for k in items:
                g[k] = f[k][...][items_idx]
        print(dst.split("/")[-1], "games", len(games), "states", len(rows))


def cut(src, dst, start, end):
    take(src, dst, range(start, end))


if __name__ == "__main__":
    cut(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
