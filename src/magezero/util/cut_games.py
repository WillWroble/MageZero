"""Copy games [start, end) of a LabeledStateWriter shard to a new shard: python cut_games.py src dst start end"""
import sys

import h5py

GROUPS = {"/offsets": ("/indices", "/values"), "/edge_offsets": ("/edge_child", "/edge_parent", "/edge_label"),
          "/policy_offsets": ("/policy_node", "/policy_visits")}


def cut(src, dst, start, end):
    with h5py.File(src) as f, h5py.File(dst, "w") as g:
        games = f["/game_offsets"][...]
        end = min(end, len(games) - 1)
        first, last = int(games[start]), int(games[end])
        g["/game_offsets"] = games[start:end + 1] - first
        g["/row"] = f["/row"][first:last]
        for off, items in GROUPS.items():
            o = f[off][first:last + 1]
            g[off] = o - o[0]
            for k in items:
                g[k] = f[k][o[0]:o[-1]]
        print(dst.split("/")[-1], "games", end - start, "states", last - first)


if __name__ == "__main__":
    cut(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
