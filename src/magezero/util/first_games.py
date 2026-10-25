"""Copy a LabeledStateWriter shard keeping only its first n games: python first_games.py src dst n"""
import sys
import h5py

GROUPS = {"/offsets": ("/indices", "/values"), "/edge_offsets": ("/edge_child", "/edge_parent", "/edge_label"),
          "/policy_offsets": ("/policy_node", "/policy_visits")}
src, dst, n = sys.argv[1], sys.argv[2], int(sys.argv[3])
with h5py.File(src) as f, h5py.File(dst, "w") as g:
    games = f["/game_offsets"][...]
    n = min(n, len(games) - 1)
    states = int(games[n])
    g["/game_offsets"] = games[:n + 1]
    g["/row"] = f["/row"][:states]
    for off, items in GROUPS.items():
        o = f[off][:states + 1]
        g[off] = o
        for k in items:
            g[k] = f[k][:o[-1]]
    print(dst.split("/")[-1], "games", n, "states", states)
