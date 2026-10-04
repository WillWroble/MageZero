# dataset.py
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from model import ActionType, Graphs, NodeType, PRIORITY_TYPES, TARGET_TYPES, Targets
from vocab import feature_id

BATCH_SIZE = 64

# LabeledStateWriter layout: each offsets array (one entry per state + 1) indexes its per-item arrays
GROUPS = {
    "/offsets": ("/indices", "/values"),
    "/edge_offsets": ("/edge_child", "/edge_parent", "/edge_label"),
    "/policy_offsets": ("/policy_node", "/policy_visits"),
}
KEYS = ("/row", "/indices", "/values", "/offsets", "/edge_child", "/edge_parent", "/edge_label", "/edge_offsets",
        "/policy_node", "/policy_visits", "/policy_offsets")
ROW_WIDTH = 6

# a typed node's id is the hash of its type name and the root's id is 0; every other node is a leaf
TYPE_IDS = {0: NodeType.ROOT}
for _t in NodeType:
    if _t not in (NodeType.ROOT, NodeType.LEAF):
        TYPE_IDS[feature_id(_t.name)] = _t


def node_types(ids: np.ndarray) -> np.ndarray:
    types = np.full(ids.shape, NodeType.LEAF, dtype=np.int8)
    for type_id, t in TYPE_IDS.items():
        types[ids == type_id] = t
    return types


def states_of(offsets: np.ndarray) -> np.ndarray:
    """State index of each item, from its offsets array."""
    return np.repeat(np.arange(len(offsets) - 1), np.diff(offsets))


def map_ids(ids, node_type, edge_label, vocab, edge_vocab):
    """(node_row, edge_row) for the network: each leaf's vocab row (-1 for typed nodes and leaves
    outside the vocab, whose edges the network drops) and each edge label's row + 1 (0 for labels
    outside the vocab). Training data and inference requests map the same way."""
    node_row = np.where(node_type == NodeType.LEAF, vocab.lookup(ids), -1)
    return node_row.astype(np.int32), (edge_vocab.lookup(edge_label) + 1).astype(np.int32)


def shard_problem(s: dict):
    """Why a shard's arrays don't form a valid set of graphs, or None."""
    n = s["/row"].shape[0]
    if s["/row"].ndim != 2 or s["/row"].shape[1] != ROW_WIDTH:
        return f"/row has shape {s['/row'].shape}, expected [states, {ROW_WIDTH}]"
    for off_key, item_keys in GROUPS.items():
        off = s[off_key]
        if off.shape[0] != n + 1:
            return f"{off_key} has {off.shape[0]} entries for {n} states (expected {n + 1})"
        if off[0] != 0:
            return f"{off_key} starts at {off[0]}, not 0"
        drops = np.flatnonzero(np.diff(off) < 0)
        if len(drops):
            return f"{off_key} decreases at state {drops[0]} ({off[drops[0]]} -> {off[drops[0] + 1]})"
        for key in item_keys:
            if s[key].shape[0] != off[-1]:
                return f"{key} has {s[key].shape[0]} entries but {off_key} ends at {off[-1]}"
    nodes = np.diff(s["/offsets"])
    for key, off_key in (("/edge_child", "/edge_offsets"), ("/edge_parent", "/edge_offsets"),
                         ("/policy_node", "/policy_offsets")):
        state = states_of(s[off_key])
        bad = np.flatnonzero((s[key] < 0) | (s[key] >= nodes[state]))
        if len(bad):
            i = bad[0]
            return f"{key} has node index {s[key][i]} in state {state[i]}, which has {nodes[state[i]]} nodes"
    return None


class H5Graphs(Dataset):
    """
    Preloads all HDF5 shards in a directory into RAM for fast random access (LabeledStateWriter layout):
      /indices, /values             int32 [nodes]       node feature id, numeric value
      /offsets                      int64 [N+1]
      /edge_child, /edge_parent     int32 [edges]       node index local to the state
      /edge_label                   int32 [edges]       hashed edge label
      /edge_offsets                 int64 [N+1]
      /policy_node, /policy_visits  int32 [candidates]  node index local to the state, MCTS visits
      /policy_offsets               int64 [N+1]
      /row                          float32 [N, 6]      resultLabel, stateScore, isPlayer, actionType,
                                                        useFalseVisits, useTrueVisits
    Ids stay raw until apply_vocab maps them to embedding rows; __getitem__ needs the mapped form.
    """

    def __init__(self, dir_path: str, vocab=None, edge_vocab=None):
        p = Path(dir_path)
        self.files = [str(pp) for pp in sorted(list(p.glob("*.h5")) + list(p.glob("*.hdf5")))]

        shards = []
        for path in self.files:
            try:
                with h5py.File(path, "r") as f:
                    shard = {key: f[key][...] for key in KEYS}
            except (OSError, KeyError) as e:
                print(f"[warn] skipping unreadable shard {path}: {e}")
                continue
            problem = shard_problem(shard)
            if problem:
                print(f"[warn] skipping inconsistent shard {path}: {problem}")
                continue
            shards.append(shard)

        def cat(key, dtype):
            return np.concatenate([s[key] for s in shards]).astype(dtype, copy=False) if shards else np.empty(0, dtype)

        def cat_offsets(key):
            parts, base = [np.zeros(1, dtype=np.int64)], 0
            for s in shards:
                parts.append(s[key][1:].astype(np.int64) + base)
                base += int(s[key][-1])
            return np.concatenate(parts)

        self.row = np.concatenate([s["/row"] for s in shards]).astype(np.float32) if shards \
            else np.empty((0, ROW_WIDTH), dtype=np.float32)
        self.N = len(self.row)
        self.ids = cat("/indices", np.int32)
        self.values = cat("/values", np.int32)
        self.offsets = cat_offsets("/offsets")
        self.edge_child = cat("/edge_child", np.int32)
        self.edge_parent = cat("/edge_parent", np.int32)
        self.edge_label = cat("/edge_label", np.int32)
        self.edge_offsets = cat_offsets("/edge_offsets")
        self.cand_node = cat("/policy_node", np.int32)
        self.cand_visits = cat("/policy_visits", np.int32)
        self.cand_offsets = cat_offsets("/policy_offsets")
        self.node_type = node_types(self.ids)
        self.check_candidates()

        if vocab is not None:
            self.apply_vocab(vocab, edge_vocab)

    def apply_vocab(self, vocab, edge_vocab) -> None:
        self.node_row, self.edge_row = map_ids(self.ids, self.node_type, self.edge_label, vocab, edge_vocab)

    def check_candidates(self) -> None:
        """Report policy candidates that aren't nodes of their head's types. The head still scores
        them, but it means the encoder gave an action's object an unexpected type."""
        state = states_of(self.cand_offsets)
        types = self.node_type[self.cand_node + self.offsets[state]]
        action = self.row[state, 3]
        for action_type, allowed in ((ActionType.PRIORITY, PRIORITY_TYPES), (ActionType.CHOOSE_TARGET, TARGET_TYPES)):
            bad = (action == action_type.value) & ~np.isin(types, allowed)
            if bad.any():
                found, counts = np.unique(types[bad], return_counts=True)
                print(f"[warn] {int(bad.sum())} {action_type.name} candidates are not "
                      f"{'/'.join(t.name for t in allowed)} nodes: "
                      + ", ".join(f"{NodeType(int(t)).name} x{c}" for t, c in zip(found, counts)))

    def __len__(self) -> int:
        return int(self.N)

    def __getitem__(self, k: int):
        n = slice(self.offsets[k], self.offsets[k + 1])
        e = slice(self.edge_offsets[k], self.edge_offsets[k + 1])
        c = slice(self.cand_offsets[k], self.cand_offsets[k + 1])
        return (self.node_type[n], self.node_row[n], self.values[n],
                self.edge_child[e], self.edge_parent[e], self.edge_row[e],
                self.cand_node[c], self.cand_visits[c], self.row[k])


def concat(parts, dtype=np.int64):
    return torch.from_numpy(np.concatenate(parts).astype(dtype))


def shift(local, starts):
    """State-local node indices -> batch-wide ones, given each state's first node."""
    return concat([idx + start for idx, start in zip(local, starts)])


def collate_graphs(node_type, node_row, values, edge_child, edge_parent, edge_row):
    """Concatenate per-state node and edge arrays into one batch of graphs. Training batches and
    inference requests both go through here."""
    starts = np.cumsum([0] + [len(t) for t in node_type], dtype=np.int64)
    return Graphs(concat(node_type), concat(node_row), concat(values), torch.from_numpy(starts),
                  shift(edge_child, starts), shift(edge_parent, starts), concat(edge_row))


def collate_batch(batch):
    """A training batch: the states' graphs and their labels."""
    *graph, cand_node, cand_visits, rows = zip(*batch)
    graphs = collate_graphs(*graph)
    rows = np.stack(rows)
    targets = Targets(torch.from_numpy(rows[:, 0].copy()), torch.from_numpy(rows[:, 3].astype(np.int64)),
                      torch.from_numpy(rows[:, 4:6].copy()), shift(cand_node, np.asarray(graphs.node_offsets)),
                      concat(cand_visits, np.float32),
                      torch.from_numpy(np.repeat(np.arange(len(batch)), [len(c) for c in cand_node])))
    return graphs, targets
