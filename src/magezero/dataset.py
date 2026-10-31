# dataset.py
import re
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from model import ActionType, Graphs, NodeType, PRIORITY_TYPES, TARGET_TYPES, Targets
from util.cut_games import select_games
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


def lookup(vocab, ids, chunk=50_000_000) -> np.ndarray:
    """vocab.lookup as int16 rows, in chunks: its int64 temporaries would be several times the data's size."""
    assert len(vocab) < np.iinfo(np.int16).max
    rows = np.empty(len(ids), np.int16)
    for a in range(0, len(ids), chunk):
        rows[a:a + chunk] = vocab.lookup(ids[a:a + chunk])
    return rows


def map_ids(ids, node_type, edge_label, vocab, edge_vocab):
    """(node_row, edge_row) for the network: each leaf's vocab row (-1 for typed nodes and leaves
    outside the vocab, whose edges the network drops) and each edge label's row + 1 (0 for labels
    outside the vocab). Training data and inference requests map the same way."""
    node_row = lookup(vocab, ids)
    node_row[node_type != NodeType.LEAF] = -1
    edge_row = lookup(edge_vocab, edge_label)
    edge_row += 1
    return node_row, edge_row


def read_split(path: str):
    """(train, vals) from a model directory's split.txt. train is a list of (shard path, games or None for all);
    vals maps each val set's name to one, in file order. The file is a list of sections, each a header line and then
    regexes, one per line, each matched against the whole path of every .h5/.hdf5 file under the current directory:
        train gen0 holdout 5:    a train section (header starting with "train"; its name, here gen0, is optional).
        <regex>                  With "holdout K", every Kth game of each of its shards (games K-1, 2K-1, ...) is
        train gen1:              left out of training and becomes the val set of the same name
        <regex>
        pimc:                    a val section: a named val set, never trained on; "holdout K" takes only every
        <regex>                  Kth game of its shards (the same games a train section with that K holds out)
    A shard may be in only one section. # starts a comment."""
    sections, current = [], None
    for line in Path(path).read_text().splitlines():
        line = line.split("#")[0].strip()
        if line.endswith(":"):
            words = line[:-1].split()
            holdout = int(words[-1]) if len(words) > 2 and words[-2] == "holdout" else None
            words = words[:-2] if holdout else words
            is_train = words[0] == "train"
            if len(words) > (2 if is_train else 1):
                raise SystemExit(f"{path}: bad section header: {line}")
            current = (words[-1], is_train, holdout, [])
            sections.append(current)
        elif line:
            if current is None:
                raise SystemExit(f"{path}: regex before the first section: {line}")
            current[3].append(re.compile(line))
    files = sorted(str(p) for p in list(Path(".").rglob("*.h5")) + list(Path(".").rglob("*.hdf5")))

    def split(f, k):
        """(kept, held out) games of shard f with holdout k."""
        with h5py.File(f, "r") as h:
            games = range(len(h["/game_offsets"]) - 1)
        return [i for i in games if i % k != k - 1], [i for i in games if i % k == k - 1]

    train, vals, owner = [], {}, {}
    for name, is_train, holdout, regexes in sections:
        shards = [f for f in files if any(r.fullmatch(f) for r in regexes)]
        for f in shards:
            if f in owner:
                raise SystemExit(f"{path}: {f} is in both {owner[f]} and {name}")
            owner[f] = name
        if holdout is None:
            (train if is_train else vals.setdefault(name, [])).extend((f, None) for f in shards)
            continue
        for f in shards:
            kept, held = split(f, holdout)
            if is_train:
                train.append((f, kept))
            vals.setdefault(name, []).append((f, held))
    return train, vals


def td_labels(row, game_offsets, td):
    """Value labels rebuilt with discount td, as XMage's ParallelDataGenerator builds them: within each game,
    label[i] = td * label[i + 1] + (1 - td) * stateScore[i], starting from the result (+1 / -1, the sign of the game's
    last stored label). td 1 gives the result itself."""
    label = np.empty(len(row), np.float32)
    for a, b in zip(game_offsets[:-1], game_offsets[1:]):
        x = 1.0 if row[b - 1, 0] > 0 else -1.0
        for i in range(b - 1, a - 1, -1):
            x = td * x + (1 - td) * float(row[i, 1])
            label[i] = x
    return label


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
      /edge_child, /edge_parent     int32 [edges]       node index local to the state (kept as int16)
      /edge_label                   int32 [edges]       hashed edge label
      /edge_offsets                 int64 [N+1]
      /policy_node, /policy_visits  int32 [candidates]  node index local to the state, MCTS visits
      /policy_offsets               int64 [N+1]
      /row                          float32 [N, 6]      resultLabel, stateScore, isPlayer, actionType,
                                                        useFalseVisits, useTrueVisits
    Ids stay raw until apply_vocab maps them to embedding rows; __getitem__ needs the mapped form.
    hide_opponent_hand: see hide_opponent_hand.
    td: if given, the value labels (resultLabel) are rebuilt with this discount (td_labels) instead of the stored ones.
    source: a directory (every shard under it) or a list of (shard path, games or None for all), as read_split gives.
    """

    def __init__(self, source, vocab=None, edge_vocab=None, hide_opponent_hand: bool = False, td: float = None):
        if isinstance(source, (str, Path)):
            source = [(str(p), None) for p in sorted(list(Path(source).rglob("*.h5")) + list(Path(source).rglob("*.hdf5")))]
        self.files = [path for path, _ in source]

        shards = []
        for path, games in source:
            try:
                with h5py.File(path, "r") as f:
                    shard = {key: f[key][...] for key in KEYS} if games is None else select_games(f, games)
                    if games is None and td is not None:
                        shard["/game_offsets"] = f["/game_offsets"][...]  # td_labels needs the games
            except (OSError, KeyError) as e:
                print(f"[warn] skipping unreadable shard {path}: {e}")
                continue
            problem = shard_problem(shard)
            if problem:
                print(f"[warn] skipping inconsistent shard {path}: {problem}")
                continue
            if td is not None:
                shard["/row"] = shard["/row"].copy()
                shard["/row"][:, 0] = td_labels(shard["/row"], shard["/game_offsets"], td)
            shards.append(shard)

        def cat(key, dtype):
            """One key of every shard, concatenated as dtype; each shard's own copy is freed as it goes."""
            return np.concatenate([s.pop(key).astype(dtype, copy=False) for s in shards]) if shards else np.empty(0, dtype)

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
        for s in shards:  # node values fit int16 (the model buckets everything past -5..61 together)
            s["/values"] = np.clip(s["/values"], -32768, 32767)
        self.values = cat("/values", np.int16)
        self.offsets = cat_offsets("/offsets")
        self.edge_child = cat("/edge_child", np.int16)  # state-local node indices (a state has hundreds of nodes)
        self.edge_parent = cat("/edge_parent", np.int16)
        self.edge_label = cat("/edge_label", np.int32)
        self.edge_offsets = cat_offsets("/edge_offsets")
        self.cand_node = cat("/policy_node", np.int32)
        self.cand_visits = cat("/policy_visits", np.int32)
        self.cand_offsets = cat_offsets("/policy_offsets")
        self.node_type = node_types(self.ids)
        if hide_opponent_hand:
            self.hide_opponent_hand()
        self.check_candidates()

        if vocab is not None:
            self.apply_vocab(vocab, edge_vocab)

    def hide_opponent_hand(self) -> None:
        """Each state as StateEncoder encodes it without perfectInfo: the cards in the hand of the player who isn't
        deciding (the one without IsDecisionPlayer) are cut from its hand zone, and every node then unreachable from
        the root goes. The hand zone and HandSize stay. Policy candidates always stay (the decider sees what it
        chooses from); a hidden card also reachable another way (e.g. a decision source) keeps its subtree."""
        e_state = states_of(self.edge_offsets)
        child, parent = self.edge_child + self.offsets[e_state], self.edge_parent + self.offsets[e_state]
        t = self.node_type
        decider = np.zeros(len(t), bool)
        decider[parent[self.ids[child] == feature_id("IsDecisionPlayer")]] = True
        hand = np.zeros(len(t), bool)
        hand[parent[(self.ids[child] == feature_id("Hand")) & (t[parent] == NodeType.ZONE)]] = True
        hidden = np.zeros(len(t), bool)
        hidden[child[hand[child] & (t[parent] == NodeType.PLAYER) & ~decider[parent]]] = True

        cand = self.cand_node + self.offsets[states_of(self.cand_offsets)]
        keep = t == NodeType.ROOT
        keep[cand] = True
        edges = ~(hidden[parent] & (t[child] == NodeType.CARD))
        while True:  # reachable without the hidden hands' card edges
            n = keep.sum()
            keep[child[edges & keep[parent]]] = True
            if keep.sum() == n:
                break
        edges &= keep[parent]

        state = states_of(self.offsets)
        offsets = np.concatenate([[0], np.cumsum(np.bincount(state[keep], minlength=self.N))])
        local = (np.cumsum(keep) - 1 - offsets[state]).astype(np.int16)  # each kept node's new index in its state
        self.edge_child, self.edge_parent = local[child[edges]], local[parent[edges]]
        self.edge_label = self.edge_label[edges]
        self.edge_offsets = np.concatenate([[0], np.cumsum(np.bincount(e_state[edges], minlength=self.N))])
        self.cand_node = local[cand]
        self.ids, self.values, self.node_type, self.offsets = self.ids[keep], self.values[keep], t[keep], offsets
        print(f"hid {int(hidden.sum())} opponent hands: {int((~keep).sum())} nodes removed "
              f"({(~keep).mean():.1%}), {int((~edges).sum())} edges")

    def apply_vocab(self, vocab, edge_vocab) -> None:
        """Maps ids to embedding rows, then drops the raw ids and edge labels (the bulk of the memory)."""
        self.node_row, self.edge_row = map_ids(self.ids, self.node_type, self.edge_label, vocab, edge_vocab)
        del self.ids, self.edge_label

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
