import numpy as np
import torch

from model import NodeType

CACHE_SIZE = 2**19  # slots; each holds one node's output of every local pass (d=256, 2 passes: 1 GiB in fp32)


class NodeCache:
    """Local-layer outputs of earlier states' internal nodes, keyed by XMage's subtree hash (FeatureGraph.Node.hash).
    A node's local outputs depend only on the subgraph below it, so a node whose hash was seen before reads every
    pass's output from its slot instead of computing it. Slots are evicted least recently used. Slots are assigned
    in lookup and written by the forward pass that follows, so lookups and forward passes must alternate (one worker)."""

    def __init__(self, model, device, size=CACHE_SIZE):
        self.table = torch.empty(len(model.local_layers), size, model.arch["d_model"], device=device)
        self.hash = np.zeros(size, np.int64)  # slot -> hash
        self.clear()

    def clear(self):
        """Forget every node, e.g. after a forward pass failed to write the slots its lookup assigned."""
        self.slot_of = {}                                  # hash -> slot
        self.used = np.full(len(self.hash), -1, np.int64)  # slot -> last lookup that used it; -1 when free
        self.free = np.arange(len(self.hash))
        self.tick = 0

    def lookup(self, node_type: np.ndarray, node_hash: np.ndarray):
        """(slot, compute) per node of one batch: each internal node's slot (-1 at leaves), and whether the batch
        computes it (the first node of each hash the cache doesn't hold yet)."""
        self.tick += 1
        internal = np.flatnonzero(node_type != NodeType.LEAF)
        hashes, first, inverse = np.unique(node_hash[internal], return_index=True, return_inverse=True)
        slots = np.fromiter((self.slot_of.get(h, -1) for h in hashes.tolist()), np.int64, len(hashes))
        miss = np.flatnonzero(slots < 0)
        self.used[slots[slots >= 0]] = self.tick
        slots[miss] = new = self.take(len(miss))
        self.slot_of.update(zip(hashes[miss].tolist(), new.tolist()))
        self.hash[new] = hashes[miss]
        self.used[new] = self.tick

        slot = np.full(len(node_type), -1, np.int64)
        slot[internal] = slots[inverse.reshape(-1)]
        compute = np.zeros(len(node_type), bool)
        compute[internal[first[miss]]] = True
        return torch.from_numpy(slot), torch.from_numpy(compute)

    def take(self, n: int) -> np.ndarray:
        """n free slots. When too few are free, frees the least recently used eighth of the cache (and at least n)."""
        if len(self.free) < n:
            k = min(len(self.used), len(self.free) + max(n, len(self.used) // 8))
            old = np.argpartition(self.used, k - 1)[:k]
            if (self.used[old] == self.tick).any():
                raise RuntimeError("batch has more unseen nodes than the node cache can hold")
            for h in self.hash[old[self.used[old] >= 0]].tolist():
                del self.slot_of[h]
            self.used[old] = -1
            self.free = old
        out, self.free = self.free[:n], self.free[n:]
        return out


def mix64(x: np.ndarray) -> np.ndarray:
    """StateEncoder.mix64."""
    x = x.astype(np.uint64)
    with np.errstate(over="ignore"):
        x = (x ^ (x >> np.uint64(30))) * np.uint64(0xbf58476d1ce4e5b9)
        x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94d049bb133111eb)
    return x ^ (x >> np.uint64(31))


def subtree_hashes(ids, values, edge_child, edge_parent, edge_label) -> np.ndarray:
    """One recorded state's node hashes, for benchmarks and tests (the h5s don't store them): FeatureGraph.getStateHash's
    16 rounds of mixHash, with the recorded edge label ids standing in for the label strings' hash64."""
    with np.errstate(over="ignore"):
        h = np.where(values != 0, mix64(ids.astype(np.int64)) ^ values.astype(np.int64).astype(np.uint64),
                     ids.astype(np.int64).astype(np.uint64))
        label = edge_label.astype(np.int64).astype(np.uint64)
        for _ in range(16):
            mixed = np.zeros(len(ids), np.uint64)
            np.add.at(mixed, edge_parent, h[edge_child] ^ label)
            h = mix64(mixed ^ h)
    return h.view(np.int64)
