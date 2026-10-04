"""
vocab.py — dense vocabularies for the graph encoder's embedding tables.

XMage hashes every leaf name and every edge label of a state graph to an id in [0, 2^31-1)
(StateEncoder.indexFor(hash64(s))). A FeatureVocab lists the ids a model actually uses, in row
order, and an embedding table has one row per listed id:

  * pruned: an id gets a row only if it occurs in more than k states of the training data. Leaves
    outside the vocab are dropped together with their edges (safe, leaves have no children); edge
    labels outside it become an unlabeled edge. Unlike the flat encoder's ignore list, ids that occur
    in exactly the same states are not merged: in the graph they hang off different objects (a
    card's name and its ability's text always co-occur but describe different nodes);
  * append-only: once an id has a row it keeps that row, so a checkpoint's id->row mapping stays
    valid as later generations add ids. The vocabs are saved inside the checkpoint;
  * discovery-order independent init: an id's initial row is drawn from a stream keyed by the id,
    so it starts from the same row whether the vocab is built fresh or the id is appended by a
    later generation;
  * id-width agnostic: ids only need to fit in int64. Which encoding produced the ids is recorded
    with the vocab and checked on load, because the same row means a different feature under a
    different encoding.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch

VOCAB_FORMAT_VERSION = 2

# How XMage turned a feature into the id stored in the data. A vocab's rows are only meaningful
# under the encoding they were built with, so this travels in the checkpoint and is checked on load.
# Version 2: graph encoder, ids are leaf names and edge labels (version 1 hashed flat feature paths).
HASH_ALGORITHM = "xmage_feature_hash"
HASH_VERSION = 2

# Keyed generator for an id's initial embedding row, so the row an id starts from depends only on
# the id and not on when it was first seen.
FEATURE_INIT_KEY = 0x4D5A45524F5F4645415455524553   # "MZERO_FEATURES"

# StateEncoder's hash constants (Java longs, as unsigned 64-bit)
_MASK = (1 << 64) - 1
_GOLDEN = 0x9E3779B185EBCA87
_TABLE_SIZE = 2**31 - 1


def _mix64(z: int) -> int:
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _MASK
    return z ^ (z >> 31)


def feature_id(name: str) -> int:
    """The id XMage stores for a leaf name or edge label: StateEncoder.indexFor(hash64(name))."""
    data = name.encode("utf-8")
    h = _mix64(_GOLDEN ^ (len(data) * _GOLDEN & _MASK))
    full = len(data) // 8 * 8
    for i in range(0, full, 8):
        h ^= _mix64(int.from_bytes(data[i:i + 8], "little"))
        h = (((h << 27 | h >> 37) & _MASK) * _GOLDEN + 0x165667B19E3779F9) & _MASK
    h ^= _mix64(int.from_bytes(data[full:], "little"))
    h = ((h ^ (h >> 33)) * 0xFF51AFD7ED558CCD) & _MASK
    h = ((h ^ (h >> 33)) * 0xC4CEB9FE1A85EC53) & _MASK
    h ^= h >> 33
    if h == 1 << 63:  # Long.MIN_VALUE: Java's -h overflows and the remainder keeps the sign
        return -((1 << 63) % _TABLE_SIZE)
    return (h if h < 1 << 63 else (1 << 64) - h) % _TABLE_SIZE


def initial_rows(ids, embedding_dim: int) -> np.ndarray:
    """N(0, 1) rows (nn.Embedding's own init) drawn from a per-id keyed stream."""
    ids = np.asarray(ids, dtype=np.int64)
    out = np.empty((len(ids), embedding_dim), dtype=np.float32)
    for i, fid in enumerate(ids):
        rng = np.random.Generator(np.random.Philox(key=FEATURE_INIT_KEY, counter=int(fid)))
        out[i] = rng.standard_normal(embedding_dim, dtype=np.float32)
    return out


class FeatureVocab:
    def __init__(self, ids: Iterable[int] = (), feature_hash_bins: Optional[int] = None,
                 hash_algorithm: str = HASH_ALGORITHM, hash_version: int = HASH_VERSION):
        self.ids = np.asarray(list(ids) if not isinstance(ids, np.ndarray) else ids, dtype=np.int64)
        if len(np.unique(self.ids)) != len(self.ids):
            raise ValueError("feature vocab ids must be unique")
        self.feature_hash_bins = feature_hash_bins
        self.hash_algorithm = hash_algorithm
        self.hash_version = hash_version
        self._index()

    def _index(self) -> None:
        self._order = np.argsort(self.ids, kind="stable")
        self._sorted = self.ids[self._order]

    def __len__(self) -> int:
        return int(self.ids.shape[0])

    def extend(self, ids: Iterable[int]) -> int:
        """Append ids not yet in the vocab (in ascending order) and return how many were added.
        Existing ids keep their rows."""
        ids = np.unique(np.asarray(list(ids) if not isinstance(ids, np.ndarray) else ids, dtype=np.int64))
        new = ids[self.lookup(ids) < 0]
        if len(new):
            self.ids = np.concatenate([self.ids, new])
            self._index()
        return int(len(new))

    def lookup(self, ids) -> np.ndarray:
        """Row for each id, or -1 for ids not in the vocab."""
        ids = np.asarray(ids, dtype=np.int64)
        if len(self) == 0 or ids.size == 0:
            return np.full(ids.shape, -1, dtype=np.int64)
        pos = np.searchsorted(self._sorted, ids)
        pos_c = np.minimum(pos, len(self) - 1)
        hit = self._sorted[pos_c] == ids
        return np.where(hit, self._order[pos_c], -1)

    def encoding(self) -> dict:
        return {"hash_algorithm": self.hash_algorithm, "hash_version": self.hash_version,
                "feature_hash_bins": self.feature_hash_bins}

    def require_encoding(self, feature_hash_bins: Optional[int], hash_algorithm: str = HASH_ALGORITHM,
                         hash_version: int = HASH_VERSION) -> None:
        """Fail closed when this vocab's rows were built under a different feature encoding."""
        want = {"hash_algorithm": hash_algorithm, "hash_version": hash_version,
                "feature_hash_bins": feature_hash_bins}
        have = self.encoding()
        if have["feature_hash_bins"] is None:       # vocabs may be built before bins are known
            have = {**have, "feature_hash_bins": want["feature_hash_bins"]}
        if have != want:
            raise ValueError(f"feature vocab was built for {have} but this run uses {want}; its rows "
                             f"would mean different features. Rebuild the vocab or keep the old encoding.")

    def state_dict(self) -> dict:
        return {"format_version": VOCAB_FORMAT_VERSION,
                "ids": torch.from_numpy(self.ids.copy()),
                "feature_hash_bins": self.feature_hash_bins,
                "hash_algorithm": self.hash_algorithm,
                "hash_version": self.hash_version}

    @classmethod
    def from_state_dict(cls, d: dict) -> "FeatureVocab":
        if d.get("format_version") not in (1, VOCAB_FORMAT_VERSION):
            raise ValueError(f"unsupported feature vocab format: {d.get('format_version')}")
        ids = d["ids"].numpy() if isinstance(d["ids"], torch.Tensor) else np.asarray(d["ids"])
        # format 1 predates the encoding record; it can only have been written by the flat encoder
        return cls(ids, d.get("feature_hash_bins"),
                   d.get("hash_algorithm", HASH_ALGORITHM), d.get("hash_version", 1))


def feature_state_counts(ids, states) -> tuple[np.ndarray, np.ndarray]:
    """(sorted distinct ids, number of states each occurs in). states[i] is the state ids[i] occurs
    in; an id repeated within a state (numeric leaves share their name's id) counts once."""
    ids = np.asarray(ids, dtype=np.int64) - np.iinfo(np.int32).min   # ids are Java ints -> [0, 2^32)
    pairs = np.unique(np.asarray(states, dtype=np.int64) << 32 | ids)
    present, counts = np.unique(pairs & 0xFFFFFFFF, return_counts=True)
    return present + np.iinfo(np.int32).min, counts


def kept_feature_ids(ids, states, k: int = 10) -> np.ndarray:
    """Sorted ids that occur in more than k states."""
    present, counts = feature_state_counts(ids, states)
    return present[counts > k]
