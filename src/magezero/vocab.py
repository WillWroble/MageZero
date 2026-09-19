"""
vocab.py — dense feature vocabulary for the embedding table.

XMage hashes every game feature to an id in [0, feature_hash_bins). The original model allocates
one embedding row per possible id (2M rows x 512 floats = ~4 GB, ~16 GB with Adam), even though a
deck only ever uses a few thousand of them: everything else is removed by the ignore list.

A FeatureVocab lists the ids a model actually uses, in row order, and the embedding table has
exactly one row per listed id:

  * lossless: the vocab is the complement of the ignore list, so ids outside it are exactly the
    ids the full-table path already drops, and every kept id still gets its own row. A converted
    model produces identical outputs, and training updates the used rows identically (Adam never
    moves a row that receives no gradient);
  * append-only: once an id has a row it keeps that row, so a checkpoint's feature->row mapping
    stays valid as later generations add features. The vocab is saved inside the checkpoint;
  * id-width agnostic: ids only need to fit in int64, so XMage can hash into far more than 2M
    bins (e.g. 2^31) to avoid collisions between distinct features without growing the model.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import torch
from scipy.sparse import coo_matrix

VOCAB_FORMAT_VERSION = 1


class FeatureVocab:
    def __init__(self, ids: Iterable[int] = (), feature_hash_bins: Optional[int] = None):
        self.ids = np.asarray(list(ids) if not isinstance(ids, np.ndarray) else ids, dtype=np.int64)
        if len(np.unique(self.ids)) != len(self.ids):
            raise ValueError("feature vocab ids must be unique")
        self.feature_hash_bins = feature_hash_bins
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

    def map_bags(self, indices, offsets) -> tuple[np.ndarray, np.ndarray]:
        """Map a flat list of ids split into bags by `offsets` to rows, dropping unknown ids.
        Returns (rows, new_offsets) with the same number of bags."""
        indices = np.asarray(indices, dtype=np.int64)
        offsets = np.asarray(offsets if len(offsets) else [0], dtype=np.int64)
        rows = self.lookup(indices)
        keep = rows >= 0
        # kept-count prefix sums give each bag's new start
        kept_before = np.concatenate([[0], np.cumsum(keep)])
        return rows[keep], kept_before[offsets]

    def state_dict(self) -> dict:
        return {"format_version": VOCAB_FORMAT_VERSION,
                "ids": torch.from_numpy(self.ids.copy()),
                "feature_hash_bins": self.feature_hash_bins}

    @classmethod
    def from_state_dict(cls, d: dict) -> "FeatureVocab":
        if d.get("format_version") != VOCAB_FORMAT_VERSION:
            raise ValueError(f"unsupported feature vocab format: {d.get('format_version')}")
        ids = d["ids"].numpy() if isinstance(d["ids"], torch.Tensor) else np.asarray(d["ids"])
        return cls(ids, d.get("feature_hash_bins"))


def _factorize(ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(sorted distinct ids, position of each id in that list). Uses a lookup table when ids fit
    in a small range (the 2M hash space does): O(n), much faster than sorting tens of millions
    of tokens. Falls back to np.unique for wide ids."""
    hi = int(ids.max()) + 1 if ids.size else 0
    if hi <= (1 << 26):
        present = np.bincount(ids, minlength=hi) > 0
        observed = np.flatnonzero(present)
        lut = np.empty(hi, dtype=np.int64)
        lut[observed] = np.arange(len(observed))
        return observed, lut[ids]
    return np.unique(ids, return_inverse=True)


def kept_feature_ids(indices: np.ndarray, idxptr: np.ndarray, k: int = 10) -> np.ndarray:
    """The ids that survive the ignore-list rule, computed over observed ids only.

    Same rule as dataset.create_redundancy_ignore_list: drop ids active in <= k states, and of
    ids active in exactly the same set of states keep only the smallest. That function scans every
    one of the 2M possible columns; this one only looks at the ids that appear, so it works for any
    id width and is much faster. Returns sorted kept ids (the complement of the ignore list).
    """
    indices = np.asarray(indices, dtype=np.int64)
    idxptr = np.asarray(idxptr, dtype=np.int64)
    n_states = len(idxptr) - 1
    if n_states <= 0 or indices.size == 0:
        return np.empty(0, dtype=np.int64)
    rows = np.repeat(np.arange(n_states, dtype=np.int64), np.diff(idxptr))
    observed, cols = _factorize(indices)
    x = coo_matrix((np.ones(len(cols), dtype=np.uint8), (rows, cols)),
                   shape=(n_states, len(observed)), dtype=np.uint8).tocsc()
    x.sum_duplicates()
    x.sort_indices()
    kept = []
    seen_patterns: set = set()
    # observed is ascending, so the first column seen with a pattern is the smallest id
    for j in range(len(observed)):
        start, end = x.indptr[j], x.indptr[j + 1]
        if end - start <= k:
            continue
        key = x.indices[start:end].tobytes()
        if key in seen_patterns:
            continue
        seen_patterns.add(key)
        kept.append(observed[j])
    return np.asarray(kept, dtype=np.int64)
