"""
Equivalence tests for the dense feature vocab (src/magezero/vocab.py).

The claim being tested: a dense-vocab model is the same model as the full-table one, with the
unused rows removed. Concretely:
  1. the vocab built from data is exactly the complement of the ignore list;
  2. a converted checkpoint gives bit-identical outputs to the original on the same states;
  3. training steps give bit-identical results (Adam never moves rows with no gradient);
  4. the dataset loader yields the same features per state either way;
  5. the vocab/embedding are append-only across generations;
  6. a bag is a set: repeated ids (including two feature names colliding on one hash id) feed the
     model the same tokens as the full-table path, which deduplicates through a BitMap, and the
     dataset loader maps a state exactly as the server does, so training and play agree;
  7. a feature's initial row depends only on its id, so a fresh dense model, a later generation
     that appends the feature, and the full-table model all start it from the same row;
  8. a vocab built under a different feature encoding is refused rather than reinterpreted;
  9. the loader hands the vocab the ids XMage wrote, so ids from a hash space wider than the
     full table's bin count reach it intact.

Runs on CPU. Tables default to 200k rows for speed; set MZ_TEST_TABLE_ROWS=2000000 to test
the production table size (needs ~4 GB RAM for inference, ~16 GB for the training test).

  python -m pytest tests/test_dense_vocab.py       or       python tests/test_dense_vocab.py
"""
import os
import sys
import tempfile

import h5py
import numpy as np
import torch
from pyroaring import BitMap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "magezero"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "magezero", "util"))

from dataset import GLOBAL_MAX, H5Indexed, collate_batch, create_redundancy_ignore_list  # noqa: E402
from model import NetTransformer  # noqa: E402
from vocab import FeatureVocab, initial_rows, kept_feature_ids  # noqa: E402
from convert_dense_vocab import convert  # noqa: E402

TABLE_ROWS = int(os.environ.get("MZ_TEST_TABLE_ROWS", 200_000))


# ── fixtures ────────────────────────────────────────────────

def synthetic_states(n_states=400, seed=0, id_space=TABLE_ROWS):
    """Feature sets with the structure the ignore rule cares about: common ids, rare ids
    (<= 10 states), exact duplicates of other ids, and noise."""
    rng = np.random.default_rng(seed)
    common = rng.choice(id_space, 300, replace=False)
    rare = rng.choice(np.setdiff1d(np.arange(id_space), common), 60, replace=False)
    dup_of = {int(c): int(d) for c, d in zip(common[:40], rng.choice(
        np.setdiff1d(np.arange(id_space), np.concatenate([common, rare])), 40, replace=False))}
    states = []
    for _ in range(n_states):
        s = set(rng.choice(common, rng.integers(20, 80), replace=False).tolist())
        if rng.random() < 0.02:
            s.add(int(rng.choice(rare)))
        s |= {dup_of[c] for c in s if c in dup_of}   # duplicate columns always co-occur
        states.append(sorted(s))
    return states


def as_flat(states):
    idxptr = np.concatenate([[0], np.cumsum([len(s) for s in states])]).astype(np.int64)
    indices = np.concatenate([np.asarray(s, dtype=np.int64) for s in states])
    return indices, idxptr


def legacy_kept(states):
    ds = [(torch.tensor(s, dtype=torch.int32), None, None, None, None) for s in states]
    ignore = create_redundancy_ignore_list(ds)
    observed = np.unique(np.concatenate([np.asarray(s) for s in states]))
    return np.asarray(sorted(set(observed.tolist()) - ignore), dtype=np.int64), BitMap(ignore)


def batch_legacy(states, ignore):
    kept = [torch.tensor([i for i in s if i not in ignore], dtype=torch.int32) for s in states]
    idx, off, *_ = collate_batch([(k, torch.zeros(4), torch.zeros(1), torch.zeros(1), torch.zeros(1, dtype=torch.long))
                                  for k in kept])
    return idx, off


def batch_dense(states, vocab):
    rows = [torch.tensor(vocab.lookup(s)[vocab.lookup(s) >= 0], dtype=torch.int32) for s in states]
    idx, off, *_ = collate_batch([(r, torch.zeros(4), torch.zeros(1), torch.zeros(1), torch.zeros(1, dtype=torch.long))
                                  for r in rows])
    return idx, off


def full_and_dense_models(states, seed=0):
    torch.manual_seed(seed)
    full = NetTransformer(num_embeddings=TABLE_ROWS)
    _, ignore = legacy_kept(states)
    converted = convert({"model_state_dict": full.state_dict()}, ignore)
    vocab = FeatureVocab.from_state_dict(converted["feature_vocab"])
    dense = NetTransformer(num_embeddings=len(vocab))
    dense.load_state_dict(converted["model_state_dict"])
    return full, dense, ignore, vocab


# ── tests ───────────────────────────────────────────────────

def test_vocab_is_complement_of_ignore_list():
    states = synthetic_states()
    expected, _ = legacy_kept(states)
    got = kept_feature_ids(*as_flat(states))
    assert np.array_equal(got, expected), (len(got), len(expected))
    assert 0 < len(got) < len(np.unique(as_flat(states)[0]))  # something was actually dropped


def test_converted_model_outputs_identical():
    states = synthetic_states()
    full, dense, ignore, vocab = full_and_dense_models(states)
    full.eval(); dense.eval()
    with torch.no_grad():
        out_full = full(*batch_legacy(states, ignore))
        out_dense = dense(*batch_dense(states, vocab))
    for a, b in zip(out_full, out_dense):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert dense.embedding.num_embeddings == len(vocab) < TABLE_ROWS


def test_training_steps_identical():
    states = synthetic_states(n_states=64)
    full, dense, ignore, vocab = full_and_dense_models(states)
    batches = (batch_legacy(states, ignore), batch_dense(states, vocab))
    results = []
    for model, (idx, off) in zip((full, dense), batches):
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        torch.manual_seed(123)  # same dropout masks in both runs
        model.train()
        for _ in range(3):
            loss = sum(o.float().pow(2).mean() for o in model(idx, off))
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            results.append((model(idx, off), model.embedding.weight.detach().clone()))
    (out_full, w_full), (out_dense, w_dense) = results
    for a, b in zip(out_full, out_dense):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(w_full[torch.from_numpy(vocab.ids)], w_dense, rtol=0, atol=0)


def test_dataset_loader_matches_ignore_path():
    states = synthetic_states(n_states=120, seed=1)
    kept, ignore = legacy_kept(states)
    vocab = FeatureVocab(kept)
    indices, idxptr = as_flat(states)
    with tempfile.TemporaryDirectory() as d:
        with h5py.File(os.path.join(d, "session1_test.hdf5"), "w") as f:
            f["/indices"] = indices.astype(np.int32)
            f["/offsets"] = idxptr
            f["/row"] = np.zeros((len(states), 8), dtype=np.float32)
        legacy = H5Indexed(d, set(ignore))
        dense = H5Indexed(d, vocab=vocab)
        assert len(legacy) == len(dense) == len(states)
        for k in range(len(states)):
            ids_legacy = legacy[k][0].numpy()
            ids_dense = vocab.ids[dense[k][0].numpy()]
            assert np.array_equal(ids_legacy, ids_dense), k


def write_states(d, states, name="session1_test.hdf5"):
    indices, idxptr = as_flat(states)
    with h5py.File(os.path.join(d, name), "w") as f:
        f["/indices"] = indices.astype(np.int32)
        f["/offsets"] = idxptr
        f["/row"] = np.zeros((len(states), 8), dtype=np.float32)


def test_dataset_loader_has_the_same_set_semantics_as_the_server():
    """A state with a repeated id has to become the same tokens in training as in play; otherwise
    the mean-pool weighs that feature twice while training and once while playing."""
    vocab = FeatureVocab([5, 6, 7], feature_hash_bins=TABLE_ROWS)
    states = [[5, 5, 6, 9, 7, 7], [6], [5, 5, 5]]
    with tempfile.TemporaryDirectory() as d:
        write_states(d, states)
        ds = H5Indexed(d, vocab=vocab)
        assert [ds[k][0].tolist() for k in range(len(states))] == [[0, 1, 2], [1], [0]]
        for k, state in enumerate(states):                      # identical to the server's mapping
            rows, _ = vocab.map_bags(state, [0])
            assert ds[k][0].tolist() == rows.tolist()


def test_ids_wider_than_the_full_table_reach_the_vocab():
    """FeatureVocab keys on the id XMage wrote. The loader must not fold it into the full table's
    bin count first, or a wider hash space collapses back onto 2M before the vocab sees it."""
    wide = GLOBAL_MAX + 123                                     # would fold to 123
    # distinct co-occurrence patterns, so the ignore rule keeps all three as separate features
    rng = np.random.default_rng(3)
    states = [sorted({7} | ({wide} if rng.random() < 0.6 else set()) | ({123} if rng.random() < 0.6 else set()))
              for _ in range(60)]
    with tempfile.TemporaryDirectory() as d:
        write_states(d, states)
        raw = H5Indexed(d)                                      # no fold_bins: ids as written
        ids = kept_feature_ids(raw.indices_t.numpy(), raw.idxptr_t.numpy())
        assert wide in ids.tolist() and 123 in ids.tolist()      # distinct features, distinct rows

        vocab = FeatureVocab(ids, feature_hash_bins=2 ** 31)
        dense = H5Indexed(d, vocab=vocab)
        both = next(k for k, st in enumerate(states) if wide in st and 123 in st)
        assert sorted(vocab.ids[dense[both][0].numpy()].tolist()) == [7, 123, wide]

        folded = H5Indexed(d, fold_bins=GLOBAL_MAX)             # the full-table path still folds
        assert sorted(folded[both][0].tolist()) == [7, 123, 123]


def test_vocab_is_append_only():
    vocab = FeatureVocab([50, 10, 30])
    rows_before = vocab.lookup([10, 30, 50]).tolist()
    assert vocab.extend([30, 20, 999_999_999]) == 2          # ids beyond 2M are fine
    assert vocab.lookup([10, 30, 50]).tolist() == rows_before
    assert vocab.lookup([20, 999_999_999, 7]).tolist() == [3, 4, -1]
    restored = FeatureVocab.from_state_dict(vocab.state_dict())
    assert np.array_equal(restored.ids, vocab.ids)

    torch.manual_seed(0)
    model = NetTransformer(num_embeddings=3)
    before = model.embedding.weight.detach().clone()
    model.resize_embedding(len(vocab))
    assert model.embedding.num_embeddings == 5
    torch.testing.assert_close(model.embedding.weight[:3], before, rtol=0, atol=0)


def test_map_bags_offsets():
    vocab = FeatureVocab([5, 6, 7])
    rows, offsets = vocab.map_bags([5, 1, 6, 2, 3, 7], [0, 2, 3, 5])   # bags: [5,1] [6] [2,3] [7]
    assert rows.tolist() == [0, 1, 2] and offsets.tolist() == [0, 1, 2, 2]


def test_duplicate_ids_in_a_bag_match_the_full_table_path():
    """The full-table server maps a state through BitMap(indices), which drops repeats. A repeat
    would otherwise be pooled twice, so the dense path has to collapse it too. Includes the case
    the collision makes real: two distinct features hashing to one id, both active in one state."""
    vocab = FeatureVocab([5, 6, 7], feature_hash_bins=TABLE_ROWS)
    ignore = BitMap(set(range(TABLE_ROWS)) - {5, 6, 7})
    torch.manual_seed(0)
    model = NetTransformer(num_embeddings=TABLE_ROWS).eval()
    dense = NetTransformer(num_embeddings=len(vocab)).eval()
    with torch.no_grad():                      # same rows in both tables
        dense.load_state_dict({k: v for k, v in model.state_dict().items() if k != "embedding.weight"},
                              strict=False)
        dense.embedding.weight.copy_(model.embedding.weight[torch.tensor(vocab.ids)])

    raw = [5, 5, 6, 9, 7, 7, 7]                # 5 and 7 repeat; 9 is outside the vocab
    full_ids = sorted(BitMap(raw) - ignore)    # what the full-table server feeds the model
    rows, offsets = vocab.map_bags(raw, [0])
    with torch.no_grad():
        want = model(torch.tensor(full_ids), torch.tensor([0]))
        got = dense(torch.tensor(rows), torch.tensor(offsets))
    for a, b in zip(want, got):
        torch.testing.assert_close(a, b, rtol=0, atol=0)

    # and per bag, not just globally
    rows, offsets = vocab.map_bags([5, 5, 6, 7, 7], [0, 3])
    assert rows.tolist() == [0, 1, 2] and offsets.tolist() == [0, 2]


def test_initial_rows_depend_only_on_the_feature_id():
    """A feature's starting row comes from its id, so when it first appears - in a fresh vocab or
    appended by a later generation - it starts from the same row either way."""
    dim = 8
    fresh = initial_rows([10, 20, 30], dim)
    assert fresh.shape == (3, dim)
    np.testing.assert_array_equal(fresh[1], initial_rows([20], dim)[0])
    np.testing.assert_array_equal(fresh[::-1], initial_rows([30, 20, 10], dim))
    assert not np.array_equal(fresh[0], fresh[1])

    # a later generation appends 30; it lands on the row a fresh vocab would have given it
    vocab = FeatureVocab([10, 20])
    model = NetTransformer(num_embeddings=len(vocab))
    d = model.embedding.embedding_dim
    added = vocab.extend([30])
    model.resize_embedding(len(vocab), initial_rows(vocab.ids[len(vocab) - added:], d))
    torch.testing.assert_close(model.embedding.weight[2],
                               torch.from_numpy(initial_rows([30], d)[0]), rtol=0, atol=0)


def test_vocab_refuses_a_different_feature_encoding():
    vocab = FeatureVocab([1, 2, 3], feature_hash_bins=2_000_000)
    vocab.require_encoding(2_000_000)                      # same encoding: fine
    for bad in ({"feature_hash_bins": 2 ** 31}, {"hash_version": 2}, {"hash_algorithm": "other"}):
        try:
            vocab.require_encoding(**{"feature_hash_bins": 2_000_000, **bad})
        except ValueError:
            pass
        else:
            raise AssertionError(f"expected a refusal for {bad}")

    # a vocab saved before the encoding record is read as this encoding, not refused
    old = FeatureVocab.from_state_dict({"format_version": 1, "ids": torch.tensor([1, 2, 3]),
                                        "feature_hash_bins": 2_000_000})
    old.require_encoding(2_000_000)
    assert np.array_equal(old.ids, vocab.ids)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
