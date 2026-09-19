"""
Equivalence tests for the dense feature vocab (src/magezero/vocab.py).

The claim being tested: a dense-vocab model is the same model as the full-table one, with the
unused rows removed. Concretely:
  1. the vocab built from data is exactly the complement of the ignore list;
  2. a converted checkpoint gives bit-identical outputs to the original on the same states;
  3. training steps give bit-identical results (Adam never moves rows with no gradient);
  4. the dataset loader yields the same features per state either way;
  5. the vocab/embedding are append-only across generations.

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

from dataset import H5Indexed, collate_batch, create_redundancy_ignore_list  # noqa: E402
from model import NetTransformer  # noqa: E402
from vocab import FeatureVocab, kept_feature_ids  # noqa: E402
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


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print(f"PASS {name}")
