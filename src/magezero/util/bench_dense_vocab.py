"""
Benchmark the full embedding table against the dense feature vocab on real self-play data.

  python src/magezero/util/bench_dense_vocab.py --data data/<deck>/ver<N>/training
  python src/magezero/util/bench_dense_vocab.py --data ... --batch 512 --steps 50   # upstream batch on a big GPU

Every measurement runs in its own subprocess so peak memory is not shared between variants.
Both variants see the same batches of the same states. Reports (JSON + a markdown table):
  * building the ignore list (2M-column scan) vs the vocab (observed ids only)
  * parameter / optimizer-state size, training step time and peak memory
  * inference time per state and peak memory
  * checkpoint size on disk and load time
Device: cuda if available, else mps, else cpu (override with --device).
"""
import argparse
import gzip
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def pick_device(name):
    import torch
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        # the eval-mode transformer fast path uses nested-tensor ops MPS lacks
        torch.backends.mha.set_fastpath_enabled(False)
        return torch.device("mps")
    return torch.device("cpu")


def peak_rss_mb():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r / 2**20 if sys.platform == "darwin" else r / 1024  # bytes on macOS, KB on Linux


def device_peak_mb(device):
    import torch
    if device.type == "cuda":
        return torch.cuda.max_memory_allocated() / 2**20
    if device.type == "mps":
        return torch.mps.driver_allocated_memory() / 2**20
    return None


def sync(device):
    import torch
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def load_variant(args, variant):
    """Returns (model, dataset) for 'full' (ignore list + full table) or 'dense' (vocab)."""
    import numpy as np
    from pyroaring import BitMap
    from dataset import H5Indexed, create_redundancy_ignore_list
    from model import NetTransformer
    from vocab import FeatureVocab, kept_feature_ids
    raw = H5Indexed(args.data)
    kept = kept_feature_ids(raw.indices_t.numpy(), raw.idxptr_t.numpy())
    if variant == "dense":
        vocab = FeatureVocab(kept)
        return NetTransformer(num_embeddings=len(vocab)), H5Indexed(args.data, vocab=vocab)
    # the upstream path: ignore list over the 2M id space, one row per possible id
    ignore = create_redundancy_ignore_list(raw)
    return NetTransformer(num_embeddings=2_000_000), H5Indexed(args.data, ignore)


def fixed_batches(ds, batch, n, seed=0):
    import torch
    from torch.utils.data import DataLoader
    from dataset import collate_batch
    g = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=batch, shuffle=True, generator=g, collate_fn=collate_batch)
    out = []
    for b in dl:
        out.append(b)
        if len(out) >= n:
            break
    return out


def run_child(args):
    import torch
    device = pick_device(args.device)
    torch.manual_seed(0)
    res = {"variant": args.variant, "phase": args.phase, "device": str(device)}
    t = time.perf_counter()
    model, ds = load_variant(args, args.variant)
    res["setup_s"] = time.perf_counter() - t
    res["params_total"] = sum(p.numel() for p in model.parameters())
    res["params_embedding"] = model.embedding.weight.numel()
    res["embedding_rows"] = model.embedding.num_embeddings
    model = model.to(device)
    batches = fixed_batches(ds, args.batch, args.steps + args.warmup)

    if args.phase == "train":
        opt = torch.optim.Adam(model.parameters(), lr=1e-4)
        model.train()
        times = []
        for i, (idx, off, *_rest) in enumerate(batches):
            idx, off = idx.to(device), off.to(device)
            t = time.perf_counter()
            loss = sum(o.float().pow(2).mean() for o in model(idx, off))
            opt.zero_grad(); loss.backward(); opt.step(); sync(device)
            if i >= args.warmup:
                times.append(time.perf_counter() - t)
        res["train_step_s"] = sum(times) / len(times)
        res["optimizer_state_mb"] = sum(v.numel() * v.element_size() for s in opt.state.values()
                                        for v in s.values() if torch.is_tensor(v)) / 2**20
    else:
        model.eval()
        times, states = [], 0
        with torch.no_grad():
            for i, (idx, off, *_rest) in enumerate(batches):
                idx, off = idx.to(device), off.to(device)
                t = time.perf_counter()
                model(idx, off); sync(device)
                if i >= args.warmup:
                    times.append(time.perf_counter() - t); states += off.shape[0]
        res["infer_ms_per_state"] = 1000 * sum(times) / states
        with tempfile.TemporaryDirectory() as d:  # checkpoint as train.py writes it
            raw_p, gz_p = os.path.join(d, "m.pt"), os.path.join(d, "m.pt.gz")
            torch.save({"model_state_dict": model.state_dict()}, raw_p)
            with open(raw_p, "rb") as fi, gzip.open(gz_p, "wb", compresslevel=1) as fo:
                shutil.copyfileobj(fi, fo, length=16 * 2**20)
            res["checkpoint_mb"] = os.path.getsize(gz_p) / 2**20
            t = time.perf_counter()
            with gzip.open(gz_p, "rb") as f:
                torch.load(f, map_location="cpu")
            res["checkpoint_load_s"] = time.perf_counter() - t
    res["device_peak_mb"] = device_peak_mb(device)
    res["process_peak_rss_mb"] = peak_rss_mb()
    print("RESULT " + json.dumps(res))


def time_ignore_vs_vocab(args):
    from dataset import H5Indexed, create_redundancy_ignore_list
    from vocab import kept_feature_ids
    raw = H5Indexed(args.data)
    t = time.perf_counter(); ignore = create_redundancy_ignore_list(raw); t_full = time.perf_counter() - t
    t = time.perf_counter(); kept = kept_feature_ids(raw.indices_t.numpy(), raw.idxptr_t.numpy()); t_dense = time.perf_counter() - t
    import numpy as np
    observed = np.unique(raw.indices_t.numpy())
    legacy_kept = np.setdiff1d(observed, np.fromiter(ignore, dtype=np.int64))
    return {"states": len(raw), "tokens_per_state": float(raw.indices_t.numel() / max(len(raw), 1)),
            "observed_ids": int(len(observed)), "kept_ids": int(len(kept)),
            "ignore_list_s": t_full, "vocab_s": t_dense, "same_kept_set": bool(np.array_equal(legacy_kept, kept))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory of HDF5 shards")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--phases", default="infer,train")
    ap.add_argument("--variants", default="full,dense")
    ap.add_argument("--out", default=None, help="write results JSON here")
    ap.add_argument("--variant"); ap.add_argument("--phase")  # child mode
    args = ap.parse_args()
    if args.variant:
        return run_child(args)

    results = {"ignore_vs_vocab": time_ignore_vs_vocab(args), "runs": []}
    for phase in args.phases.split(","):
        for variant in args.variants.split(","):
            cmd = [sys.executable, __file__, "--variant", variant, "--phase", phase] + \
                  [x for k, v in vars(args).items() if k in ("data", "device", "batch", "steps", "warmup")
                   for x in (f"--{k.replace('_', '-')}", str(v))]
            p = subprocess.run(cmd, capture_output=True, text=True)
            line = next((l for l in p.stdout.splitlines() if l.startswith("RESULT ")), None)
            if line is None:
                err = (p.stderr or p.stdout).strip().splitlines()[-1:] or ["no output"]
                results["runs"].append({"variant": variant, "phase": phase, "error": err[0]})
            else:
                results["runs"].append(json.loads(line[7:]))
            print(json.dumps(results["runs"][-1]), flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(results, f, indent=1)
    print(json.dumps(results["ignore_vs_vocab"], indent=1))


if __name__ == "__main__":
    main()
