import time

import numpy as np
import torch

from cache import NodeCache, subtree_hashes
from dataset import H5Graphs, collate_graphs
from model import load_checkpoint, DEVICE

BATCH_SIZES = (1, 8, 32, 128, 512)
REPEATS = 20
STREAM_SIZES = (1, 8, 32, 128, 512)
STREAM_STEPS = 40


def bench(deck: str, version: int):
    """Inference time and peak memory per batch size, on real test states (fp16 autocast, as the server runs).
    Each pass goes as in the server: a CPU batch moved to the GPU, the forward pass, the outputs copied back.
    'gpu plan' builds the forward pass's index work on the GPU (forward computes it), 'cpu plan' builds it on
    the CPU first (as the server does). Each size is warmed up, then timed as the mean of REPEATS passes."""
    model, vocab, edge_vocab = load_checkpoint(f"models/{deck}/ver{version}/model.pt.gz")
    model = model.to(DEVICE).eval()
    ds = H5Graphs(f"data/{deck}/ver{version}/testing", vocab, edge_vocab)
    print(f"arch: {model.arch}")

    def gpu_plan(g):
        return model(g.to(DEVICE))

    def cpu_plan(g):
        return model(g.to(DEVICE), model.plan(g).to(DEVICE))

    for b in BATCH_SIZES:
        g = collate_graphs(*zip(*(ds[k][:6] for k in range(b))))
        times = []
        for run in (gpu_plan, cpu_plan):
            with torch.no_grad(), torch.amp.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                run(g)
                torch.cuda.reset_peak_memory_stats()
                start = time.perf_counter()
                for _ in range(REPEATS):
                    [t.cpu() for t in run(g)]
            times.append((time.perf_counter() - start) * 1000 / REPEATS)
        print(f"bench batch={b}  gpu plan {times[0]:.1f} ms  cpu plan {times[1]:.1f} ms  "
              f"{b * 1000 / times[1]:.0f} states/s  {torch.cuda.max_memory_allocated() / 2**30:.2f} GB")
    bench_cache(model, ds)


def bench_cache(model, ds):
    """The node cache on a stream of batches, as the server runs it: each batch takes the next 4 consecutive states
    (1 at batch 1) of batch/4 games spread over the test set (fewer steps when the set is too small for them not
    to overlap), and goes through the server's lookup, plan, forward and copy back, without and with the cache (one
    cache per batch size). Times are means over the steps after the first 2.
    Also the largest output difference, and the top-1 changes where the plain top led by more than 1e-3."""
    hashes = np.concatenate([subtree_hashes(ds.ids[ds.offsets[k]:ds.offsets[k + 1]], ds.values[ds.offsets[k]:ds.offsets[k + 1]],
                                            *(x[ds.edge_offsets[k]:ds.edge_offsets[k + 1]] for x in (ds.edge_child, ds.edge_parent, ds.edge_label)))
                             for k in range(len(ds))])
    for b in STREAM_SIZES:
        cache = NodeCache(model, DEVICE)
        games = max(1, b // 4)
        per = b // games
        steps = min(STREAM_STEPS, len(ds) // b)
        starts = np.arange(games) * (len(ds) // games)
        times, computed, internal, diff, flips = np.zeros((2, 3)), 0, 0, 0.0, 0
        for step in range(steps):
            idx = [s + step * per + j for s in starts for j in range(per)]
            g = collate_graphs(*zip(*(ds[k][:6] for k in idx)))
            h = np.concatenate([hashes[ds.offsets[k]:ds.offsets[k + 1]] for k in idx])
            outs = []
            for i, cached in enumerate((False, True)):
                with torch.no_grad(), torch.amp.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                    t = [time.perf_counter()]
                    c = cache.lookup(g.node_type.numpy(), h) if cached else None
                    t.append(time.perf_counter())
                    gd = g.to(DEVICE)
                    plan = model.plan(g if b <= 32 else gd, c).to(DEVICE)
                    torch.cuda.synchronize()
                    t.append(time.perf_counter())
                    outs.append([x.float().cpu() for x in model(gd, plan, cache.table if cached else None)])
                    t.append(time.perf_counter())
                if step >= 2:
                    times[i] += np.diff(t) * 1000 / (steps - 2)
            computed += int(c[1].sum())
            internal += int((c[0] >= 0).sum())
            for x, y in zip(*outs):
                fin = torch.isfinite(x)
                diff = max(diff, float((x[fin] - y[fin]).abs().max()))
            for x, y in zip(outs[0][:2], outs[1][:2]):
                for s in range(len(idx)):
                    xs, ys = x[g.node_offsets[s]:g.node_offsets[s + 1]], y[g.node_offsets[s]:g.node_offsets[s + 1]]
                    top = int(ys.argmax())
                    flips += int((xs == xs.max()).sum() == 1 and xs.max() - xs[top] > 1e-3)
        print(f"cache batch={b} ({steps} steps)  plain {times[0].sum():.1f} ms  cached {times[1].sum():.1f} ms "
              f"(lookup {times[1, 0]:.1f}, plan {times[1, 1]:.1f} vs {times[0, 1]:.1f}, forward {times[1, 2]:.1f} vs "
              f"{times[0, 2]:.1f})  computed {computed / internal:.1%} of internal nodes  max |diff| {diff:.3f}  "
              f"top-1 changes {flips}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    bench(args.deck, args.version)
