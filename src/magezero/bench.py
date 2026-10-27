import time

import numpy as np
import torch

from cache import NodeCache, subtree_hashes
from dataset import H5Graphs, collate_graphs
from graphed import GraphedNet
from model import load_checkpoint, DEVICE

BATCH_SIZES = (1, 8, 32, 128, 512)
REPEATS = 20
STREAM_SIZES = (1, 8, 32, 128, 512)
STREAM_STEPS = 40
GRAPH_SIZES = (1, 8, 32, 128)


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
    to overlap), and goes through the server's lookup, plan, forward and copy back: plain, cached, and cached through
    CUDA graphs (up to GRAPH_SIZES; each cached path has its own cache). Times are means over the steps after the
    first 2; graphed steps that captured a graph are left out and reported apart.
    Also the largest output difference from plain, and the top-1 changes where the plain top led by more than 1e-3."""
    hashes = np.concatenate([subtree_hashes(ds.ids[ds.offsets[k]:ds.offsets[k + 1]], ds.values[ds.offsets[k]:ds.offsets[k + 1]],
                                            *(x[ds.edge_offsets[k]:ds.edge_offsets[k + 1]] for x in (ds.edge_child, ds.edge_parent, ds.edge_label)))
                             for k in range(len(ds))])
    for b in STREAM_SIZES:
        paths = ["plain", "cached"] + (["graphed"] if b in GRAPH_SIZES else [])
        caches = {"cached": NodeCache(model, DEVICE), "graphed": NodeCache(model, DEVICE)}
        graphed = GraphedNet(model, caches["graphed"], DEVICE)
        games = max(1, b // 4)
        per = b // games
        steps = min(STREAM_STEPS, len(ds) // b)
        starts = np.arange(games) * (len(ds) // games)
        times = {p: np.zeros(3) for p in paths}
        timed = {p: 0 for p in paths}
        capture_ms, computed, internal, diff, flips = 0.0, 0, 0, {p: 0.0 for p in paths}, {p: 0 for p in paths}
        for step in range(steps):
            idx = [s + step * per + j for s in starts for j in range(per)]
            g = collate_graphs(*zip(*(ds[k][:6] for k in idx)))
            h = np.concatenate([hashes[ds.offsets[k]:ds.offsets[k + 1]] for k in idx])
            outs = {}
            for path in paths:
                captures = graphed.captures
                with torch.no_grad(), torch.amp.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
                    t = [time.perf_counter()]
                    c = None if path == "plain" else caches[path].lookup(g.node_type.numpy(), h)
                    t.append(time.perf_counter())
                    if path == "graphed":
                        plan = model.plan(g, c)
                        t.append(time.perf_counter())
                        outs[path] = list(graphed(g, plan))
                    else:
                        gd = g.to(DEVICE)
                        plan = model.plan(g if b <= 32 else gd, c).to(DEVICE)
                        torch.cuda.synchronize()
                        t.append(time.perf_counter())
                        outs[path] = [x.float().cpu() for x in model(gd, plan, None if c is None else caches[path].table)]
                    t.append(time.perf_counter())
                if graphed.captures > captures:
                    capture_ms += (t[-1] - t[0]) * 1000
                elif step >= 2:
                    times[path] += np.diff(t) * 1000
                    timed[path] += 1
                if path == "cached":
                    computed += int(c[1].sum())
                    internal += int((c[0] >= 0).sum())
            for path in paths[1:]:
                for x, y in zip(outs["plain"], outs[path]):
                    fin = torch.isfinite(x)
                    diff[path] = max(diff[path], float((x[fin] - y[fin]).abs().max()))
                for x, y in zip(outs["plain"][:2], outs[path][:2]):
                    for s in range(len(idx)):
                        xs, ys = x[g.node_offsets[s]:g.node_offsets[s + 1]], y[g.node_offsets[s]:g.node_offsets[s + 1]]
                        top = int(ys.argmax())
                        flips[path] += int((xs == xs.max()).sum() == 1 and xs.max() - xs[top] > 1e-3)
        line = f"cache batch={b} ({steps} steps, {computed / internal:.1%} of internal nodes computed)"
        for path in paths:
            m = times[path] / max(timed[path], 1)
            line += f"  {path} {m.sum():.1f} ms (lookup {m[0]:.1f}, plan {m[1]:.1f}, forward {m[2]:.1f})"
            if path != "plain":
                line += f" max |diff| {diff[path]:.3f} top-1 changes {flips[path]}"
        if "graphed" in paths:
            line += f"  {graphed.captures} captures, {capture_ms:.0f} ms in all"
        print(line)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    bench(args.deck, args.version)
