import time

import torch

from dataset import H5Graphs, collate_graphs
from graphed import GraphedNet
from model import load_checkpoint, DEVICE

BATCH_SIZES = (1, 8, 32, 64, 128, 192, 512)
REPEATS = 20


def ties(scores, offsets):
    """States whose top-scored node is shared by another node (to within 1e-4): any of them is a correct top-1."""
    return {s for s in range(len(offsets) - 1)
            if len(torch.topk(scores[offsets[s]:offsets[s + 1]], 2).values) > 1
            and float(torch.topk(scores[offsets[s]:offsets[s + 1]], 2).values.diff().abs()) < 1e-4}


def agreement(a, b, offsets):
    """Max abs difference per output (finite entries) and top-1 disagreements per state, ties excluded."""
    diff = [float((x[torch.isfinite(x)] - y[torch.isfinite(y)]).abs().max()) for x, y in zip(a, b)]
    flips = 0
    for h in (0, 1):
        tied = ties(a[h], offsets) | ties(b[h], offsets)
        flips += sum(int(a[h][offsets[s]:offsets[s + 1]].argmax() != b[h][offsets[s]:offsets[s + 1]].argmax())
                     for s in range(len(offsets) - 1) if s not in tied)
    flips += int((a[2].argmax(1) != b[2].argmax(1)).sum())
    return diff, flips


def bench(deck: str, version: int):
    """Inference time per batch size on real test states, as the server runs it (CPU batch in, CPU outputs back,
    fp16 autocast): eager forward building its plan on the GPU, eager with a ready CPU plan, and CUDA graphs
    (GraphedNet, ready plan). The server builds plans per request in the HTTP threads, so they're not timed. Also
    how far graphed outputs are from eager ones, next to how far two eager runs are from each other (index_add on
    the GPU is not bit-reproducible)."""
    model, vocab, edge_vocab = load_checkpoint(f"models/{deck}/ver{version}/model.pt.gz")
    model = model.to(DEVICE).eval()
    graphed = GraphedNet(model, DEVICE)
    ds = H5Graphs(f"data/{deck}/ver{version}/testing", vocab, edge_vocab)
    print(f"arch: {model.arch}")

    def eager(g, plan=None):
        with torch.no_grad(), torch.amp.autocast(DEVICE.type):
            return [t.float().cpu() for t in model(g.to(DEVICE), plan and plan.to(DEVICE))]

    for b in BATCH_SIZES:
        g = collate_graphs(*zip(*(ds[k][:6] for k in range(b))))
        plan = model.plan(g)  # the server builds this per request, off the worker's critical path
        t0 = time.perf_counter()
        ref, again, out = eager(g), eager(g), [t.float() for t in graphed(g, plan)]
        capture = (time.perf_counter() - t0) * 1000
        noise, _ = agreement(ref, again, g.node_offsets)
        diff, flips = agreement(ref, out, g.node_offsets)
        times = []
        for run in (lambda: eager(g), lambda: eager(g, plan), lambda: graphed(g, plan)):
            run()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(REPEATS):
                run()
            times.append((time.perf_counter() - start) * 1000 / REPEATS)
        print(f"bench batch={b}  eager {times[0]:.1f} ms  eager+plan {times[1]:.1f} ms  graphed {times[2]:.1f} ms  "
              f"({b * 1000 / times[2]:.0f} states/s)  first call incl. capture {capture:.0f} ms")
        print(f"      max |graphed - eager| priority {diff[0]:.4f} target {diff[1]:.4f} use {diff[2]:.4f} value {diff[3]:.4f}  "
              f"(eager vs eager: {noise[0]:.4f} {noise[1]:.4f} {noise[2]:.4f} {noise[3]:.4f})  top-1 flips (ties excluded): {flips}")
    print(f"graphs captured: {len(graphed.graphs)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    bench(args.deck, args.version)
