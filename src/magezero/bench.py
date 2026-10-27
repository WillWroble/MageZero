import time

import torch

from dataset import H5Graphs, collate_graphs
from model import load_checkpoint, DEVICE

BATCH_SIZES = (1, 8, 32, 128, 512)
REPEATS = 20


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


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    bench(args.deck, args.version)
