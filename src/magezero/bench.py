import time

import torch

from dataset import H5Graphs, collate_graphs
from model import load_checkpoint, DEVICE

BATCH_SIZES = (1, 8, 32, 128, 512)
REPEATS = 20


def bench(deck: str, version: int):
    """Inference forward-pass time and peak memory per batch size, on real test states (fp16 autocast,
    as the server runs). Each size is warmed up first, then timed as the mean of REPEATS passes."""
    model, vocab, edge_vocab = load_checkpoint(f"models/{deck}/ver{version}/model.pt.gz")
    model = model.to(DEVICE).eval()
    ds = H5Graphs(f"data/{deck}/ver{version}/testing", vocab, edge_vocab)
    print(f"arch: {model.arch}")
    for b in BATCH_SIZES:
        g = collate_graphs(*zip(*(ds[k][:6] for k in range(b)))).to(DEVICE)
        with torch.no_grad(), torch.amp.autocast(DEVICE.type, enabled=DEVICE.type == "cuda"):
            model(g)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(REPEATS):
                model(g)
            torch.cuda.synchronize()
        ms = (time.perf_counter() - start) * 1000 / REPEATS
        print(f"bench batch={b}  {ms:.1f} ms  {b * 1000 / ms:.0f} states/s  "
              f"{torch.cuda.max_memory_allocated() / 2**30:.2f} GB")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()
    bench(args.deck, args.version)
