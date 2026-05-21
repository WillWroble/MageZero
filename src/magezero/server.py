import os
import threading
import time
from queue import Queue, Empty

import torch
import waitress
from pyroaring import BitMap
from flask import Flask, request, Response
import msgpack
from model import NetTransformer, Net, load_model, GLOBAL_MAX, ACTIONS_MAX

# Device setup
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Threading config
TORCH_THREADS = 1 #max(1, os.cpu_count() // 2)
torch.set_num_threads(TORCH_THREADS)

# Batching config
MAX_BATCH = 16
MAX_WAIT_MS = 0

#module state
server_model = None
IGNORE_BM = None
VALID_RANGE = None

app = Flask(__name__)

req_counter = 0
req_counter_lock = threading.Lock()

def init(deck: str, version: int, port: int):
    global server_model, IGNORE_BM, VALID_RANGE

    model_dir = f"models/{deck}/ver{version}"
    ignore_path = f"{model_dir}/ignore.roar"
    model_path = f"{model_dir}/best.pt.gz"

    with open(ignore_path, "rb") as f:
        IGNORE_BM = BitMap.deserialize(f.read())

    VALID_RANGE = BitMap(range(GLOBAL_MAX))

    server_model = NetTransformer(GLOBAL_MAX, ACTIONS_MAX).to(DEVICE).eval()
    ckpt = load_model(model_path)
    server_model.load_state_dict(ckpt["model_state_dict"])

    threading.Thread(target=worker_loop, daemon=True).start()

    print(f"[INIT] deck={deck} ver={version} port={port} device={DEVICE}")
    waitress.serve(app, host="127.0.0.1", port=port, threads=6)

class Pending:
    __slots__ = ("idx", "off", "evt", "out", "req_id", "pre_count", "post_count", "t_recv", "t_done", "num_bags")

    def __init__(self, req_id, indices, offsets):
        self.req_id = req_id
        self.pre_count = len(indices)
        self.t_recv = time.perf_counter()
        self.evt = threading.Event()
        self.out = None
        self.t_done = 0.0

        indices, offsets, num_bags = apply_ignore(indices, offsets)
        self.post_count = len(indices)
        self.num_bags = num_bags

        self.idx = torch.tensor(indices, dtype=torch.long)
        self.off = torch.tensor(offsets, dtype=torch.long)


def apply_ignore(indices: list[int], offsets: list[int] | None):
    if not offsets:
        offsets = [0]

    if len(offsets) == 1:
        # Single bag - pure bitmap ops in C
        kept_bm = (BitMap(indices) - IGNORE_BM) & VALID_RANGE
        return list(kept_bm), [0], 1

    # Multi-bag
    n = len(indices)
    new_indices = []
    new_offsets = [0]

    for b in range(len(offsets)):
        start = offsets[b]
        end = offsets[b + 1] if b + 1 < len(offsets) else n

        kept_bm = (BitMap(indices[start:end]) - IGNORE_BM) & VALID_RANGE
        new_indices.extend(kept_bm)
        new_offsets.append(len(new_indices))

    new_offsets = new_offsets[:-1]
    return new_indices, new_offsets, len(new_offsets)


Q: "Queue[Pending]" = Queue(maxsize=4096)


def worker_loop():
    while True:
        p0 = Q.get()
        batch = [p0]

        # Collect more requests up to MAX_BATCH or MAX_WAIT_MS
        deadline = time.perf_counter() + (MAX_WAIT_MS / 1000.0)
        while len(batch) < MAX_BATCH or not Q.empty():
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                remaining = 0
                if Q.empty():
                    break
            try:
                batch.append(Q.get(timeout=remaining))
            except Empty:
                break

        bag_counts = [p.num_bags for p in batch]

        # Fast path: single request
        if len(batch) == 1:
            idx = batch[0].idx.to(DEVICE, non_blocking=True)
            off = batch[0].off.to(DEVICE, non_blocking=True)
        else:
            # Concatenate indices
            idx = torch.cat([p.idx for p in batch]).to(DEVICE, non_blocking=True)

            # Vectorized offset adjustment
            all_off = torch.cat([p.off for p in batch])
            idx_lens = torch.tensor([len(p.idx) for p in batch])
            bag_counts_t = torch.tensor(bag_counts)
            adjustments = torch.repeat_interleave(
                torch.cat([torch.tensor([0]), idx_lens.cumsum(0)[:-1]]),
                bag_counts_t
            )
            off = (all_off + adjustments).to(DEVICE, non_blocking=True)

        # Single forward pass
        with torch.no_grad():
            pA, pB, tgt, bin2, val = server_model(idx, off)

        # Move to CPU once
        pA = pA.cpu()
        pB = pB.cpu()
        tgt = tgt.cpu()
        bin2 = bin2.cpu()
        val = val.cpu()

        # Split results back to individual requests
        row = 0
        for p, num_bags in zip(batch, bag_counts):
            if num_bags == 1:
                p.out = {
                    "policy_player": pA[row].tolist(),
                    "policy_opponent": pB[row].tolist(),
                    "policy_target": tgt[row].tolist(),
                    "policy_binary": bin2[row].tolist(),
                    "value": float(val[row].item()),
                }
            else:
                p.out = [
                    {
                        "policy_player": pA[row + i].tolist(),
                        "policy_opponent": pB[row + i].tolist(),
                        "policy_target": tgt[row + i].tolist(),
                        "policy_binary": bin2[row + i].tolist(),
                        "value": float(val[row + i].item()),
                    }
                    for i in range(num_bags)
                ]
            row += num_bags
            p.t_done = time.perf_counter()
            p.evt.set()

        print(f"[BATCH] size={len(batch)}, total_bag_size={row}")


#threading.Thread(target=worker_loop, daemon=True).start()


@app.post("/evaluate")
def evaluate():
    global req_counter

    data = msgpack.unpackb(request.data, raw=False)
    with req_counter_lock:
        req_counter += 1

    indices = data.get("indices", [])
    offsets = data.get("offsets", [])
    pending = Pending(req_counter, indices, offsets)

    print(f"[REQ {pending.req_id}] indices={pending.pre_count}, kept={pending.post_count}, bag_size={pending.num_bags}")

    Q.put(pending)
    pending.evt.wait()

    total_ms = (pending.t_done - pending.t_recv) * 1000.0
    print(f"[REQ {pending.req_id}] done: {total_ms:.1f}ms")

    return Response(msgpack.packb(pending.out, use_bin_type=True), mimetype="application/x-msgpack")


@app.get("/healthz")
def healthz():
    return "ok", 200


if __name__ == "__main__":
    import argparse
    import waitress
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--port", type=int, default=50052)
    args = parser.parse_args()
    init(args.deck, args.version, args.port)