import threading
import time
import traceback
from queue import Queue, Empty

import msgpack
import numpy as np
import torch
import waitress
from flask import Flask, request, Response

from cache import NodeCache
from dataset import map_ids, node_types
from model import Graphs, load_checkpoint, AMP, DEVICE, NodeType

# Threading config
TORCH_THREADS = 1 #max(1, os.cpu_count() // 2)
torch.set_num_threads(TORCH_THREADS)

# Batching config
MAX_BATCH = 128
CPU_PLAN_MAX = 32  # states per batch up to which NetGraph.plan is built on the CPU (T4: faster below, slower above)
WORKERS = 1  # batching threads; must stay 1 while the node cache is used (see NodeCache)

#module state
server_model = None
VOCAB = None
EDGE_VOCAB = None
CACHE = None

app = Flask(__name__)

req_counter = 0
req_counter_lock = threading.Lock()

def init(model_dir: str, port: int):
    global server_model, VOCAB, EDGE_VOCAB, CACHE

    model, VOCAB, EDGE_VOCAB = load_checkpoint(f"{model_dir}/model.pt.gz")
    server_model = model.to(DEVICE).eval()
    CACHE = NodeCache(server_model, DEVICE)

    for _ in range(WORKERS):
        threading.Thread(target=worker_loop, daemon=True).start()

    print(f"[INIT] model={model_dir} port={port} device={DEVICE} "
          f"leaf_rows={len(VOCAB)} edge_rows={len(EDGE_VOCAB)}")
    waitress.serve(app, host="127.0.0.1", port=port, threads=8)


def as_array(x, dtype=np.int32):
    """One request array: raw little-endian int32 bytes (binary requests) or a list of ints."""
    return np.frombuffer(x, dtype="<i4").astype(dtype) if isinstance(x, bytes) else np.asarray(x, dtype=dtype)


def request_arrays(data):
    """An /evaluate request as whole-request arrays, with ids mapped to embedding rows the same way the training
    data is. RemoteModelEvaluator sends every state's nodes and edges concatenated, each array either as a list
    or (binary requests) as raw int32 bytes:
      indices, values                       per node
      offsets                               per state, its first node
      edge_child, edge_parent, edge_label   per edge; child and parent are local to the state
      edge_offsets                          per state, its first edge
    Returns node_type, node_row, values, edge_child, edge_parent, edge_row (edges local to the request) and each
    state's first node plus the node count, all int64 so the worker only concatenates."""
    ids, values = as_array(data["indices"]), as_array(data["values"])
    edge_child, edge_parent = as_array(data["edge_child"]), as_array(data["edge_parent"])
    node_type = node_types(ids)
    node_row, edge_row = map_ids(ids, node_type, as_array(data["edge_label"]), VOCAB, EDGE_VOCAB)
    starts = np.append(as_array(data["offsets"], np.int64), len(ids))
    edge_starts = np.append(as_array(data["edge_offsets"], np.int64), len(edge_child))
    state_start = np.repeat(starts[:-1], np.diff(edge_starts))
    return (node_type.astype(np.int64), node_row.astype(np.int64), values.astype(np.int64),
            edge_child + state_start, edge_parent + state_start, edge_row.astype(np.int64), starts)


def collate_requests(batch):
    """One batch of graphs from whole requests (the same tensors collate_graphs builds from their states), and
    every node's subtree hash if each request sent them (else None)."""
    bases = np.cumsum([0] + [len(p.arrays[0]) for p in batch], dtype=np.int64)

    def cat(parts):
        return torch.from_numpy(np.concatenate(parts))

    node_type, node_row, values = (cat([p.arrays[i] for p in batch]) for i in range(3))
    edge_child, edge_parent = (cat([p.arrays[i] + base for p, base in zip(batch, bases)]) for i in (3, 4))
    offsets = cat([p.arrays[6][:-1] + base for p, base in zip(batch, bases)] + [bases[-1:]])
    graphs = Graphs(node_type, node_row, values, offsets, edge_child, edge_parent, cat([p.arrays[5] for p in batch]))
    hashed = all(p.node_hash is not None for p in batch)
    return graphs, np.concatenate([p.node_hash for p in batch]) if hashed else None


class Pending:
    __slots__ = ("req_id", "arrays", "node_hash", "binary", "states", "nodes", "kept", "evt", "out", "error", "t_recv", "t_done")

    def __init__(self, req_id, data):
        self.req_id = req_id
        self.t_recv = time.perf_counter()
        self.evt = threading.Event()
        self.out = None
        self.error = None
        self.t_done = 0.0

        self.arrays = request_arrays(data)
        # per node, its subtree hash (raw int64): the node cache's key. Clients that don't send it go uncached
        self.node_hash = np.frombuffer(data["node_hash"], "<i8") if "node_hash" in data else None
        self.binary = bool(data.get("binary"))  # reply with raw float32 arrays instead of per-state maps
        self.states = len(self.arrays[6]) - 1
        self.nodes = len(self.arrays[0])
        # nodes the network uses: typed nodes, and leaves with a vocab row
        self.kept = int(((self.arrays[0] != NodeType.LEAF) | (self.arrays[1] >= 0)).sum())


Q: "Queue[Pending]" = Queue(maxsize=4096)


def worker_loop():
    while True:
        p0 = Q.get()
        batch = [p0]

        # Collect more requests up to MAX_BATCH
        while len(batch) < MAX_BATCH:
            try:
                batch.append(Q.get(block=False))
            except Empty:
                break

        states = sum(p.states for p in batch)
        t_start = time.perf_counter()
        t_collated = t_forward = t_start
        computed = None
        try:
            # Single forward pass over every state of every request
            graphs, node_hash = collate_requests(batch)
            cached = None if node_hash is None else CACHE.lookup(graphs.node_type.numpy(), node_hash)
            t_collated = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast(DEVICE.type, enabled=AMP):
                # small batches: the index work runs on the CPU, so the GPU runs the forward pass without stalls;
                # large ones: the CPU work costs more than the stalls (bench.py compares both)
                g = graphs.to(DEVICE)
                plan = server_model.plan(graphs if states <= CPU_PLAN_MAX else g, cached).to(DEVICE)
                out = server_model(g, plan, None if cached is None else CACHE.table)
            if cached is not None:
                computed = f"{int(cached[1].sum())}/{int((cached[0] >= 0).sum())}"
            priority, target, use, value = (t.float().cpu().numpy() for t in out[:4])
            t_forward = time.perf_counter()

            # Split results back to individual requests: per-node scores in the order each state was sent
            starts = np.asarray(graphs.node_offsets)
            row = 0
            for p in batch:
                rows = slice(row, row + p.states)
                if p.binary:
                    nodes = slice(starts[row], starts[row + p.states])
                    p.out = {"priority": priority[nodes].astype("<f4").tobytes(),
                             "target": target[nodes].astype("<f4").tobytes(),
                             "use": use[rows].astype("<f4").tobytes(),
                             "value": value[rows].astype("<f4").tobytes()}
                else:
                    p.out = []
                    for r in range(rows.start, rows.stop):
                        nodes = slice(starts[r], starts[r + 1])
                        p.out.append({
                            "policy_priority": priority[nodes].tolist(),
                            "policy_target": target[nodes].tolist(),
                            "policy_binary": use[r].tolist(),
                            "value": float(value[r]),
                        })
                row += p.states
        except Exception as e:
            # fail these requests instead of the worker, so later requests still get served
            traceback.print_exc()
            CACHE.clear()  # the failed batch's new slots were never written
            for p in batch:
                p.error = f"{type(e).__name__}: {e}"

        for p in batch:
            p.t_done = time.perf_counter()
            p.evt.set()

        t_end = time.perf_counter()
        print(f"[BATCH] size={len(batch)}, total_states={states}, "
              f"queued={(t_start - min(p.t_recv for p in batch)) * 1000:.1f}ms "
              f"collate={(t_collated - t_start) * 1000:.1f}ms forward={(t_forward - t_collated) * 1000:.1f}ms "
              f"split={(t_end - t_forward) * 1000:.1f}ms computed={computed}")


@app.post("/evaluate")
def evaluate():
    global req_counter

    data = msgpack.unpackb(request.data, raw=False)
    with req_counter_lock:
        req_counter += 1
        req_id = req_counter

    pending = Pending(req_id, data)

    print(f"[REQ {req_id}] nodes={pending.nodes}, kept={pending.kept}, states={pending.states}")

    Q.put(pending)
    pending.evt.wait()

    if pending.error is not None:
        print(f"[REQ {req_id}] failed: {pending.error}")
        return Response(pending.error, status=500, mimetype="text/plain")

    total_ms = (pending.t_done - pending.t_recv) * 1000.0
    print(f"[REQ {req_id}] done: {total_ms:.1f}ms")

    # binary requests: one map of raw float32 arrays (priority/target per node in the order sent, use as
    # [false, true] per state, value per state); otherwise an array with one result map per state
    return Response(msgpack.packb(pending.out, use_bin_type=True), mimetype="application/x-msgpack")


@app.get("/healthz")
def healthz():
    return "ok", 200


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="model directory (model.pt.gz)")
    parser.add_argument("--port", type=int, default=50052)
    args = parser.parse_args()
    init(args.model, args.port)
