import threading
import time
import traceback
from queue import Queue, Empty

import msgpack
import numpy as np
import torch
import waitress
from flask import Flask, request, Response

from dataset import collate_graphs, map_ids, node_types
from model import load_checkpoint, AMP, DEVICE, NodeType

# Threading config
TORCH_THREADS = 1 #max(1, os.cpu_count() // 2)
torch.set_num_threads(TORCH_THREADS)

# Batching config
MAX_BATCH = 128
CPU_PLAN_MAX = 32  # states per batch up to which NetGraph.plan is built on the CPU (T4: faster below, slower above)
WORKERS = 1  # batching threads; with 2, one batch's CPU work overlaps the other's forward pass

#module state
server_model = None
VOCAB = None
EDGE_VOCAB = None

app = Flask(__name__)

req_counter = 0
req_counter_lock = threading.Lock()

def init(deck: str, version: int, port: int):
    global server_model, VOCAB, EDGE_VOCAB

    model_path = f"models/{deck}/ver{version}/model.pt.gz"
    model, VOCAB, EDGE_VOCAB = load_checkpoint(model_path)
    server_model = model.to(DEVICE).eval()

    for _ in range(WORKERS):
        threading.Thread(target=worker_loop, daemon=True).start()

    print(f"[INIT] deck={deck} ver={version} port={port} device={DEVICE} "
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
    state's first node plus the node count."""
    ids, values = as_array(data["indices"]), as_array(data["values"])
    edge_child, edge_parent = as_array(data["edge_child"]), as_array(data["edge_parent"])
    node_type = node_types(ids)
    node_row, edge_row = map_ids(ids, node_type, as_array(data["edge_label"]), VOCAB, EDGE_VOCAB)
    starts = np.append(as_array(data["offsets"], np.int64), len(ids))
    edge_starts = np.append(as_array(data["edge_offsets"], np.int64), len(edge_child))
    state_start = np.repeat(starts[:-1], np.diff(edge_starts))
    return node_type, node_row, values, edge_child + state_start, edge_parent + state_start, edge_row, starts


def collate_requests(batch):
    """One batch of graphs from whole requests: collate_graphs over the requests, with node offsets per state."""
    graphs = collate_graphs(*zip(*(p.arrays[:6] for p in batch)))
    bases = np.asarray(graphs.node_offsets)
    offsets = np.concatenate([p.arrays[6][:-1] + base for p, base in zip(batch, bases)] + [bases[-1:]])
    return graphs._replace(node_offsets=torch.from_numpy(offsets.astype(np.int64)))


class Pending:
    __slots__ = ("req_id", "arrays", "binary", "states", "nodes", "kept", "evt", "out", "error", "t_recv", "t_done")

    def __init__(self, req_id, data):
        self.req_id = req_id
        self.t_recv = time.perf_counter()
        self.evt = threading.Event()
        self.out = None
        self.error = None
        self.t_done = 0.0

        self.arrays = request_arrays(data)
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
        try:
            # Single forward pass over every state of every request
            graphs = collate_requests(batch)
            t_collated = time.perf_counter()
            with torch.no_grad(), torch.amp.autocast(DEVICE.type, enabled=AMP):
                # small batches: the index work runs on the CPU, so the GPU runs the forward pass without stalls;
                # large ones: the CPU work costs more than the stalls (bench.py compares both)
                plan = server_model.plan(graphs).to(DEVICE) if states <= CPU_PLAN_MAX else None
                out = server_model(graphs.to(DEVICE), plan)
            priority, target, use, value = (t.float().cpu().numpy() for t in out)
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
            for p in batch:
                p.error = f"{type(e).__name__}: {e}"

        for p in batch:
            p.t_done = time.perf_counter()
            p.evt.set()

        t_end = time.perf_counter()
        print(f"[BATCH] size={len(batch)}, total_states={states}, "
              f"queued={(t_start - min(p.t_recv for p in batch)) * 1000:.1f}ms "
              f"collate={(t_collated - t_start) * 1000:.1f}ms forward={(t_forward - t_collated) * 1000:.1f}ms "
              f"split={(t_end - t_forward) * 1000:.1f}ms")


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
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--port", type=int, default=50052)
    args = parser.parse_args()
    init(args.deck, args.version, args.port)
