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
MAX_BATCH = 64

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

    threading.Thread(target=worker_loop, daemon=True).start()

    print(f"[INIT] deck={deck} ver={version} port={port} device={DEVICE} "
          f"leaf_rows={len(VOCAB)} edge_rows={len(EDGE_VOCAB)}")
    waitress.serve(app, host="127.0.0.1", port=port, threads=6)


def request_states(data):
    """Split an /evaluate request into per-state arrays in the form collate_graphs takes, with ids
    mapped to embedding rows the same way the training data is. RemoteModelEvaluator sends every
    state's nodes and edges concatenated:
      indices, values                       per node
      offsets                               per state, its first node
      edge_child, edge_parent, edge_label   per edge; child and parent are local to the state
      edge_offsets                          per state, its first edge"""
    ids = np.asarray(data["indices"], dtype=np.int32)
    values = np.asarray(data["values"], dtype=np.int32)
    edge_child = np.asarray(data["edge_child"], dtype=np.int32)
    edge_parent = np.asarray(data["edge_parent"], dtype=np.int32)
    node_type = node_types(ids)
    node_row, edge_row = map_ids(ids, node_type, np.asarray(data["edge_label"], dtype=np.int32), VOCAB, EDGE_VOCAB)

    node_start = np.append(np.asarray(data["offsets"], dtype=np.int64), len(ids))
    edge_start = np.append(np.asarray(data["edge_offsets"], dtype=np.int64), len(edge_child))
    states = []
    for i in range(len(node_start) - 1):
        n = slice(node_start[i], node_start[i + 1])
        e = slice(edge_start[i], edge_start[i + 1])
        states.append((node_type[n], node_row[n], values[n], edge_child[e], edge_parent[e], edge_row[e]))
    return states


class Pending:
    __slots__ = ("req_id", "states", "nodes", "kept", "evt", "out", "error", "t_recv", "t_done")

    def __init__(self, req_id, data):
        self.req_id = req_id
        self.t_recv = time.perf_counter()
        self.evt = threading.Event()
        self.out = None
        self.error = None
        self.t_done = 0.0

        self.states = request_states(data)
        self.nodes = sum(len(s[0]) for s in self.states)
        # nodes the network uses: typed nodes, and leaves with a vocab row
        self.kept = sum(int(((s[0] != NodeType.LEAF) | (s[1] >= 0)).sum()) for s in self.states)


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

        states = [s for p in batch for s in p.states]
        try:
            # Single forward pass over every state of every request
            graphs = collate_graphs(*zip(*states))
            with torch.no_grad(), torch.amp.autocast(DEVICE.type, enabled=AMP):
                out = server_model(graphs.to(DEVICE))
            priority, target, use, value = (t.float().cpu().numpy() for t in out)

            # Split results back to individual requests: per-node scores in the order each state was sent
            starts = np.asarray(graphs.node_offsets)
            row = 0
            for p in batch:
                p.out = []
                for _ in p.states:
                    nodes = slice(starts[row], starts[row + 1])
                    p.out.append({
                        "policy_priority": priority[nodes].tolist(),
                        "policy_target": target[nodes].tolist(),
                        "policy_binary": use[row].tolist(),
                        "value": float(value[row]),
                    })
                    row += 1
        except Exception as e:
            # fail these requests instead of the worker, so later requests still get served
            traceback.print_exc()
            for p in batch:
                p.error = f"{type(e).__name__}: {e}"

        for p in batch:
            p.t_done = time.perf_counter()
            p.evt.set()

        print(f"[BATCH] size={len(batch)}, total_states={len(states)}")


@app.post("/evaluate")
def evaluate():
    global req_counter

    data = msgpack.unpackb(request.data, raw=False)
    with req_counter_lock:
        req_counter += 1
        req_id = req_counter

    pending = Pending(req_id, data)

    print(f"[REQ {req_id}] nodes={pending.nodes}, kept={pending.kept}, states={len(pending.states)}")

    Q.put(pending)
    pending.evt.wait()

    if pending.error is not None:
        print(f"[REQ {req_id}] failed: {pending.error}")
        return Response(pending.error, status=500, mimetype="text/plain")

    total_ms = (pending.t_done - pending.t_recv) * 1000.0
    print(f"[REQ {req_id}] done: {total_ms:.1f}ms")

    # always an array with one result map per state, as RemoteModelEvaluator reads it
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
