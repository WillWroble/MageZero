import threading
from collections import OrderedDict

import torch

from model import Graphs, NodeType, Plan

MAX_GRAPHS = 64  # captured graphs kept (least recently used dropped); they share one memory pool


# Padded size of each index array per padded node: 1.5x its mean over real states (v0.3 data). Batches above a
# ratio move up a node class, so these only trade padding against how often that happens.
RATIOS = {"leaves": 0.9, "edges": 3.4, "leaf_edges": 2.8, "internal": 0.61,
          "ABILITY": (0.34, 1.07), "CARD": (0.10, 0.79), "PERMANENT": (0.083, 1.0), "STACK_OBJECT": (0.01, 0.03),
          "ZONE": (0.069, 0.28), "PLAYER": (0.012, 0.23), "ROOT": (0.006, 0.053)}


def node_class(n: int) -> int:
    """Padded node count: the smallest half-octave step (256 * 2**(k/2)) above n."""
    k = 0
    while round(256 * 2 ** (k / 2)) <= n:
        k += 1
    return round(256 * 2 ** (k / 2))


def pow2_class(n: int, low: int) -> int:
    """The next power of two above n (so there is always a spare slot), at least `low`."""
    return max(low, 1 << n.bit_length())


def pad(x: torch.Tensor, size: int, fill: int) -> torch.Tensor:
    return torch.cat([x, x.new_full((size - len(x),), fill)])


def padded(model, g: Graphs, plan: Plan):
    """g and its plan padded to sizes set by a few classes (padded nodes, states, global width), every stage present,
    so batches of similar size share one graph. Padding goes to one dummy node (the last), one dummy state (the last)
    and each array's last slot: real nodes never read a padded entry, and outputs for real nodes and states are the
    first N and B rows. Returns (inputs, width, key): the flat input tensors, the padded global width, the cache key."""
    N, B = len(g.node_type), len(g.node_offsets) - 1
    present = {name: rest for name, *rest in plan.stages}
    empty = torch.zeros(0, dtype=torch.long)
    stages = [present.get(name, (empty,) * 4) for name in model.groups]

    def cap(ratio):  # padded size for a ratio; at least 8 and always one spare slot
        return max(8, int(ratio * n_pad) + 1)

    n_pad = node_class(N)
    while (len(plan.leaves) >= cap(RATIOS["leaves"]) or len(plan.child) >= cap(RATIOS["edges"])
           or len(plan.leaf_edges) >= cap(RATIOS["leaf_edges"]) or len(plan.internal) >= cap(RATIOS["internal"])
           or any(len(st[0]) >= cap(RATIOS[name][0]) or len(st[1]) >= cap(RATIOS[name][1])
                  for name, st in zip(model.groups, stages))):
        n_pad = node_class(n_pad)
    b_pad, width = pow2_class(B, 8), pow2_class(plan.width, 64)
    dummy = n_pad - 1
    e_pad = cap(RATIOS["edges"])
    inputs = [pad(g.node_type, n_pad, NodeType.LEAF), pad(g.node_row, n_pad, 0), pad(g.node_value, n_pad, 0),
              pad(g.node_offsets, b_pad + 1, N),
              pad(plan.leaves, cap(RATIOS["leaves"]), dummy), pad(plan.child, e_pad, dummy),
              pad(plan.parent, e_pad, dummy), pad(plan.label, e_pad, 0),
              pad(plan.leaf_edges, cap(RATIOS["leaf_edges"]), e_pad - 1)]
    for name, (st_nodes, child, label, seg) in zip(model.groups, stages):
        s_pad, c_pad = cap(RATIOS[name][0]), cap(RATIOS[name][1])
        inputs += [pad(st_nodes, s_pad, dummy), pad(child, c_pad, dummy), pad(label, c_pad, 0), pad(seg, c_pad, s_pad - 1)]
    i_pad = cap(RATIOS["internal"])
    inputs += [pad(plan.internal, i_pad, dummy), pad(plan.gi, i_pad, b_pad - 1), pad(plan.pos, i_pad, 1),
               pad(plan.counts, b_pad, 0)]
    return inputs, width, (n_pad, b_pad, width)


def unpack(model, inputs, width):
    """The (Graphs, Plan) forward takes, from padded's flat input list."""
    node_type, node_row, node_value, node_offsets, leaves, child, parent, label, leaf_edges = inputs[:9]
    rest = inputs[9:]
    stages = [(name, *rest[4 * i:4 * i + 4]) for i, name in enumerate(model.groups)]
    internal, gi, pos, counts = rest[4 * len(model.groups):]
    g = Graphs(node_type, node_row, node_value, node_offsets, child, parent, label)
    return g, Plan(leaves, child, parent, label, leaf_edges, stages, internal, gi, pos, counts, width)


class GraphedNet:
    """NetGraph inference through CUDA graphs. Each batch is padded to a size class (see padded); the first batch of
    each class is captured as a CUDA graph, later ones copy their inputs in and replay it, so the forward pass's ~750
    kernels launch as one. Returns CPU float outputs for the real nodes and states, as the model would. Thread-safe
    (one replay at a time). For per-type local layers ("sequential" or "parallel"); see supports."""

    @staticmethod
    def supports(model) -> bool:
        return model.arch["local"] in ("sequential", "parallel")

    def __init__(self, model, device):
        self.model, self.device = model, device
        self.graphs = OrderedDict()  # key -> (graph, static inputs, static outputs)
        self.pool = torch.cuda.graph_pool_handle()
        self.lock = threading.Lock()

    def run(self, inputs, width):
        g, plan = unpack(self.model, inputs, width)
        with torch.no_grad(), torch.autocast(self.device.type, cache_enabled=False):
            return self.model(g, plan)

    def capture(self, inputs, width):
        static = [t.to(self.device) for t in inputs]
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):  # warm-up runs, as CUDA graph capture requires
            for _ in range(2):
                self.run(static, width)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=self.pool):
            out = self.run(static, width)
        return graph, static, out

    def __call__(self, g: Graphs, plan: Plan):
        """g and its plan (model.plan, on the CPU) -> priority, target, use, value for the real nodes and states."""
        N, B = len(g.node_type), len(g.node_offsets) - 1
        inputs, width, key = padded(self.model, g, plan)
        with self.lock:
            entry = self.graphs.get(key)
            if entry is None:
                entry = self.graphs[key] = self.capture(inputs, width)
                if len(self.graphs) > MAX_GRAPHS:
                    self.graphs.popitem(last=False)
            self.graphs.move_to_end(key)
            graph, static, out = entry
            for s, t in zip(static, inputs):
                s.copy_(t, non_blocking=True)
            graph.replay()
            priority, target, use, value = (t.float().cpu() for t in out)
        return priority[:N], target[:N], use[:B], value[:B]
