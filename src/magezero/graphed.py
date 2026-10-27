import threading
from collections import OrderedDict

import torch

from model import Graphs, NodeType, Plan

MAX_GRAPHS = 64  # captured graphs kept (least recently used dropped); they share one memory pool

# Padded size of each index array per padded node: 1.5x its mean over real states (v0.3 data); for a stage, of its
# nodes read from the cache. Batches above a ratio move up a node class, so these only trade padding against how
# often that happens.
RATIOS = {"leaves": 0.9, "edges": 3.4, "leaf_edges": 2.8, "internal": 0.61,
          "ABILITY": 0.34, "CARD": 0.10, "PERMANENT": 0.083, "STACK_OBJECT": 0.01, "ZONE": 0.069, "PLAYER": 0.012,
          "ROOT": 0.006}
CHILDREN_PER_NODE = 16  # padded child edges per padded computed node, in every stage


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


def padded(model, g: Graphs, plan: Plan, scratch: int):
    """g and its cached plan (model.plan with a NodeCache lookup) padded to sizes set by a few classes, every stage
    present, so batches of similar size share one graph:
      nodes: a node class N', with every array that scales with the batch (leaves, edges, global tokens, each
             stage's nodes read from the cache) capped at a ratio of it
      computed: one class C for every stage's computed nodes (the most any stage has), CHILDREN_PER_NODE * C edges
      states and global width: powers of two
    Padding goes to one dummy node (the last), one dummy state (the last), each stage's last computed position and
    the cache's scratch slot: real nodes never read a padded entry, and outputs for real nodes and states are the
    first N and B rows. Returns (inputs, width, key): the flat input tensors, the padded global width, the cache key."""
    N, B = len(g.node_type), len(g.node_offsets) - 1
    present = {name: rest for name, *rest in plan.stages}
    empty = torch.zeros(0, dtype=torch.long)
    stages = [present.get(name, (empty,) * 7) for name in model.groups]

    c_pad = pow2_class(max(len(st[0]) for st in stages), 8)
    while any(len(st[1]) > CHILDREN_PER_NODE * c_pad for st in stages):
        c_pad *= 2
    e_pad_c = CHILDREN_PER_NODE * c_pad

    def cap(ratio):  # padded size for a ratio; at least 8 and always one spare slot
        return max(8, int(ratio * n_pad) + 1)

    n_pad = node_class(N)
    while (len(plan.leaves) >= cap(RATIOS["leaves"]) or len(plan.child) >= cap(RATIOS["edges"])
           or len(plan.leaf_edges) >= cap(RATIOS["leaf_edges"]) or len(plan.internal) >= cap(RATIOS["internal"])
           or any(len(st[5]) >= cap(RATIOS[name]) for name, st in zip(model.groups, stages))):
        n_pad = node_class(n_pad)
    b_pad, width = pow2_class(B, 8), pow2_class(plan.width, 64)
    dummy = n_pad - 1
    e_pad = cap(RATIOS["edges"])
    inputs = [pad(g.node_type, n_pad, NodeType.LEAF), pad(g.node_row, n_pad, 0), pad(g.node_value, n_pad, 0),
              pad(g.node_offsets, b_pad + 1, N),
              pad(plan.leaves, cap(RATIOS["leaves"]), dummy), pad(plan.child, e_pad, dummy),
              pad(plan.parent, e_pad, dummy), pad(plan.label, e_pad, 0),
              pad(plan.leaf_edges, cap(RATIOS["leaf_edges"]), e_pad - 1)]
    for name, (nodes, child, label, seg, slots, fill, fill_slots) in zip(model.groups, stages):
        f_pad = cap(RATIOS[name])
        inputs += [pad(nodes, c_pad, dummy), pad(child, e_pad_c, dummy), pad(label, e_pad_c, 0),
                   pad(seg, e_pad_c, c_pad - 1), pad(slots, c_pad, scratch),
                   pad(fill, f_pad, dummy), pad(fill_slots, f_pad, scratch)]
    i_pad = cap(RATIOS["internal"])
    inputs += [pad(plan.internal, i_pad, dummy), pad(plan.gi, i_pad, b_pad - 1), pad(plan.pos, i_pad, 1),
               pad(plan.counts, b_pad, 0)]
    return inputs, width, (n_pad, b_pad, width, c_pad)


def unpack(model, inputs, width):
    """The (Graphs, Plan) forward takes, from padded's flat input list."""
    node_type, node_row, node_value, node_offsets, leaves, child, parent, label, leaf_edges = inputs[:9]
    rest = inputs[9:]
    stages = [(name, *rest[7 * i:7 * i + 7]) for i, name in enumerate(model.groups)]
    internal, gi, pos, counts = rest[7 * len(model.groups):]
    g = Graphs(node_type, node_row, node_value, node_offsets, child, parent, label)
    return g, Plan(leaves, child, parent, label, leaf_edges, stages, internal, gi, pos, counts, width)


class GraphedNet:
    """The cached forward pass through CUDA graphs. Each batch is padded to a size class (see padded); the first batch
    of each class is captured as a CUDA graph, later ones copy their inputs in and replay it, so the forward pass's
    kernels launch as one. Returns CPU float outputs for the real nodes and states, as the model would. For per-type
    local layers ("sequential" or "parallel"); see supports."""

    @staticmethod
    def supports(model) -> bool:
        return model.arch["local"] in ("sequential", "parallel")

    def __init__(self, model, cache, device):
        self.model, self.cache, self.device = model, cache, device
        self.graphs = OrderedDict()  # key -> (graph, static inputs, static outputs)
        self.pool = torch.cuda.graph_pool_handle()
        self.lock = threading.Lock()
        self.captures = 0

    def run(self, inputs, width):
        g, plan = unpack(self.model, inputs, width)
        with torch.no_grad(), torch.autocast(self.device.type, cache_enabled=False):
            return self.model(g, plan, self.cache.table)

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
        self.captures += 1
        return graph, static, out

    def __call__(self, g: Graphs, plan: Plan):
        """g and its cached plan (on the CPU) -> priority, target, use, value for the real nodes and states."""
        N, B = len(g.node_type), len(g.node_offsets) - 1
        inputs, width, key = padded(self.model, g, plan, self.cache.scratch)
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
