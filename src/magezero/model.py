import gzip
import math
from enum import Enum, IntEnum
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from vocab import FeatureVocab

"""
MageZero graph network for AlphaZero style MCTS. Input: StateEncoder's state graph, nodes typed
(ROOT, PLAYER, ZONE, STACK_OBJECT, PERMANENT, CARD, ABILITY) or LEAF, edges child -> parent.
  embeddings: typed node = type embedding; leaf = leaf vocab row + numeric value bucket
  local layers: 2 bottom-up passes (separate weights), each one LocalLayer per type, run in order
      ABILITY -> CARD -> PERMANENT -> STACK_OBJECT -> ZONE -> PLAYER -> ROOT
      every node attends over [itself, children + edge label embedding] (d=256, nhead=4, ff=512)
  global layers: 2-layer TransformerEncoder over [CLS, the state's internal nodes]
      ├── MLP(512 -> 256 -> 1)  per node  priority       (read at ABILITY nodes)
      ├── MLP(512 -> 256 -> 1)  per node  target         (read at CARD/PERMANENT/STACK_OBJECT/PLAYER nodes)
      ├── MLP(512 -> 256 -> 2)  CLS       choose_use     (false, true)
      └── MLP(512 -> 256 -> 1)  CLS       value          (tanh, -1..1)
"""


GLOBAL_MAX = 2**31-1

D_MODEL = 256
N_HEADS = 4
D_FF = 512
DROPOUT = 0.0
N_PASSES = 2
GLOBAL_LAYERS = 2
HIDDEN_MLP = 256
# checkpoints saved before the arch was stored in them
LEGACY_ARCH = dict(d_model=512, d_ff=1024, dropout=0.25)
# options added later, as checkpoints saved before them were trained
LEGACY_OPTIONS = dict(policy_domain="all")

# numeric value buckets: exact -5..20 (P/T, counters, mana, costs, small counts), then 21-25, 26-30,
# 31-40, 41-60, 61+ (life totals, library and zone sizes); everything below -5 is one bucket
VALUE_BOUNDS = list(range(-5, 22)) + [26, 31, 41, 61]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
AMP = DEVICE.type == "cuda"  # autocast + GradScaler


class ActionType(Enum):
    PRIORITY = 0
    CHOOSE_TARGET = 3
    CHOOSE_USE = 5


class NodeType(IntEnum):
    """Java FeatureGraph.Node.Type, plus the root."""
    ROOT = 0
    PLAYER = 1
    ZONE = 2
    STACK_OBJECT = 3
    PERMANENT = 4
    CARD = 5
    ABILITY = 6
    LEAF = 7


STAGES = (NodeType.ABILITY, NodeType.CARD, NodeType.PERMANENT, NodeType.STACK_OBJECT,
          NodeType.ZONE, NodeType.PLAYER, NodeType.ROOT)
PRIORITY_TYPES = (NodeType.ABILITY,)
TARGET_TYPES = (NodeType.CARD, NodeType.PERMANENT, NodeType.STACK_OBJECT, NodeType.PLAYER)

HEADS = ("priority", "choose_target", "choose_use", "value")
POLICY_HEADS = HEADS[:3]


class Graphs(NamedTuple):
    """A batch of state graphs, concatenated: the network's input."""
    node_type: torch.Tensor     # [N] NodeType
    node_row: torch.Tensor      # [N] leaf vocab row; -1 for typed nodes and leaves outside the vocab
    node_value: torch.Tensor    # [N] numeric value (0 for non-numeric)
    node_offsets: torch.Tensor  # [B+1] first node of each state
    edge_child: torch.Tensor    # [E] node index (batch-wide)
    edge_parent: torch.Tensor   # [E] node index (batch-wide)
    edge_label: torch.Tensor    # [E] edge vocab row + 1; 0 for labels outside the vocab

    def to(self, device):
        return Graphs(*(t.to(device, non_blocking=True) for t in self))


class Targets(NamedTuple):
    """Training labels for a batch of states."""
    value: torch.Tensor         # [B] result label
    action_type: torch.Tensor   # [B] ActionType value
    use_visits: torch.Tensor    # [B, 2] CHOOSE_USE visits (false, true)
    cand_node: torch.Tensor     # [C] policy candidate node index (batch-wide)
    cand_visits: torch.Tensor   # [C] MCTS visits
    cand_graph: torch.Tensor    # [C] state of each candidate

    def to(self, device):
        return Targets(*(t.to(device, non_blocking=True) for t in self))


class Plan(NamedTuple):
    """NetGraph.plan's output: index tensors for one batch of graphs, and the global sequence width."""
    leaves: torch.Tensor      # nodes with a leaf embedding
    child: torch.Tensor       # kept edges (both ends have embeddings)
    parent: torch.Tensor
    label: torch.Tensor
    leaf_edges: torch.Tensor  # kept edges whose child is a leaf
    stages: list              # (layer name, nodes, children, edge labels, each child's parent position), plus
                              # with a node cache (cache slots of the nodes, nodes read from the cache, their slots)
    internal: torch.Tensor    # global tokens
    gi: torch.Tensor          # state of each global token
    pos: torch.Tensor         # position of each global token in its state's sequence
    counts: torch.Tensor      # global tokens per state
    width: int                # padded sequence length (1 + the largest count)

    def to(self, device):
        def move(x):
            return x.to(device, non_blocking=True) if isinstance(x, torch.Tensor) else x
        return Plan(*(move(x) for x in self[:5]), [tuple(move(x) for x in st) for st in self.stages],
                    *(move(x) for x in self[6:]))


def graph_index(offsets: torch.Tensor) -> torch.Tensor:
    """State of each node, from a batch's node offsets."""
    return torch.repeat_interleave(torch.arange(offsets.numel() - 1, device=offsets.device), offsets.diff())


def segment_logsumexp(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """Log-sum-exp over the rows of x that share an index in seg (values 0..n-1); returns n rows."""
    index = seg.view(-1, *([1] * (x.dim() - 1))).expand_as(x)
    with torch.no_grad():  # the max only stabilizes; the result and its gradient don't depend on it
        m = x.new_full((n, *x.shape[1:]), float("-inf")).scatter_reduce(0, index, x, "amax")
    return m + x.new_zeros((n, *x.shape[1:])).index_add(0, seg, (x - m[seg]).exp()).log()


class LocalLayer(nn.TransformerEncoderLayer):
    """A pre-norm transformer layer run over each node's [node, children...] sequence, keeping only
    the node's output token. The children's output tokens would be discarded, so only the node's
    query is computed: attention from the node to itself and its children, then the FFN on the node.
    Same parameters, init and result as nn.TransformerEncoderLayer, without padding to the largest
    child count and without running the FFN on children."""

    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__(d_model, n_heads, d_ff, dropout, activation="gelu", batch_first=True, norm_first=True)

    def forward(self, node: torch.Tensor, child: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """node: [S, d] this stage's nodes; child: [E, d] child tokens (child embedding + edge label
        embedding); seg: [E] position in `node` of each child's parent. Returns the nodes' new embeddings."""
        S, d = node.shape
        H, dh = self.self_attn.num_heads, self.self_attn.head_dim
        w, b = self.self_attn.in_proj_weight, self.self_attn.in_proj_bias
        q, k, v = F.linear(self.norm1(node), w, b).chunk(3, dim=-1)
        k_child, v_child = F.linear(self.norm1(child), w[d:], b[d:]).chunk(2, dim=-1)
        seg = torch.cat([torch.arange(S, device=seg.device), seg])  # each node also attends to itself
        k = torch.cat([k, k_child]).view(-1, H, dh)
        v = torch.cat([v, v_child]).view(-1, H, dh)
        score = (q.reshape(S, H, dh)[seg] * k).sum(-1).float() / math.sqrt(dh)  # [S+E, H]
        attn = (score - segment_logsumexp(score, seg, S)[seg]).exp()
        attn = F.dropout(attn, self.self_attn.dropout, self.training)
        out = torch.zeros(S, H, dh, device=node.device).index_add_(
            0, seg, (attn.to(v.dtype).unsqueeze(-1) * v).float())
        node = node + self.dropout1(self.self_attn.out_proj(out.view(S, d)))
        return node + self._ff_block(self.norm2(node))


def mlp(d_model: int, out_dim: int, *tail: nn.Module) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_model, HIDDEN_MLP), nn.ReLU(), nn.Linear(HIDDEN_MLP, out_dim), *tail)


class NetGraph(nn.Module):
    """Architecture options (saved in checkpoints as `arch`):
      local: "sequential"  one layer per type, run bottom-up in STAGES order (children earlier in the
                           order arrive updated in the same pass)
             "parallel"    one layer per type, every stage reads the embeddings from the pass's start
             "joint"       one layer for all internal nodes per pass (a single call)
             "flat"        no local layers: each internal node adds the mean of its own leaves (with their
                           edge labels) once; no hierarchy and no object-to-object edges
      global_tokens: "internal"  global layers see [CLS, internal nodes]
                     "all"       they also see the leaves; with n_passes=0 the state is a bag of tokens
      policy_domain: "legal"  policy softmax over the legal candidates only, as MCTS uses the priors
                     "all"    over every node of the head's types (learns legality; checkpoints from before
                              this option)
      legality_weight: > 0 adds an auxiliary head that classifies each node of a policy head's types as a legal
                       candidate or not (BCE, this weight in the total loss), returned as a fifth output
      legality_heads: "shared"  one legality head for every node type
                      "split"   one read at ABILITY nodes and one at the target types, like the policy heads"""

    def __init__(self, num_embeddings: int = 0, num_edge_labels: int = 0, d_model: int = D_MODEL,
                 n_heads: int = N_HEADS, d_ff: int = D_FF, dropout: float = DROPOUT, n_passes: int = N_PASSES,
                 global_layers: int = GLOBAL_LAYERS, local: str = "sequential", global_tokens: str = "internal",
                 policy_domain: str = "legal", legality_weight: float = 0.0, legality_heads: str = "shared"):
        super().__init__()
        self.arch = dict(d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout, n_passes=n_passes,
                         global_layers=global_layers, local=local, global_tokens=global_tokens,
                         policy_domain=policy_domain, legality_weight=legality_weight, legality_heads=legality_heads)
        self.embedding = nn.Embedding(num_embeddings, d_model)                         # leaves
        self.edge_embedding = nn.Embedding(num_edge_labels + 1, d_model, padding_idx=0)  # row 0: unknown label
        self.type_embedding = nn.Embedding(len(NodeType), d_model)
        self.value_embedding = nn.Embedding(len(VALUE_BOUNDS) + 1, d_model)
        self.register_buffer("value_bounds", torch.tensor(VALUE_BOUNDS), persistent=False)

        self.groups = {"INTERNAL": STAGES} if local == "joint" else {t.name: (t,) for t in STAGES}
        self.local_layers = nn.ModuleList(
            nn.ModuleDict({name: LocalLayer(d_model, n_heads, d_ff, dropout) for name in self.groups})
            for _ in range(0 if local == "flat" else n_passes))

        self.cls = nn.Parameter(torch.randn(d_model))
        global_layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                                  batch_first=True, norm_first=True)
        self.global_layers = nn.TransformerEncoder(global_layer, global_layers, norm=nn.LayerNorm(d_model),
                                                   enable_nested_tensor=False)

        self.priority_head = mlp(d_model, 1)
        self.target_head = mlp(d_model, 1)
        self.binary_head = mlp(d_model, 2)
        self.value_head = mlp(d_model, 1, nn.Tanh())
        if legality_weight > 0:
            self.legality_head = mlp(d_model, 1)
            if legality_heads == "split":
                self.target_legality_head = mlp(d_model, 1)

    def resize_embedding(self, num_embeddings: int, init_rows=None, name: str = "embedding") -> None:
        """Grow an embedding table to `num_embeddings` rows, keeping existing rows unchanged.
        Used when a new generation adds ids to a vocab. `init_rows` supplies the added rows
        (vocab.initial_rows draws each from its id, so an id starts from the same row whatever
        generation first sees it); without it they get nn.Embedding's default init."""
        old = getattr(self, name)
        if num_embeddings == old.num_embeddings:
            return
        if num_embeddings < old.num_embeddings:
            raise ValueError("vocabs are append-only; an embedding table cannot shrink")
        new = nn.Embedding(num_embeddings, old.embedding_dim, padding_idx=old.padding_idx).to(old.weight.device)
        with torch.no_grad():
            new.weight[:old.num_embeddings] = old.weight
            if init_rows is not None:
                added = num_embeddings - old.num_embeddings
                new.weight[old.num_embeddings:] = torch.as_tensor(
                    init_rows, dtype=new.weight.dtype, device=new.weight.device)[:added]
        setattr(self, name, new)

    def plan(self, g: Graphs, cached=None) -> "Plan":
        """The forward pass's index work, which depends only on the graphs' structure. Built where `g` lives;
        the server builds it on the CPU, where its nonzero() and max() calls don't stall the GPU.
        cached: (slot, compute) per node from NodeCache.lookup. The local layers then run only on the nodes
        to compute, store them at their slots, and read every other internal node from its slot."""
        B, N = g.node_offsets.numel() - 1, g.node_type.numel()
        leaves = (g.node_row >= 0).nonzero().squeeze(1)

        # leaves outside the vocab have no embedding, so their edges are dropped
        keep = (g.node_type[g.edge_child] != NodeType.LEAF) | (g.node_row[g.edge_child] >= 0)
        child, parent, label = g.edge_child[keep], g.edge_parent[keep], g.edge_label[keep]
        leaf_edges = (g.node_type[child] == NodeType.LEAF).nonzero().squeeze(1)

        # one stage per group of types: its nodes, their children, and each child's parent position
        if cached is not None:
            slot, compute = (t.to(g.node_type.device) for t in cached)
        stages = []
        for name, types in self.groups.items():
            run = torch.isin(g.node_type, torch.tensor(types, device=g.node_type.device))
            if cached is not None:
                fill = (run & ~compute).nonzero().squeeze(1)
                run &= compute
            nodes = run.nonzero().squeeze(1)
            if len(nodes) or cached is not None and len(fill):
                edges = run[parent].nonzero().squeeze(1)
                stage = (name, nodes, child[edges], label[edges], torch.searchsorted(nodes, parent[edges]))
                stages.append(stage if cached is None else stage + (slot[nodes], fill, slot[fill]))

        # global tokens: [CLS, tokens] per state
        internal = g.node_type != NodeType.LEAF
        if self.arch["global_tokens"] == "all":
            internal |= g.node_row >= 0
        internal = internal.nonzero().squeeze(1)
        gi = graph_index(g.node_offsets)[internal]
        counts = torch.bincount(gi, minlength=B)
        pos = torch.arange(len(internal), device=gi.device) - (counts.cumsum(0) - counts)[gi] + 1
        return Plan(leaves, child, parent, label, leaf_edges, stages, internal, gi, pos, counts, int(counts.max()) + 1)

    def forward(self, g: Graphs, plan: "Plan" = None, cache: torch.Tensor = None):
        """Returns per-node priority and target scores [N] (-inf at leaves), choose_use logits
        [B, 2] (false, true) and value [B]. `plan`: self.plan(g), built ahead (see plan); computed here if omitted.
        `cache`: NodeCache.table [passes, slots, d], with a plan built from its lookup."""
        if plan is None:
            plan = self.plan(g)
        B, N = g.node_offsets.numel() - 1, g.node_type.numel()
        child, parent, label = plan.child, plan.parent, plan.label

        # initial embeddings: typed nodes their type, leaves their name and numeric value
        value = torch.bucketize(g.node_value[plan.leaves], self.value_bounds, right=True)
        h = self.type_embedding(g.node_type).index_add(
            0, plan.leaves, self.embedding(g.node_row[plan.leaves]) + self.value_embedding(value))

        if self.arch["local"] == "flat":
            p = parent[plan.leaf_edges]
            count = torch.bincount(p, minlength=N).clamp(min=1).unsqueeze(1)
            h = h + h.new_zeros(h.shape).index_add(
                0, p, h[child[plan.leaf_edges]] + self.edge_embedding(label[plan.leaf_edges])) / count

        # local layers, bottom-up passes. sequential: children earlier in the order arrive updated in this
        # pass, children of the same type or later in the order (attachments, targets, linked exile) from the last one
        # cached nodes are read from the cache at their own stage, so later stages see them as if computed.
        # Inference writes rows in place (sequential only: parallel stages read the pass's starting embeddings);
        # an out-of-place copy rewrites every node, leaves included, at each stage
        inplace = self.arch["local"] == "sequential" and not torch.is_grad_enabled()
        for p, layers in enumerate(self.local_layers):
            src = h
            for name, nodes, c, lbl, seg, *fill in plan.stages:
                if self.arch["local"] == "sequential":
                    src = h
                if len(nodes):
                    out = layers[name](src[nodes], src[c] + self.edge_embedding(lbl), seg)
                    h = h.index_copy_(0, nodes, out) if inplace else h.index_copy(0, nodes, out)
                    if fill:
                        cache[p].index_copy_(0, fill[0], out)
                if fill and len(fill[1]):
                    h = h.index_copy_(0, fill[1], cache[p][fill[2]]) if inplace else h.index_copy(0, fill[1], cache[p][fill[2]])

        # global layers: self-attention over [CLS, tokens], one padded sequence per state
        internal, gi, pos = plan.internal, plan.gi, plan.pos
        x = h.new_zeros(B, plan.width, h.shape[1])
        x[:, 0] = self.cls
        x[gi, pos] = h[internal]
        pad = torch.arange(plan.width, device=h.device) > plan.counts.unsqueeze(1)
        x = self.global_layers(x, src_key_padding_mask=pad)
        cls, z = x[:, 0], x[gi, pos]

        def per_node(head):
            return torch.full((N,), float("-inf"), device=h.device).index_copy(
                0, internal, head(z).squeeze(-1).float())

        out = per_node(self.priority_head), per_node(self.target_head), self.binary_head(cls), self.value_head(cls).squeeze(-1)
        if self.arch["legality_weight"] <= 0:
            return out
        legality = per_node(self.legality_head)
        if self.arch["legality_heads"] == "split":  # target types read their own head
            legality = torch.where(g.node_type == NodeType.ABILITY, legality, per_node(self.target_legality_head))
        return out + (legality,)


class NetMLP(nn.Module):
    """Small MLP baseline after Dama's d64 MLPNet (gorge `bots/gonet`), on the same graphs and outputs as NetGraph.
    No attention and no learned message passing: the graph only supplies each object's leaves and its ancestors.
      object  c = relu(phi(relu(raw(type + mean of its leaves (+ edge label)))))     every internal node
      entity  e = LN(c + ctx(sum of c over its ancestors))                           the bag-of-nodes bridge:
                                                                                     zone, owner, source card
      state   s = root(e_ROOT) + pools(0.25-sum and max of e per node type), then residual blocks, LN
      heads   value, choose_use from s; priority and target per node from [s, e]
    Policy heads are meant to be trained over legal candidates only (policy_domain "legal"), as the MLP was."""

    def __init__(self, num_embeddings: int = 0, num_edge_labels: int = 0, d_model: int = 64, blocks: int = 1,
                 policy_domain: str = "legal", net: str = "mlp"):
        super().__init__()
        self.arch = dict(net=net, d_model=d_model, blocks=blocks, policy_domain=policy_domain)
        d = d_model
        self.embedding = nn.Embedding(num_embeddings, d)
        self.edge_embedding = nn.Embedding(num_edge_labels + 1, d, padding_idx=0)
        self.type_embedding = nn.Embedding(len(NodeType), d)
        self.value_embedding = nn.Embedding(len(VALUE_BOUNDS) + 1, d)
        self.register_buffer("value_bounds", torch.tensor(VALUE_BOUNDS), persistent=False)
        self.raw, self.phi, self.ctx = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.norm = nn.LayerNorm(d)
        self.root, self.pools = nn.Linear(d, d), nn.Linear(2 * len(STAGES) * d, d)
        self.blocks = nn.ModuleList(nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.ReLU(), nn.Linear(d, d))
                                    for _ in range(blocks))
        self.out_norm = nn.LayerNorm(d)
        head = lambda n_in, n_out, *tail: nn.Sequential(nn.Linear(n_in, d), nn.ReLU(), nn.Linear(d, n_out), *tail)
        self.priority_head, self.target_head = head(2 * d, 1), head(2 * d, 1)
        self.binary_head, self.value_head = head(d, 2), head(d, 1, nn.Tanh())

    resize_embedding = NetGraph.resize_embedding

    def forward(self, g: Graphs, plan=None):
        B, N = g.node_offsets.numel() - 1, g.node_type.numel()
        leaf = g.node_type == NodeType.LEAF
        keep = ~leaf[g.edge_child] | (g.node_row[g.edge_child] >= 0)
        child, parent, label = g.edge_child[keep], g.edge_parent[keep], g.edge_label[keep]
        node_graph = graph_index(g.node_offsets)

        # objects: type + mean of their leaves (name and value, plus the edge label)
        leaves = (g.node_row >= 0).nonzero().squeeze(1)
        value = torch.bucketize(g.node_value[leaves], self.value_bounds, right=True)
        x = torch.zeros(N, self.arch["d_model"], device=g.node_type.device).index_add(
            0, leaves, self.embedding(g.node_row[leaves]) + self.value_embedding(value))
        lf = leaf[child]
        count = torch.bincount(parent[lf], minlength=N).clamp(min=1).unsqueeze(1)
        bag = x.new_zeros(x.shape).index_add(0, parent[lf], x[child[lf]] + self.edge_embedding(label[lf])) / count
        c = F.relu(self.phi(F.relu(self.raw(self.type_embedding(g.node_type) + bag))))

        # bridge: each object adds the sum of its ancestors' c, top-down (ROOT first), so a card knows its zone and
        # owner, and an ability its card
        up = ~lf
        anc = torch.zeros_like(c)
        for t in reversed(STAGES[:-1]):
            e = up & (g.node_type[child] == t)
            anc = anc.index_add(0, child[e], (c + anc)[parent[e]])
        e = self.norm(c + self.ctx(anc))

        # state: root object + per-type pools
        pools = []
        for t in STAGES:
            m = g.node_type == t
            pools.append(e.new_zeros(B, e.shape[1]).index_add(0, node_graph[m], 0.25 * e[m]))
            pools.append(e.new_zeros(B, e.shape[1]).index_reduce(0, node_graph[m], e[m], "amax", include_self=False))
        root = g.node_type == NodeType.ROOT
        s = self.root(e.new_zeros(B, e.shape[1]).index_add(0, node_graph[root], e[root])) + self.pools(torch.cat(pools, 1))
        for block in self.blocks:
            s = s + block(s)
        s = self.out_norm(s)

        internal = (~leaf).nonzero().squeeze(1)
        z = torch.cat([s[node_graph[internal]], e[internal]], 1)

        def per_node(head):
            return torch.full((N,), float("-inf"), device=s.device).index_copy(0, internal, head(z).squeeze(-1).float())

        return per_node(self.priority_head), per_node(self.target_head), self.binary_head(s), self.value_head(s).squeeze(-1)


def build_model(num_embeddings: int = 0, num_edge_labels: int = 0, **arch) -> nn.Module:
    """NetMLP for arch {"net": "mlp", ...}, NetGraph otherwise."""
    return (NetMLP if arch.get("net") == "mlp" else NetGraph)(num_embeddings, num_edge_labels, **arch)


def node_policy_loss(scores, domain, states, node_graph, t: Targets):
    """KL from the normalized MCTS visits to a softmax over each state's domain nodes, for `states`
    ([B] bool). correct: states whose top-scored domain node (legal or not) is one of MCTS's most
    visited candidates."""
    B = states.shape[0]
    idx = (domain & states[node_graph]).nonzero().squeeze(1)
    domain_scores, domain_graph = scores[idx].float(), node_graph[idx]
    lse = segment_logsumexp(domain_scores, domain_graph, B)
    in_states = states[t.cand_graph]
    cand, visits, graph = t.cand_node[in_states], t.cand_visits[in_states], t.cand_graph[in_states]
    s = scores[cand].float()
    p = visits / torch.zeros(B, device=s.device).index_add(0, graph, visits)[graph]
    n = states.sum()
    loss = (torch.xlogy(p, p) - p * (s - lse[graph])).sum() / n.clamp(min=1)
    with torch.no_grad():
        node_visits = torch.zeros(len(scores), device=s.device).index_add(0, cand, visits)[idx]
        best_s = torch.full((B,), float("-inf"), device=s.device).scatter_reduce(0, domain_graph, domain_scores, "amax")
        best_v = torch.zeros(B, device=s.device).scatter_reduce(0, graph, visits, "amax")
        top = domain_scores == best_s[domain_graph]
        hit = (top & (node_visits > 0) & (node_visits == best_v[domain_graph])).float()
        correct = (torch.zeros(B, device=s.device).index_add(0, domain_graph, hit) > 0).sum()
    return loss, n, correct


def use_loss(logits, t: Targets):
    """KL from the normalized CHOOSE_USE visits to the 2-way softmax."""
    states = (t.action_type == ActionType.CHOOSE_USE.value) & (t.use_visits.sum(1) > 0)
    p = t.use_visits[states]
    p = p / p.sum(1, keepdim=True)
    logq = torch.log_softmax(logits[states].float(), 1)
    n = states.sum()
    loss = (torch.xlogy(p, p) - p * logq).sum() / n.clamp(min=1)
    with torch.no_grad():
        correct = (p.gather(1, logq.argmax(1, keepdim=True)).squeeze(1) == p.max(1).values).sum()
    return loss, n, correct


def losses(out, g: Graphs, t: Targets, legal_only: bool = False, legality_weight: float = 0.0) -> dict:
    """Per head: (loss, states it covers, states where the top choice matches MCTS's).
    Value: MSE over every state. Policy heads: over the states of their decision type that have
    visits, a softmax over every node of the head's types (no legality mask; candidates are always
    included) trained toward the normalized visits. legal_only: the softmax is over the candidates only.
    legality_weight: with a fifth output (per-node legality logits), adds "legality": BCE over the nodes of each
    policy head's types in the states of its decision type, label = is a candidate, times this weight."""
    priority, target, use_logits, value = out[:4]
    node_graph = graph_index(g.node_offsets)
    is_cand = torch.zeros_like(g.node_type, dtype=torch.bool).index_fill(0, t.cand_node, True)
    visited = torch.zeros_like(t.value).index_add(0, t.cand_graph, t.cand_visits) > 0

    def of_types(types):
        if legal_only:
            return is_cand
        return torch.isin(g.node_type, torch.tensor(types, device=g.node_type.device)) | is_cand

    heads = {
        "priority": node_policy_loss(priority, of_types(PRIORITY_TYPES),
                                     (t.action_type == ActionType.PRIORITY.value) & visited, node_graph, t),
        "choose_target": node_policy_loss(target, of_types(TARGET_TYPES),
                                          (t.action_type == ActionType.CHOOSE_TARGET.value) & visited, node_graph, t),
        "choose_use": use_loss(use_logits, t),
        "value": (F.mse_loss(value.float(), t.value), t.value.shape[0], 0),
    }
    if legality_weight > 0 and len(out) > 4:
        def nodes(action, types):
            return torch.isin(g.node_type, torch.tensor(types, device=g.node_type.device)) & \
                (t.action_type == action.value)[node_graph]
        domain = (nodes(ActionType.PRIORITY, PRIORITY_TYPES) | nodes(ActionType.CHOOSE_TARGET, TARGET_TYPES)).nonzero().squeeze(1)
        bce = F.binary_cross_entropy_with_logits(out[4][domain].float(), is_cand[domain].float())
        heads["legality"] = (legality_weight * bce, int(t.value.shape[0]), 0)
    return heads


def accumulate(stats: dict, heads: dict) -> None:
    """Add one batch's per-head (loss, states, correct) to running sums."""
    for name, (loss, n, correct) in heads.items():
        s = stats.setdefault(name, [0.0, 0, 0])
        n = int(n)
        s[0] += float(loss) * n
        s[1] += n
        s[2] += int(correct)


def averages(stats: dict) -> dict:
    """Mean loss per head, over the states each head covered."""
    out = {}
    for name in HEADS:
        total, n, _ = stats.get(name, (0.0, 0, 0))
        out[name] = total / max(n, 1)
    return out


def decision_states(stats: dict) -> int:
    """States that trained a policy head."""
    n = 0
    for name in POLICY_HEADS:
        n += stats.get(name, (0.0, 0, 0))[1]
    return n


def load_model(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return torch.load(f, map_location="cpu")
    return torch.load(path, map_location="cpu")


def load_checkpoint(path):
    """(model, leaf vocab, edge label vocab) from a graph-encoder checkpoint."""
    checkpoint = load_model(path)
    if "edge_vocab" not in checkpoint:
        raise ValueError(f"{path} is not a graph-encoder checkpoint (no edge vocab); it was trained on the flat encoder")
    vocab = FeatureVocab.from_state_dict(checkpoint["feature_vocab"])
    edge_vocab = FeatureVocab.from_state_dict(checkpoint["edge_vocab"])
    vocab.require_encoding(GLOBAL_MAX)
    edge_vocab.require_encoding(GLOBAL_MAX)
    model = build_model(len(vocab), len(edge_vocab), **{**LEGACY_OPTIONS, **checkpoint.get("arch", LEGACY_ARCH)})
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, vocab, edge_vocab
