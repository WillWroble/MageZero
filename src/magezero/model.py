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
  2 bottom-up passes (separate weights), each one NodeLayer per type, run in order
      ABILITY -> CARD -> PERMANENT -> STACK_OBJECT -> ZONE -> PLAYER -> ROOT
      every node attends over [itself, children + edge label embedding] (d=512, nhead=4, ff=1024)
  2-layer TransformerEncoder over [CLS, the state's internal nodes]
      ├── MLP(512 -> 256 -> 1)  per node  priority       (read at ABILITY nodes)
      ├── MLP(512 -> 256 -> 1)  per node  target         (read at CARD/PERMANENT/STACK_OBJECT/PLAYER nodes)
      ├── MLP(512 -> 256 -> 2)  CLS       choose_use     (false, true)
      └── MLP(512 -> 256 -> 1)  CLS       value          (tanh, -1..1)
"""


GLOBAL_MAX = 2**31-1

D_MODEL = 512
N_HEADS = 4
D_FF = 1024
DROPOUT = 0.25
N_PASSES = 2
MIXER_LAYERS = 2
HIDDEN_MLP = 256

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


def graph_index(offsets: torch.Tensor) -> torch.Tensor:
    """State of each node, from a batch's node offsets."""
    return torch.repeat_interleave(torch.arange(offsets.numel() - 1, device=offsets.device), offsets.diff())


def segment_logsumexp(x: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    """Log-sum-exp over the rows of x that share an index in seg (values 0..n-1); returns n rows."""
    index = seg.view(-1, *([1] * (x.dim() - 1))).expand_as(x)
    with torch.no_grad():  # the max only stabilizes; the result and its gradient don't depend on it
        m = x.new_full((n, *x.shape[1:]), float("-inf")).scatter_reduce(0, index, x, "amax")
    return m + x.new_zeros((n, *x.shape[1:])).index_add(0, seg, (x - m[seg]).exp()).log()


class NodeLayer(nn.TransformerEncoderLayer):
    """A pre-norm transformer layer run over each node's [node, children...] sequence, keeping only
    the node's output token. The children's output tokens would be discarded, so only the node's
    query is computed: attention from the node to itself and its children, then the FFN on the node.
    Same parameters, init and result as nn.TransformerEncoderLayer, without padding to the largest
    child count and without running the FFN on children."""

    def __init__(self):
        super().__init__(D_MODEL, N_HEADS, D_FF, DROPOUT, activation="gelu", batch_first=True, norm_first=True)

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


def mlp(out_dim: int, *tail: nn.Module) -> nn.Sequential:
    return nn.Sequential(nn.Linear(D_MODEL, HIDDEN_MLP), nn.ReLU(), nn.Linear(HIDDEN_MLP, out_dim), *tail)


class NetGraph(nn.Module):
    def __init__(self, num_embeddings: int = 0, num_edge_labels: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, D_MODEL)                         # leaves
        self.edge_embedding = nn.Embedding(num_edge_labels + 1, D_MODEL, padding_idx=0)  # row 0: unknown label
        self.type_embedding = nn.Embedding(len(NodeType), D_MODEL)
        self.value_embedding = nn.Embedding(len(VALUE_BOUNDS) + 1, D_MODEL)
        self.register_buffer("value_bounds", torch.tensor(VALUE_BOUNDS), persistent=False)

        self.passes = nn.ModuleList(
            nn.ModuleDict({t.name: NodeLayer() for t in STAGES}) for _ in range(N_PASSES))

        self.cls = nn.Parameter(torch.randn(D_MODEL))
        mixer_layer = nn.TransformerEncoderLayer(D_MODEL, N_HEADS, D_FF, DROPOUT, activation="gelu",
                                                 batch_first=True, norm_first=True)
        self.mixer = nn.TransformerEncoder(mixer_layer, MIXER_LAYERS, norm=nn.LayerNorm(D_MODEL),
                                           enable_nested_tensor=False)

        self.priority_head = mlp(1)
        self.target_head = mlp(1)
        self.use_head = mlp(2)
        self.value_head = mlp(1, nn.Tanh())

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

    def forward(self, g: Graphs):
        """Returns per-node priority and target scores [N] (-inf at leaves), choose_use logits
        [B, 2] (false, true) and value [B]."""
        B, N = g.node_offsets.numel() - 1, g.node_type.numel()

        # initial embeddings: typed nodes their type, leaves their name and numeric value
        leaves = (g.node_row >= 0).nonzero().squeeze(1)
        value = torch.bucketize(g.node_value[leaves], self.value_bounds, right=True)
        h = self.type_embedding(g.node_type).index_add(
            0, leaves, self.embedding(g.node_row[leaves]) + self.value_embedding(value))

        # leaves outside the vocab have no embedding, so their edges are dropped
        keep = (g.node_type[g.edge_child] != NodeType.LEAF) | (g.node_row[g.edge_child] >= 0)
        child, parent, label = g.edge_child[keep], g.edge_parent[keep], g.edge_label[keep]

        # one stage per type: every node of that type, its children, and each child's parent position
        parent_type = g.node_type[parent]
        stages = []
        for t in STAGES:
            nodes = (g.node_type == t).nonzero().squeeze(1)
            if len(nodes):
                edges = (parent_type == t).nonzero().squeeze(1)
                stages.append((t.name, nodes, child[edges], label[edges], torch.searchsorted(nodes, parent[edges])))

        # bottom-up passes: children earlier in the order arrive updated in this pass, children of
        # the same type or later in the order (attachments, targets, linked exile) from the last one
        for layers in self.passes:
            for name, nodes, c, lbl, seg in stages:
                h = h.index_copy(0, nodes, layers[name](h[nodes], h[c] + self.edge_embedding(lbl), seg))

        # final self-attention over [CLS, internal nodes], one padded sequence per state
        internal = (g.node_type != NodeType.LEAF).nonzero().squeeze(1)
        gi = graph_index(g.node_offsets)[internal]
        counts = torch.bincount(gi, minlength=B)
        pos = torch.arange(len(internal), device=h.device) - (counts.cumsum(0) - counts)[gi] + 1
        x = h.new_zeros(B, int(counts.max()) + 1, h.shape[1])
        x[:, 0] = self.cls
        x[gi, pos] = h[internal]
        pad = torch.arange(x.shape[1], device=h.device) > counts.unsqueeze(1)
        x = self.mixer(x, src_key_padding_mask=pad)
        cls, z = x[:, 0], x[gi, pos]

        def per_node(head):
            return torch.full((N,), float("-inf"), device=h.device).index_copy(
                0, internal, head(z).squeeze(-1).float())

        return per_node(self.priority_head), per_node(self.target_head), self.use_head(cls), self.value_head(cls).squeeze(-1)


def node_policy_loss(scores, domain, states, node_graph, t: Targets):
    """KL from the normalized MCTS visits to a softmax over each state's domain nodes, for `states`
    ([B] bool). correct: states whose highest-scored candidate is one of the most visited."""
    B = states.shape[0]
    idx = (domain & states[node_graph]).nonzero().squeeze(1)
    lse = segment_logsumexp(scores[idx].float(), node_graph[idx], B)
    in_states = states[t.cand_graph]
    cand, visits, graph = t.cand_node[in_states], t.cand_visits[in_states], t.cand_graph[in_states]
    s = scores[cand].float()
    p = visits / torch.zeros(B, device=s.device).index_add(0, graph, visits)[graph]
    n = states.sum()
    loss = (torch.xlogy(p, p) - p * (s - lse[graph])).sum() / n.clamp(min=1)
    with torch.no_grad():
        best_s = torch.full((B,), float("-inf"), device=s.device).scatter_reduce(0, graph, s, "amax")
        best_v = torch.zeros(B, device=s.device).scatter_reduce(0, graph, visits, "amax")
        hit = ((s == best_s[graph]) & (visits == best_v[graph])).float()
        correct = (torch.zeros(B, device=s.device).index_add(0, graph, hit) > 0).sum()
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


def losses(out, g: Graphs, t: Targets) -> dict:
    """Per head: (loss, states it covers, states where the top choice matches MCTS's).
    Value: MSE over every state. Policy heads: over the states of their decision type that have
    visits, a softmax over every node of the head's types (no legality mask; candidates are always
    included) trained toward the normalized visits."""
    priority, target, use_logits, value = out
    node_graph = graph_index(g.node_offsets)
    is_cand = torch.zeros_like(g.node_type, dtype=torch.bool).index_fill(0, t.cand_node, True)
    visited = torch.zeros_like(t.value).index_add(0, t.cand_graph, t.cand_visits) > 0

    def of_types(types):
        return torch.isin(g.node_type, torch.tensor(types, device=g.node_type.device)) | is_cand

    return {
        "priority": node_policy_loss(priority, of_types(PRIORITY_TYPES),
                                     (t.action_type == ActionType.PRIORITY.value) & visited, node_graph, t),
        "choose_target": node_policy_loss(target, of_types(TARGET_TYPES),
                                          (t.action_type == ActionType.CHOOSE_TARGET.value) & visited, node_graph, t),
        "choose_use": use_loss(use_logits, t),
        "value": (F.mse_loss(value.float(), t.value), t.value.shape[0], 0),
    }


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
            return torch.load(f)
    return torch.load(path)


def load_checkpoint(path):
    """(model, leaf vocab, edge label vocab) from a graph-encoder checkpoint."""
    checkpoint = load_model(path)
    if "edge_vocab" not in checkpoint:
        raise ValueError(f"{path} is not a graph-encoder checkpoint (no edge vocab); it was trained on the flat encoder")
    vocab = FeatureVocab.from_state_dict(checkpoint["feature_vocab"])
    edge_vocab = FeatureVocab.from_state_dict(checkpoint["edge_vocab"])
    vocab.require_encoding(GLOBAL_MAX)
    edge_vocab.require_encoding(GLOBAL_MAX)
    model = NetGraph(len(vocab), len(edge_vocab))
    model.load_state_dict(checkpoint["model_state_dict"])
    return model, vocab, edge_vocab
