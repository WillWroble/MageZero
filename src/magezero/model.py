import torch
from torch import nn
import math
import gzip
from enum import Enum

"""
MageZero Neural Network architecture for AlphaZero style MCTS:
2M x 512 embedding table (sparse indices -> ~1300 tokens)
└── 2-layer TransformerEncoder (d=512, nhead=4, ff=1024)
    └── mean pool -> 512D
        ├── MLP(512 -> 256 -> 128)  priority_player   (PlayerA priority actions)
        ├── MLP(512 -> 256 -> 128)  priority_opponent (PlayerB priority actions)
        ├── MLP(512 -> 256 -> 128)  target            (target choices, both players)
        ├── MLP(512 -> 256 -> 2)    binary            (attack/block use decisions)
        └── MLP(512 -> 256 -> 1)    value             (tanh, -1..1)
"""


GLOBAL_MAX = 2**31-1

PRIORITY_A_MAX = 128
PRIORITY_B_MAX = 128
TARGETS_MAX = 128
BINARY_MAX = 2


class ActionType(Enum):
    PRIORITY = 0
    CHOOSE_TARGET = 3
    CHOOSE_USE = 5

def head_weight(K: int) -> float:
    if K <= 1:
        raise ValueError("K must be >= 2 for cross-entropy.")
    return math.log(2.0) / math.log(float(K))

#per head weights
lambda_pA = head_weight(PRIORITY_A_MAX)
lambda_pB = head_weight(PRIORITY_B_MAX)
lambda_t = head_weight(TARGETS_MAX)
lambda_b = head_weight(BINARY_MAX)

class NetTransformer(nn.Module):
    def __init__(self, num_embeddings=2048, policy_size_pA=PRIORITY_A_MAX, policy_size_pB=PRIORITY_B_MAX, policy_size_t=TARGETS_MAX, policy_size_b=BINARY_MAX):
        super().__init__()

        embedding_dim = 512
        hidden_dim_mlp = 256
        self.input_dropout = 0.3


        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparse=False,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=4,
            dim_feedforward=1024, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.embedding_dropout = nn.Dropout(p=0.2)

        self.player_priority_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_pA),
        )
        self.opponent_priority_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_pB),
        )
        self.target_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_t),
        )
        self.binary_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, policy_size_b),
        )
        self.value_head = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp), nn.ReLU(),
            nn.Linear(hidden_dim_mlp, 1), nn.Tanh(),
        )

    def resize_embedding(self, num_embeddings: int, init_rows=None) -> None:
        """Grow the embedding table to `num_embeddings` rows, keeping existing rows unchanged.
        Used by dense-vocab training when a new generation adds features to the vocab. `init_rows`
        supplies the added rows (vocab.initial_rows draws each from its feature id, so a feature
        starts from the same row whenever it is first seen, whatever generation that is); without
        it they get nn.Embedding's default init."""
        old = self.embedding
        if num_embeddings == old.num_embeddings:
            return
        if num_embeddings < old.num_embeddings:
            raise ValueError("the feature vocab is append-only; the embedding table cannot shrink")
        new = nn.Embedding(num_embeddings, old.embedding_dim, sparse=old.sparse).to(old.weight.device)
        with torch.no_grad():
            new.weight[:old.num_embeddings] = old.weight
            if init_rows is not None:
                added = num_embeddings - old.num_embeddings
                new.weight[old.num_embeddings:] = torch.as_tensor(
                    init_rows, dtype=new.weight.dtype, device=new.weight.device)[:added]
        self.embedding = new

    def forward(self, indices, offsets):
        B = offsets.shape[0]
        ends = torch.cat([offsets[1:], torch.tensor([indices.shape[0]], device=offsets.device)])
        lengths = ends - offsets
        max_len = lengths.max().item()

        # reconstruct padded sequences
        padded = indices.new_zeros(B, max_len)
        mask = torch.zeros(B, max_len, dtype=torch.bool, device=indices.device)

        for i in range(B):
            l = lengths[i]
            padded[i, :l] = indices[offsets[i]:offsets[i] + l]
            mask[i, :l] = True

        if self.training and self.input_dropout > 0:
            drop = torch.rand(B, max_len, device=indices.device) < self.input_dropout
            mask = mask & ~drop


        emb = self.embedding(padded)  # (B, max_len, 512)
        emb = self.transformer(emb, src_key_padding_mask=~mask)  # (B, max_len, 512)

        # mean pool over real tokens
        pool_count = mask.sum(1).clamp(min=1).unsqueeze(-1).float()
        emb = (emb * mask.unsqueeze(-1)).sum(1) / pool_count

        emb = self.embedding_dropout(emb)

        return (
            self.player_priority_head(emb),
            self.opponent_priority_head(emb),
            self.target_head(emb),
            self.binary_head(emb),
            self.value_head(emb).squeeze(-1),
        )




def load_model(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return torch.load(f)
    return torch.load(path)

def normalize_policy_labels(raw: torch.Tensor) -> torch.Tensor:
    total = raw.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return raw / total