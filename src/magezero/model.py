import torch
from torch import nn
import math
import gzip
from enum import Enum

"""
MageZero Neural Network architecture for AlphaZero style MCTS:
2M sparse embedding bag -> 512D embedding layer -> 256D hidden layer -> (3 x 128D policy heads + 2D binary policy head + 1D value head)

Policy heads are for each decision type (disjoint action spaces) they are:
128D PriorityA (priority actions for PlayerA - which is this agent)
128D PriorityB (priority actions for PlayerB - which is the opponent)
128D Choose Target (target choices for both players)
2D Choose Use (binary decisions for either player - this is used for selecting attackers and blockers)
"""
ACTIONS_MAX = 128
GLOBAL_MAX = 2000000



PRIORITY_A_MAX = 128
PRIORITY_B_MAX = 128
TARGETS_MAX = 128
BINARY_MAX = 2


class ActionType(Enum):
    PRIORITY = 0
    CHOOSE_TARGET = 3
    CHOOSE_USE = 5

def head_weight(K: int) -> float:
    """
    Analytic loss weight to equalize baseline CE scales:
    lambda_K = ln(2) / ln(K)
    """
    if K <= 1:
        raise ValueError("K must be >= 2 for cross-entropy.")
    return math.log(2.0) / math.log(float(K))

#per head weights
lambda_pA = head_weight(PRIORITY_A_MAX)
lambda_pB = head_weight(PRIORITY_B_MAX)
lambda_t = head_weight(TARGETS_MAX)
lambda_b = head_weight(BINARY_MAX)


class Net(nn.Module):
    def __init__(self, num_embeddings, policy_size_A):
        super().__init__()


        embedding_dim = 512  # Output of EmbeddingBag
        hidden_dim_mlp = 256  # Output of the main MLP block
        self.embedding_bag = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode='sum',
            sparse=True,
            max_norm=1
        )
        self.embedding_bias = nn.Parameter(torch.zeros(embedding_dim))
        self.input_dropout = 0
        #self.embedding_norm = nn.LayerNorm(embedding_dim)
        self.embedding_dropout = nn.Dropout(p=0.5)
        self.l1_penalty = None

        self.fc_after_embedding = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp),  # From 512 to 256
            nn.ReLU(),
        )
        #policy heads (4 x 256->128 + 1 x 256->2)
        self.player_priority_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.opponent_priority_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.target_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.binary_head = nn.Linear(hidden_dim_mlp, 2)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim_mlp, 1),  # From 256 to 1
            nn.Tanh()
        )

    def forward(self, indices, offsets):
        input_weights = None
        if self.training and self.input_dropout > 0:
            keep_mask = torch.rand_like(indices, dtype=torch.float32) > self.input_dropout
            keep_mask = keep_mask.to(torch.float32)
            input_weights = keep_mask / (1.0 - self.input_dropout)

        emb = self.embedding_bag(indices, offsets, per_sample_weights=input_weights)

        if self.training:
            self.l1_penalty = emb.abs().sum() * 1e-7

        #emb = emb + self.embedding_bias
        #emb = F.relu(emb)
        #emb = self.embedding_norm(emb)


        emb = self.embedding_dropout(emb)
        h = self.fc_after_embedding(emb)
        return self.player_priority_head(h), self.opponent_priority_head(h), self.target_head(h), self.binary_head(h), self.value_head(h).squeeze(-1)

class NetTransformer(nn.Module):
    def __init__(self, num_embeddings, policy_size_A):
        super().__init__()

        embedding_dim = 512
        hidden_dim_mlp = 256
        self.input_dropout = 0.3


        self.embedding = nn.Embedding(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            sparse=True,
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim, nhead=4,
            dim_feedforward=512, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        self.embedding_dropout = nn.Dropout(p=0.2)

        self.fc_after_embedding = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp),  # From 512 to 256
            nn.ReLU(),
        )
        #policy heads (4 x 256->128 + 1 x 256->2)
        self.player_priority_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.opponent_priority_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.target_head = nn.Linear(hidden_dim_mlp, policy_size_A)
        self.binary_head = nn.Linear(hidden_dim_mlp, 2)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim_mlp, 1),  # From 256 to 1
            nn.Tanh()
        )

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
            drop = drop & mask  # only drop real tokens
            mask = mask & ~drop  # removed from attention


        emb = self.embedding(padded)  # (B, max_len, 512)
        emb = self.transformer(emb, src_key_padding_mask=~mask)  # (B, max_len, 512)

        # mean pool over real tokens
        #emb = (emb * mask.unsqueeze(-1)).sum(1) / lengths.unsqueeze(-1).float()  # (B, 512)
        pool_count = mask.sum(1).clamp(min=1).unsqueeze(-1).float()
        emb = (emb * mask.unsqueeze(-1)).sum(1) / pool_count

        emb = self.embedding_dropout(emb)
        h = self.fc_after_embedding(emb)
        return self.player_priority_head(h), self.opponent_priority_head(h), self.target_head(h), self.binary_head(h), self.value_head(h).squeeze(-1)



def load_model(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rb') as f:
            return torch.load(f)
    return torch.load(path)

def normalize_policy_labels(raw: torch.Tensor) -> torch.Tensor:
    total = raw.sum(dim=1, keepdim=True).clamp(min=1e-8)
    return raw / total