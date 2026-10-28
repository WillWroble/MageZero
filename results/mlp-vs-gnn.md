# NetGraph vs a small MLP across generations (RunPod)

Dama's claim (https://adams-shaun.github.io/gorge/bots/magezero-replication/, background in
`handoff-mlp-vs-netgraph.md`): on the ver4 gen0 mirror test, a 166K-parameter d64 MLP matches NetGraph on value
and target and beats it on priority (non-pass top-1 0.837 vs 0.457). He attributes the priority gap to the
training objective: NetGraph's softmax covers every ability node, legal or not.

## Setup

- **Models.**
  - `gnn`: NetGraph default (d256, FF 512, 2 passes, 8.95M params), policy softmax over all nodes of the head's types.
  - `gnnlegal`: the same model, policy softmax over the legal candidates only (`policy_domain: "legal"`).
  - `mlp`: `NetMLP` d64, 1 block (209K params incl. embeddings), legal candidates only, lr 1e-3.
  - `mlp256`: `NetMLP` d256, 2 blocks (2.2M params), lr 1e-3.
- **NetMLP** is a reconstruction of Dama's `MLPNet` from the forward pass in gorge's `bots/gonet/gonet.go`; his
  bridge from our graphs to MLP inputs isn't published. Each object is an MLP over the mean of its leaves; it adds the
  sum of its ancestors' vectors (zone, owner, source card); the state is per-type sum/max pools plus the root,
  through residual blocks. No attention, no learned message passing. CPU batch 1: 3.7 ms vs 31 ms for NetGraph.
- **Data** (built by `runpod/prepare.sh` on the network volume):
  - `gen0/train`: ver4 gen0 Pauper pool, 8 matchups, both seats (420K decision states).
  - `gen0/test`: ver4 gen0 mirror test set (63.5K states). This is the set in Dama's table.
  - `gen1/train`, `gen1/test`: every ver4 and ver5 gen1 shard except the `_pimc` ones, both seats. The last 20%
    of each shard's games are held out (314K train, 139K test states). Caveat: within a shard, game length and
    win rate drift (ver5 Elves: 72 states/game and 0.85 win rate in the first fifth, 245 and 0.35 in the last),
    so the held-out tail skews toward long games.
- **Training:** `train.py`, 6 epochs, batch 64, Adam (1e-4 for NetGraph), seed 0. One RTX PRO 6000 running 4
  runs at once.
- **Metrics** (`test.py`, last epoch). Every policy metric is restricted to the legal candidates, so all models
  are scored the same way:
  - prio KL / tgt KL: KL from MCTS visits to the softmax over candidates.
  - prio acc: top-1 by ability name. non-pass / pass: the same, split by whether MCTS's top move was Pass.
  - tgt acc: top candidate is among the most visited.

## gen0 training (mz `mlp-g0`)

| run | test | value | prio KL | prio acc | non-pass | pass | tgt KL | tgt acc | use acc |
|---|---|---|---|---|---|---|---|---|---|
| gnn | gen0 mirror | 0.037 | 0.190 | 0.640 | 0.452 | 0.913 | 0.074 | 0.788 | 0.887 |
| gnnlegal | gen0 mirror | 0.039 | **0.052** | **0.795** | **0.864** | 0.695 | **0.063** | **0.794** | 0.886 |
| mlp | gen0 mirror | 0.038 | 0.053 | 0.790 | 0.852 | 0.700 | 0.071 | 0.763 | 0.886 |
| mlp256 | gen0 mirror | 0.039 | 0.054 | 0.789 | 0.848 | 0.702 | 0.074 | 0.757 | 0.886 |
| gnn | gen1 held-out | 0.094 | 0.619 | 0.655 | 0.284 | 0.915 | 0.305 | **0.700** | 0.737 |
| gnnlegal | gen1 held-out | **0.077** | **0.107** | **0.731** | **0.749** | 0.719 | **0.200** | 0.685 | **0.814** |
| mlp | gen1 held-out | 0.105 | 0.110 | 0.717 | 0.683 | 0.740 | 0.212 | 0.645 | 0.728 |
| mlp256 | gen1 held-out | 0.096 | 0.107 | 0.712 | 0.699 | 0.721 | 0.210 | 0.642 | 0.740 |

Gen0 mirror over epochs (prio acc / non-pass): gnnlegal 0.779/0.833, 0.790/0.851, 0.781/0.826, 0.790/0.793,
0.794/0.834, 0.795/0.864; mlp 0.736/0.786, 0.771/0.837, 0.785/0.814, 0.783/0.859, 0.789/0.832, 0.790/0.852.
Non-pass moves ±0.04 between epochs, because it trades off against pass accuracy.

Conclusions (one seed):
- **The priority gap is the training objective.** Trained over legal candidates, NetGraph scores 0.864 non-pass
  (Dama's MLP: 0.837). The default GNN's all-node softmax leaves it biased to Pass (pass 0.91, non-pass 0.45).
  Restricting the default checkpoint to legal moves only at test time doesn't fix it: 0.432 → 0.440 on 60 mirror
  games. MCTS only uses the legal children's priors, so the legal-only loss is the objective that matters.
- **Dama's claim holds in distribution.** On the gen0 mirror, the MLP matches the legal GNN on value and priority
  (within 0.001–0.005). The GNN is ahead on targets (0.794 vs 0.763 acc, 0.063 vs 0.071 KL). MLP width doesn't
  help (d256 ≈ d64).
- **Not on the next generation.** Trained on gen0 and tested on held-out gen1 games, the GNN generalizes better on
  every head but pass: value 0.077 vs 0.096–0.105, non-pass 0.749 vs 0.683–0.699, target acc 0.685 vs 0.642–0.645,
  choose_use 0.814 vs 0.728–0.740. Needs a second seed (running) before it counts.
