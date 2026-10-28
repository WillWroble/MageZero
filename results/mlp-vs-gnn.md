# NetGraph vs a small MLP across generations (RunPod)

Dama's claim (https://adams-shaun.github.io/gorge/bots/magezero-replication/, background in
`handoff-mlp-vs-netgraph.md`): on the ver4 gen0 mirror test, a 166K-parameter d64 MLP matches NetGraph on value
and target and beats it on priority (non-pass top-1 0.837 vs 0.457). He attributes the priority gap to the
training objective: NetGraph's softmax covers every ability node, legal or not.

## Setup

- **Models.**
  - `gnn`: NetGraph default (d256, FF 512, 2 passes, 8.95M params), policy softmax over all nodes of the head's types.
  - `gnnlegal`: the same model, policy softmax over the legal candidates only (`policy_domain: "legal"`).
  - `mlp`: `NetMLP` d64, 1 block (172K params with gen0's vocab; Dama: 166K), legal candidates only, lr 1e-3.
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
  choose_use 0.814 vs 0.728–0.740. Needs a second seed before it counts (below: only value and target hold up).

### gen0, second seed and controls (`mlp-g0b`)

| run | test | value | prio KL | prio acc | non-pass | pass | tgt KL | tgt acc | use acc |
|---|---|---|---|---|---|---|---|---|---|
| gnnlegal seed 1 | gen0 mirror | 0.041 | 0.057 | 0.787 | 0.807 | 0.757 | 0.059 | 0.802 | 0.886 |
| mlp seed 1 | gen0 mirror | 0.038 | 0.054 | 0.790 | 0.836 | 0.724 | 0.073 | 0.762 | 0.886 |
| gnnlegal d64 (667K params) | gen0 mirror | 0.040 | 0.053 | 0.785 | 0.805 | 0.757 | 0.067 | 0.785 | 0.886 |
| mlp lr 3e-4 | gen0 mirror | 0.040 | 0.052 | 0.794 | 0.862 | 0.694 | 0.072 | 0.761 | 0.886 |
| gnnlegal seed 1 | gen1 held-out | 0.077 | 0.109 | 0.680 | 0.528 | 0.786 | 0.192 | 0.696 | 0.735 |
| mlp seed 1 | gen1 held-out | 0.101 | 0.100 | 0.716 | 0.681 | 0.739 | 0.190 | 0.675 | 0.730 |
| gnnlegal d64 | gen1 held-out | 0.093 | 0.106 | 0.716 | 0.626 | 0.780 | 0.192 | 0.695 | 0.734 |
| mlp lr 3e-4 | gen1 held-out | 0.097 | 0.108 | 0.719 | 0.713 | 0.724 | 0.214 | 0.640 | 0.731 |

- On the gen0 mirror, both seeds agree: value and priority are level, and the GNN leads on targets (acc 0.794/0.802
  vs 0.763/0.762). The MLP's learning rate doesn't change that (lr 3e-4: 0.761).
- A d64 GNN (667K params, 3× the MLP) keeps most of the target lead (0.785) with the MLP's value and priority.
- gen0 → gen1 transfer: only two results repeat across seeds. The GNN's value is better (0.077 on both seeds vs
  0.101–0.105), and so are targets (0.685/0.696 vs 0.645/0.675). Out-of-distribution priority swings with the
  seed (GNN non-pass 0.749 → 0.528, use 0.814 → 0.735), as in the earlier diversity runs.

## gen1 training (`mlp-g1`)

| run | test | value | prio KL | prio acc | non-pass | pass | tgt KL | tgt acc | use acc |
|---|---|---|---|---|---|---|---|---|---|
| gnn | gen1 held-out | 0.058 | 0.152 | 0.715 | 0.403 | 0.924 | 0.147 | 0.758 | 0.816 |
| gnnlegal | gen1 held-out | 0.074 | **0.076** | **0.756** | **0.612** | 0.853 | **0.144** | **0.758** | 0.798 |
| mlp | gen1 held-out | 0.079 | 0.080 | 0.744 | 0.580 | 0.854 | 0.153 | 0.733 | 0.802 |
| mlp256 | gen1 held-out | 0.086 | 0.081 | 0.742 | 0.564 | 0.862 | 0.154 | 0.738 | 0.806 |
| gnn | gen0 mirror | 0.063 | 0.309 | 0.576 | 0.330 | 0.932 | 0.131 | 0.710 | 0.885 |
| gnnlegal | gen0 mirror | **0.766** | 0.083 | 0.721 | 0.687 | 0.771 | 0.089 | 0.703 | 0.846 |
| mlp | gen0 mirror | 0.058 | 0.083 | 0.704 | 0.655 | 0.776 | 0.079 | 0.727 | 0.868 |
| mlp256 | gen0 mirror | 0.084 | 0.085 | 0.680 | 0.567 | 0.844 | 0.084 | 0.735 | 0.886 |

- In distribution on gen1, the legal GNN is ahead of the MLP on priority (KL 0.076 vs 0.080, non-pass 0.612 vs
  0.580) and targets (acc 0.758 vs 0.733). Value and choose_use are level. One seed; seeds 1–2 are running.
- gen1 has no mirror games. On the gen0 mirror, the gen1-trained legal GNN's value head collapsed (MSE 0.766;
  every other run 0.06–0.08). One seed. This is the failure mode seen before in `experiments.md`: value on an unseen
  matchup from a model without that matchup.

### gen1 training, seeds 1–2 (`mlp-g1b`)

| run | test | value | prio KL | prio acc | non-pass | pass | tgt KL | tgt acc | use acc |
|---|---|---|---|---|---|---|---|---|---|
| gnnlegal s1 | gen1 held-out | 0.064 | 0.075 | 0.756 | 0.574 | 0.878 | 0.144 | 0.768 | 0.813 |
| gnnlegal s2 | gen1 held-out | 0.063 | 0.075 | 0.754 | 0.605 | 0.854 | 0.144 | 0.764 | 0.809 |
| mlp s1 | gen1 held-out | 0.083 | 0.081 | 0.747 | 0.611 | 0.838 | 0.153 | 0.743 | 0.816 |
| mlp s2 | gen1 held-out | 0.078 | 0.081 | 0.737 | 0.613 | 0.819 | 0.151 | 0.744 | 0.812 |
| gnnlegal s1 | gen0 mirror | 0.714 | 0.086 | 0.697 | 0.644 | 0.774 | 0.094 | 0.648 | 0.861 |
| gnnlegal s2 | gen0 mirror | 0.599 | 0.080 | 0.734 | 0.727 | 0.745 | 0.086 | 0.742 | 0.794 |
| mlp s1 | gen0 mirror | 0.065 | 0.075 | 0.731 | 0.687 | 0.794 | 0.080 | 0.737 | 0.886 |
| mlp s2 | gen0 mirror | 0.054 | 0.076 | 0.729 | 0.682 | 0.797 | 0.079 | 0.732 | 0.885 |

Three seeds each, gen1 in distribution (gnnlegal vs mlp, ranges): prio KL 0.075–0.076 vs 0.080–0.081, target acc
0.758–0.768 vs 0.733–0.744, target KL 0.144 vs 0.151–0.153, value 0.063–0.074 vs 0.078–0.083. Non-pass overlaps
(0.574–0.612 vs 0.580–0.613), use is level. **On gen1, in distribution, the GNN is consistently but slightly
better on priority KL, targets and value.** The gen1-trained GNN's mirror value collapse repeats on all three
seeds (0.60–0.77 vs 0.054–0.065 for the MLP).

## True OOD opponents at gen0 (`mlp-ood`)

Training: `gen0-nomirror/train`, i.e. gen0/train without the mirror shards (7 Pauper matchups, both seats). Tests:
the gen0 mirror (now an unseen opponent) and `gen0-mono/test`, the first 100 games of each gen0 Standard Mono
matchup, both seats (54K states; no gen0 training set has Mono). In `ver4/training/gen0/PlayerB`,
`Pauper-Faeries_vs_Standard-MonoW.hdf5` is a byte-identical copy of PlayerA's shard; it is left out.

| run | test | value | prio KL | prio acc | non-pass | pass | tgt KL | tgt acc | use acc |
|---|---|---|---|---|---|---|---|---|---|
| gnnlegal s0 | mirror (unseen) | 0.863 | 0.074 | 0.756 | 0.780 | 0.721 | 0.083 | 0.749 | 0.883 |
| gnnlegal s1 | mirror (unseen) | 0.947 | 0.081 | 0.732 | 0.739 | 0.722 | 0.088 | 0.666 | 0.855 |
| mlp s0 | mirror (unseen) | 0.059 | 0.059 | 0.779 | 0.843 | 0.686 | 0.075 | 0.754 | 0.886 |
| mlp s1 | mirror (unseen) | 0.052 | 0.059 | 0.775 | 0.831 | 0.693 | 0.078 | 0.751 | 0.885 |
| gnnlegal s0 | Mono (unseen) | 0.226 | 0.109 | 0.778 | 0.801 | 0.748 | 0.116 | 0.753 | 0.715 |
| gnnlegal s1 | Mono (unseen) | 0.194 | 0.109 | 0.768 | 0.779 | 0.753 | 0.122 | 0.738 | 0.767 |
| mlp s0 | Mono (unseen) | 0.079 | 0.105 | 0.790 | 0.856 | 0.705 | 0.122 | 0.725 | 0.730 |
| mlp s1 | Mono (unseen) | 0.075 | 0.100 | 0.779 | 0.806 | 0.745 | 0.127 | 0.724 | 0.729 |

For reference, with the mirror in training (gen0 table above): gnnlegal 0.039–0.041 value, 0.052–0.057 prio KL;
mlp 0.038, 0.053–0.054.

- **On unseen opponents the MLP wins on value and priority, on both seeds.** Mirror value 0.052–0.059 vs
  0.86–0.95; Mono 0.075–0.079 vs 0.19–0.23. Priority KL on the mirror 0.059 vs 0.074–0.081. The MLP loses almost
  nothing when the mirror leaves training (value 0.038 → 0.052–0.059); the GNN's value breaks.
- Targets: level on the mirror (s0), MLP ahead on s1; GNN ahead on Mono (0.738–0.753 vs 0.724–0.725).
- **The GNN's value failure is a shortcut on seat identity.** On 60 mirror games per seat (`nm-gnnlegal-s0`), the
  GNN predicts about +0.76 for every PlayerA state and −0.68 for every PlayerB state (correlation with the result
  only 0.41 / 0.31); the MLP tracks the result (0.91 / 0.92). Faeries is always PlayerA in the training files, so
  the player-name leaf identifies the Faeries seat. Hiding the "PlayerA"/"PlayerB" leaves at inference flips
  PlayerB's mean to +0.74 (MSE 1.05 → 0.49), so the name drives the sign; what is left is a +0.7 bias on both
  seats, i.e. other deck-identity cues (plausibly Faeries cards on the deciding side). The value head learned
  "which deck am I" rather than the board.

## gen0 + gen1 training (`mlp-g01`)

Both test sets are in distribution here (the gen0 mirror shard and every gen1 matchup are in training).

| run | test | value | prio KL | prio acc | non-pass | pass | tgt KL | tgt acc | use acc |
|---|---|---|---|---|---|---|---|---|---|
| gnn | gen1 held-out | 0.056 | 0.170 | 0.702 | 0.321 | 0.958 | 0.155 | 0.751 | 0.814 |
| gnnlegal | gen1 held-out | **0.053** | **0.074** | **0.755** | 0.572 | 0.878 | **0.140** | **0.762** | **0.827** |
| mlp | gen1 held-out | 0.055 | 0.080 | 0.739 | **0.614** | 0.823 | 0.149 | 0.743 | 0.812 |
| mlp256 | gen1 held-out | 0.072 | 0.079 | 0.744 | 0.573 | 0.859 | 0.149 | 0.738 | 0.819 |
| gnn | gen0 mirror | 0.035 | 0.199 | 0.622 | 0.398 | 0.946 | 0.076 | 0.794 | 0.886 |
| gnnlegal | gen0 mirror | 0.037 | **0.051** | **0.798** | 0.824 | 0.760 | **0.059** | **0.801** | 0.885 |
| mlp | gen0 mirror | 0.036 | 0.054 | 0.789 | **0.827** | 0.734 | 0.071 | 0.765 | 0.886 |
| mlp256 | gen0 mirror | 0.044 | 0.055 | 0.785 | 0.813 | 0.744 | 0.073 | 0.756 | 0.886 |

One seed. Same pattern as gen0-only and gen1-only: in distribution, the legal GNN is slightly ahead on priority
KL (0.074 vs 0.080, 0.051 vs 0.054) and clearly ahead on targets (0.762 vs 0.743, 0.801 vs 0.765); value is level.
With gen1 added, value on gen1 drops for both (0.063–0.083 → 0.053–0.055).
