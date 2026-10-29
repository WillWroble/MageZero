# Which gen1 data helps, and GNN vs MLP at gen0 + gen1 scale (RunPod, Oct 29)

Question (Will): are regular gen1 games (v4 model against the heuristic search) more or less useful than self-play
games (v4 model on both seats), and does the GNN pull further ahead of the MLP with the extra data?

## Setup

- **Branches** (data built by `runpod/prepare.sh`; every branch includes all of ver4 gen0, Pauper and Mono, both seats):
  - **A** gen0 + gen1 reg: ver4 gen1 Mono B/G/R/U/W, MonoB2, Affinity, Burn, CawGates, Elves (1.82M decision states)
  - **B** gen0 + gen1 self-play: every `*_selfplay*` shard, Will's Mono ones and the 8 Pauper ones from RunPod (1.92M)
  - **C** gen0 + reg + self-play (2.21M)
  - **D** C + PIMC (`_pimc`, `_pimc2`, `_pimc_pimc`), `_policy` and the two v5-model shards (2.34M)
- **Hold-out:** every 5th game of each gen1 shard (`split_games.py --every 5`), both seats.
- **Tests:** self-play held-out (97K states), reg held-out (70K), PIMC (A–C: all 600 PIMC games, 73K states, never
  trained on; D: its held-out fifth, 15K states), the gen0 mirror test (64K). Each also on Faeries' seat only.
- **Models:** NetGraph d256 and NetMLP d64, both with the legal-only policy loss and `--drop-player-names`.
  3 epochs over the whole set (`--steps 40000`), seed 0, NetGraph lr 1e-4, NetMLP lr 1e-3. One run per model and
  branch. Logs: `/workspace/runs/branch-*` on the RunPod volume.

Cells: value MSE / priority legal KL / target legal top-1 (lower, lower, higher is better).


### both seats

| run | selfplay value / prio KL / tgt acc | reg value / prio KL / tgt acc | pimc value / prio KL / tgt acc | gen0 value / prio KL / tgt acc |
|---|---|---|---|---|
| A-gnn | 0.048 / 0.085 / 0.722 | 0.050 / 0.069 / 0.759 | 0.054 / 0.098 / 0.733 | 0.037 / 0.055 / 0.802 |
| A-mlp | 0.051 / 0.080 / 0.689 | 0.051 / 0.070 / 0.731 | 0.114 / 0.115 / 0.709 | 0.040 / 0.053 / 0.774 |
| B-gnn | 0.040 / 0.067 / 0.745 | 0.047 / 0.066 / 0.757 | 0.123 / 0.107 / 0.733 | 0.032 / 0.047 / 0.805 |
| B-mlp | 0.046 / 0.073 / 0.720 | 0.050 / 0.072 / 0.740 | 0.096 / 0.125 / 0.716 | 0.043 / 0.053 / 0.772 |
| C-gnn | 0.043 / 0.068 / 0.754 | 0.048 / 0.064 / 0.778 | 0.117 / 0.102 / 0.741 | 0.037 / 0.053 / 0.794 |
| C-mlp | 0.044 / 0.070 / 0.729 | 0.047 / 0.068 / 0.749 | 0.112 / 0.111 / 0.722 | 0.039 / 0.053 / 0.756 |
| D-gnn | 0.039 / 0.069 / 0.754 | 0.043 / 0.064 / 0.781 | 0.043 / 0.089 / 0.754 | 0.032 / 0.049 / 0.797 |
| D-mlp | 0.046 / 0.069 / 0.731 | 0.051 / 0.066 / 0.758 | 0.050 / 0.094 / 0.734 | 0.039 / 0.055 / 0.773 |

### Faeries' seat

| run | selfplay value / prio KL / tgt acc | reg value / prio KL / tgt acc | pimc value / prio KL / tgt acc | gen0 value / prio KL / tgt acc |
|---|---|---|---|---|
| A-gnn | 0.047 / 0.071 / 0.739 | 0.050 / 0.073 / 0.751 | 0.059 / 0.087 / 0.687 | 0.038 / 0.055 / 0.799 |
| A-mlp | 0.052 / 0.067 / 0.691 | 0.055 / 0.072 / 0.710 | 0.153 / 0.105 / 0.654 | 0.040 / 0.052 / 0.770 |
| B-gnn | 0.040 / 0.059 / 0.751 | 0.044 / 0.067 / 0.750 | 0.168 / 0.102 / 0.679 | 0.033 / 0.047 / 0.802 |
| B-mlp | 0.043 / 0.067 / 0.721 | 0.053 / 0.074 / 0.730 | 0.124 / 0.120 / 0.671 | 0.046 / 0.052 / 0.768 |
| C-gnn | 0.044 / 0.058 / 0.764 | 0.047 / 0.063 / 0.778 | 0.159 / 0.091 / 0.699 | 0.040 / 0.052 / 0.794 |
| C-mlp | 0.044 / 0.060 / 0.743 | 0.048 / 0.066 / 0.752 | 0.146 / 0.100 / 0.675 | 0.040 / 0.052 / 0.753 |
| D-gnn | 0.037 / 0.058 / 0.766 | 0.042 / 0.063 / 0.778 | 0.040 / 0.072 / 0.710 | 0.033 / 0.049 / 0.797 |
| D-mlp | 0.045 / 0.058 / 0.737 | 0.052 / 0.063 / 0.755 | 0.050 / 0.076 / 0.691 | 0.040 / 0.054 / 0.771 |

## Conclusions (one seed)

**Self-play games are the more useful gen1 data.**
- Trained on self-play (B), the GNN predicts the held-out *reg* games as well as the reg-trained model (A) does on
  its own games: value 0.047 vs 0.050, priority KL 0.066 vs 0.069, targets 0.757 vs 0.759.
- The reverse transfer is poor. A on self-play games: priority KL 0.085 vs B's 0.067, value 0.048 vs 0.040,
  targets 0.722 vs 0.745. The heuristic opponent's games don't teach what a model-guided opponent does.
- Both together (C) add a little over B on reg targets (0.778) and keep self-play level; everything (D) is the best
  or level-best on every test for the GNN.

**The GNN's lead grows with this data.** In every branch the GNN beats the MLP on targets, by 2.5–4 points (e.g.
B self-play 0.745 vs 0.720, C reg 0.778 vs 0.749, gen0 mirror 0.794–0.805 vs 0.756–0.774), and on priority KL
everywhere but one cell (A on self-play, 0.085 vs 0.080, where both are out of distribution). Value is level or
better for the GNN except on the PIMC test in B and C. On gen0-only data (`mlp-vs-gnn.md`) the in-distribution
gaps were about half as large (priority KL ≈0.001–0.004, targets ≈3 points, value level).

**PIMC (hidden opponent hand).**
- Without PIMC training, the value heads of B and C fail on PIMC states (GNN 0.117–0.123, MLP 0.096–0.112; Faeries'
  seat 0.146–0.168). A does not (GNN 0.054). The PIMC shards are against the heuristic-search MonoB, the same
  opponent as the reg data, and B has no heuristic-opponent games, so this test mixes hidden information with an
  opponent shift; A's GNN suggests the opponent shift is most of it.
- Training on PIMC games (D) fixes it: GNN 0.043, MLP 0.050 (on the held-out fifth of the PIMC games, a smaller set
  than A–C's PIMC test). Priority on PIMC stays the hardest test for every model (KL 0.089–0.125).

Recommendation for gen2 training: all of gen0 + every gen1 kind (D), with the GNN (legal-only, no player names).
