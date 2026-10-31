# gen3 training: A–E (GNN and MLP), Oct 31

Splits in `runpod/splits/{A..E}.txt` (per-section holdout 5: every run is scored on the same held-out games of every
block). All use the legal-only policy loss and TD 0.92 labels, seed 0. Logs: `/workspace/runs/gen3-{A..E}-{gnn,mlp}`.

| run | data | start | epochs | lr GNN / MLP |
|---|---|---|---|---|
| A | gen2 | scratch | 6 | 1e-4 / 1e-3 |
| B | gen1 + gen2 | scratch | 6 | 1e-4 / 1e-3 |
| C | gen0 + gen1 + gen2 | scratch | 6 | 1e-4 / 1e-3 |
| D | gen0 + gen1 + gen2 | v4 (GNN, names kept), gen0 MLP | 3 | 1e-4 / 1e-3 |
| E | gen0 + gen1 + gen2 | as D | 3 | 3e-5 / 3e-4 |

Value MSE per held-out block (final epoch; C-gnn's gen2-vsgen1 test was still running):

| run | gen0 | gen1-reg | gen1-selfplay | gen2-offline | gen2-vsgen1 | gen2-selfplay | pimc | policy | gen0-test | ub |
|---|---|---|---|---|---|---|---|---|---|---|
| A-gnn | 0.060 | 0.056 | 0.038 | 0.048 | 0.040 | 0.045 | 0.047 | 0.064 | 0.037 | 0.049 |
| A-mlp | 0.064 | 0.056 | 0.041 | 0.053 | 0.046 | 0.048 | 0.088 | 0.067 | 0.041 | 0.046 |
| B-gnn | 0.051 | 0.045 | 0.035 | 0.044 | 0.041 | 0.043 | 0.055 | 0.057 | 0.038 | 0.050 |
| B-mlp | 0.055 | 0.048 | 0.039 | 0.046 | 0.044 | 0.047 | 0.112 | 0.059 | 0.036 | 0.043 |
| C-gnn | 0.039 | 0.040 | 0.032 | 0.043 | – | 0.045 | 0.065 | 0.055 | 0.033 | 0.044 |
| C-mlp | 0.040 | 0.044 | 0.038 | 0.046 | 0.047 | 0.050 | 0.121 | 0.061 | 0.035 | 0.048 |
| D-gnn | 0.039 | 0.044 | 0.034 | 0.045 | 0.041 | 0.044 | 0.075 | 0.058 | 0.034 | 0.040 |
| D-mlp | 0.040 | 0.045 | 0.037 | 0.045 | 0.047 | 0.051 | 0.119 | 0.060 | 0.037 | 0.042 |
| E-gnn | **0.036** | 0.041 | **0.032** | **0.042** | **0.038** | **0.041** | 0.065 | **0.055** | **0.031** | 0.051 |
| E-mlp | 0.039 | 0.043 | 0.037 | 0.044 | 0.047 | 0.050 | 0.118 | 0.058 | 0.035 | 0.042 |

Priority legal top-1 / target top-1 on gen2-offline: A 0.779 / 0.784 (MLP 0.772 / 0.749), B 0.780 / 0.796 (0.774 /
0.759), C 0.786 / 0.796 (0.776 / 0.766), D 0.793 / 0.801 (0.776 / 0.773), E **0.798 / 0.805** (0.776 / 0.773).

Conclusions (held-out metrics only; these have not predicted play in this project):
- E (warm start from v4, lr 3e-5) is the best GNN on nearly every block; D (same start, lr 1e-4) is no better than C
  (scratch). Warm starting helps only with the lower learning rate.
- Keeping the buffer costs nothing on gen2: C/D/E match or beat A and B on the gen2 blocks and are far better on
  gen0/gen1. A (gen2 only) is worst everywhere but gen2.
- GNN over MLP: value 0.002–0.009 better on the gen2 blocks, targets +3–5 points, priority +1–2 points. The MLP's
  value fails on PIMC states (0.118 vs 0.065) and is better on the unseen Standard16-UB opponent (0.042 vs 0.051).
- All of A–E use the legal-only loss, which the ver7/ver12 games now implicate in worse play.
