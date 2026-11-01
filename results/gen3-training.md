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

## F: E's recipe one generation on (gen4 candidate), GNN and MLP (Nov 1)

`runpod/splits/F.txt`: E's blocks plus the gen3 games (ver13) in four blocks, holdout 5 each; 4.37M training states
per epoch. GNN warm-started from ver13 (lr 3e-5), MLP from E-mlp (lr 3e-4), 3 epochs, legal-only loss.
Logs: `/workspace/runs/gen4-F-{gnn,mlp}`.

GNN / MLP per block, final epoch (gap = MLP − GNN; E's gap in brackets where the block existed):

| block | value | gap (E) | prio KL | gap (E) | prio top-1 | gap (E) | target top-1 | gap (E) |
|---|---|---|---|---|---|---|---|---|
| gen0 | 0.036 / 0.041 | .005 (.003) | 0.061 / 0.073 | .012 (.013) | 0.813 / 0.791 | .022 (.025) | 0.833 / 0.798 | .035 (.042) |
| gen1-reg | 0.038 / 0.042 | .004 (.002) | 0.056 / 0.066 | .010 (.008) | 0.794 / 0.772 | .022 (.020) | 0.800 / 0.772 | .028 (.026) |
| gen1-selfplay | 0.032 / 0.035 | .003 (.005) | 0.053 / 0.061 | .008 (.007) | 0.788 / 0.768 | .020 (.019) | 0.795 / 0.759 | .036 (.031) |
| gen2-offline | 0.039 / 0.043 | .004 (.002) | 0.048 / 0.056 | .008 (.008) | 0.800 / 0.777 | .023 (.022) | 0.806 / 0.777 | .029 (.032) |
| gen2-vsgen1 | 0.038 / 0.042 | .004 (.009) | 0.051 / 0.058 | .007 (.005) | 0.786 / 0.761 | .025 (.023) | 0.798 / 0.769 | .029 (.032) |
| gen2-selfplay | 0.042 / 0.045 | .003 (.009) | 0.057 / 0.065 | .008 (.007) | 0.770 / 0.746 | .024 (.024) | 0.800 / 0.769 | .031 (.031) |
| gen3-offline | 0.035 / 0.042 | .007 | 0.045 / 0.054 | .009 | 0.809 / 0.788 | .021 | 0.813 / 0.785 | .028 |
| gen3-vsgen1 | 0.041 / 0.044 | .003 | 0.050 / 0.057 | .007 | 0.793 / 0.774 | .019 | 0.801 / 0.769 | .032 |
| gen3-vsgen2 | 0.045 / 0.047 | .002 | 0.053 / 0.062 | .009 | 0.777 / 0.753 | .024 | 0.810 / 0.780 | .030 |
| gen3-selfplay | 0.045 / 0.047 | .002 | 0.055 / 0.063 | .008 | 0.777 / 0.752 | .025 | 0.810 / 0.777 | .033 |
| pimc (val) | 0.055 / 0.122 | .067 (.053) | 0.086 / 0.106 | | 0.690 / 0.660 | | 0.771 / 0.740 | |
| ub (val) | 0.046 / 0.044 | −.002 (−.009) | 0.079 / 0.077 | | 0.747 / 0.735 | | 0.728 / 0.722 | |

Conclusion: the GNN–MLP gaps do not grow with the extra generation. Policy and target gaps are flat (priority KL
≈0.008, top-1 ≈0.02, targets ≈0.03 on every block, as in E); the value gap on the gen2 blocks shrank (≈0.004 vs up
to 0.009 in E) because the MLP gained more from the new data, and on the new gen3 blocks it is 0.002–0.007. The MLP
still fails on PIMC states. F-gnn improves on E-gnn mainly on gen1-reg, gen2-offline and PIMC value.
