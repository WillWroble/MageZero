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

### F in play: vs the offline search (Nov 1)

Faeries = F-gnn (ver14) or F-mlp, value only, 200 games per deck (`runpod/configs/norm-*.yml`, 4090 pods,
`/workspace/runs/normF4*`). ver13 and v4 from `generations.md`.

| deck | F-gnn (ver14) | F-mlp | ver13 | v4 |
|---|---|---|---|---|
| mirror | 72.5% | 73.0% | 72.0% | 72.0% |
| Burn | 62.5% | 59.5% | 69.5% | 66.5% |
| MonoB | 26.0% | 28.0% | 31.2% | 31.6% |
| MonoW | 47.0% | 41.0% | 53.5% | 57.5% |
| Rally | 27.0% | 26.5% | 27.0% | 30.0% |
| MonoG | 20.5% | 24.0% | 25.5% | 16.0% |
| mean | 42.6% | 42.0% | 46.5% | 45.6% |

Conclusion: F-gnn (epoch 3) is below its warm start ver13 on Burn, MonoB, MonoW and MonoG (−5 to −7 points, ≈1.5–2 SE
each) and level with F-mlp; with its value loss best at epoch 1 on the model-vs-model blocks, F-gnn overfit. The
F2 retrains (per-10K-step checkpoints) are for picking each model's best point.

## F2: F retrained with a validation and checkpoint every 10K steps (Nov 1–2)

Same runs as F (same split, warm starts, lr, seed) plus `--val-every 10000`, 7 points per epoch; checkpoints
`model-e<N>-s<step>.pt.gz` in `/workspace/runs/gen4-F2-{gnn,mlp}/F`. Value loss averaged over the five
model-vs-model blocks (gen2-vsgen1, gen2-selfplay, gen3-vsgen1/vsgen2/selfplay):

| point | F2-gnn | F2-mlp | F-gnn |
|---|---|---|---|
| e1 10K | .0428 | .0474 | |
| e1 end | .0394 | .0464 | .0392 |
| e2 end | .0388 | .0472 | .0398 |
| e3 30K | **.0384** | .0456 | |
| e3 end | .0386 | .0448 | .0422 |
| e4 50K | | .0444 | |
| e6 60K | | **.0436** | |
| e6 end | | .0444 | |

F2 matches F through epoch 1 and diverges after: every DataLoader iteration (the val loaders' too) draws from
torch's global RNG, so F2's extra validations changed the shuffle order of epochs 2–3. F's epoch-3 rise is
therefore one data order, not a property of 3 epochs: run-to-run noise is ≈0.004 here. The MLP is flat from epoch 3
(≈.0445, ±.001 between points); its training loss barely moves (value .043 → .040 over 6 epochs), so it is
capacity-bound rather than slow. At each model's best point the GNN leads by .003–.007 value, .006–.011 priority
loss and .008–.015 target loss on every block except ub (MLP better) and PIMC (MLP value ≈2×).

F2-gnn e3 end (ver15) and F2-mlp e6 end vs the offline search (200 games, value only,
`/workspace/runs/normF2e3-gnn-*`, `normF2-{gnn,mlp}-*`):

| deck | F2-gnn (ver15) | F2-mlp | F-gnn (ver14) | F-mlp | ver13 | v4 |
|---|---|---|---|---|---|---|
| mirror | 72.0% | 73.0% | 72.5% | 73.0% | 72.0% | 72.0% |
| Burn | 68.5% | 59.5% | 62.5% | 59.5% | 69.5% | 66.5% |
| MonoB | 36.0% | 29.5% | 26.0% | 28.0% | 31.2% | 31.6% |
| MonoW | 53.5% | 45.5% | 47.0% | 41.0% | 53.5% | 57.5% |
| Rally | 31.0% | 17.0% | 27.0% | 26.5% | 27.0% | 30.0% |
| MonoG | 32.0% | 22.5% | 20.5% | 24.0% | 25.5% | 16.0% |
| mean | 48.8% | 41.2% | 42.6% | 42.0% | 46.5% | 45.6% |

Conclusion: the second F-gnn run (ver15) is the best model on these decks (+2.3 over ver13, +6.2 over ver14), so
ver14's weakness came with its data order, not the recipe; single-run comparisons carry this much noise. The
F2 GNN–MLP gap (value .0058, 7.6 points of win rate) is far larger than the F gap (value .0026, 0.6 points):
small value-loss differences carry large win-rate differences. F2-mlp ≈ F-mlp (same value loss) except Rally
(17.0% vs 26.5%, ≈2.7 SE).

### F2-mlp256: is the MLP capacity-bound? (Nov 2)

`NetMLP` d256, 2 blocks (2.26M params vs 0.21M for d64 and 9.11M for the GNN), cold start on the F split,
lr 3e-4, legal-only loss, names dropped, 6 epochs, `--val-every 20000`; `/workspace/runs/gen4-F2-mlp256/F`.
Mean over the five model-vs-model blocks:

| model | point | value | priority | target | pimc value | ub value |
|---|---|---|---|---|---|---|
| mlp256 | e6 end | .0428 | .0584 | .0496 | .118 | .039 |
| mlp d64 (F2) | e6 end | .0444 | .0604 | .0512 | .114 | .044 |
| gnn (F2, ver15) | e3 end | .0386 | .0532 | .0424 | .060 | .050 |

mlp256 value per epoch end: .0516, .0478, .0458, .0446, .0428, .0428 (flat over epoch 6).

Conclusion: at 4.37M states width helps the MLP (unlike on gen0 alone): ≈.0015 on value, .002 on priority and
target over the d64. It closes about a third of the value gap to the GNN and little of the policy gap, and does
not fix PIMC value, so most of the GNN's lead is architectural, not size. Not tested in play.
