# Architecture experiments (Kaggle, graph-encoder-test)

Data: `Pauper-Faeries/ver4` (v0.3), Kaggle dataset `willwroble/magezero-pauper-faeries-ver4`.
Training: 421K states (gen0, both seats, 8 matchups). Testing: 63.5K states (gen0 mirror, both seats, 800 games).
GPU: Kaggle T4, one run per GPU. Timings compare runs with each other, not with the RTX 4070.
Logs: `results/kaggle/<kernel>/<run>.log`. Kernel: `kaggle/launch.py`.

## Summary so far (after mz-s8)

Seed noise on the val total is about 0.04 (d256: 1.103 vs 1.063), so only differences well above that count.

1. **Dropout is the lever.** 0.25 → 0.1 → 0 gives val total 1.10 → 0.98 → 0.905 at d256 (6 epochs), and 1.116 → 1.025
   at d512. With dropout 0 the val loss still falls at epoch 12 (0.862). The data is not small enough to need it.
2. **Size doesn't matter at dropout 0.25** (d128 1.134, d256 1.06–1.10, d512 1.116, 4 global layers 1.10), so the
   cheaper model wins. d256 has a quarter of d512's parameters and 2.3× its throughput (≈3.9k vs 1.7k states/s on a T4).
3. **Passes:** 1 sequential pass is close to 2 at dropout 0.25, but ≈0.1 worse at dropout 0. `joint` (one shared layer)
   is worse than per-type layers. Bag-of-tokens is degenerate: without edges, nodes of a type are indistinguishable.
4. **Inference:** batch-1 latency is 11–18 ms for every graph model on a T4 and barely moves with size, so small
   batches are overhead-bound. Throughput at large batches scales with size: d128 ≈6.7k, d256 ≈3.9k, d512 ≈1.7k states/s.

**Default (chosen):** `d_model 256, d_ff 512, dropout 0`, 2 sequential passes, 2 global layers. Val total 0.905 on
both seeds. d128 at dropout 0 is within noise (0.920) with 1.75× the throughput.

5. **Graph structure matters.** Flat objects (each node keeps its own leaves, no hierarchy or object links): 1.472.
   Parallel passes (no bottom-up chaining within a pass): 1.143. Both lose most on priority and value.

**Open:** with less dropout the priority loss improves but non-pass recall (top-1 by ability name) does not
(0.35–0.45 vs 0.52–0.60 for base). Recall moves ±0.08 between epochs, so it needs a better metric or more seeds
before reading anything into it. The tie-counting in policy accuracy also inflates degenerate models.

## Runs

| kernel | run | arch | epochs | conclusion |
|---|---|---|---|---|
| mz-pilot | base | default | 300 steps | 0.18 s/step (≈20 min/epoch); inference 17 ms at batch 1, ≈1.9k states/s from batch 128 |
| mz-pilot | joint1 | `local: joint, n_passes: 1` | – | killed: two runs loading data at once ran out of RAM; the launcher now staggers loading |
| mz-s1-edges | base | default (34.7M params) | 6 | val total 1.116 (prio 0.682, tgt 0.317, use 0.059, value 0.058); prio acc 0.658, non-pass 0.518 (0.602 at epoch 5), tgt acc 0.673. Reference. 1,180 s/epoch. Bench: 18.4 ms @1, ≈1.7k states/s @128 |
| mz-s1-edges | bag | `n_passes: 0, global_tokens: all` (5.2M) | 6 | degenerate: priority stuck at uniform (3.476 from epoch 1). Without edges every ABILITY node has the same embedding, so the heads cannot tell nodes apart. Target acc 0.75 is a metric artifact: accuracy counts ties as hits. Bench: 5.7 ms @1 |
| mz-s2-passes | seq1 | `n_passes: 1` (20.0M params) | 6 | val total 1.134 (prio 0.700, tgt 0.315, use 0.061, value 0.059); prio acc 0.637, non-pass 0.462, tgt acc 0.673. Still improving at epoch 6. Bench: 14.7 ms @1, ≈3.0k states/s @512 |
| mz-s2-passes | joint2 | `local: joint` (9.4M params) | 6 | val total 1.320 (prio 0.811, tgt 0.326, value 0.125); non-pass 0.331, tgt acc 0.669. Worse priority and value than seq1. Bench: 6.9 ms @1 (half of seq1), but ≈2.0k states/s @512 |
| mz-s3-size | d256 | `d_model: 256, d_ff: 512` (9.0M) | 6 | val total 1.103 (prio 0.677, tgt 0.312, value **0.044**); non-pass 0.582, tgt acc 0.704. Matches or beats base at a quarter of the size. 670 s/epoch. Bench: 17.1 ms @1, ≈3.9k states/s @512 (2.3× base) |
| mz-s3-size | glob4 | `global_layers: 4` (38.9M) | 6 | val total 1.102 (best 1.092 at epoch 5); non-pass 0.554, tgt acc 0.696. Learns priority faster (epoch-1 prio 0.826 vs 1.020 for base), ends level. 1,520 s/epoch. Bench ≈1.4k states/s |
| mz-s4-reg | drop01 | `dropout: 0.1` (34.7M) | 6 | val total **1.025** (best 1.018 at epoch 5; prio 0.638, tgt 0.291, value 0.040); non-pass 0.519, tgt acc 0.724. Clearly better than base (1.116), beyond epoch noise: less dropout generalizes better on the held-out mirror |
| mz-s4-reg | seq1-e12 | `n_passes: 1`, 12 epochs | 12 | val total 1.071 at epoch 12 (1.137 at 6), still slowly improving; non-pass ≈0.51. More epochs help, less than lower dropout |
| mz-s5-small | d128 | `d_model: 128, d_ff: 256` (2.4M) | 6 | val total 1.134 (prio 0.713, tgt 0.318, value 0.045); non-pass 0.490, tgt acc 0.664. Near base at 1/15 the size. Bench: 16.6 ms @1, ≈6.8k states/s @512 (4× base) |
| mz-s5-small | d256-seq1 | `d_model: 256, d_ff: 512, n_passes: 1` (5.3M) | 6 | val total 1.131, still falling fast at epoch 6 (prio 0.861 → 0.744 → 0.687); non-pass 0.41. 475 s/epoch. Bench: 11.3 ms @1, ≈5.8k states/s @512 |
| mz-s6-d256-drop | d256-drop01 | `d_model: 256, d_ff: 512, dropout: 0.1` | 6 | val total 0.976 (prio 0.591, tgt 0.291, value 0.040); tgt acc 0.728, non-pass 0.42 (noisy: 0.36–0.53 across epochs) |
| mz-s6-d256-drop | d256-drop0 | `d_model: 256, d_ff: 512, dropout: 0` (9.0M) | 6 | val total **0.905**, falling every epoch (prio 0.542, tgt 0.275, value 0.039); tgt acc **0.740**, non-pass 0.42–0.49. Best so far. Train/val priority gap (0.38 vs 0.54) is the mirror shift, not overfitting. Bench ≈3.9k states/s @512 |
| mz-s7-noise | d256-seed1 | d256 again, `--seed 1` | 6 | val total 1.063 vs 1.103 for seed 0: **seed noise ≈0.04**. Size differences (1.06–1.13) are within noise; the dropout gains (0.09–0.2) are not |
| mz-s7-noise | d128-drop01 | `d_model: 128, d_ff: 256, dropout: 0.1` (2.4M) | 6 | val total 1.022 (vs 1.134 at dropout 0.25); non-pass 0.537, tgt acc 0.703, value 0.040. Bench ≈6.6k states/s @512 |
| mz-s8-drop0 | d256-drop0-e12 | `d_model: 256, d_ff: 512, dropout: 0`, 12 epochs | 12 | val total **0.862** at epoch 12 (0.911 at 6, matching s6's 0.905), still slowly improving; prio 0.529, tgt 0.247, value 0.039; tgt acc **0.755**, non-pass 0.35–0.45 |
| mz-s8-drop0 | d256-seq1-drop0 | `d_model: 256, d_ff: 512, n_passes: 1, dropout: 0` (5.3M) | 6 | val total 1.006 vs 0.905 with 2 passes: without dropout the second pass is worth ≈0.1. Bench: 11.7 ms @1, ≈6.0k states/s @512 |
| mz-s9-flat | flat | `local: flat, d_model: 256, d_ff: 512, dropout: 0` (1.6M) | 6 | val total **1.472** vs 0.905 for the full graph (prio 0.913 vs 0.542, tgt 0.399 vs 0.275, value 0.102 vs 0.039); non-pass ≈0.2. Fair bag-of-objects ablation: each node keeps its own leaves, but without the hierarchy it also loses zone and owner, which likely explains value. Graph structure matters. Bench: 6.0 ms @1, ≈9.9k states/s @512 |
| mz-s9-flat | d128-drop0 | `d_model: 128, d_ff: 256, dropout: 0` (2.4M) | 6 | val total 0.920, within noise of d256-drop0 (0.905); tgt acc 0.727, value 0.037. Bench ≈6.8k states/s @512 (1.75× d256) |
| mz-s10-parallel | parallel | `local: parallel` at the default (d256, FF 512, dropout 0) | 6 | val total **1.143** vs 0.905 sequential (prio 0.745 vs 0.548, value 0.060 vs 0.039, non-pass 0.28 vs 0.38–0.42). Sequential carries information up the whole hierarchy in one pass; parallel moves it one level per pass. Not worth fusing for speed |
| mz-s10-parallel | seq-seed1 | default, `--seed 1` | 6 | val total 0.905, identical to seed 0 (0.905): the default is stable across seeds |

## v0.1 vs v0.2 (UWTempo, `willwroble/mz-uwtempo-v01-v02`)

Matched opponents: train on one ≈100-game shard per Standard16 matchup (11 decks), test on MonoW (≈100 games).
v0.1 = `main`'s flat transformer and training code, unchanged (`ver30`, 114K states). v0.2 = GNN on `ver31` (144K
states), at its v0.2 settings (d512, dropout 0.25) and at the new default. 10 epochs, batch 64, Adam 1e-4 for all.
Policy losses are not comparable (128 action slots vs a softmax over nodes); value MSE and top-1 accuracies are.
Always-pass priority baseline: 0.79 on the v0.1 test set, 0.76 on the v0.2 one.

| run | value MSE (epochs 6–10) | priority acc (vs always-pass) | target acc | choose_use acc |
|---|---|---|---|---|
| mz-uw-v01 v01 (flat, v0.1) | 0.069–0.095, final 0.083 | 0.83 (+0.04) | 0.29–0.35 | 0.80–0.83 |
| mz-uw-v01-v02 v02 (GNN, d512 dropout 0.25) | 0.092–0.109, final 0.092 | 0.35–0.69, final 0.65 (below baseline; unstable) | ≈0.60 | ≈0.90 |
| mz-uw-v01 v02-default (GNN, d256 dropout 0) | 0.072–0.084, final 0.080 | 0.79–0.80 (+0.04) | 0.66–0.67 | ≈0.89–0.90 |

Conclusion: priority is the fairest comparison and comes out even (both ≈4 points above always-pass). Value and
choose_use labels come from search (TD targets), and `ver30` spans several generations; the subset here is the
earliest session per matchup (370–401, assumed gen0), but engine, sim budget and TD discount still differ, so
value is not comparable. Targeting favours the GNN (≈0.66 vs ≈0.33), but v0.1 picks from 128 hashed slots
while the GNN picks the object, so the metrics differ too. At the v0.2 settings the GNN's priority was unstable
on this out-of-distribution opponent; the new default fixes that.

## v0.2 vs v0.3 (Pauper-Faeries, `willwroble/mz-faeries-v02-v03`)

Same model (the default), same opponents, gen0 only, both seats, mirror held out. v0.2 = `ver3` gen0 (1,238 games,
245K states; test: gen0 mirror, 96 games). v0.3 = `ver4` gen0 non-mirror shards cut to their first 100 games with
`util/cut_games.py` (1,400 games, 113K states; test: gen0 mirror PlayerA, 400 games). v0.2 records about twice
as many states per game. Kernel: mz-fae-v02-v03.

| run | always-pass baseline | priority acc (epoch 6) | non-pass recall (epochs 4–6) | target acc | value MSE |
|---|---|---|---|---|---|
| v02 (`ver3`) | 0.778 (3,150 of 14,180 non-pass) | 0.833 (+0.055) | 0.32–0.38 | 0.74–0.77 | 0.064 |
| v03 (`ver4`, first 100 games) | 0.412 (8,974 of 15,251 non-pass) | 0.539 (+0.127) | 0.17–0.28 | 0.64–0.66 | 0.056 |

Conclusion: no evidence that v0.3 learns policy better at matched game counts. v0.3's test set is a much harder
priority task (59% non-pass vs 22%); it gains more over always-pass, but its non-pass recall is lower, and it trained
on half as many states. Losses and target accuracy aren't comparable across encodings (different candidate sets).
With all of `ver4` (421K states) v0.3 reaches non-pass 0.42–0.52, so data volume matters more than the version here.

## Data diversity vs volume (Faeries vs Standard Mono, gen0, `willwroble/mz-faeries-diversity`)

Every arm: Faeries seat only, ≈1,600 games (≈95K states), default model, 6 epochs, seed 0. In-distribution tests are
games 1,600–2,000 of the MonoB and MonoG shards, which no arm trained on. MonoW (400 games) and the Pauper mirror
(Faeries seat, 400 games) are unseen opponents. Always-pass baselines: MonoB 0.44, MonoG 0.47, MonoW 0.39, mirror 0.41.
Kernels mz-div-narrow, mz-div-wide. (bench.py failed afterwards because the arms have no per-arm testing folder;
evaluations are unaffected.)

Priority loss / non-pass recall / target loss / value MSE (bold: in distribution for that arm):

| arm | MonoB held-out | MonoG held-out | MonoW | Pauper mirror |
|---|---|---|---|---|
| narrow-B (B × 1,600) | **0.416 / 0.49 / 0.217 / 0.045** | 0.519 / 0.44 / 0.461 / 0.146 | 0.775 / 0.37 / 0.307 / 0.167 | 0.743 / 0.30 / 0.674 / 0.676 |
| narrow-G (G × 1,600) | 0.544 / 0.54 / 0.350 / 0.092 | **0.400 / 0.57 / 0.246 / 0.032** | 0.784 / 0.53 / 0.314 / 0.081 | 0.753 / 0.46 / 0.502 / 0.701 |
| 2 decks (B, G × 800) | **0.443 / 0.54 / 0.231 / 0.046** | **0.433 / 0.49 / 0.252 / 0.036** | 0.760 / 0.45 / 0.366 / 0.083 | 0.794 / 0.33 / 0.586 / 0.668 |
| 4 decks (B, G, R, U × 400) | **0.531 / 0.42 / 0.245 / 0.051** | **0.510 / 0.40 / 0.270 / 0.039** | 1.073 / 0.30 / 0.316 / 0.087 | 1.219 / 0.21 / 0.437 / 0.594 |

Training priority loss at epoch 6: narrow-B 0.430, narrow-G 0.400, deck2 0.417, deck4 0.489, all still falling.

Conclusion (one seed, 6 epochs):
- In distribution, more games of the same matchup wins on loss, as expected. Halving them (2 decks) costs little
  (≈0.03 priority loss); quartering them (4 decks) costs ≈0.11. No sign that diversity makes known-opponent
  predictions more robust.
- Out of distribution, 2 decks matches the better single deck; 4 decks is worst on priority for both unseen
  opponents. That is the opposite of the expected generalization gain, and 4 decks is also the least converged
  arm (highest training loss), so the likeliest reading is that four opponents at this budget need more epochs.
- Value on the mirror is poor for every arm (0.59–0.70): models trained only on lopsided Mono games carry a
  "losing" prior into an even matchup. Target accuracy on the mirror is the one place 4 decks helps (0.64 vs 0.54).
