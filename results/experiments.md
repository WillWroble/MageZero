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

## Combat: v0.2 vs v0.3 (target head per step, `util/step_accuracy.py`)

Checkpoints from mz-fae-v02-v03 on their own gen0 mirror. Attacks and blocks are CHOOSE_TARGET decisions in
DECLARE_ATTACKERS / DECLARE_BLOCKERS. "Random" = expected top-1 accuracy of picking uniformly among the legal
candidates; the model column restricts its argmax to the legal candidates too.

| version | step | candidates/state | random | model | gain |
|---|---|---|---|---|---|
| v0.2 | attacks | 1.2 | 0.883 | 0.932 | +0.05 |
| v0.2 | blocks | 1.6 | 0.772 | 0.788 | +0.02 |
| v0.3 | attacks | 1.9 | 0.528 | 0.779 | **+0.25** |
| v0.3 | blocks | 2.2 | 0.580 | 0.641 | +0.06 |

Conclusion: most v0.2 combat decisions were not real choices (≈1 legal option). v0.3 makes attacking a real
decision and the model learns it well; blocking is still close to random.

### Diversity at 12 epochs (mz-div-e12)

Each arm continued from its 6-epoch checkpoint for 6 more epochs (Adam state restarts at epoch 7, the same for all
arms). 0.65 GPU hours. Training priority loss at epoch 12: narrow-B 0.396, narrow-G 0.366, deck2 0.384, deck4 0.417
(deck4 still falling fastest). Priority loss / non-pass recall / priority legal accuracy / target legal accuracy /
value MSE (bold: in distribution):

| arm | MonoB held-out | MonoG held-out | MonoW | Pauper mirror |
|---|---|---|---|---|
| narrow-B | **0.400 / 0.50 / 0.700 / 0.803 / 0.048** | 0.481 / 0.45 / 0.694 / 0.677 / 0.352 | 0.820 / 0.40 / 0.637 / 0.683 / 0.385 | 0.706 / 0.30 / 0.579 / 0.622 / 0.806 |
| narrow-G | 0.637 / 0.39 / 0.646 / 0.722 / 0.112 | **0.382 / 0.48 / 0.706 / 0.784 / 0.030** | 0.959 / 0.40 / 0.626 / 0.748 / 0.084 | 0.788 / 0.33 / 0.583 / 0.670 / 0.660 |
| 2 decks | **0.407 / 0.54 / 0.719 / 0.797 / 0.050** | **0.388 / 0.53 / 0.726 / 0.792 / 0.036** | 0.789 / 0.47 / 0.662 / 0.756 / 0.068 | 0.679 / 0.35 / 0.602 / 0.593 / 0.610 |
| 4 decks | **0.432 / 0.54 / 0.710 / 0.777 / 0.051** | **0.412 / 0.53 / 0.724 / 0.778 / 0.037** | 1.098 / 0.44 / 0.649 / 0.722 / 0.085 | 1.035 / 0.27 / 0.569 / 0.714 / 0.501 |

Conclusion (one seed):
- In distribution, the loss gap closed to ≈0.01 (2 decks) and ≈0.03 (4 decks) behind the narrow arm, with a half
  and a quarter of its games. On top-1 decisions the diverse arms are level or better: non-pass recall 0.54 vs
  0.50 (MonoB) and 0.53 vs 0.48 (MonoG), priority legal accuracy 0.71–0.73 vs 0.70–0.71. Diverse data did not
  make known-opponent predictions worse, and the decisions came out slightly more robust.
- Value is where narrow data breaks: narrow-B's value on unseen Mono decks is 0.35–0.39 MSE vs 0.04–0.09 for
  every arm that saw two or more decks.
- 2 vs 4 decks on unseen opponents is mixed: 2 decks has the better priority head (loss 0.68–0.79 vs 1.0–1.1,
  top-1 0.01–0.08 better), 4 decks the better mirror targets (0.714 vs 0.593 legal accuracy) and mirror value.
  4 decks' priority is poorly calibrated rather than much less accurate, it is the least converged arm, and it
  also changes composition (adds MonoR, MonoU). One seed: no clear winner between the two.

### Diversity, second seed (mz-div-seed1)

Same arms, `--seed 1`, 12 epochs straight (no restart), all four arms at once (two per GPU), 0.93 GPU hours.
Priority loss / non-pass recall / priority legal accuracy / target legal accuracy / value MSE:

| arm | MonoB held-out | MonoG held-out | MonoW | Pauper mirror |
|---|---|---|---|---|
| narrow-B | **0.404 / 0.59 / 0.736 / 0.800 / 0.042** | 0.461 / 0.58 / 0.739 / 0.698 / 0.205 | 0.770 / 0.51 / 0.687 / 0.712 / 0.259 | 0.961 / 0.36 / 0.599 / 0.583 / 0.723 |
| narrow-G | 0.646 / 0.55 / 0.712 / 0.714 / 0.371 | **0.394 / 0.58 / 0.745 / 0.787 / 0.033** | 0.980 / 0.55 / 0.698 / 0.752 / 0.271 | 0.852 / 0.42 / 0.629 / 0.714 / 0.894 |
| 2 decks | **0.411 / 0.56 / 0.724 / 0.788 / 0.044** | **0.394 / 0.54 / 0.732 / 0.790 / 0.035** | 1.349 / 0.39 / 0.620 / 0.752 / 0.100 | 2.499 / 0.25 / 0.550 / 0.693 / 0.690 |
| 4 decks | **0.403 / 0.57 / 0.716 / 0.753 / 0.048** | **0.390 / 0.55 / 0.730 / 0.756 / 0.036** | 0.693 / 0.54 / 0.686 / 0.727 / 0.074 | 0.952 / 0.30 / 0.578 / 0.707 / 0.510 |

Conclusion across both seeds:
- **Value needs several training opponents (clean).** Single-opponent arms: 0.21–0.39 value MSE on unseen Mono decks
  in 4 of 4 cases on seed 1 and 2 of 4 on seed 0. Multi-deck arms: 0.07–0.10 on both seeds.
- **No in-distribution cost and no in-distribution gain (clean null).** At 12 epochs, 2 and 4 decks match the narrow
  arm's priority loss on its own opponent (within 0.01–0.03) with a half or a quarter of its games. The top-1
  differences flip sign between seeds.
- **Priority on unseen opponents is dominated by seed noise.** Rankings flip between seeds (MonoW: deck4 1.098 →
  0.693, deck2 0.789 → 1.349; mirror: deck2 0.679 → 2.499). The priority head's out-of-distribution calibration is
  unstable; diversity's effect on it can't be measured at this noise level.

## Inference: index work on the CPU (`NetGraph.plan`, mz-bench-plan)

`NetGraph.plan(g)` holds the forward pass's graph-structure index work (≈20 nonzero/mask/max calls, each a
CPU–GPU sync when done on the GPU). Outputs are bit-for-bit identical either way. T4, ver4 checkpoint, gen1 MonoB
states, server path (CPU batch → GPU → outputs back), mean of 20:

| batch | plan on GPU | plan on CPU |
|---|---|---|
| 1 | 25.4 ms | 19.7 ms (−22%) |
| 8 | 24.2 ms | 23.8 ms |
| 32 | 28.9 ms | 28.1 ms |
| 128 | 31.4 ms | 38.2 ms (+22%) |
| 512 | 112.8 ms | 145.8 ms (+29%) |

Conclusion: removing the syncs saves ≈5 ms at small batches; at large batches the CPU index work (on Kaggle's slow
CPU, not overlapped) costs more than the syncs did. The server uses the CPU plan up to `CPU_PLAN_MAX = 32` states.

## Inference: subtree cache of local-layer outputs (`cache.py`, mz-bench-cache)

A node's local-layer outputs depend only on the subgraph below it, and XMage already hashes every node's subtree
(`FeatureGraph.Node.hash`, 16 WL rounds in `getStateHash`). `RemoteModelEvaluator` now sends those hashes
(`node_hash`, raw int64). The server keeps a persistent LRU cache, shared by all games: a GPU table [passes, 2^19
slots, d] in fp32, about 1 GiB. Per batch, each hash the cache doesn't hold is computed once (its first node). Every
other internal node reads every pass's output from its slot, at its own stage, so the sequential semantics hold.
Stages with nothing to compute are skipped. Requests without hashes (old clients) take the old path, and their
responses are byte-identical to before.

Correctness (CPU, fp32, ver4 checkpoint, 50 gen1 MonoB games; recorded states don't store hashes, so
`cache.subtree_hashes` recomputes them):
- 1,280 states as a stream of 8 games × 4 consecutive states per batch, cached vs plain forward: max |diff| 2e-5,
  0 top-1 changes (12 argmax swaps, all exact float ties, gap ≤ 4e-7). The same holds with a 800-slot cache that
  evicts constantly.
- Server end to end: old-format list responses byte-identical to the original server; hashed binary requests within
  1.3e-5 of unhashed ones.
- Java round trip (`RemoteModelEvaluator` against a stub `FeatureGraph` carrying the hashes, 200 states, one at a time
  and batched): all 86,898 returned numbers within 1.9e-5 of the uncached reference, 0 top-1 changes.

Speed (T4, `bench.py` `bench_cache`). Server path: lookup + plan + forward + copy back, fp16 autocast. Each batch
takes 4 consecutive states from each of batch/4 games (1 state at batch 1). Times in ms, means after 2 warm-up
steps:

| batch | plain | cached | lookup | forward plain → cached | internal nodes computed |
|---|---|---|---|---|---|
| 1 | 20.5 | 13.2 (−36%) | 0.2 | 19.3 → 11.4 | 9.7% |
| 8 | 24.4 | 21.1 (−14%) | 0.3 | 22.2 → 18.4 | 5.6% |
| 32 | 28.4 | 25.2 (−11%) | 0.6 | 24.1 → 20.1 | 4.7% |
| 128 | 35.8 | 31.4 (−12%) | 1.6 | 32.0 → 25.1 | 4.6% |
| 512 | 134.3 | 87.1 (−35%) | 5.6 | 128.9 → 74.6 | 4.6% |

Under fp16, cached vs plain differ by up to 0.023 (fp16 noise). Top-1 changed in 2 of about 1,400 states, where
the plain top led by between 1e-3 and the noise level.

Conclusion: the hit rate is as predicted (about 95% of internal nodes reused), but the time saved is not. Up to
batch 128 the local layers are launch-bound, not compute-bound: cached stages still launch the same kernels for a
handful of nodes (ROOT and PLAYER always miss). The cache pays off at large batches (−42% forward at 512), where the
rest is the global layers and the embeddings. Next lever for small batches: fewer launches (CUDA graphs on the cached
path, or fusing the stages that have only a few misses).
