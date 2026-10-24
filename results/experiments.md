# Architecture experiments (Kaggle, graph-encoder-test)

Data: `Pauper-Faeries/ver4` (v0.3), Kaggle dataset `willwroble/magezero-pauper-faeries-ver4`.
Training: 421K states (gen0, both seats, 8 matchups). Testing: 63.5K states (gen0 mirror, both seats, 800 games).
GPU: Kaggle T4, one run per GPU. Timings compare runs with each other, not with the RTX 4070.
Logs: `results/kaggle/<kernel>/<run>.log`. Kernel: `kaggle/launch.py`.

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
