# Architecture experiments (Kaggle, graph-encoder-test)

Data: `Pauper-Faeries/ver4` (v0.3), Kaggle dataset `willwroble/magezero-pauper-faeries-ver4`.
Training: 421K states (gen0, both seats, 8 matchups). Testing: 63.5K states (gen0 mirror, both seats, 800 games).
GPU: Kaggle T4, one run per GPU. Timings compare runs with each other, not with the RTX 4070.
Logs: `results/kaggle/<kernel>/<run>.log`. Kernel: `kaggle/launch.py`.

| kernel | run | arch | epochs | conclusion |
|---|---|---|---|---|
| mz-pilot | base | default | 300 steps | 0.18 s/step (≈20 min/epoch); inference 17 ms at batch 1, ≈1.9k states/s from batch 128 |
| mz-pilot | joint1 | `local: joint, n_passes: 1` | – | killed: two runs loading data at once ran out of RAM; the launcher now staggers loading |
| mz-s2-passes | seq1 | `n_passes: 1` (20.0M params) | 6 | val total 1.134 (prio 0.700, tgt 0.315, use 0.061, value 0.059); prio acc 0.637, non-pass 0.462, tgt acc 0.673. Still improving at epoch 6. Bench: 14.7 ms @1, ≈3.0k states/s @512 |
| mz-s2-passes | joint2 | `local: joint` (9.4M params) | 6 | val total 1.320 (prio 0.811, tgt 0.326, value 0.125); non-pass 0.331, tgt acc 0.669. Worse priority and value than seq1. Bench: 6.9 ms @1 (half of seq1), but ≈2.0k states/s @512 |
