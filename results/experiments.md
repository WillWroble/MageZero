# Architecture experiments (Kaggle, graph-encoder-test)

Data: `Pauper-Faeries/ver4` (v0.3), Kaggle dataset `willwroble/magezero-pauper-faeries-ver4`.
Training: 421K states (gen0, both seats, 8 matchups). Testing: 63.5K states (gen0 mirror, both seats, 800 games).
GPU: Kaggle T4, one run per GPU. Timings compare runs with each other, not with the RTX 4070.
Logs: `results/kaggle/<kernel>/<run>.log`. Kernel: `kaggle/launch.py`.

| kernel | run | arch | epochs | conclusion |
|---|---|---|---|---|
| mz-pilot | base | default | 300 steps | 0.18 s/step (≈20 min/epoch); inference 17 ms at batch 1, ≈1.9k states/s from batch 128 |
| mz-pilot | joint1 | `local: joint, n_passes: 1` | – | killed: two runs loading data at once ran out of RAM; the launcher now staggers loading |
