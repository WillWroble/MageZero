# Length-adaptive TD (`--td first:W`)

Question: does normalizing the TD discount by game length (result weight W at a game's first state, gamma = W^(1/N))
reduce the value drift behind the early land-drop skipping, independent of the TD strength itself?

Recipe: v4's (cold, ver4 gen0 PlayerA+B minus PlayerB MonoW, `policy_domain all`, 4 epochs, seed 0, `--val-every 10000`).
Val = `ver4/testing/gen0`. Pods: w=0.007 on an RTX 5090 (~9.5 min/epoch), w=0.03 on an RTX PRO 6000 (~14 min/epoch,
CPU-bound). Runs: `/workspace/runs/v4-tdfirst0007`, `/workspace/runs/v4-tdfirst003`. Cost ≈ $1.2 + $2.3.

End of epoch 4 (val; value MSE is against each run's own labels, so not comparable across W):

| run | priority | target | choose_use | value | priority legal acc |
|---|---|---|---|---|---|
| first:0.007 (≈TD 0.92 at N=70) | .527 | .261 | .049 | .053 | .664 |
| first:0.03 (≈TD 0.95 at N=70) | .524 | .256 | .048 | .071 | .668 |

Policy heads are the same within noise. Play test (early land-drop skip rate vs v4's shards) pending.
