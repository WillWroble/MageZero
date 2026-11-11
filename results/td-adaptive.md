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

Policy heads are the same within noise.

## Play: Faeries vs offline Burn (200 games, RTX 4090, XMage `b6ca7b35`, value only, 300 sims, max_pending 4)

Early land drops (fewer than 4 own lands, land play legal, priority states in own main phases). "Pass top" = the
search's most-visited move is Pass. "Turn skipped" = CanPlayLand still set at the turn's last recorded state (has a
floor from unrecorded land plays: the heuristic shows 30%). v4 and gen0 are Will's local games (older XMage); ver15
is the search shard from the earlier land-drop analysis.

| model | win | pass top (states) | turn skipped (turns) |
|---|---|---|---|
| gen0 heuristic | 56% (100 games) | 4.1% (463) | 29.7% (212) |
| v4 (TD 0.92) | 66.5% | 17.2% (1,059) | 36.5% (433) |
| first:0.03 | 71.0% | 18.8% (1,068) | 32.0% (419) |
| ver15 | 68.5% | 25.9% (1,231) | |

Conclusion: normalizing the TD by game length (w=0.03) does not reduce the search's preference for passing on early
land drops against Burn: 18.8% vs v4's 17.2%, both far above the heuristic's 4%.
