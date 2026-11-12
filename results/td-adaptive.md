# Length-adaptive TD (`--td first:W`)

Question: does normalizing the TD discount by game length (result weight W at a game's first state, gamma = W^(1/N))
reduce the value drift behind the early land-drop skipping, independent of the TD strength itself?

Recipe: v4's (cold, ver4 gen0 PlayerA+B minus PlayerB MonoW, `policy_domain all`, 4 epochs, seed 0, `--val-every 10000`).
Val = `ver4/testing/gen0`. Pods: w=0.007 on an RTX 5090 (~9.5 min/epoch), w=0.03 on an RTX PRO 6000 (~14 min/epoch,
CPU-bound). Runs: `/workspace/runs/v4-tdfirst{0007,003,01,025}`. Cost ≈ $1.2 + $2.3; w=0.1 and 0.25 on RTX 4090s (~11.5 min/epoch), ≈ $0.9 each.

End of epoch 4 (val; value MSE is against each run's own labels, so not comparable across W):

| run | priority | target | choose_use | value | priority legal acc |
|---|---|---|---|---|---|
| first:0.007 (≈TD 0.92 at N=70) | .527 | .261 | .049 | .053 | .664 |
| first:0.03 (≈TD 0.95 at N=70) | .524 | .256 | .048 | .071 | .668 |
| first:0.1 (≈TD 0.968) | .544 | .264 | .048 | .123 | |
| first:0.25 (≈TD 0.98) | .556 | .262 | .047 | .206 | |
| mix:0.5 | .533 | .264 | .049 | .157 | |

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
| first:0.007 | 62.0% | 24.3% (1,172) | |
| first:0.1 | 60.5% | 10.3% (949) | |
| first:0.25 | 59.5% | 20.3% (1,033) | |
| ver15 | 68.5% | 25.9% (1,231) | |

w=0.1 and 0.25: w=0.1 on an RTX 4090, w=0.25 on an RTX 5090. Decisions within a game are correlated, so the
Pass share has an error of a few points, not ±1.

Conclusion: normalizing the TD by game length (w=0.03) does not reduce the search's preference for passing on early
land drops against Burn: 18.8% vs v4's 17.2%, both far above the heuristic's 4%.

With w=0.1 and 0.25 added: Pass share 24.3 / 18.8 / 10.3 / 20.3% for w = 0.007 / 0.03 / 0.1 / 0.25, not monotone in w;
only w=0.1 is clearly below v4 (17.2%), and it wins less (60.5% vs 66.5%). Win rate falls for w ≥ 0.1.

### First 3 land drops (fewer than 3 own lands), vs first 4: Pass top, all vs offline Burn

| model | win | <3 lands (states) | <4 lands |
|---|---|---|---|
| gen0 heuristic | 56% | 3.1% (349) | 4.1% |
| v4 | 66.5% | 14.6% (760) | 17.2% |
| ver10 | 61.5% | 22.8% (904) | 27.5% |
| ver13 | 69.5% | 23.2% (880) | 23.7% |
| ver15 (search shard, 70.5% won) | | 21.7% (864) | 26.0% |
| first:0.007 | 62.0% | 20.9% (845) | 24.3% |
| first:0.03 | 71.0% | 15.0% (753) | 18.8% |
| first:0.06 | 63.5% | 13.4% (746) | 18.7% |
| first:0.1 | 60.5% | 7.4% (698) | 10.3% |
| first:0.25 | 59.5% | 19.9% (791) | 20.3% |
| mix:0.5 (label = ½ result + ½ root value, no TD) | 61.0% | 7.5% (660) | 9.9% |

The ranking is the same as for the first 4 drops, every Pass share 0–5 points lower.
