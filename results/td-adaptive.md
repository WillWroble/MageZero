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
| mix:0 (root value only) | .557 | .261 | .051 | .010 |
| mix:1 (game result only) | .574 | .272 | .051 | .672 |

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
| first:0.03, 1000 sims | 78.5% | 10.9% (716) | 13.0% |
| first:0.03, 1000 sims, no duplicate pruning (both sides) | 77.4% (199 games) | 12.1% (728) | 14.8% |

The ranking is the same as for the first 4 drops, every Pass share 0–5 points lower.

1000 sims (offline Burn stays at 300; XMage `b6ca7b35`, RTX 4090, `max_pending` 4): win +7.5 points over 300 sims and Pass top drops from 15.0% to 10.9% on the first 3 drops. Turning duplicate pruning off on both sides changes neither beyond noise. More search partly corrects the value net's early passing; it doesn't remove it (heuristic: 3.1%).

### Burn piloting Burn vs offline Burn: Pass top

Will's local run: the Pauper ver1 model (`experiment_data/Pauper/ver1/Pauper-Burn_vs_Pauper-Burn1000{A,B}.hdf5`),
1000 sims on both seats, duplicate pruning on, 200 games. ver15 rows: the gen5 Burn-mirror shards (300 sims).

| Burn pilot | win | <3 lands (states) | <4 lands | all land-legal |
|---|---|---|---|---|
| Pauper ver1, 1000 sims | 55.5% | 31.7% (1,175) | 30.4% | 31.3% |
| its opponent, offline Burn 1000 sims | 44.5% | 9.1% (996) | 8.9% | 9.6% |
| first:0.03, 1000 sims (offline 300), 100 games | 56% | 30.0% (534) | 32.9% | 35.0% |
| first:0.1, 1000 sims (offline 300), 100 games | 55% | 19.5% (497) | 18.4% | 19.4% |
| first:0.03, 1000 sims, c_puct 0.5 (offline 300, c_puct 1), 100 games | 57% | 26.0% (454) | 27.5% | 31.5% |
| first:0.1, 1000 sims, c_puct 0.5 (offline 300, c_puct 1), 100 games | 56% | 14.9% (450) | 14.4% | 15.6% |
| Pauper ver1, 300 sims (both seats) | 37.5% | 57.5% (1,712) | 55.2% | 56.0% |
| its opponent, offline Burn 300 sims | 62.5% | 7.7% (891) | 8.1% | 10.6% |
| ver15 vs offline Burn | 47% | 54.6% | | 53.0% |
| offline Burn vs ver15 | 53% | 8.9% | | 9.2% |
| ver15 self-play | | | | 43.0% |

Pauper ver1 skips early land drops about half as often as ver15 piloting Burn, still about 3.5x the heuristic.
The v4-recipe twins (trained on Faeries gen0, which includes the Burn seat's perspective) at 1000 sims: w=0.03 passes as often as Pauper ver1 (30%), w=0.1 about 10 points less (19.5%); both win 55–56%, level with Pauper ver1 (±10 at 100 games). Pilot RTX 4090, XMage `b6ca7b35`, `max_pending` 4.
c_puct 0.5 on the pilot: Pass top about 4 points lower for both models, win rate unchanged (±10 at 100 games).
Pauper ver1 at 300 sims (Will's run, 200 games): Pass top 57.5% on the first 3 drops, level with ver15 (54.8%), and wins 37.5% (ver15 47%). Its 1000-sim run: 31.7%, 55.5%. The skipping is not a ver15 quirk; more search roughly halves it.
