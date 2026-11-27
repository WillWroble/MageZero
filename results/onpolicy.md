# Policy only: ver15's policy without search

Faeries is `on_policy` (`ComputerPlayerPolicyOnly`): at every decision it plays the network's top-scored legal move
(priority, target and choose-use priors on, no noise), with no search. The opponent is the offline heuristic search
(300 sims). 200 games per deck. Kaggle T4, 4 vCPUs, 8 game threads (`kaggle/generate.py`, configs in
`kaggle/configs/onpolicy-<deck>.yml`). Logs in `results/kaggle/onpolicy-<deck>/`.

Reference: the same model with value-only search (300 sims, priors off), from `results/gen3-training.md`.

| deck | XMage commit | policy only | ver15 with search |
|---|---|---|---|
| MonoB | `a800f4fe` (priors.opponent true) | 12.5% (25/200) | 36.0% |
| mirror | `3ae6456e` | 44.0% (88/200) | 72.0% |
| Burn | `3ae6456e` | 52.0% (104/200) | 68.5% |
| Rally | `3ae6456e` | 10.0% (20/200) | 31.0% |

- MonoB: the run needed `priors.opponent: true`. With it false, every game failed at `a800f4fe` on a null
  `node.policy` in `bestAction` (logs in the first commit of `results/kaggle/onpolicy-monob`). Faeries plays its
  spells (Humbling Elder, Spellstutter, ninjutsu, Of One Mind and Strix loot appear in the game log), so the result
  isn't the player passing. One error line in the game log.
- mirror: `priors.opponent: false`; no errors in the game log. The policy player's own log lines show it casting
  Spellstutter, Counterspell, Ninja, Humbling Elder and Snap.
- Burn: `priors.opponent: false`; no errors. Faeries casts Faerie Seer, Spellstutter, Harrier Strix and Ninja and
  uses ninjutsu in the policy player's log lines.
- Rally: `priors.opponent: false`; no errors. 0 wins in the first 26 games, then 20 over 200. The policy player logs
  as many casts as against the mirror and Burn (32 vs 40 and 30 in the last 2,000 log lines), including
  Spellstutter, Snap, Of One Mind and Moon-Circuit Hacker, so it isn't passing. Faeries' shard records 199.5
  states per game, Rally's 56.3.

## Will's local runs (Nov 7, `experiment_data/Pauper-Faeries/ver15/*_{policy,priority}_only*.hdf5`)

Same setup (ver15 policy only, greedy, vs the offline search), 200 games each, win rate from the shards' final
labels. gen0 = heuristic search vs heuristic search, gen1 = v4 with value search (`generations.md`); with search =
ver15, or ver13 where marked.

| deck | policy only | gen0 | gen1 | with search |
|---|---|---|---|---|
| MonoB | 15.0%, 9.5% (and Kaggle 12.5%): **12.3%** pooled over 600 | 7.4% | 31.6% | 36.0% |
| MonoG | 11.5% | 7.5% | 16.0% | 32.0% |
| MonoR | 8.5% | 10.2% | 17.5% | 25.0% (ver13) |
| MonoU | 61.5% | 53.6% | 75.4% | 79.0% (ver13) |
| MonoW | 25.3% (198 games) | 21.0% | 57.5% | 53.5% |
| mean | 23.8% | 20.0% | 39.6% | 45.1% |

Other ver15 runs vs offline MonoB (value search unless noted): 1000 sims 44.6% (186 games, +8.6 over 300 sims);
priority priors temp 2.5 with `c_puct` 0.5, 33.0%. vs Rally, priority priors temp 2.5: 26.5% (31.0% without).

Conclusion: the policy alone beats gen0's heuristic search on 4 of 5 decks (+3.8 on average) but is ≈21 points
below value search with the same model, and more search still helps (+8.6 at 1000 sims). Priors tilt the search
toward this much weaker policy, which fits them not helping. The policy-only Faeries shards record ≈180–235 states
per game against ≈70–80 in search games (the opponent's seat is unchanged), so they hold many more decisions per
game; check what they contain before training on them.

### Pauper pool (Will, Nov 7, `experiment_data/Pauper-Faeries/ver15/policy_only/`)

ver15 policy only vs the offline search, 200 games each (Elves 197). Win rates from PlayerA's shards; PlayerB's
agree. gen0 / v4 / ver13 vs offline from `generations.md` (ver13: first run of the mirror and Rally).

| deck | policy only | gen0 | v4 (gen1) | ver13 (gen3) |
|---|---|---|---|---|
| Affinity | 66.5% | 40.0% | 63.5% | 77.0% |
| Burn | 45.5% | 56.0% | 66.5% | 69.5% |
| CawGates | 67.0% | 61.8% | 81.1% | 83.0% |
| Elves | 55.3% | 35.5% | 65.0% | 75.0% |
| mirror | 39.0% | 50.9% | 72.0% | 68.5% |
| Rally | 13.5% | 15.2% | 30.0% | 27.0% |
| Spy | 55.0% | 46.7% | 59.0% | 71.5% |
| mean | 48.8% | 43.7% | 62.4% | 67.4% |

Conclusion: as on the Mono decks, the policy alone is about level with or above gen0's heuristic search (+5.1 on
average; ahead on Affinity, CawGates, Elves, Spy; behind on Burn, the mirror, Rally) and ≈19 points below value
search (ver13). Its weakest decks are the interactive ones: the mirror (−11.9 vs gen0) and Burn (−10.5).

## P2-2 policy only (Nov 23)

Same setup as ver15's runs above (`runpod/configs/onpolicy/Faeries-vs-<deck>.yml`, identical to
`kaggle/configs/onpolicy-*.yml` but for output paths: Faeries `on_policy`, greedy, priority/target/binary priors on, no
noise, vs the offline search at 300 sims, opponent's hand visible), with P2-2 (`pauper-P2-2v2-gnn/P2-2/model-e6-s60000`,
`results/gen5-data.md`), XMage `b6ca7b35`, two RTX 4090s on Ryzen 9 7950X hosts (`/workspace/runs/onpolicy-p22/`).

| deck | P2-2 policy only | ver15 policy only | Δ |
|---|---|---|---|
| Affinity | 61.5% | 66.5% | −5.0 |
| Burn | 49.0% | 45.5% | +3.5 |
| CawGates | 63.5% (125/197) | 67.0% | −3.5 |
| Elves | 53.4% (102/191) | 55.3% | −1.9 |
| mirror | **54.5%** | 39.0% | **+15.5** |
| Rally | 14.0% | 13.5% | +0.5 |
| Spy | 49.5% (98/198) | 55.0% | −5.5 |
| **mean (7)** | **49.3%** | 48.8% | +0.5 |
| Wildfire | 82.0% | – | |

- Level with ver15 on average; the one clear change is the mirror (+15.5, ≈3 SE), where ver15's policy was weakest.
  The other six decks move by −5.5 to +3.5, within ≈1.5 SE each.
- 14 failed games (Elves 9, CawGates 3, Spy 2), none in the other five matchups.
- Cost ≈ $3.80 (balance $34.40 → $30.59).

### Land skips of the policy-only players (Nov 23)

Policy-only shards record no visit counts, so the move played isn't stored; the greedy player plays the top-scored
legal candidate, so each model was re-run (CPU, fp32) on its own games' early land-drop states (`util/land_pass.py`'s
selection: own main phase, a land play legal) and the share where its top candidate is Pass counted. All priority
states are recorded here, so there are ≈6-8K such states per 200 games against ≈1K in search games.

| deck | P2-2, < 3 lands (states) | ver15, < 3 lands (states) | P2-2, < 4 | ver15, < 4 |
|---|---|---|---|---|
| Affinity | 0.5% (6342) | 2.6% (6536) | 0.8% | 4.8% |
| Burn | 0.2% (7037) | 4.8% (6783) | 0.4% | 7.0% |
| CawGates | 1.0% (6051) | 3.8% (6157) | 2.1% | 7.7% |
| Elves | 0.8% (7638) | 2.4% (7625) | 1.0% | 3.8% |
| mirror | 0.2% (6539) | 1.7% (6506) | 0.5% | 3.0% |
| Rally | 0.6% (6310) | 1.2% (6457) | 1.8% | 2.7% |
| Spy | 1.0% (5852) | 9.3% (6024) | 1.5% | 10.2% |
| **mean (7)** | **0.6%** | **3.7%** | **1.2%** | **5.6%** |
| Wildfire | 0.7% (5491) | – | 0.8% | – |

P2-2's policy skips early land drops about six times less often than ver15's, against every deck. The 8.4% Pass top of
P2-2's Burn mirror with search (`results/gen5-data.md`) comes from the search, not from its policy, which skips at
0.2% in policy-only Faeries vs Burn (a different matchup and seat, so only indicative). Script:
`<scratchpad>/polland.py` (land_pass's selection without its visit-count filter, land_policy's scoring); not committed.

## P2-2 vs ver15 head-to-head mirrors (Nov 27)

P2-2 (`pauper-P2-2v2-gnn/P2-2/model-e6-s60000`, player A, port 50052) against ver15 (`gen4-F2-gnn/F`, player B,
port 50053) in the Pauper mirrors, 200 games each, both seats with the same settings, RTX 4090s.

### Policy only (abandoned)

Configs `runpod/configs/h2h-pol/<deck>.yml` (both seats `on_policy`: top legal move, priority/target/binary priors, no
noise). Both seats policy-only exposed three bugs in WillWroble/mage, fixed on branch `policy-choice-fix`. Earlier
policy-only runs never hit them because only Faeries, which has no "any color" mana, played policy-only.

1. `b6ca7b35`: the policy-only player doesn't search MAKE_CHOICE, so `ComputerPlayerMCTS.choose` added a null choice
   (NPE). Fixed in `5198ff40`: fall back to the heuristic `ComputerPlayer.choose`.
2. Same for CHOOSE_NUM (X costs, modes): null `amountAction`. Fixed in `ea9c1046`.
3. `ComputerPlayer.chooseHelper` took the alphabetically first option before its unpaid-cost color hint, so "add one
   mana of any color" always made Black, the policy's cast then couldn't be paid, and `ComputerPlayerPolicyOnly.priority`
   threw "failed to activate chosen ability". Fixed in `3b270a33` (hint first). This changes the heuristic AI's mana
   color choice wherever it pays a cost; searching players choose colors in the search and are unaffected.

Failed games: on `b6ca7b35` CawGates 200/200, Spy 197, Elves 68, Burn 76, Affinity 23; on `ea9c1046` Spy 158/200,
Burn 34; on `3b270a33` a 50-game Spy smoke test still failed 28 (remaining: "failed to activate chosen ability" on
"Cast Overgrown Battlement" and a flashback "sacrifice three creatures" cost; the pods keep only 500 log lines, so the
rest are unseen). The only clean result was Faeries on `ea9c1046`: P2-2 74/200 = 37.0% ± 3.4 (42.0% on `b6ca7b35`,
also 0 failed). Cost ≈ $5.

### Search budget 10

Configs `runpod/configs/h2h-s10/<deck>.yml` (`324498b`): the policy-only configs with both seats `mcts`,
`search_budget: 10`, priority/target/binary priors on, noise off. XMage `3b270a33`. Faeries skipped.

| deck | P2-2 wins | P2-2 win rate | failed |
|---|---|---|---|
| Affinity | 154/200 | 77.0% ± 3.0 | 0 |
| Burn | 111/169 | 65.7% ± 3.7 | 31 |
| CawGates | 59/152 | 38.8% ± 4.0 | 48 |
| Elves | 106/199 | 53.3% ± 3.5 | 1 |
| Rally | 124/200 | 62.0% ± 3.4 | 0 |
| Spy | 93/174 | 53.4% ± 3.8 | 26 |
| Wildfire | 165/200 | 82.5% ± 2.7 | 0 |
| **mean (7)** | | **61.8% ± 1.3** | |

- P2-2 beats ver15 at a 10-simulation budget: 61.8% over 7 mirrors, ahead in 6 of 7, behind only in CawGates.
- Failures remain in Burn, Spy and CawGates (all "Error in unit tests" in the visible log tails, one NPE in Spy); the
  dropped games could bias those three decks. A 50-game Spy smoke test first: 4 failed, P2-2 31/46.
- ≈ 1 h on two pods, ≈ $1.80.
