# Policy only: ver15's policy without search

Faeries is `on_policy` (`ComputerPlayerPolicyOnly`): at every decision it plays the network's top-scored legal move
(priority, target and choose-use priors on, no noise), with no search. The opponent is the offline heuristic search
(300 sims). 200 games per deck. Kaggle T4, 4 vCPUs, 8 game threads (`kaggle/generate.py`, configs in
`kaggle/configs/onpolicy-<deck>.yml`). Logs in `results/kaggle/onpolicy-<deck>/`.

Reference: the same model with value-only search (300 sims, priors off), from `results/gen3-training.md`.

| deck | XMage commit | policy only | ver15 with search |
|---|---|---|---|
| MonoB | `a800f4fe` (priors.opponent true) | 12.5% (25/200) | 36.0% |

- MonoB: the run needed `priors.opponent: true`. With it false, every game failed at `a800f4fe` on a null
  `node.policy` in `bestAction` (logs in the first commit of `results/kaggle/onpolicy-monob`). Faeries plays its
  spells (Humbling Elder, Spellstutter, ninjutsu, Of One Mind and Strix loot appear in the game log), so the result
  isn't the player passing. One error line in the game log.

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
