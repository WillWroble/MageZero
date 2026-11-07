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
