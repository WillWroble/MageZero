# How Faeries' play changes across generations

Probed straight from the h5s with `src/magezero/util/behaviour.py` (Faeries' seat, PlayerA shards). "Chosen" is how
often an ability is search's top move in priority states where it is available and there is more than one option.

## gen0 → gen1 vs Standard MonoB (`ver4`)

gen0: offline MCTS, 2,000 games. gen1: the default model trained locally on all of `ver4` gen0 (Pauper + Mono, both
seats), value only (priors off), 300 sims, 60 s timeout, TD blend 0.92, 171 games.

| | gen0 | gen1 |
|---|---|---|
| Faeries win rate | 7.4% | 31.6% (±3.6%) |
| states per game, Faeries / MonoB | 60.1 / 47.5 | 84.1 / 47.2 |
| priority decisions with a choice, per game | 24.1 | 33.3 |
| top move is Pass (all priority states) | 43.9% | 65.0% |
| corr(search score, value label) | 0.89 | 0.95 |

### Seer vs Miscreant (priority states where both can be cast; share of states where each is the top move)

| | gen0 | gen1 |
|---|---|---|
| no Miscreant on Faeries' battlefield | Seer 0.47 / Miscreant 0.33 (750 states) | Seer 0.17 / Miscreant 0.44 (80) |
| a Miscreant already out | Seer 0.16 / Miscreant 0.70 (145) | Seer 0.08 / Miscreant 0.54 (26) |

The same flip as v0.2's gen0 → gen1 (`magezero-faeries-findings.md` §6): gen0 makes the human choice (Seer unless
Miscreant would draw), gen1 prefers Miscreant regardless. Policy priors were off, so this comes from the value head
guiding search. gen1's samples are small.

### Value labels by Faeries' board (correlational; groups differ in more than the board)

| board | gen0 median / mean / from won games | gen1 median / mean / from won games |
|---|---|---|
| Seer only | −0.192 / −0.208 / 0.09 | −0.204 / −0.184 / 0.32 |
| one Miscreant only | −0.184 / −0.199 / 0.08 | −0.249 / −0.157 / 0.37 |
| 2+ Miscreants | +0.013 / +0.003 / 0.19 | −0.100 / −0.076 / 0.38 |

### Chosen when available, and own-turn share of those choices

| ability | gen0 chosen / own turn | gen1 chosen / own turn |
|---|---|---|
| Pass | 0.33 / 0.81 | 0.50 / 0.75 |
| Play Island | 0.69 / 1.00 | 0.65 / 1.00 |
| Faerie Seer | 0.61 / 1.00 | 0.37 / 1.00 |
| Faerie Miscreant | 0.63 / 1.00 | 0.53 / 1.00 |
| Ninja of the Deep Hours (cast) | 0.88 / 1.00 | 0.28 / 1.00 |
| ninjutsu A / B | 0.65 / 0.61 | 0.27 / 0.40 |
| Harrier Strix | 0.58 / 1.00 | 0.32 / 1.00 |
| Snap | 0.52 / 0.97 | 0.34 / 0.97 |
| Saiba (flash) | 0.63 / 0.93 | 0.35 / 0.86 |
| Humbling Elder (flash) | 0.42 / 0.69 | 0.23 / 0.66 |
| Spellstutter (flash) | 0.38 / 0.84 | 0.24 / 0.68 |
| Of One Mind | 0.39 / 1.00 | 0.32 / 1.00 |
| Counterspell | 0.39 / 0.03 | 0.37 / 0.05 |
| Force Spike | 0.41 / 0.05 | 0.23 / 0.07 |
| Dispel | 0.32 / 0.20 | 0.25 / 0.38 |
| Strix loot | 0.22 / 0.80 | 0.15 / 0.49 |

gen1 is far more patient: every threat is cast less often when available (Ninja 0.88 → 0.28, ninjutsu 0.6 → 0.3),
while land drops and Counterspell are unchanged. The flash and instant plays it does make shift toward the
opponent's turn (Spellstutter own-turn 0.84 → 0.68, Strix loot 0.80 → 0.49). It wins four times as often.

## gen0 → gen1, all five Standard Mono matchups

| opponent | gen0 win | gen1 win | gen1 games | states/game gen0 → gen1 |
|---|---|---|---|---|
| MonoB | 7.4% | 31.6% | 171 | 60.1 → 84.1 |
| MonoG | 7.5% | 16.0% | 200 | 58.5 → 109.4 |
| MonoR | 10.2% | 17.5% | 200 | 48.8 → 81.6 |
| MonoU | 53.6% | 75.4% | 199 | 69.8 → 87.9 |
| MonoW | 21.0% | 57.5% | 186 | 61.6 → 110.0 |

Chosen when available, gen0 → gen1:

| ability | MonoB | MonoG | MonoR | MonoU | MonoW |
|---|---|---|---|---|---|
| Pass | 0.33 → 0.50 | 0.29 → 0.47 | 0.32 → 0.52 | 0.34 → 0.49 | 0.28 → 0.46 |
| Play Island | 0.69 → 0.65 | 0.69 → 0.67 | 0.74 → 0.64 | 0.70 → 0.60 | 0.71 → 0.61 |
| Faerie Seer | 0.61 → 0.37 | 0.61 → 0.43 | 0.63 → 0.41 | 0.51 → 0.30 | 0.64 → 0.41 |
| Faerie Miscreant | 0.63 → 0.53 | 0.59 → 0.54 | 0.59 → 0.48 | 0.65 → 0.52 | 0.62 → 0.51 |
| Ninja (cast) | 0.88 → 0.28 | 0.88 → 0.36 | 0.85 → 0.42 | 0.85 → 0.37 | 0.94 → 0.46 |
| ninjutsu A | 0.65 → 0.27 | 0.56 → 0.30 | 0.62 → 0.30 | 0.68 → 0.27 | 0.68 → 0.35 |
| ninjutsu B | 0.61 → 0.40 | 0.63 → 0.53 | 0.60 → 0.45 | 0.62 → 0.33 | 0.67 → 0.45 |
| Harrier Strix | 0.58 → 0.32 | 0.56 → 0.38 | 0.58 → 0.32 | 0.45 → 0.29 | 0.57 → 0.37 |
| Snap | 0.52 → 0.34 | 0.59 → 0.28 | 0.50 → 0.25 | 0.16 → 0.12 | 0.49 → 0.25 |
| Saiba | 0.63 → 0.35 | 0.59 → 0.35 | 0.59 → 0.30 | 0.57 → 0.39 | 0.63 → 0.42 |
| Humbling Elder | 0.42 → 0.23 | 0.54 → 0.28 | 0.46 → 0.21 | 0.49 → 0.29 | 0.54 → 0.31 |
| Spellstutter | 0.38 → 0.24 | 0.41 → 0.28 | 0.42 → 0.23 | 0.54 → 0.32 | 0.54 → 0.36 |
| Of One Mind | 0.39 → 0.32 | 0.38 → 0.39 | 0.30 → 0.25 | 0.32 → 0.23 | 0.38 → 0.32 |
| Counterspell | 0.39 → 0.37 | 0.50 → 0.46 | 0.37 → 0.37 | 0.26 → 0.26 | 0.39 → 0.32 |
| Force Spike | 0.41 → 0.23 | 0.44 → 0.29 | 0.39 → 0.26 | 0.25 → 0.19 | 0.36 → 0.26 |
| Dispel | 0.32 → 0.25 | 0.04 → 0.08 | 0.38 → 0.47 | 0.36 → 0.43 | 0.17 → 0.24 |
| Strix loot | 0.22 → 0.15 | 0.23 → 0.21 | 0.18 → 0.14 | 0.34 → 0.18 | 0.30 → 0.25 |

Own-turn share of choices, gen0 → gen1:

| ability | MonoB | MonoG | MonoR | MonoU | MonoW |
|---|---|---|---|---|---|
| Humbling Elder | 0.69 → 0.66 | 0.66 → 0.56 | 0.68 → 0.50 | 0.86 → 0.84 | 0.76 → 0.71 |
| Spellstutter | 0.84 → 0.68 | 0.76 → 0.59 | 0.84 → 0.61 | 0.92 → 0.87 | 0.90 → 0.80 |
| Strix loot | 0.80 → 0.49 | 0.71 → 0.56 | 0.76 → 0.58 | 0.90 → 0.83 | 0.86 → 0.80 |

Seer / Miscreant top-move share with no Miscreant out, gen0 → gen1: MonoB 0.47/0.33 → 0.17/0.44 (80 states),
MonoG 0.50/0.27 → 0.28/0.35 (94), MonoR 0.49/0.30 → 0.31/0.36 (88), MonoU 0.29/0.47 → 0.21/0.40 (131),
MonoW 0.46/0.29 → 0.30/0.37 (73).

Conclusion: the patience is general, not a read of one matchup. Every opponent, including the aggressive MonoR,
gets more passing, fewer threats (Ninja 0.85–0.94 → 0.28–0.46) and unchanged Counterspell use, and gen1 wins more
against all five. Flash plays move to the opponent's turn against MonoB, G and R. The Miscreant preference appears
in every matchup. Since this is value-only search, a likely cause is a value head that rates cards held in hand
higher than the gen0 heuristic did; not checked.

### Opponent-specific adaptation (gen0 → gen1)

Across abilities, gen1's per-matchup chosen rates correlate with gen0's (median correlation of matchup deviations
+0.74), and the median spread across matchups (std / mean) rises only from 0.09 to 0.14. The largest
matchup differences already existed in gen0 and mostly reflect what is legal or useful (Dispel ≈0 vs MonoG, Snap
rare vs MonoU, Counterspell lowest vs MonoU). After removing each ability's average gen0 → gen1 change, the largest
matchup-specific shifts are: Ninja −0.10 vs MonoB; ninjutsu +0.08/+0.09 vs MonoG and −0.07/−0.10 vs MonoU; Snap
+0.16 vs MonoU and −0.11 vs MonoG; Dispel −0.11 vs MonoB; Strix loot −0.09 vs MonoU. Each gen1 cell rests on
≈70–500 states, so these are about two standard errors: suggestive only.

Conclusion: one generation of value-only search learned a general style (patient, reactive), not per-opponent
plans; the few matchup-specific shifts are suggestive.

## gen0 model on gen1 data (the classic test)

The gen0 model that generated gen1 (`experiment_models/Pauper-Faeries/ver4/model.pt.gz`, epoch 4, trained on all of
`ver4` gen0) scored with `test.py --testing` on each gen1 Mono matchup and seat, CPU. Reference: the same model on
the gen0 mirror test set. Always-pass = share of priority states whose MCTS top move is Pass.

| test set | states | always-pass | priority acc | gain | non-pass recall | prio loss | target legal acc | use acc | value MSE |
|---|---|---|---|---|---|---|---|---|---|
| gen0 mirror (reference) | 63,518 | 0.41 | 0.638 | +0.23 | 0.43 | 0.524 | 0.790 | 0.886 | 0.035 |
| gen1 Faeries vs MonoB | 14,388 | 0.65 | 0.659 | +0.01 | 0.38 | 0.496 | 0.726 | 0.793 | 0.065 |
| gen1 Faeries vs MonoG | 21,817 | 0.68 | 0.707 | +0.02 | 0.35 | 0.453 | 0.729 | 0.891 | 0.041 |
| gen1 Faeries vs MonoR | 16,317 | 0.70 | 0.696 | 0.00 | 0.38 | 0.461 | 0.732 | 0.814 | 0.065 |
| gen1 Faeries vs MonoU | 17,474 | 0.59 | 0.631 | +0.04 | 0.40 | 0.486 | 0.783 | 0.746 | 0.049 |
| gen1 Faeries vs MonoW | 20,449 | 0.60 | 0.635 | +0.03 | 0.36 | 0.531 | 0.677 | 0.755 | 0.040 |
| gen1 MonoB (opponent seat) | 8,067 | 0.33 | 0.633 | +0.30 | 0.50 | 0.501 | 0.812 | 0.961 | 0.077 |
| gen1 MonoG (opponent seat) | 12,187 | 0.27 | 0.586 | +0.31 | 0.46 | 0.594 | 0.832 | 0.728 | 0.034 |
| gen1 MonoR (opponent seat) | 10,509 | 0.50 | 0.669 | +0.17 | 0.45 | 0.409 | 0.861 | 0.956 | 0.051 |
| gen1 MonoU (opponent seat) | 11,213 | 0.41 | 0.701 | +0.30 | 0.34 | 0.405 | 0.767 | 0.854 | 0.052 |
| gen1 MonoW (opponent seat) | 23,546 | 0.43 | 0.631 | +0.20 | 0.44 | 0.373 | 0.767 | 0.733 | 0.047 |

Conclusion: on gen1 Faeries decisions the gen0 model's priority top-1 is barely above always-pass (+0.00 to +0.04,
vs +0.23 on gen0 data), while the opponent seats stay well predicted. Its errors are actions where gen1's search
passes (vs MonoB: predicts an action in 19% of gen1's pass states, vs 6% on the gen0 mirror). Priority loss stays
level because the soft visit distributions still overlap. Value is worst where gen1 outperformed gen0 most (MonoB,
MonoR: 0.065), despite gen1's labels being largely this model's own evaluations.

## gen1 vs MonoB from a Mono-only gen0 model (`ver5`)

**Which model made which `ver5` gen1 shard (Will, Oct 29):** only `Pauper-Faeries_vs_Standard-MonoB.hdf5` (26.5%)
and `Pauper-Faeries_vs_Standard-MonoW.hdf5` (40.6%), with their PlayerB pairs, come from the Mono-only v5 model.
Every other `ver5` gen1 shard (`MonoB2`, `_pimc`, `_pimc2`, `_pimc_pimc`, `_policy`, Affinity, Burn, CawGates,
Elves) was generated by the v4 model. This section compares the two v5-model shards with `ver4`, so it stands.

`ver5` model: the default trained on gen0 Mono games only (both seats, all five decks, 1.12M states), 4 epochs; same
search settings as `ver4` gen1 except a longer timeout (which favours Faeries), 200 games.

| | v4 gen1 (Pauper + Mono model) | v5 gen1 (Mono-only model) |
|---|---|---|
| Faeries win rate | 31.6% (171 games) | 26.5% (200 games; difference ≈1 SE) |
| states/game Faeries / MonoB | 84.1 / 47.2 | 88.1 / 55.3 |
| decisions with < 300 visits | 0.32 | 0.26 (longer timeout) |

Chosen when available, gen0 / v4 gen1 / v5 gen1: Pass 0.33 / 0.50 / 0.50; Ninja 0.88 / 0.28 / 0.38; ninjutsu A
0.65 / 0.27 / 0.36, B 0.61 / 0.40 / 0.50; Harrier Strix 0.58 / 0.32 / 0.42; Saiba 0.63 / 0.35 / 0.47; Of One Mind
0.39 / 0.32 / 0.46; Elder 0.42 / 0.23 / 0.30; Miscreant 0.63 / 0.53 / 0.60; Seer 0.61 / 0.37 / 0.41; Spellstutter
0.38 / 0.24 / 0.23; Counterspell 0.39 / 0.37 / 0.38; Snap 0.52 / 0.34 / 0.26; Island 0.69 / 0.65 / 0.59.
Spellstutter own-turn share 0.84 / 0.68 / 0.75; Elder 0.69 / 0.66 / 0.73.

Seer / Miscreant (no Miscreant out): v5 0.30 / 0.56 (50 states). Median value label by board, v4 / v5: Seer only
−0.20 / −0.40, one Miscreant −0.25 / −0.26, 2+ Miscreants −0.10 / −0.08.

Conclusion: the patience shift does not need the Pauper games (same pass rate, threats still held). v5 sits
between gen0 and v4, much closer to v4: it casts threats ≈0.1 more often and moves flash plays less, each about
two standard errors but all in the same direction. The Pauper games made gen1 slightly more patient and possibly
slightly stronger against MonoB; the win-rate gap is within noise and confounded by the timeout.

### v5 gen1 vs MonoW

| | gen0 | v4 gen1 (Pauper + Mono) | v5 gen1 (Mono only) |
|---|---|---|---|
| Faeries win rate | 21.0% (1,996) | 57.5% (186) | 40.6% (192) |
| states/game Faeries / MonoW | 61.6 / – | 110.0 / 127.4 | 104.5 / 133.1 |
| decisions with < 300 visits (Faeries) | – | 0.62 | 0.60 |

The 17-point gap is ≈3.3 SE, and v5's longer timeout favours Faeries, so v5 is clearly weaker against MonoW
although both models trained on gen0 MonoW games.

Chosen when available, gen0 / v4 / v5: Pass 0.28 / 0.46 / 0.43; Ninja 0.94 / 0.46 / 0.58; ninjutsu A 0.68 / 0.35 /
0.39, B 0.67 / 0.45 / 0.55; Seer 0.64 / 0.41 / 0.52; Miscreant 0.62 / 0.51 / 0.61; Elder 0.54 / 0.31 / 0.42;
Spellstutter 0.54 / 0.36 / 0.40; Saiba 0.63 / 0.42 / 0.48; Force Spike 0.36 / 0.26 / 0.33; Island 0.71 / 0.61 / 0.56.
Own-turn share: Spellstutter 0.90 / 0.80 / 0.86, Elder 0.76 / 0.71 / 0.82.
Seer / Miscreant (no Miscreant out): gen0 0.46 / 0.29, v4 0.30 / 0.37, v5 0.45 / 0.32 (74 states): v5 kept gen0's
Seer preference against MonoW but flipped against MonoB. v5 median value labels: Seer only −0.44, one Miscreant
−0.27, 2+ Miscreants −0.15.

Conclusion: dropping the ≈2,600 Pauper-pool games (421K states, both seats) costs a lot against MonoW (57.5% → 40.6%) and little or nothing
significant against MonoB, with both opponents in training. Behaviour sits between gen0 and v4 in both matchups.
Diverse opponent data made gen1's play against known opponents stronger, not just its predictions.

## PIMC with hidden information (ver5 gen1 MonoB `_pimc` shards)

**Caveat (Oct 29):** the "ver5 gen1 (perfect info)" row (26.5%) is the v5 (Mono-only) model, while the `_pimc` shards
were generated by the v4 model. The same-model perfect-info comparison is `MonoB2` (v4 model, 200 games, Faeries
WR 31.5%) or ver4's MonoB (31.6%); the PIMC drop (12.5–13.0%) holds against either.

Faeries ran PIMC with the encoder's hidden-information mode on (PlayerA), against a MonoB opponent with perfect
information (PlayerB). Compared with the perfect-info MonoB shards (win rate from Faeries' seat, 95% interval):

| shards | games | Faeries WR | Faeries states/game | opponent hand cards in Faeries' states |
|---|---|---|---|---|
| ver4 gen1 (perfect info) | 171 | 31.6% ± 7.0 | 84 | 3.63 |
| ver5 gen1 (perfect info) | 200 | 26.5% ± 6.1 | 88 | 3.31 |
| ver5 gen1 PIMC, hidden | 200 | **12.5% ± 4.6** | 75 | 0.00 |

- Faeries' recorded states hold the hidden-information encoding, not PIMC's determinized worlds. The
  opponent's hand zone keeps only its name leaf; its cards are gone.
- `H5Graphs(hide_opponent_hand=True)` is an exact no-op on these states. Applied to perfect-info states, it gives
  the opponent's hand zone the same shape. This check caught a bug, now fixed: the first version also cut the hand
  zone's name edge.

Conclusion: these states are exactly what a hidden-info model sees at play time, so the `_pimc` shards can be
trained on directly. Their win rate halves against a perfect-info opponent with the same budget.

### PIMC play vs perfect-info play (Faeries vs MonoB, gen1)

From the h5s (`results/pimc_play.py`, `src/magezero/util/behaviour.py`). Faeries' priority states with a choice;
"top move" = most visited.

| | ver4 perfect info | ver5 perfect info | ver5 PIMC hidden |
|---|---|---|---|
| WR | 31.6% | 26.5% | **12.5%** |
| priority choices per game, own / opponent's turn | 27.0 / 6.3 | 28.9 / 6.8 | 32.8 / **3.7** |
| flash creature or Strix loot castable on opponent's turn, states per game | 4.5 | 4.9 | **1.9** |
| Spellstutter per game: own turn / onto a spell / opponent's turn, empty stack | 0.80 / 0.15 / 0.22 | 0.93 / 0.10 / 0.21 | **1.19 / 0.02** / 0.10 |
| counterspells per game; used when castable on opponent's turn | 1.11; 52% | 1.29; 64% | 1.00; 58% |
| top move's visit share (own / opponent's turn) | 0.64 / 0.65 | 0.64 / 0.65 | 0.63 / 0.65 |
| search root value vs result, first third of the game | −0.08 vs −0.33 | −0.13 vs −0.47 | **−0.14 vs −0.71** |

- **PIMC Faeries acts at sorcery speed.** It reaches the opponent's turn with instant-speed plays available less
  than half as often. Spellstutter, Elder and Strix loot are chosen on its own turn 91%, 80% and 80% of the time,
  against 49–75%.
- **Spellstutter is cast as a creature, not a counter:** 0.02 times per game onto a spell, against 0.10–0.15.
- **Counterspells are used at the same rate when castable,** but castable less often.
- **Search confidence is unchanged, but root values overstate the result** by 0.57 early in the game, against
  0.25–0.34.

## gen1 → gen2 vs Standard MonoB (`ver4/training/gen2`, uploaded Oct 29)

Assumed: generated by the ver7 model (branch D: gen0 + all gen1, legal-only policy loss, no player names), Faeries
seat only, against offline MCTS MonoB. The opponent seat's share of decisions under 300 visits (0.42) matches the
offline opponents of gen1 (0.46–0.57), not self-play (0.23). Probed with `util/behaviour.py` and `results/pimc_play.py`.

| | gen0 | gen1 (v4) | gen1 `MonoB2` (v4) | gen2 (ver7) |
|---|---|---|---|---|
| games | 2,000 | 171 | 200 | 200 |
| Faeries WR (95% CI) | 7.4% | 31.6% ± 7.0 | 31.5% ± 6.4 | **32.0% ± 6.5** |
| states/game Faeries / MonoB | 60.1 / 47.5 | 84.1 / 47.2 | 100.0 / 55.0 | 76.9 / 46.4 |
| priority choices per game, own / opp turn | – | 27.0 / 6.3 | 32.2 / 8.4 | 26.9 / 5.7 |
| counterspells per game; used when castable on opp turn | – | 1.11; 52% | 1.38; 58% | 1.12; 59% |
| root value vs result, early game | – | −0.08 vs −0.33 | −0.08 vs −0.39 | −0.11 vs −0.36 |

Chosen when available, gen0 / gen1 / gen1 `MonoB2` / gen2 (gen2 states in brackets):

| ability | chosen | own-turn share |
|---|---|---|
| Pass | 0.33 / 0.50 / 0.52 / 0.48 (6508) | 0.81 / 0.75 / 0.73 / 0.76 |
| Humbling Elder | 0.42 / 0.23 / 0.23 / **0.37** (798) | 0.69 / 0.66 / 0.66 / 0.76 |
| Strix loot | 0.22 / 0.15 / 0.15 / 0.21 (575) | 0.80 / 0.49 / 0.60 / **0.77** |
| Spellstutter | 0.38 / 0.24 / 0.26 / 0.20 (1178) | 0.84 / 0.68 / 0.74 / 0.70 |
| Counterspell | 0.39 / 0.37 / 0.37 / 0.30 (388) | 0.03 / 0.05 / 0.05 / 0.01 |
| Miscreant | 0.63 / 0.53 / 0.54 / 0.46 (624) | 1.00 |
| Seer | 0.61 / 0.37 / 0.36 / 0.38 (747) | 1.00 |
| Ninja / ninjutsu A / B | 0.88 / 0.65 / 0.61 → 0.28 / 0.27 / 0.40 → 0.25 / 0.30 / 0.45 → 0.24 / 0.23 / 0.46 | 1.00 |
| Harrier Strix, Snap, Saiba, Of One Mind, Force Spike, Island | within ±0.06 of gen1 | unchanged |

Seer / Miscreant top-move share with no Miscreant out: gen0 0.47 / 0.33, gen1 0.17 / 0.44 (80 states), `MonoB2`
0.18 / 0.49 (82), gen2 **0.30 / 0.27** (135).

How well each model predicts gen2's search (`test.py`, CPU; priority accuracy over legal moves):

| model | seat | priority top-1 | non-pass / pass | target legal acc | use acc | value MSE |
|---|---|---|---|---|---|---|
| v4 (made gen1) | Faeries | 0.700 | 0.43 / 0.88 | 0.751 | 0.871 | 0.072 |
| ver7 (made gen2) | Faeries | 0.760 | 0.64 / 0.84 | 0.795 | 0.873 | 0.061 |
| v4 | MonoB | 0.628 | 0.51 / 0.89 | 0.821 | 0.950 | 0.080 |
| ver7 | MonoB | 0.746 | 0.75 / 0.74 | 0.840 | 0.902 | 0.067 |

Conclusion: gen2 is no stronger than gen1 against MonoB (32.0% vs 31.6% and 31.5%), although ver7 predicts every
head better than v4. The gen0 → gen1 patience shift stays; gen2 partly walks back two gen1 habits (Elder cast more
often, Seer again level with Miscreant) and moves Strix loot back to its own turn, each ≥3 SE. Its own predictions
match its search far better than v4's did on gen1 (non-pass 0.64 vs v4's 0.38 on gen1), so gen2 is mostly
the model's prior play; the WR plateau suggests MonoB is near what 300 sims can get with this deck and model, or
that better predictions no longer translate into better search.

### Search value (root `stateScore`) vs result, Faeries seat

| | gen0 | gen1 | gen1 `MonoB2` | gen2 |
|---|---|---|---|---|
| mean root value vs mean result | −0.06 vs −0.82 | −0.14 vs −0.33 | −0.17 vs −0.39 | −0.21 vs −0.36 |
| AUC (root value → win), early / mid / late third | 0.71 / 0.82 / 0.94 | 0.75 / 0.92 / 0.99 | 0.73 / 0.85 / 0.97 | 0.77 / 0.88 / 0.98 |
| root value bin ≈ −0.34 → actual result | −0.97 | −0.84 | −0.68 | −0.70 |
| root value bin ≈ +0.33 → actual result | −0.69 | +0.51 | +0.40 | +0.64 |
| mean step-to-step change | 0.041 | 0.027 | 0.026 | 0.028 |

Conclusion: no real change from gen1. gen2's root values are a little more pessimistic (bias +0.19/+0.22 → +0.16)
and as discriminative as gen1's (within the gap between the two gen1 shards). In all three model generations the
root value is compressed toward 0: a −0.34 position ends in a loss 85% of the time, a +0.33 position wins 70–82%.
The MonoB seat (offline heuristic) is unchanged, as expected.

## Value labels vs search root values, gen0 and gen1 (`results/value_histograms.png`)

Every ver4/ver5 training shard, bucketed (`util/value_hist.py` on a RunPod CPU pod, `results/value_hist_plot.py`).
Labels are the TD(0.92) targets; root = the search's `stateScore`. Both from the seat's perspective.
`ver4/training/gen0/PlayerB/Pauper-Faeries_vs_Standard-MonoW.hdf5` is the same file as the PlayerA one (same LFS
object, Faeries' seat) and is left out. `runpod/prepare.sh` already leaves it out of `gen0-all`, so branches A–D
and ver7–ver9 never trained on it twice.

| bucket | seat | games | seat WR | states | label mean | share |label| > 0.5 | root mean | share |root| > 0.5 |
|---|---|---|---|---|---|---|---|---|
| gen0 Pauper | Faeries seat | 2576 | 44.9% | 185,035 | +0.06 | 0.50 | +0.08 | 0.38 |
| gen0 Pauper | opponent seat | 2576 | 55.1% | 235,634 | +0.11 | 0.45 | +0.09 | 0.36 |
| gen0 Mono | Faeries seat | 9993 | 19.9% | 597,293 | -0.13 | 0.44 | -0.02 | 0.29 |
| gen0 Mono | opponent seat | 9993 | 80.1% | 527,734 | +0.35 | 0.54 | +0.23 | 0.37 |
| gen1 reg Pauper | Faeries seat | 780 | 68.7% | 77,644 | +0.13 | 0.51 | +0.10 | 0.43 |
| gen1 reg Pauper | opponent seat | 780 | 31.3% | 86,011 | +0.02 | 0.43 | +0.05 | 0.35 |
| gen1 reg Mono | Faeries seat | 1548 | 37.0% | 148,219 | -0.21 | 0.47 | -0.19 | 0.38 |
| gen1 reg Mono | opponent seat | 1548 | 63.0% | 113,386 | +0.33 | 0.56 | +0.29 | 0.43 |
| gen1 self-play Pauper | Faeries seat | 1600 | 34.4% | 153,815 | -0.12 | 0.48 | -0.09 | 0.39 |
| gen1 self-play Pauper | opponent seat | 1600 | 65.6% | 164,369 | +0.17 | 0.47 | +0.13 | 0.39 |
| gen1 self-play Mono | Faeries seat | 1199 | 18.6% | 98,256 | -0.29 | 0.50 | -0.22 | 0.38 |
| gen1 self-play Mono | opponent seat | 1199 | 81.4% | 60,700 | +0.39 | 0.63 | +0.28 | 0.44 |
| gen1 PIMC (MonoB) | Faeries seat | 600 | 14.7% | 42,124 | -0.38 | 0.52 | -0.30 | 0.39 |
| gen1 PIMC (MonoB) | opponent seat | 600 | 85.3% | 30,541 | +0.43 | 0.55 | +0.29 | 0.38 |
| gen1 policy (MonoB) | Faeries seat | 199 | 28.6% | 18,355 | -0.20 | 0.44 | -0.17 | 0.33 |
| gen1 policy (MonoB) | opponent seat | 199 | 71.4% | 10,034 | +0.27 | 0.48 | +0.19 | 0.29 |

Conclusion: root values are concentrated around 0 in every bucket (only 29–44% of states beyond ±0.5), and the labels
are wider only because the result enters near each game's end. gen0's root values ignore the matchup: Faeries'
roots vs Mono peak slightly above 0 (mean −0.02) at a 20% win rate. gen1's model-guided roots track it (−0.19 to
−0.30 in the losing Mono/PIMC buckets), but stay much closer to 0 than the results, so the labels keep that compression.

### The same labels recomputed with TD 0.97 and 0.99 (`results/value_histograms_label_0.99_label.png`, `..._0.97_label.png`)

`util/value_hist.py` rebuilds each game's labels from its root values and result, as `ParallelDataGenerator` does
(recomputing at 0.92 reproduces the stored labels to 4e-8). Label mean / share beyond ±0.5, against the mean result:

| bucket | seat | result (2·WR − 1) | TD 0.92 | TD 0.97 | TD 0.99 |
|---|---|---|---|---|---|
| gen0 Pauper | Faeries seat | -0.10 | +0.06 / 0.50 | +0.02 / 0.66 | -0.03 / 0.90 |
| gen0 Pauper | opponent seat | +0.10 | +0.11 / 0.45 | +0.14 / 0.59 | +0.18 / 0.84 |
| gen0 Mono | Faeries seat | -0.60 | -0.13 / 0.44 | -0.26 / 0.64 | -0.40 / 0.92 |
| gen0 Mono | opponent seat | +0.60 | +0.35 / 0.54 | +0.46 / 0.76 | +0.56 / 0.96 |
| gen1 reg Pauper | Faeries seat | +0.37 | +0.13 / 0.51 | +0.17 / 0.63 | +0.22 / 0.84 |
| gen1 reg Pauper | opponent seat | -0.37 | +0.02 / 0.43 | -0.03 / 0.55 | -0.10 / 0.79 |
| gen1 reg Mono | Faeries seat | -0.26 | -0.21 / 0.47 | -0.22 / 0.61 | -0.24 / 0.85 |
| gen1 reg Mono | opponent seat | +0.26 | +0.33 / 0.56 | +0.36 / 0.71 | +0.37 / 0.87 |
| gen1 self-play Pauper | Faeries seat | -0.31 | -0.12 / 0.48 | -0.17 / 0.61 | -0.24 / 0.85 |
| gen1 self-play Pauper | opponent seat | +0.31 | +0.17 / 0.47 | +0.21 / 0.60 | +0.28 / 0.82 |
| gen1 self-play Mono | Faeries seat | -0.63 | -0.29 / 0.50 | -0.38 / 0.67 | -0.50 / 0.92 |
| gen1 self-play Mono | opponent seat | +0.63 | +0.39 / 0.63 | +0.49 / 0.83 | +0.58 / 0.98 |
| gen1 PIMC (MonoB) | Faeries seat | -0.71 | -0.38 / 0.52 | -0.47 / 0.70 | -0.57 / 0.95 |
| gen1 PIMC (MonoB) | opponent seat | +0.71 | +0.43 / 0.55 | +0.56 / 0.77 | +0.69 / 0.98 |
| gen1 policy (MonoB) | Faeries seat | -0.43 | -0.20 / 0.44 | -0.25 / 0.61 | -0.33 / 0.88 |
| gen1 policy (MonoB) | opponent seat | +0.43 | +0.27 / 0.48 | +0.38 / 0.73 | +0.49 / 0.97 |

Conclusion: at 0.99 the horizon (≈100 decisions) exceeds a whole game (42–100 states per seat), so labels become
nearly the game result: 79–98% of states beyond ±0.5 (vs 43–63% at 0.92) and bucket means near 2·WR − 1. 0.97 sits
between. Higher TD removes the self-referencing compression, at the cost of noisier (near-binary) targets. (Label
means are per state and the result per game, so long games weigh more: hence gen1 reg Mono at −0.24 vs −0.26.)

## gen2 with priority priors on (ver7, vs offline MonoB and MonoW, uploaded Oct 30)

Only the opponents' seats were uploaded (`gen2/PlayerA/Standard-MonoB_vs_Pauper-Faeries_ppolicy.hdf5` is a MonoB-seat
file; neither shard has a Faeries decision), so Faeries' play comes from Will's console grep of the MonoB run
(`MCTSNode.bestChild` lines, 20:57–22:19, ≈4,600 Faeries priority decisions; `results/mcts_log.py`).

| matchup | gen2 value only | gen2 priors on |
|---|---|---|
| vs MonoB: Faeries WR | 32.0% ± 6.5 | 25.0% ± 6.0 |
| vs MonoW: Faeries WR | 48.2% ± 6.9 | **16.5% ± 5.1** |
| vs MonoW: opponent states/game | 96 | 61 |

Faeries, chosen when available (priority states), gen0 / gen1 / gen2 / gen2 with priors: Pass 0.33 / 0.50 / 0.48 /
**0.38**; Ninja 0.88 / 0.28 / 0.24 / **0.51**; Miscreant 0.63 / 0.53 / 0.46 / 0.64; Seer 0.61 / 0.37 / 0.38 / 0.56;
Elder 0.42 / 0.23 / 0.37 / 0.43; Harrier Strix 0.58 / 0.32 / 0.33 / 0.38; Saiba 0.63 / 0.35 / 0.37 / 0.41;
Counterspell 0.39 / 0.37 / 0.30 / 0.37; Spellstutter 0.38 / 0.24 / 0.20 / 0.28; Island 0.69 / 0.65 / 0.67 / 0.72.

In 17% of Faeries' priority decisions the most visited action is not the best scored (MonoB's offline search: 4%).
Almost all are an action chosen over a higher-scored Pass (Island, Spellstutter, Seer, Miscreant, Elder, ...), by a
small score margin (median 0.010) while Pass still gets ≈40% of the visits.
**Update (Oct 30):** the full ver10 value-only log (19K lines, all threads, ≈5,300 Faeries priority decisions)
has the mismatch at 5% (an earlier one-thread slice, 58 decisions, showed 12%), so the 17% under priors does come
from the priors.

Conclusion: with priors on, Faeries reverts toward gen0's play (casts threats on curve, passes less): the priority
head pulls visits toward acting. ver7's policy trained mostly on gen0 games (≈65% of its states), whose heuristic
search acted far more, so its priors carry gen0's style. gen1's gain over gen0 was the patience that value-only
search found, and the priors undo it; the MonoW collapse fits that.

## gen2 vs MonoB: ver9 (no gen0) with and without priors, and ver7 priors `_nodedup` (uploaded Oct 30)

Faeries' seat, `util/behaviour.py` and `results/pimc_play.py`. ver7 priors-on is from the console log (no Faeries shard).

| | gen1 (v4) | ver7, value only | ver7, priors (log) | ver7, priors `_nodedup` | ver9, value only | ver9, priors |
|---|---|---|---|---|---|---|
| Faeries WR | 31.6% | 32.0% ± 6.5 | 25.0% ± 6.0 | 28.5% ± 6.3 | 22.0% ± 5.7 | 28.5% ± 6.3 |
| Pass chosen | 0.50 | 0.48 | 0.38 | 0.45 | 0.45 | 0.40 |
| Ninja / Seer / Miscreant | 0.28 / 0.37 / 0.53 | 0.24 / 0.38 / 0.46 | 0.51 / 0.56 / 0.64 | 0.58 / 0.61 / 0.69 | 0.50 / 0.51 / 0.52 | 0.38 / 0.48 / 0.62 |
| priority choices on opp's turn / game | 6.3 | 5.7 | – | 3.2 | 4.7 | 3.7 |
| flash/loot castable on opp's turn, states / game | 4.5 | 3.8 | – | 1.8 | 3.4 | 2.4 |
| Spellstutter onto a spell / game | 0.15 | 0.12 | – | 0.04 | 0.09 | 0.09 |
| Spellstutter / Elder own-turn share | 0.68 / 0.66 | 0.70 / 0.76 | – | 0.93 / 0.91 | 0.74 / 0.78 | 0.85 / 0.83 |

ver10 (no gen0, no PIMC, no `_policy`), value only, from Will's console: **26.0%** (200 games, ±6.1). With priority priors: **29.5%** (200 games, ±6.3).

Conclusion: no variant beats ver7 value-only or gen1 (all within ≈2 SE). Dropping gen0 did not restore gen1's
patience: ver9's value-only search casts threats as often as ver7 with priors (Ninja 0.50, Seer 0.51). Every gen2
variant, most of all with priors, plays more at sorcery speed than gen1: fewer decisions on the opponent's turn,
fewer flash windows kept open, Spellstutter rarely onto a spell. That is the pattern of the PIMC games (§ PIMC play:
1.9 flash states/game, 0.02 Spellstutter onto a spell, 12.5% WR), which are in both ver7's and ver9's training
data; whether they cause it is untested.

### ver10 value only, and `_ptpolicy_gen1only` (uploaded Oct 30)

`_gen1nopimconly` = ver10, value only; `_ptpolicy_gen1only` = assumed ver9 with priority and target priors.

| | gen1 (v4) | ver7, value | ver9, value | ver10, value | ver9, priority + target priors |
|---|---|---|---|---|---|
| Faeries WR | 31.6% | 32.0% | 22.0% | 26.0% | 23.5% |
| Pass chosen | 0.50 | 0.48 | 0.45 | 0.49 | 0.41 |
| Ninja / Seer / Miscreant | 0.28 / 0.37 / 0.53 | 0.24 / 0.38 / 0.46 | 0.50 / 0.51 / 0.52 | 0.39 / 0.44 / 0.57 | 0.49 / 0.51 / 0.63 |
| priority choices on opp's turn / game | 6.3 | 5.7 | 4.7 | 5.6 | 3.9 |
| flash/loot castable on opp's turn / game | 4.5 | 3.8 | 3.4 | 4.3 | 2.3 |
| Spellstutter onto a spell / game | 0.15 | 0.12 | 0.09 | 0.10 | 0.07 |

Conclusion: ver10 brings back most of gen1's reactive style (passes as often, keeps nearly as many flash windows
open on the opponent's turn) yet wins 26%, so the sorcery-speed shift does not explain the gen2 plateau. Priors
(priority, or priority + target) again push toward acting on Faeries' own turn.

### How decisive the search is (console logs, `results/mcts_gaps.py`)

Faeries priority decisions; option scores are the search's mean values per root child.

| log | decisions | best − second score, median | share within 0.02 | best − worst option, median | Pass is best-scored |
|---|---|---|---|---|---|
| ver7, priors on | 4,643 | 0.033 | 35% | 0.048 | 50% |
| ver10, value only | 5,289 | 0.028 | 38% | 0.041 | 47% |

Conclusion: the search barely separates its options. The best and worst option of a decision typically differ by
0.04–0.05 in value (on a ±1 scale), and over a third of decisions are within 0.02 between the top two, about the
noise of 300 simulations. Which move is played is then close to a coin flip at many decisions, whichever model
evaluates them, which fits every gen2 model landing at the same win rate.

## gen3 data vs the Pauper pool: the gen2 model (ver10) against the offline heuristic search

RunPod (`runpod/configs/gen3-offline-*.yml`, value only, 300 sims, 200 games each; shards in
`/workspace/runs/gen3-offline`). gen1 = v4 vs offline: Affinity, Burn, CawGates, Elves from the repo's gen1 shards;
mirror, Rally, Spy, Wildfire from Will's `Mage.MageZero/WinRates.txt` lines 542–547 (v4 vs offline, confirmed
Oct 30; those games are not in the repo). gen0 = heuristic vs heuristic (ver4 gen0 shards).

| deck | gen0 | gen1 (v4) | gen2 model (ver10) | gen2 − gen1 |
|---|---|---|---|---|
| Affinity | 40% (100) | 63.5% | 77.0% | +13.5 |
| Burn | 56% (100) | 66.5% | 61.5% | −5.0 |
| CawGates | 61.8% | 81.1% | 74.5% | −6.6 |
| Elves | 35.5% | 65.0% | running | |
| Faeries mirror | 50.9% | 72.0% | 49.5% | −22.5 |
| Rally | 15.2% | 30.0% | 17.0% | −13.0 |
| Spy | 46.7% | 59.0% | 54.0% | −5.0 |
| Wildfire | 57.7% | 72.0% | running | |

Conclusion (six decks so far): ver10 averages 55.6% against v4's 62.0% (≈3 SE), a regression from gen1, largest in
the mirror and against Rally, where it falls back to gen0's level.
