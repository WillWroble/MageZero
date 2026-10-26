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
EOF
cd /home/user/MageZero && git add src/magezero/util/behaviour.py && git add -f results/generations.md && git commit -q -m "util/behaviour.py and gen0 vs gen1 behaviour against MonoB

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01L2xxgZSAQNvWem2yZsXCsZ" && git pull -q --no-rebase origin graph-encoder-test 2>&1|tail -1; git push -q origin graph-encoder-test 2>&1|tail -1; git log --oneline -1
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
