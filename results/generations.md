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