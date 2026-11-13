# Value perspective experiments (v4 recipe)

All runs: v4's recipe (ver4 gen0 PlayerA + PlayerB minus PlayerB's MonoW, `policy_domain all`, stored TD 0.92 labels,
4 epochs, seed 0, `--val-every 10000`), tested on the gen0 mirror test set (`gen0-test`, PlayerA / PlayerB shards).

## |value| (`v4-abs`, `--td abs`): does the sign cost v4 capacity?

Target: |stored label|. Compared with v4's own prediction made absolute (`abs_baseline.py`), MSE against |label|:

| model | PlayerA | PlayerB |
|---|---|---|
| v4, abs of its prediction | .025 | .024 |
| v4-abs, best check (epoch 4, step 10K) | .021 | .020 |
| v4-abs, end of epoch 4 | .024 | .023 |

v4 gets the sign wrong on 11.7% / 10.6% of states. A net that never predicts the sign fits |label| no better than v4
at the end of training (.001 apart; .004 at its best check). Conclusion: the sign, i.e. perspective, is not where
v4's value error comes from; the hard part is how decisive a position is. RTX 4090, ~1.2 h, ~$1.05.

## Value head and encoding variants, end of epoch 4 on gen0-test (PlayerA / PlayerB)

| run | change from v4 | value MSE | prio KL | prio top-1 | params | pod, cost |
|---|---|---|---|---|---|---|
| v4 (abs_baseline.py) | | .037 / .033 | | | 9.10M | |
| v4-rootvalue | value head reads the ROOT token, not CLS | .036 / .033 | .168 / .175 | .676 / .664 | 9.12M | PRO 6000, ~$3.60 |
| v4-noname | `--drop-player-names` | .037 / .032 | .173 / .171 | .654 / .647 | 9.12M | PRO 6000, ~$2.80 |
| v4-entityattn | Dama's entity encoder (objects + ancestor context, owner-aware), 15 attention layers d256 FF512 over [CLS, entities]; legal-only policy loss | .038 / .037 | .050 / .051 | .799 / .794 | 9.04M | 4090, ~$1.20 |
| v4-downdown | both local passes top-down (`directions: [down, down]`): each node attends over its leaves and parents | .034 / .033 | .068 / .068 | .782 / .774 | 9.12M | 4090, ~$1 |
| v4-playervalue | value = tanh(own PLAYER token's score − opponent's) (`value_token: players`) | .034 / .032 | .170 / .172 | .659 / .652 | 9.12M | 4090, ~$1 |
| v4-updown | pass 1 bottom-up, pass 2 top-down (`directions: [up, down]`) | .034 / .032 | .069 / .066 | | 9.12M | 5090 (slow pod), ~$6 |

Prio KL / top-1 of entityattn are not comparable with the all-node runs (legal-only loss; cf. `gnnlegal` .052 / .795
in mlp-vs-gnn.md). Conclusions: reading value from ROOT changes nothing; dropping the player names costs nothing on
held-out value; the owner-aware entity net with message passing replaced by global attention is level on value
(.001-.004 worse) and level with the legal-loss GNN on policy.

## Opponent-turn land test (P1's Burn games, positions where P1 skipped its last drop, Mountain in hand)

Share of positions where the value goes up / down. "Flag" clears CanPlayLand only; "played" also moves the Mountain
from hand to battlefield (hand size -1, permanent count +1).

| network | 300-sim: flag | played, all | played, lost games | 2 lands | 1000-sim: flag | played, all | played, lost |
|---|---|---|---|---|---|---|---|
| w=0.03 | 0 / 0% | 29 / 70% | 23 / 75% | 24 / 74% | 0 / 0% | 48 / 50% | 42 / 55% |
| ver15 | 34 / 49% | 46 / 51% | 43 / 54% | 39 / 58% | 41 / 44% | 73 / 24% | 71 / 27% |
| P1 | 17 / 66% | 44 / 53% | 41 / 56% | 37 / 60% | 12 / 76% | 71 / 26% | 70 / 28% |
| v4-playervalue | 27 / 55% | 39 / 59% | 34 / 64% | 31 / 67% | 23 / 58% | 63 / 36% | 57 / 42% |
| v4-updown | 4 / 0% | 62 / 35% | 60 / 37% | 58 / 39% | 6 / 0% | 62 / 35% | 57 / 39% |
| v4-downdown | 0 / 19% | 86 / 14% | 86 / 14% | 84 / 15% | 0 / 22% | 78 / 22% | 75 / 24% |

Two top-down passes give the best land judgement, and the only one that holds in lost games; one top-down pass is
between it and the bottom-up models. The per-player value head alone doesn't change it.

## Pending
- `v15-noname` (ver15 recipe from v4-noname, `--drop-player-names`), 
