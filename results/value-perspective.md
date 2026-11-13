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
| v4-downdown-players | down-down + per-player value | .032 / .031 | .063 / .061 | | 9.12M | 4090, ~$0.80 |
| v4-owner-players | owner injection into the global tokens + per-player value | .033 / .031 | .138 / .126 | .682 / .685 | | 4090 |
| v4-dd-players-td003 | down-down + per-player value, `--td first:0.03` (different labels: value MSE not comparable) | .044 / .041 | .065 / .063 | .790 / .782 | 9.12M | 4090 |
| v4-dd-players-legal | down-down + per-player value, legal-only policy loss | .076 / .070 | .052 / .055 | .800 / .794 | 9.12M | 4090 |
| v4-dd-players-legalhead | as legal, plus a shared legality head (weight 1.0) | .042 / (.041 both) | .047 / | .805 / | 9.12M | 4090 |

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

## Down-down piloting Burn vs offline Burn (300 sims both seats, pruning on, XMage `b6ca7b35`, 200 games)

| pilot | win | Pass top, first 3 drops | first 4 | games with a skip | won with skip / without |
|---|---|---|---|---|---|
| ver15 (gen5 shard) | 47% | 54.8% | 53.5% | 106 | 23% / 54% |
| P1 (Will's run) | 37.5% | 57.5% | 55.2% | 105 | 24% / 53% |
| v4-downdown (gen0 Faeries data only) | 43.0% | 19.3% | 17.6% | 23 | 39% / 44% |
| v4-downdown-players (gen0 only) | 46.0% | 24.1% | 24.3% | 42 | 29% / 51% |

Down-down skips a third as often as ver15 and P1, from v4's data alone. RTX 5090, ~1.3 h, ~$1.75.

## ver15 recipe variants (warm start from the v4 variant, gen0-gen3, lr 3e-5, 3 epochs; value MSE / prio legal KL)

| set | ver15 | v15-noname | v15-entityattn |
|---|---|---|---|
| gen0 | .037 / .060 | .039 / .062 | .039 / .070 |
| gen1-reg | .040 / .056 | .039 / .057 | .039 / .062 |
| gen1-selfplay | .031 / .053 | .031 / .054 | .032 / .057 |
| gen2-offline | .040 / .048 | .040 / .049 | .042 / .053 |
| gen2-vsgen1 | .036 / .051 | .035 / .052 | .039 / .055 |
| gen2-selfplay | .037 / .057 | .038 / .059 | .042 / .063 |
| gen3-offline | .035 / .045 | .035 / .046 | .037 / .052 |
| gen3-vsgen1 | .039 / .050 | .037 / .051 | .040 / .055 |
| gen3-vsgen2 | .041 / .053 | .041 / .055 | .044 / .059 |
| gen3-selfplay | .040 / .055 | .042 / .057 | .046 / .061 |
| pimc | .060 / .085 | .070 / .087 | .065 / .098 |
| policy | .052 / .066 | .052 / .067 | .054 / .072 |
| gen0-test | .031 / .052 | .032 / .051 | .033 / .056 |
| ub | .050 / .078 | .041 / .076 | .045 / .070 |

Dropping player names is level with ver15 except the small pimc (worse) and ub (better) sets. The entity transformer
is .002-.006 worse on value and on policy almost everywhere. PRO 6000, ~$14 each.

Legal-only policy loss breaks value generalisation on down-down + per-player value: training value .032, held-out
.072-.082 from epoch 2 on (all-node loss: .032 held out). The legality head recovers half of it (.041) and gives the
best policy so far (KL .047). legalhead's pod was reaped before test.log was written; its numbers come from gnn.log
(PlayerB line missing; .041 is the combined end-of-training validation).
