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
| v4-dd-players-legal-noname | legal, `--drop-player-names` | .036 / .034 | .045 / .046 | .816 / .814 | 9.12M | PRO 6000 |
| v4-dd-players-legalhead-noname | legalhead, `--drop-player-names` | .035 / .034 | .045 / .046 | .822 / .819 | 9.12M | 4090 |

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
| v4-dd-players-legal-noname | 73 / 17% | 56 / 39% | 54 / 41% | 58 / 38% | 69 / 23% | 81 / 16% | 83 / 14% |

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
Dropping the player names removes the overfit: legal-only goes from .073 to .035 held out (legality head .035), with
the best policy yet (KL .045). The names let the legal-only net memorise value by seat. Both ~$2 of training; the
pods then sat idle 4.4 h while this session had no RunPod access (~$16 wasted).

## Ablation of the down-down variants (P1's Burn mirror, 300 + 1000 sims, 5,000 states; `attrib.py`)

Value MSE against the stored labels (TD 0.92) with nothing removed, then with one thing removed (its subtree goes
with it). Gradient x input: share of the value attribution that sits on PLAYER nodes and their leaves.

| model | baseline | − IsDecisionPlayer | − IsActivePlayer | − LifeTotal | − ZONE | − PERMANENT | − CARD | PLAYER share |
|---|---|---|---|---|---|---|---|---|
| P1 (trained on this matchup) | .048 | .644 | +.006 | +.314 | .207 | .132 | .083 | 38% |
| v4-noname | .090 | .349 | +.010 | +.095 | .205 | .172 | .096 | 35% |
| v4-downdown | .078 | .834 | +.003 | +.151 | .320 | .193 | .121 | 26% |
| v4-downdown-players | .087 | .973 | +.010 | +.088 | .183 | .179 | .129 | 40% |
| v4-dd-players-td003 | .088 | .991 | +.008 | +.177 | .182 | .207 | .111 | 38% |
| v4-dd-players-legal | .314 | .827 | −.047 | +.067 | .228 | .169 | .448 | 26% |
| v4-dd-players-legalhead | .221 | .918 | +.003 | +.199 | .188 | .126 | .381 | 32% |
| v4-dd-players-legal-noname | .076 | .667 | +.061 | +.106 | .174 | .166 | .087 | 33% |
| v4-dd-players-legalhead-noname | .076 | .642 | +.063 | +.104 | .208 | .158 | .094 | 37% |

Removing every PLAYER node gives .281 for all per-player-value models: that is the value-0 floor (mean label²), not
a measurement. Conclusions: the legal-loss models with player names don't transfer to this unseen matchup (.31 /
.22 vs .076 without names). Every down-down model leans harder on IsDecisionPlayer than P1 does, except the two
no-name legal-loss ones (.67 / .64, level with P1); those are also the only ones that use IsActivePlayer (+.06,
6x the others). Spurious single-leaf dependencies on features absent from Burn games appear in the all-node models
(down-down: GreenMana +.139; td003: Battlefield +.194, BlueMana +.095) but not in the no-name legal ones.

Removing IsDecisionPlayer is not a fair test of the per-player-value models: the leaf also picks which PLAYER token
gets the + sign. Swap test instead (`swap_symmetry.py`, the flag moved to the other player, 2,000 of the same
states): if the network used the flag only for the sign, the value would exactly negate.

| model | value MSE | swapped vs −label | mean abs(v_swap + v) | mean abs(v) | corr(v_swap, −v) |
|---|---|---|---|---|---|
| P1 | .046 | .078 | .122 | .368 | .91 |
| v4-downdown | .076 | .095 | .128 | .332 | .91 |
| v4-downdown-players | .091 | .087 | .099 | .344 | .94 |
| v4-dd-players-legal-noname | .076 | .081 | .060 | .318 | .98 |
| v4-dd-players-legalhead-noname | .078 | .084 | .055 | .315 | .98 |

The per-player value models mostly use the flag for the sign; the no-name legal-loss ones almost only for that.

## Sink or information? Same ablations for every model (`util/ablate.py`, `runpod/ablate.sh`, 2,000 states of P1's Burn mirror)

The mana-pool leaves (WhiteMana ... ColorlessMana) are one node per state shared by both PLAYERs, value 0 in 100% of
states (autopay: mana never floats at a decision), so they carry no information. The earlier "spurious leaves" were
removal effects of these constant tokens. Here every model gets the same tests: PLAYER local-layer attention mass,
removal of the mana leaves and Battlefield, and value shuffling (the token stays, its value moves to another state).

gen0-test of the new runs (value MSE A / B, priority KL A): all-node no-name flag visible (`v4-dd-players-noname`)
.035 / .030, .072; flag hidden .034 / .031, .080; flag hidden + 4 registers (`v4-dd-players-hidden-reg4`) .031 / .029, .076.

| model | Burn MSE | mana attention (pass 0 / 1) | − all mana | − Battlefield | ~ LifeTotal | ~ BattlefieldSize | ~ ManaValue |
|---|---|---|---|---|---|---|---|
| P1 | .046 | .00 / .05 | +.001 | +.001 | +.220 | +.077 | +.007 |
| v4-noname | .091 | .00 / .03 | +.000 | +.004 | +.129 | +.037 | +.002 |
| downdown (names) | .076 | .13 / .01 | +.191 | +.000 | +.153 | +.008 | +.035 |
| dd-players (names) | .091 | .02 / .01 | −.000 | +.008 | +.146 | +.013 | +.022 |
| td003 (names) | .092 | .05 / .13 | +.182 | +.190 | +.177 | +.026 | +.013 |
| td003, no names | .077 | .20 / .02 | +.016 | +.001 | +.207 | +.022 | +.019 |
| all-node, no names | .078 | .03 / .11 | +.002 | +.035 | +.181 | +.013 | +.010 |
| all-node, no names, hidden | .078 | .21 / .01 | +.004 | +.005 | +.174 | +.009 | +.014 |
| all-node, no names, hidden, 4 registers | .079 | .04 / .08 | +.002 | +.013 | +.166 | +.017 | +.010 |
| legal, no names | .076 | .02 / .01 | +.000 | +.000 | +.180 | +.074 | +.017 |
| legal + legality head, no names | .078 | .03 / .05 | −.000 | +.001 | +.158 | +.064 | +.008 |
| legal, no names, hidden | .086 | .02 / .03 | −.000 | +.002 | +.166 | +.036 | +.014 |

Other shuffles (HandSize, GraveyardSize, LibrarySize, Power, Toughness) are ≤ +.02 for every model.

Conclusions:
- The big removal effects (+.13 to +.19) come only from the two named all-node models. Without names, all-node models
  survive removing every mana leaf (≤ +.016). "All-node leans on garbage features" was wrong: it was a names effect
  and a removal artefact.
- No-name all-node models do park 11–21% of one pass's PLAYER attention on the constant mana leaves (legal-only:
  ≤ 5%), so they use them as attention sinks, but harmlessly. With 4 global registers the hidden model's mana
  attention fell from .21 to .04 in pass 0 (one seed; the registers sit in the global layers, not the local ones).
- Information use is similar across the loss types: LifeTotal dominates everywhere (+.13 to +.22). Legal-only models
  use BattlefieldSize more (+.04 to +.07 vs +.01 to +.02), closer to P1 (+.08).
- On this unseen matchup, all-node no-name (.078) and legal no-name (.076) are level; registers change nothing (.079).
  Registers give the best gen0-test value so far (.031 / .029), within the noise of one seed.

## ddp (down-down + per-player value, names kept, all-node) in play and the land-skip rate

Faeries vs offline, 300 sims, value only, `max_pending` 4, XMage `b6ca7b35`, RunPod 4090s (`/workspace/runs/ddp-*`).
v4 from `generations.md` (XMage `f2ce8eca`; mirror, Rally, Spy, Wildfire from Will's machine).

| | Affinity | Burn | CawGates | Elves | mirror | Rally | Spy | Wildfire | mean |
|---|---|---|---|---|---|---|---|---|---|
| ddp | 67.5 | 57.5 | 80.3 (198) | 72.0 | 67.5 | 30.5 | 60.0 | 75.0 | 63.8 |
| v4 | 63.5 | 66.5 | 81.1 | 65.0 | 72.0 | 30.0 | 59.0 | 72.0 | 63.6 |

Land skips (`util/land_pass.py`, `runpod/land-pass.sh`): share of the decision player's own main-phase priority
states with a land play legal where the search's top move is Pass, at < 3 own lands (< 4 in brackets).
gen0 = heuristic vs heuristic; v4 games exist only for the first four decks.

| deck | gen0 | v4 | ddp | ver15 |
|---|---|---|---|---|
| Affinity | 3.7 (5.5) | 13.8 (19.7) | 7.1 (10.4) | 21.5 (26.9) |
| Burn | 3.4 (4.1) | 14.2 (17.5) | 8.3 (10.2) | 21.8 (26.2) |
| CawGates | 3.8 (4.3) | 5.2 (9.2) | 5.6 (6.5) | 8.3 (14.2) |
| Elves | 4.2 (4.9) | 9.3 (11.8) | 4.6 (6.5) | 9.4 (15.3) |
| mirror | 3.2 (4.3) | | 3.0 (3.8) | 7.0 (10.8) |
| Rally | 3.9 (4.3) | | 4.1 (6.7) | 6.4 (9.8) |
| Spy | 4.0 (5.8) | | 6.9 (9.5) | 12.5 (19.0) |
| Wildfire | 5.3 (6.1) | | 6.4 (9.1) | 27.4 (33.4) |

ddp is a gen1 model like v4 (trained on gen0 only). It skips about half as often as v4 on the four shared decks
and stays near gen0 on the mirror, Rally, CawGates and Elves, while winning as often (63.8% vs 63.6% mean).
ver15 skips most everywhere: the degeneration compounded v4 -> ver15; ddp starts from a lower point.

## Does the policy domain change value in play? Faeries mirror, all-node vs legal-only (Nov 13)

`v4-dd-players-noname` (all-node) vs `v4-dd-players-legal-noname` (legal-only): both down-down, per-player value,
flag visible, no names, v4 recipe, seed 0, epoch-4 checkpoints; they differ only in `policy_domain`. Value-only search
(priors off), 300 sims, `max_pending` 4, XMage `b6ca7b35`, 4090s; 400 games per test as two 200-game pods
(`mir-*`, `h2h-*`; `runpod/configs/norm-Faeries.yml`, `h2h-Faeries.yml`).

| test | all-node | legal-only |
|---|---|---|
| gen0-test value MSE (A / B), final checkpoint | .035 / .030 | .036 / .034 |
| held-out value, mean over epochs 3-4 | .0323 | .0372 |
| vs offline mirror | 74.8% (79.5, 70.0) | 78.8% (77.5, 80.0) |
| head-to-head (seat A / seat B) | 47.0% (47.5 as A, 46.5 as B) | 53.0% |

Legal-only wins both tests by 4-6 points, each about 1.2-1.3 SE: same direction, not significant alone. The all-node
loss's lower held-out value MSE (.003-.005) did not turn into wins. Cost ≈ $7 (six 4090 pods).

Land skips and value MSE on those 1,600 mirror games (`runpod/mirror-eval.sh`; land skips on each model's own
decisions, pooled over its vs-offline and head-to-head shards; Pass top at < 3 own lands, < 4 in brackets):

| | all-node | legal-only | gen0 | ddp | ver15 |
|---|---|---|---|---|---|
| land skips | 3.2% (4.3%) | 6.1% (10.1%) | 3.2% (4.3%) | 3.0% (3.8%) | 7.0% (10.8%) |

Every one of the 8 shards agrees: legal-only 5.6-6.5% (8.9-11.5%), all-node 2.1-3.8% (3.4-5.4%).

| value MSE (mean over shards) | all-node model | legal-only model |
|---|---|---|
| all-node's games vs offline (both seats, 4 shards) | **.0324** | .0335 |
| legal-only's games vs offline (4 shards) | .0351 | **.0326** |
| head-to-head games (4 shards) | **.0393** | .0404 |

Each model predicts its own games about .0025 better (8 of 8 shards); on the shared head-to-head games all-node is
.001 better (3 of 4). Legal-only skips early land drops twice as often (ver15's level, all-node stays at gen0's)
and still won more games.

### Retrained with every 5th gen0 game held out (`v4-dd-players-noname-h5`, `v4-dd-players-legal-noname-h5`, Nov 14)

Same recipe as the two models above, with `train gen0 holdout 5` (80% of gen0 trained on; the held-out games are val
set "gen0", 26 shards: every opponent, both seats). 4090 and 5090, ≈ $2.30 together.

| | all-node | legal-only |
|---|---|---|
| gen0 held-out games, value MSE: final / mean of epochs 3-4 | .036 / .0363 | .037 / .0380 |
| gen0-test, value MSE: final / mean of epochs 3-4 | .036 / .0345 | .036 / .0365 |
| priority KL, final (gen0 held out / gen0-test) | .092 / .064 | .064 / .046 |
| shards with the lower final value MSE (of 26, 1 tie) | 14 | 11 |

Per shard (final): all-node is clearly lower on Elves (.034 vs .040), CawGates (.036/.040), MonoW (.040/.046) and
MonoB (.031/.034) as Faeries; legal-only on Spy (.052 vs .063), Affinity and Burn (both seats, small shards). The
gen0-test gap of the first pair (.032 vs .035 final) shrank to nothing here (.036 vs .036): about .001-.002 of value
MSE is all the policy domain buys, within seed and data noise.
