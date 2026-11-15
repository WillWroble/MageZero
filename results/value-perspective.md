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
| v4-dd-players-noname (all-node) | 9 / 3% | 12 / 85% | 9 / 87% | 10 / 87% | 11 / 3% | 38 / 57% | 34 / 61% |
| v4-dd-players-noname-h5 (same, 80% of gen0) | 71 / 13% | 85 / 11% | 84 / 11% | 87 / 9% | 66 / 22% | 81 / 15% | 84 / 12% |
| v4-dd-players-legal-noname-h5 | 80 / 9% | 62 / 35% | 61 / 36% | 58 / 38% | 73 / 16% | 68 / 28% | 71 / 25% |
| v4-downdown-players (names kept) | 1 / 0% | 85 / 12% | 85 / 12% | 82 / 14% | 1 / 0% | 76 / 18% | 78 / 16% |

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

Land skips and value MSE on those 1,200 mirror games (800 per model: its 400 vs offline and all 400 head-to-head) (`runpod/mirror-eval.sh`; land skips on each model's own
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

Opponent-turn land test on the all-node model of the mirror games (`v4-dd-players-noname`, row added to the table
above): playing the skipped Mountain lowers its value in 85% of positions (300-sim games) and 57% (1000-sim), the
wrong way round and worse than every other model, and it barely reacts to the flag (86-88% unchanged). Its legal-only
twin values the drop correctly. Yet in its own mirror games the all-node model skipped early drops half as often
(3.2% vs 6.1%): this counterfactual (P1's Burn mirror, a deck neither model trained on) does not predict land play
in the models' own games.

The holdout retrain of the same all-node config (`-h5`: 80% of gen0, otherwise identical) flips the result: played
85% up, flag 71% up. v4-downdown-players (per-player value, names kept) is also 85% up. So neither the value head nor
the names explains the all-node no-name model's 12% / 85%: it is training variance. The test moves by ~70 points
between two trainings of one config; single-model results on it (including "only legal-noname gets the flag right")
are not evidence about an architecture.

Opponent-turn land test along the original all-node run (`v4-dd-players-noname`, saved checkpoints; up / down, 300-sim
set, 1000-sim in brackets):

| checkpoint (step) | flag | played |
|---|---|---|
| epoch 2 end (48K) | 11 / 5% (13 / 5%) | 69 / 24% (69 / 23%) |
| epoch 3, 10K (58K) | 3 / 0% (5 / 0%) | 27 / 67% (49 / 45%) |
| epoch 3, 20K (68K) | 27 / 4% (28 / 6%) | 40 / 53% (61 / 31%) |
| epoch 3 end (72K) | 15 / 0% (17 / 1%) | 19 / 75% (39 / 54%) |
| epoch 4, 10K (82K) | 55 / 1% (52 / 1%) | 14 / 82% (34 / 60%) |
| epoch 4, 20K (92K) | 13 / 1% (18 / 1%) | 20 / 75% (44 / 49%) |
| epoch 4 end (97K) | 9 / 3% (11 / 3%) | 12 / 85% (38 / 57%) |

Neighbouring checkpoints differ by up to 20-40 points, and this run drifts the wrong way from epoch 3 on (69% up at
epoch 2, 12-20% by epoch 4). The holdout retrain's final checkpoint (77K steps) is 85% up, where this run was at 19%
at a similar step: the gap between the two runs is not the step count. The test varies a lot between checkpoints and runs,
so test every candidate checkpoint; it does track skipping in play (correction, Nov 15: ver15 46% / P1 44% up skip
55-58% of Burn drops, down-down 85-86% up skips 19-24%).

### The holdout retrains in play (Nov 14; final checkpoints, which are also their best or within .0002 of it)

Same settings as the first pair; 200 games per test, head-to-head with all-node as seat A (`mir-*-h5`, `h2h-h5`).

| | all-node (-h5) | legal-only (-h5) | pooled with the first pair (600 games per test) |
|---|---|---|---|
| vs offline mirror | 64.0% | 72.0% | 71.2% vs 76.5% (+5.3, ≈2.1 SE) |
| head-to-head | 44.0% | 56.0% | 46.0% vs 54.0% (≈2.0 SE) |
| land skips, own decisions, < 3 (< 4) lands | 4.1% (8.2%) | 4.2% (7.5%) | first pair: 3.2% (4.3%) vs 6.1% (10.1%) |
| value MSE on all 6 shards of these games (mean) | .0425 | **.0393** | |

Legal-only wins all four game comparisons across two independent pairs: its value head plays better despite
all-node's ≈.001 lower held-out value MSE, and it also predicts these games better (.0393 vs .0425; legal-only's own
vs-offline games .040 vs .051). The first pair's 2x land-skip gap did not reproduce (≈4% each): it was run-specific.

## Do MLPs skip land drops too? (Nov 14)

Land skips in existing MLP games vs offline (`runpod/land-pass.sh` with `SHARDS`; Pass top at < 3 own lands, < 4 in
brackets). gen1 MLP = `mlp-gen0all` (gen0 data, as v4); F2-mlp = ver15's MLP twin (same data, e6 end).

| deck | gen0 | gen1 MLP | v4 (GNN) | F2-mlp | ver15 (GNN) |
|---|---|---|---|---|---|
| Affinity | 3.7 (5.5) | 12.0 (18.6) | 13.8 (19.7) | | 21.5 (26.9) |
| Burn | 3.4 (4.1) | 10.2 (14.6) | 14.2 (17.5) | 15.7 (20.2) | 21.8 (26.2) |
| CawGates | 3.8 (4.3) | 7.3 (12.8) | 5.2 (9.2) | | 8.3 (14.2) |
| Elves | 4.2 (4.9) | 8.1 (13.8) | 9.3 (11.8) | | 9.4 (15.3) |
| mirror | 3.2 (4.3) | 4.7 (8.0) | | 4.2 (9.2) | 7.0 (10.8) |
| Rally | 3.9 (4.3) | 4.8 (7.4) | | 4.7 (6.9) | 6.4 (9.8) |
| Spy | 4.0 (5.8) | 10.7 (18.0) | | | 12.5 (19.0) |
| Wildfire | 5.3 (6.1) | 14.4 (25.6) | | | 27.4 (33.4) |

The MLPs recreate the land skipping: gen1 MLP is at v4's level (2-4x gen0), and F2-mlp is between gen1 and ver15 on
Burn. So the behaviour comes from the data and search loop, not the GNN. F2-mlp skips ≈30% less than ver15 on the
same data.

Opponent-turn land injection test on the same MLPs (P1's Burn mirror; up / down):

| network | 300: flag | played | lost | 2 lands | 1000: flag | played | lost |
|---|---|---|---|---|---|---|---|
| gen1 MLP | 62 / 24% | 35 / 62% | 31 / 66% | 34 / 63% | 57 / 30% | 60 / 37% | 57 / 40% |
| F2-mlp | 69 / 16% | 76 / 20% | 74 / 22% | 76 / 20% | 64 / 21% | 94 / 6% | 92 / 8% |

F2-mlp aces the injection test and skips ≈30% less than ver15 on Burn (15.7% vs 21.8%), on the same data: consistent
with the injection test tracking skips in play.

## Is down-down's land-injection result real? 40 models and checkpoints (`util/land_inject.py`, Nov 14)

Share of positions whose value goes up when the skipped Mountain is played ("played"), P1's Burn mirror, 300-sim
shard (1000-sim in brackets). All final checkpoints unless noted; ver13 skipped (LFS pointer on the volume).

| group | models | played up: each | mean |
|---|---|---|---|
| up-up | v4, v4-noname, rootvalue, playervalue, tdfirst 0007/003/006/01/025, mix0/05/1b | 44 49 60 39 44 29 42 29 17 33 89 41 (73 79 85 63 72 48 65 53 27 68 92 23) | 43% (62%) |
| up-down | v4-updown | 62 (62) | |
| down-down | downdown, dd-players, td003, td003-nn, noname, noname-h5, legal, legalhead, legal-nn, legalhead-nn, legal-nn-h5, hidden, legal-hidden, reg4 | 86 85 27 63 12 85 38 32 56 62 62 90 60 62 (78 76 26 59 38 81 45 33 81 81 68 84 67 66) | 59% (63%) |

Along one run (epochs 1, 2, 3, 4; 300-sim): v4-noname 70 75 36 49; v4-playervalue 60 38 39 39; v4-downdown 79 75 82 86;
v4-downdown-players 48 68 71 85.

Down-down's 86% was the top of its distribution: down-down models average 59% on the 300-sim shard against up-up's
43% (≈16 points, a real but modest shift), and nothing on the 1000-sim shard (63% vs 62%). Both groups span ≈15-90%,
and one run can move 20-40 points between epochs (v4-downdown itself was unusually stable, 75-86%). The headline
was mostly a lucky run.

## d128 vs d256 (Nov 14): curves, play and land skips

d128 = the `-h5` recipe with `d_model: 128, d_ff: 256` (≈2.4M parameters), seeds 0 and 1, all-node and legal-only
(`v4-dd-players{,-legal}-noname-h5-d128-s{0,1}`). Held-out = gen0 holdout games (26 shards), state-weighted value MSE.
Games: Faeries mirror vs offline, 200 games, best held-out checkpoint (`CKPT` in `runpod/generate.sh`), XMage
`b6ca7b35`, 4090s. Land skips: Pass top at < 3 own lands (< 4), the model's seat in its vs-offline games.

| model | best held-out value (checkpoint) | KL | train value e4 | vs offline | land skips |
|---|---|---|---|---|---|
| all-node d256 (`-h5`) | .0361 (e3·10K) | .107 | .034 | 64.0% | 3.8% (7.9%) |
| all-node d256 (original, 2 x 200) | | | | 79.5%, 70.0% | 3.4%, 2.1% (4.7%, 3.4%) |
| all-node d128 s0 | .0351 (e4) | .091 | .035 | 76.5% | 3.6% (5.6%) |
| all-node d128 s1 | .0353 (e4) | .123 | .035 | 74.0% | 2.6% (4.8%) |
| legal d256 (`-h5`) | .0373 (e4) | .064 | .032 | 72.0% | 4.3% (8.0%) |
| legal d256 (original, 2 x 200) | | | | 77.5%, 80.0% | 6.5%, 6.2% (11.5%, 10.1%) |
| legal d128 s0 | .0370 (e4·10K) | .065 | .034 | 74.5% | 2.5% (7.0%) |
| legal d128 s1 | .0357 (e4) | .064 | .033 | 75.5% | 6.6% (13.0%) |

- d128 matches or beats d256 on held-out value (.001-.0015 better), doesn't overfit by epoch 4 (train ≈ held out), and
  is best at or near its last checkpoint: it would likely gain from more epochs.
- In play the four d128 runs sit within 74.0-76.5% (inside the ±3.5 of 200 games); d256 runs spread 64-80%.
  Pooled: d128 all-node 75.3%, legal 75.0%: at d128 the policy domain makes no difference.
- Land skips vary between runs at both sizes (d128 2.5-6.6%). All-node is consistently low across its four
  independent models (2.1-3.8%); legal-only ranges 2.5-6.6%.

Second d256 legal-only seed (`v4-dd-players-legal-noname-h5-s1`, seed 1): held-out value best .0381 (e4·10K), KL .065,
train .035 at e4. d256 legal seeds .0373 / .0381 (mean .0377) vs d128 legal seeds .0370 / .0357 (mean .0364): the seed
spread is similar at both sizes (.0008 vs .0013); d128 is ≈.0013 better.

### Ablations on the d128 / d256 seeds (`runpod/ablate.sh`, best held-out checkpoints, 2,000 states of P1's Burn mirror)

| model | Burn MSE | mana attention (pass 0 / 1) | − all mana | − Battlefield | ~ LifeTotal | ~ HandSize | ~ BattlefieldSize | ~ ManaValue |
|---|---|---|---|---|---|---|---|---|
| P1 | .046 | .00 / .05 | +.001 | +.001 | +.220 | +.012 | +.077 | +.007 |
| all-node d256 (e3·10K) | .075 | .01 / .02 | +.000 | +.002 | +.159 | +.004 | +.010 | +.014 |
| all-node d128 s0 | .077 | .03 / .02 | +.001 | −.000 | +.172 | +.001 | +.001 | +.017 |
| all-node d128 s1 | .079 | .04 / .06 | +.003 | −.000 | +.160 | −.001 | +.000 | +.027 |
| legal d256 s0 | .080 | .01 / .05 | +.000 | +.028 | +.176 | +.011 | +.077 | +.007 |
| legal d256 s1 | .084 | .06 / .02 | +.005 | +.000 | +.169 | +.012 | +.052 | +.005 |
| legal d128 s0 | .080 | .02 / .06 | +.001 | −.000 | +.153 | +.025 | +.008 | +.023 |
| legal d128 s1 | .083 | .04 / .05 | +.000 | +.001 | +.156 | +.003 | +.029 | +.028 |

No model uses the mana leaves as a sink at its best checkpoint (attention ≤ .06, removal ≤ .005). On this unseen
matchup all are level (.075-.084); LifeTotal dominates everywhere. Which secondary leaves a model relies on is a
seed effect: BattlefieldSize +.077 / +.052 for the two d256 legal seeds but +.008 / +.029 at d128, ~0 for all-node
d128; per-leaf PLAYER attention also differs widely between seeds of one config. d128 leans more on ManaValue
(+.017-.028 vs +.005-.014).

## Leaf-token dropout at d128 (Nov 14)

`leaf_dropout: 0.1` (`ca2a2ac`): in training each leaf token and its edge is dropped with p = .1, except
`IsDecisionPlayer`; nothing is dropped at evaluation. Otherwise the d128 `-h5` recipe, seeds 0 and 1, all-node and legal-only
(`v4-dd-players{,-legal}-noname-h5-d128-ld01-s{0,1}`), 4 epochs, then `--resume` from `model-e4` for 4 more epochs
with the same seed (`…-e8`). 4090s; about $8 in all (one e1-4 run landed on a slow pod, 37 min/epoch instead of 11).
Held-out = gen0 holdout games, state-weighted value MSE, best checkpoint of every 10K-step and end-of-epoch validation.

| model | no dropout, best (≤ e4) | dropout, best ≤ e4 | dropout, best ≤ e8 | prio KL at best | train value e8 |
|---|---|---|---|---|---|
| all-node s0 | .0351 (e4) | .0364 (e4) | .0343 (e8) | .086 | .037 |
| all-node s1 | .0353 (e4) | .0363 (e4) | .0338 (e8·10K) | .090 | .037 |
| legal s0 | .0370 (e4·10K) | .0375 (e4) | .0344 (e8) | .061 | .035 |
| legal s1 | .0357 (e4) | .0363 (e4) | .0341 (e8·10K) | .060 | .035 |

- At equal epochs dropout is worse on every seed (+.0005-.0013): it slows fitting, it doesn't stabilise it.
- With 8 epochs every dropout run beats every no-dropout d128 run (≤ e4), all still improving at e8 with train ≥ held out
  (no overfitting). Seeds agree within .0005 (.0003 per policy domain) vs .0002-.0013 without dropout. Confound: the
  no-dropout baselines were not trained past e4, so part of the gain may be epochs alone.
- Jumps remain: seed 1 (same data order) still spikes, e.g. legal s1 .0362 → .0393 at e6·10K and its last checkpoint
  (.0358) is worse than its best (.0341). Checkpoint selection still matters.

Ablations (`runpod/ablate.sh`, best checkpoints above, the same 2,000 Burn-mirror states as the table before):

| model | Burn MSE | mana attention (pass 0 / 1) | − all mana | − Battlefield | ~ LifeTotal | ~ HandSize | ~ BattlefieldSize | ~ ManaValue |
|---|---|---|---|---|---|---|---|---|
| all-node ld s0 | .080 | .04 / .07 | +.001 | −.000 | +.163 | −.000 | +.001 | +.011 |
| all-node ld s1 | .073 | .04 / .09 | −.000 | +.000 | +.154 | +.002 | −.000 | +.019 |
| legal ld s0 | .088 | .04 / .09 | +.000 | +.000 | +.164 | +.000 | +.000 | +.003 |
| legal ld s1 | .090 | .03 / .07 | −.001 | −.000 | +.138 | +.000 | +.004 | +.023 |

On the unseen Burn mirror dropout doesn't help: all-node .080 / .073 (vs .077 / .079), legal .088 / .090 (vs .080 / .083).
Dropout removes the reliance on secondary counters (HandSize, BattlefieldSize ≈ 0 for all four; legal d128 had
+.025 / +.029); ManaValue still varies by seed (+.003 to +.023). Mana leaves draw a little more attention in pass 1 (.07-.09)
but removing them changes nothing (≤ .001): no sink.

### Land injection on the d128 runs, every epoch (`runpod/land-inject.sh`, P1's Burn mirror)

Share of positions whose value goes up when the skipped Mountain is played, end-of-epoch checkpoints, 300-sim shard
(1000-sim in brackets). `ld` = leaf dropout 0.1; e5-e8 are the `--resume` continuations.

| run | e1 → e8 | mean jump between epochs |
|---|---|---|
| all-node s0 | 12 11 18 26 (29 25 40 54) | 5.3 (11.0) |
| all-node s1 | 98 98 89 87 (96 96 98 98) | 3.7 (0.7) |
| legal s0 | 61 58 74 48 (75 75 92 68) | 15.0 (13.7) |
| legal s1 | 78 94 60 67 (91 91 84 93) | 19.0 (5.3) |
| all-node ld s0 | 83 72 92 67 84 69 50 64 (73 60 85 64 68 60 54 56) | 17.3 (11.3) |
| all-node ld s1 | 99 99 99 98 98 97 97 96 (99 99 100 99 99 99 96 96) | 0.4 (0.7) |
| legal ld s0 | 94 94 87 98 98 99 98 96 (92 84 76 94 97 98 94 96) | 3.1 (6.3) |
| legal ld s1 | 76 84 74 76 11 12 27 49 (71 82 84 83 23 37 55 72) | 17.6 (17.6) |

Dropout does not make the injection test stable. Over e1-e4 the mean jump is 7.9 points with dropout against 10.8
without, and the seeds of one config still land far apart. Legal ld s1 drops from 76% to 11% at e5, while its held-out
value improves (.0363 → .0353). Dropout moves the level up: at e4, 67-98% against 26-98% without. The test tracks skipping in play
(ver15 / P1 fail it and skip most), so this seed and checkpoint variation is in the models, not the test.

## Checkpoint averaging (Nov 14, `util/average.py`, `runpod/average-eval.sh`)

Weights of each run's last 4 validated checkpoints averaged (last 2 epochs: e3·10K, e3, e4·10K, e4 without dropout;
e7·10K…e8 with leaf dropout 0.1), then tested like a checkpoint: held-out gen0 value (test.py, the run's split), the
Burn-mirror land injection test and the ablations (2,000 Burn-mirror states). One 4090, ≈$0.40.

| run | best single checkpoint | average of last 4 | gain | injection, played up: average (its checkpoints) | Burn MSE: best → average |
|---|---|---|---|---|---|
| all-node s0 | .0351 | .0340 | −.0011 | 19% (18 e3, 26 e4) | .077 → .080 |
| all-node s1 | .0353 | .0345 | −.0008 | 90% (89, 87) | .079 → .076 |
| legal s0 | .0370 | .0355 | −.0015 | 67% (74, 48) | .080 → .079 |
| legal s1 | .0357 | .0343 | −.0014 | 76% (60, 67) | .083 → .082 |
| all-node ld s0 | .0343 | .0329 | −.0014 | 66% (50 e7, 64 e8) | .080 → .080 |
| all-node ld s1 | .0338 | **.0320** | −.0018 | 97% (97, 96) | .073 → .073 |
| legal ld s0 | .0344 | .0332 | −.0012 | 99% (98, 96) | .088 → .085 |
| legal ld s1 | .0341 | .0326 | −.0015 | 25% (27, 49) | .090 → .088 |

- Averaging is a free, consistent gain: every run improves on its best single checkpoint by .0008-.0018, more than the
  seed spread. all-node ld s1's average (.0320) is the best held-out value of any model on this split.
- It doesn't remove seed dependence: each average lands inside the injection range of its own checkpoints (all-node s0
  stays low, legal ld s1 stays low), and keeps its run's ablation profile (legal s0 still leans on HandSize +.026,
  legal s1 on BattlefieldSize +.028; the dropout averages ≈0 on both). Burn-mirror MSE moves by ≤ .003.

## Cosine LR decay at d128 (Nov 15, `train.py --min-lr`, `1ef072a`)

The four d128 recipes (no dropout) retrained from scratch for 8 epochs with the learning rate cosine-decayed from 1e-4 to
1e-5 (`v4-dd-players{,-legal}-noname-h5-d128-cos-s{0,1}`). Held-out gen0 value, state-weighted.

| run | constant LR, best ≤ e4 | cosine, best ≤ e8 | cosine, last (e8) | e5 → e8 checkpoints | train value e8 |
|---|---|---|---|---|---|
| all-node s0 | .0351 | .0325 (e7) | .0326 | .0349 .0347 .0335 .0332 .0331 .0325 .0326 .0326 | .029 |
| all-node s1 | .0353 | .0326 (e7) | .0327 | .0346 .0354 .0342 .0329 .0329 .0326 .0331 .0327 | .029 |
| legal s0 | .0370 | .0341 (e8) | .0341 | .0356 .0344 .0344 .0346 .0343 .0343 .0343 .0341 | .026 |
| legal s1 | .0357 | .0335 (e8) | .0335 | .0348 .0341 .0343 .0341 .0340 .0342 .0336 .0335 | .026 |

- Once the rate is low the curve settles: over e6-e8 checkpoints move by ≤ .0006, and the last checkpoint is within .0001
  of the best on all four (constant LR: best and last differ by up to .0017). Early, at high LR, seed 1 still spikes
  (e3·10K, as in every seed-1 run).
- Seeds agree: all-node .0325 / .0326, legal .0341 / .0335. Best single checkpoints of any recipe so far; all-node
  matches dropout 0.1 + checkpoint averaging (.0320-.0329) without either.
- Train value (.026-.029) is now below held out (.033-.034): the start of overfitting, but held out is flat, not rising.

## Bidirectional local passes (Nov 15, directions "both", `499b41f`)

`directions: [both, both], local: parallel`: each internal node attends over its leaves, internal children and parents
(parent tokens + a learned vector), all nodes reading the pass's starting embeddings. Otherwise the d128 recipe
(constant LR, 4 epochs, no dropout), `v4-bb-par-players{,-legal}-noname-h5-d128-s{0,1}`.

| run | down-down sequential (best) | both parallel (best) | train value e4 |
|---|---|---|---|
| all-node s0 / s1 | .0351 / .0353 | .0397 / .0408 | .037 / .039 |
| legal s0 / s1 | .0370 / .0357 | .0422 / .0423 | .036 / .038 |

Clearly worse (+.005), with a larger train / held-out gap (≈.004-.006 vs ≈0): it generalizes worse, not just fits slower.
Training takes ≈745 s/epoch vs ≈640 s for down-down on 4090s (the parallel mode still runs one layer per type in turn).
Two things changed at once (bidirectional tokens and parallel updates); a sequential "both" or a parallel down-down
would separate them.

## Recipes compared on land injection and ablations (Nov 15)

d128, best held-out checkpoint of each run: no dropout (constant LR, ≤ e8), cosine LR (no dropout, ≤ e8), leaf dropout
0.1 (constant LR, ≤ e12), leaf dropout 0.2 (constant LR, ≤ e12; all-node s1 still training). Land injection = share of
P1 Burn-mirror positions (300-sim shard) whose value goes up when the skipped Mountain is played; jump = mean change
between consecutive end-of-epoch checkpoints. Ablations: 2,000 Burn-mirror states, value MSE change.

| recipe | injection at best: all s0 / s1, legal s0 / s1 | mean | mean jump | ~HandSize | ~BattlefieldSize | ~ManaValue | −Battlefield | ~LifeTotal |
|---|---|---|---|---|---|---|---|---|
| no dropout | 20 / 67 / 45 / 48 | 45 | 9.8 | ≤ +.008 | up to +.039 | +.010 to +.046 | up to +.004 | +.168 |
| cosine LR | 64 / 49 / 72 / 99 | 71 | 10.0 | up to +.023 | up to +.016 | +.015 to +.028 | up to **+.071** | +.154 to +.175 |
| dropout 0.1 | 66 / 90 / 95 / 18 | 67 | 8.7 | ≈ 0 | ≤ +.005 | ≤ +.014 | 0 | +.131 to +.171 |
| dropout 0.2 | 72 / – / 88 / 50 | 70 | 15.1 | ≈ 0 | ≤ +.003 | ≤ +.006 | 0 | +.122 to +.145 |

- Ablations: dropout is what makes seeds agree. With it, no secondary feature costs more than +.014 and the profile is the
  same across seeds (0.2 flattest). Without it, which counters a seed leans on is seed-specific (legal s1 BattlefieldSize
  +.039 / ManaValue +.046; cosine legal s1 leans on the Battlefield name leaf, +.071). Cosine LR does not fix this.
- Land injection: dropout lifts the level (mean 45% → 67-70%), but so does cosine LR (71%), and no recipe gets every seed
  above 90% (dropout 0.1 legal s1: 18%; 0.2 legal s1: 50%). Epoch-to-epoch jumps are not smaller with dropout.

## The embedding tables barely train (Nov 15)

Leaf rows start from `vocab.initial_rows` (N(0, 1) per id, reproducible), so a checkpoint can be compared with its own
init. d128 all-node s0 runs: relative change ‖W − W0‖ / ‖W0‖ per leaf row, and cos(W, W0).

| checkpoint | leaf rows: median / p90 / max change | cos(trained, init) median | edge rows median change |
|---|---|---|---|
| no dropout, e1 | .032 / .056 / .152 | .999 | .023 |
| no dropout, e4 | .062 / .105 / .265 | .998 | .054 |
| cosine LR, e7 | .060 / .103 / .284 | .998 | .050 |
| dropout 0.1, e12 | .120 / .201 / .386 | .993 | .112 |

The numeric value embedding (32 buckets, −5 … ) has no order either: cos between adjacent buckets .04-.08 (random 128-d
vectors: ≈ 0 ± .09), 5+ buckets apart ≈ 0. With N(0, 1) init (row norm ≈ 11.3) and Adam at 1e-4 (steps ≈ 1e-4 per
element), the embeddings stay close to random codes: the local layers do the learning, card similarity is never put into
the table, and "3 life" vs "4 life" or "2 lands" vs "3 lands" are unrelated vectors whose order the layers must learn.

### Sequential "both" (Nov 15): isolates the parallel effect

`directions: [both, both]`, `local: sequential` (stages ROOT → ABILITY: a node sees its parents updated this pass, its
children from the last), otherwise as above (`v4-bb-seq-players{,-legal}-noname-h5-d128-s{0,1}`).

| run | down-down sequential | both sequential | both parallel |
|---|---|---|---|
| all-node s0 / s1 | .0351 / .0353 | .0352 / .0353 | .0397 / .0408 |
| legal s0 / s1 | .0370 / .0357 | .0374 / .0377 | .0422 / .0423 |

Sequential "both" is level with down-down (legal ≈.001 worse): the .005 loss came from the parallel update, and
bidirectional tokens add nothing over down-down. Priority KL .091-.100 / .065-.066, as down-down.

### Cosine LR + dropout 0.1, and value / 20, on land injection and ablations (Nov 15)

Cosine + dropout 0.1: 12 epochs, LR 1e-4 → 1e-5, `leaf_dropout: 0.1` (`…-ld01-cos-s{0,1}`). Best held out: all-node
.0322 / .0318, legal .0335 / .0327, the best of any recipe; the last checkpoints move ≤ .0003. value / 20:
`value_encoding: scaled` (`db719b0`), 4 epochs constant LR (`…-sv20-s{0,1}`): all-node .0372 / .0355, legal .0355 / .0348.

| recipe | injection at best: all s0 / s1, legal s0 / s1 | mean | min | e9-e12 range | ~BattlefieldSize | ~ManaValue | ~LifeTotal |
|---|---|---|---|---|---|---|---|
| cosine + dropout 0.1 | 73 / 96 / 77 / 82 | 82 | 73 | 69-79, 95-97, 77-83, 67-84 | ≤ +.001 | +.009 to +.018 | +.125 to +.168 |
| value / 20 (e4) | 95 / 100 / 19 / 71 | 71 | 19 | – | up to +.037 | ≈ 0 | +.151 to +.189 |

- Cosine + dropout is the best recipe on every count so far: highest mean and minimum injection score (earlier recipes:
  mean 45-71, min 18-49), the narrowest late-epoch ranges, the best held-out value, and seed-consistent ablations
  (no secondary feature above +.018). It still doesn't put every seed above 90%.
- value / 20 ends the ManaValue reliance (≈ 0) but brings back seed-specific BattlefieldSize use (legal s1 +.037), and
  legal s0 scores 19% on injection: no better than bins at 4 epochs.
