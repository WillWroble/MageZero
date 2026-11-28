# Pauper format data from ver15: 8x8 vs the offline search and self-play (Nov 7–9)

First data for a single model for the whole Pauper format (Will's "new gen0"), all from ver15 (F2-gnn, end of
epoch 3; `/workspace/runs/gen4-F2-gnn/F`), which was only ever trained to pilot Faeries. Value-only MCTS, 300 sims,
`c_puct` 1.0, `max_pending` 4, opponent's hand visible, XMage `f2ce8eca`, RTX 4090 pods, 200 games per matchup
(a few 194–199). Configs `runpod/configs/gen5/`; shards on the volume in `/workspace/runs/gen5-offline/` and
`/workspace/runs/gen5-selfplay/` (Player{A,B}, both seats recorded). 12,795 + 7,174 games.

## ver15 piloting each deck vs the offline heuristic search of each deck

Model's win rate; row = deck ver15 pilots, column = offline deck.

| model ↓ / offline → | Affinity | Burn | CawGates | Elves | Faeries | Rally | Spy | Wildfire | mean |
|---|---|---|---|---|---|---|---|---|---|
| **Affinity** | 66.0 | 52.5 | 74.5 | 67.0 | 83.5 | 46.0 | 73.5 | 67.5 | **66.3** |
| **Burn** | 52.5 | 37.5 | 69.5 | 69.0 | 71.0 | 25.0 | 64.5 | 48.5 | **54.7** |
| **CawGates** | 65.5 | 52.0 | 77.0 | 60.2 | 87.0 | 33.5 | 66.0 | 47.7 | **61.1** |
| **Elves** | 72.5 | 42.5 | 94.5 | 72.0 | 85.5 | 54.0 | 75.5 | 63.5 | **70.0** |
| **Faeries** | 75.0 | 70.5 | 84.5 | 64.0 | 76.0 | 28.5 | 62.0 | 85.5 | **68.2** |
| **Rally** | 83.5 | 66.5 | 93.5 | 78.5 | 92.0 | 56.0 | 68.0 | 78.5 | **77.1** |
| **Spy** | 62.5 | 46.2 | 79.5 | 62.5 | 77.4 | 62.0 | 58.0 | 57.3 | **63.2** |
| **Wildfire** | 59.5 | 36.5 | 73.4 | 58.5 | 66.0 | 23.0 | 67.0 | 36.0 | **52.5** |
| mean vs this deck | 67.1 | 50.5 | 80.8 | 66.5 | 79.8 | 41.0 | 66.8 | 60.6 | 64.1 |

## Self-play: ver15 on both seats

Win rate of the row deck (mirrors: the PlayerA seat).

| row ↓ vs column → | Affinity | Burn | CawGates | Elves | Faeries | Rally | Spy | Wildfire | mean (excl. mirror) |
|---|---|---|---|---|---|---|---|---|---|
| **Affinity** | 54.0 | 35.5 | 54.3 | 45.5 | 54.5 | 29.5 | 43.5 | 53.5 | **45.2** |
| **Burn** | 64.5 | 49.5 | 71.5 | 68.0 | 64.0 | 32.5 | 65.0 | 72.5 | **62.6** |
| **CawGates** | 45.7 | 28.5 | 38.7 | 20.3 | 56.0 | 16.1 | 34.0 | 42.5 | **34.7** |
| **Elves** | 54.5 | 32.0 | 79.7 | 39.5 | 64.3 | 44.5 | 60.5 | 60.0 | **56.5** |
| **Faeries** | 45.5 | 36.0 | 44.0 | 35.7 | 49.0 | 13.0 | 41.5 | 42.5 | **36.9** |
| **Rally** | 70.5 | 67.5 | 83.9 | 55.5 | 87.0 | 53.0 | 52.8 | 82.5 | **71.4** |
| **Spy** | 56.5 | 35.0 | 66.0 | 39.5 | 58.5 | 47.2 | 51.5 | 62.0 | **52.1** |
| **Wildfire** | 46.5 | 27.5 | 57.5 | 40.0 | 57.5 | 17.5 | 38.0 | 43.0 | **40.6** |

Conclusions:
- ver15 pilots most decks better than the heuristic search: 64.4% over all 64 matchups, and in the mirrors (same
  deck both sides) it wins with CawGates 77.0, Faeries 76.0, Elves 72.0, Affinity 66.0, Spy 58.0, Rally 56.0. It
  loses with **Burn (37.5)** and **Wildfire (36.0)**: the decks whose plans are furthest from Faeries'.
- Rally is the hardest opponent (41.0% mean against the offline Rally) and the strongest deck in self-play (71.4%).
- In self-play Burn is the second strongest deck (62.6%) though the model pilots it worst against the heuristic:
  its weakness vs the offline search is piloting, not the deck.
- Mirror self-play results 38.7–54.0 (first seat) are within ≈2 SE of 50 except CawGates (38.7) and Elves (39.5).

Incidents: a RunPod volume stall (≈06:39–07:15 UTC, Nov 8) blocked the model servers' log writes and failed or hung
12 matchups; they were rerun (`gen5off2-*`) after `generate.sh` moved server logs to the container disk (`1cf7a62`).
One rerun pod failed to clone and was relaunched. Cost ≈ $255–275 (≈283 RTX 4090 pod-hours from watchdog samples,
including ≈$16 for three pods hung by the stall).

## P1: the first Pauper-format model (Nov 9)

Warm start from ver15 on `runpod/splits/P1.txt` (F's Faeries blocks + the two blocks above, 7.18M training states),
lr 3e-5, 3 epochs × 112,172 steps, legal-only policy loss, MSE value, seed 0, names kept, `--val-every 20000`, RTX
PRO 6000 (`/workspace/runs/pauper-P1b-gnn/P1`; a first attempt on an A100 SXM was ≈6–10× slower, CPU-bound, and was
stopped at 40K steps). Held-out value / priority / target / non-pass top-1:

| point | pauper-offline | pauper-selfplay | Faeries model-vs-model (mean of 5) |
|---|---|---|---|
| e1 20K | .043 / .055 / .060 / .559 | .052 / .074 / .063 / .539 | .0402 / .0534 / .0428 / .672 |
| e1 end | .039 / .051 / .055 / .579 | .049 / .070 / .058 / .549 | .0390 / .0528 / .0424 / .678 |
| e2 end | .039 / .050 / .053 / .613 | .048 / .068 / .056 / .588 | .0378 / .0532 / .0412 / .695 |
| **e3 end** | .039 / .049 / .052 / **.621** | .048 / .068 / .056 / **.594** | .0376 / **.0524** / .0418 / .689 |
| ver15 | – | – | .0386 / .0532 / .0424 / .687 |

Conclusion: the new decks' policy improves through all 3 epochs (non-pass .559 → .621 / .539 → .594) and value
settles by epoch 1–2; no overfitting. The Faeries blocks end at or better than ver15. End of epoch 3 is uploaded as
**P1**: `experiment_models/Pauper/P1` (full checkpoint) and https://huggingface.co/inkling616/MZ-Pauper-P1 (weights
and model card). Cost ≈$21 (PRO 6000) + ≈$15 (stopped A100 attempt). Not yet tested in play.

## P2 pilot: the pick piloting each deck vs the offline search of each deck (Nov 17-18)

Same 8×8 layout as the ver15 table above, 200 games each (12,799 games, 1 failed). Pilot: the pick
(`v4-dd-players-legal-noname-h5-d128-sv20-es002-ld02-cos-s1` e10, `value-perspective.md`, trained on Faeries data only,
like ver15), **priority priors 1.5 on**, target priors off. Offline opponent as before. Shards:
`/workspace/runs/gen5-p2-offline/`; the CawGates mirror is two 100-game halves (`-h1`, `-h2`). Win rate P2 (ver15) Δ:

| pilot ↓ / offline → | Affinity | Burn | CawGates | Elves | Faeries | Rally | Spy | Wildfire | mean |
|---|---|---|---|---|---|---|---|---|---|
| **Affinity** | 73.0 (66.0) +7.0 | 71.0 (52.5) +18.5 | 81.5 (74.5) +7.0 | 73.5 (67.0) +6.5 | 75.5 (83.5) −8.0 | 52.5 (46.0) +6.5 | 70.0 (73.5) −3.5 | 79.0 (67.5) +11.5 | **72.0 (66.3) +5.7** |
| **Burn** | 66.5 (52.5) +14.0 | 61.5 (37.5) +24.0 | 67.0 (69.5) −2.5 | 74.0 (69.0) +5.0 | 70.5 (71.0) −0.5 | 30.0 (25.0) +5.0 | 59.0 (64.5) −5.5 | 65.0 (48.5) +16.5 | **61.7 (54.7) +7.0** |
| **CawGates** | 73.0 (65.5) +7.5 | 71.0 (52.0) +19.0 | 84.0 (77.0) +7.0 | 66.5 (60.2) +6.3 | 83.5 (87.0) −3.5 | 38.5 (33.5) +5.0 | 74.5 (66.0) +8.5 | 73.4 (47.7) +25.7 | **70.5 (61.1) +9.4** |
| **Elves** | 80.5 (72.5) +8.0 | 63.0 (42.5) +20.5 | 94.5 (94.5) 0.0 | 82.0 (72.0) +10.0 | 84.5 (85.5) −1.0 | 58.5 (54.0) +4.5 | 83.0 (75.5) +7.5 | 84.5 (63.5) +21.0 | **78.8 (70.0) +8.8** |
| **Faeries** | 70.0 (75.0) −5.0 | 58.0 (70.5) −12.5 | 76.5 (84.5) −8.0 | 68.5 (64.0) +4.5 | 73.5 (76.0) −2.5 | 34.5 (28.5) +6.0 | 59.5 (62.0) −2.5 | 73.5 (85.5) −12.0 | **64.2 (68.2) −4.0** |
| **Rally** | 91.0 (83.5) +7.5 | 89.0 (66.5) +22.5 | 91.5 (93.5) −2.0 | 82.0 (78.5) +3.5 | 96.5 (92.0) +4.5 | 61.0 (56.0) +5.0 | 72.0 (68.0) +4.0 | 91.5 (78.5) +13.0 | **84.3 (77.1) +7.2** |
| **Spy** | 79.0 (62.5) +16.5 | 62.0 (46.2) +15.8 | 90.5 (79.5) +11.0 | 68.5 (62.5) +6.0 | 76.5 (77.4) −0.9 | 71.5 (62.0) +9.5 | 60.0 (58.0) +2.0 | 66.0 (57.3) +8.7 | **71.8 (63.2) +8.6** |
| **Wildfire** | 73.0 (59.5) +13.5 | 65.0 (36.5) +28.5 | 80.0 (73.4) +6.6 | 61.5 (58.5) +3.0 | 71.5 (66.0) +5.5 | 35.0 (23.0) +12.0 | 70.0 (67.0) +3.0 | 73.0 (36.0) +37.0 | **66.1 (52.5) +13.6** |
| mean vs this deck | 75.8 (67.1) +8.6 | 67.6 (50.5) +17.0 | 83.2 (80.8) +2.4 | 72.1 (66.5) +5.6 | 79.0 (79.8) −0.8 | 47.7 (41.0) +6.7 | 68.5 (66.8) +1.7 | 75.7 (60.6) +15.2 | **71.2 (64.1) +7.1** |

- Overall +7.1 points (SE ≈ 0.6); 48 of 64 matchups up, 15 down, 1 level. 24 differ by more than 2 SE: 21 up, 3 down.
- The biggest gains are where ver15 was weakest: Wildfire +13.6 and Burn +7.0 as pilots, the Wildfire mirror 36 → 73
  and the Burn mirror 37.5 → 61.5; against the offline Burn +17.0 and Wildfire +15.2.
- **Faeries is the only deck that got worse (−4.0)**: vs Burn −12.5, Wildfire −12.0 and CawGates −8.0. This fits the pick's
  Faeries mirror test (79.0% value-only vs 72.5% with priors): the priority priors cost Faeries and help the rest.
- Confounds: (1) model, (2) priority priors (P1 was value-only), (3) XMage `f2ce8eca` → `b6ca7b35` (6 commits: virtual
  visits and changed backprop in `MCTSNode2`/`ComputerPlayerMCTS2`, which the offline opponent also runs, plus a
  one-line `Combat` change), (4) P2 hosts filtered to fast CPUs. The offline opponent's strength may have moved too, so
  the Δ is not the pilot's gain alone.

Incidents: the volume filled up (300 GB) at ≈14:07 UTC Nov 17 and was resized to 1000 GB; 5 finished or running
matchups were lost and rerun, one rerun of the CawGates mirror crashed the JVM (SIGSEGV in
`MCTSNode.getMatchingState`) after ≈6 h and was rerun as two halves. Cost ≈ $137 (RunPod balance ≈ $504 → $367, including the files pod and volume).

Conclusion: the pick pilots the Pauper format 7 points better than ver15 on average, most on Burn and Wildfire; with
priority priors on, Faeries drops 4 points.

## P2 self-play: the pick on both seats (Nov 18-19)

All 36 matchups, 200 games each (7,198 games, 2 failed), the pick on both seats with the same settings as the P2
pilot (priority priors 1.5 on both seats, target priors off, XMage `b6ca7b35`). Shards: `/workspace/runs/gen5-p2-selfplay/`;
CawGates vs Wildfire and the Wildfire mirror are two 100-game halves (`-h1`, `-h2`), because their 200-game runs hit
the pods' 10 h cap and were lost. Win rate of the row deck, P2 (ver15) Δ (mirrors: the PlayerA seat):

| row ↓ vs column → | Affinity | Burn | CawGates | Elves | Faeries | Rally | Spy | Wildfire | mean (excl. mirror) |
|---|---|---|---|---|---|---|---|---|---|
| **Affinity** | 45.0 (54.0) −9.0 | 57.0 (35.5) +21.5 | 51.5 (54.3) −2.8 | 35.5 (45.5) −10.0 | 68.5 (54.5) +14.0 | 25.5 (29.5) −4.0 | 48.0 (43.5) +4.5 | 61.5 (53.5) +8.0 | **49.6 (45.2) +4.5** |
| **Burn** | 43.0 (64.5) −21.5 | 51.5 (49.5) +2.0 | 50.5 (71.5) −21.0 | 63.0 (68.0) −5.0 | 61.0 (64.0) −3.0 | 20.5 (32.5) −12.0 | 43.0 (65.0) −22.0 | 53.5 (72.5) −19.0 | **47.8 (62.6) −14.8** |
| **CawGates** | 48.5 (45.7) +2.8 | 49.5 (28.5) +21.0 | 53.0 (38.7) +14.3 | 20.1 (20.3) −0.2 | 55.5 (56.0) −0.5 | 15.0 (16.1) −1.1 | 43.0 (34.0) +9.0 | 50.3 (42.5) +7.8 | **40.3 (34.7) +5.5** |
| **Elves** | 64.5 (54.5) +10.0 | 37.0 (32.0) +5.0 | 79.9 (79.7) +0.2 | 48.5 (39.5) +9.0 | 71.5 (64.3) +7.2 | 41.5 (44.5) −3.0 | 53.5 (60.5) −7.0 | 58.0 (60.0) −2.0 | **58.0 (56.5) +1.5** |
| **Faeries** | 31.5 (45.5) −14.0 | 39.0 (36.0) +3.0 | 44.5 (44.0) +0.5 | 28.5 (35.7) −7.2 | 48.0 (49.0) −1.0 | 11.5 (13.0) −1.5 | 31.5 (41.5) −10.0 | 48.0 (42.5) +5.5 | **33.5 (36.9) −3.4** |
| **Rally** | 74.5 (70.5) +4.0 | 79.5 (67.5) +12.0 | 85.0 (83.9) +1.1 | 58.5 (55.5) +3.0 | 88.5 (87.0) +1.5 | 51.5 (53.0) −1.5 | 49.0 (52.8) −3.8 | 85.0 (82.5) +2.5 | **74.3 (71.4) +2.9** |
| **Spy** | 52.0 (56.5) −4.5 | 57.0 (35.0) +22.0 | 57.0 (66.0) −9.0 | 46.5 (39.5) +7.0 | 68.5 (58.5) +10.0 | 51.0 (47.2) +3.8 | 49.0 (51.5) −2.5 | 43.0 (62.0) −19.0 | **53.6 (52.1) +1.5** |
| **Wildfire** | 38.5 (46.5) −8.0 | 46.5 (27.5) +19.0 | 49.7 (57.5) −7.8 | 42.0 (40.0) +2.0 | 52.0 (57.5) −5.5 | 15.0 (17.5) −2.5 | 57.0 (38.0) +19.0 | 41.5 (43.0) −1.5 | **43.0 (40.6) +2.3** |

- The SE of a Δ between two 200-game rates is ≈5 points, so by chance the mean |Δ| would be ≈4.0. It is 7.4, and 8
  of 36 matchups move by more than 10 points (≈1.6 expected): the deck balance under P2 is really different.
- **Burn falls from the second strongest deck to below average (62.6 → 47.8, −14.8)**: CawGates +21.0, Spy +22.0,
  Affinity +21.5 and Wildfire +19.0 against it. This matches the pilot table, where P2's gains were largest against
  the offline Burn (+17.0): the pick defends against Burn much better than ver15 did, and gains less piloting it.
- Faeries is again the only deck down among the non-Burn decks (−3.4; vs Affinity −14.0, vs Spy −10.0), in line with
  the priors costing Faeries.
- Spy vs Wildfire flips (62.0 → 43.0, −19.0), in line with Wildfire's +13.6 as a pilot in the P2 pilot table.
- Rally stays the strongest deck (74.3); CawGates stays the weakest (40.3), though its mirror moves from 38.7 to a
  level 53.0. Mirrors (first seat) average 48.5 (ver15 47.3), all within 2 SE of 50 (lowest: Affinity 45.0).
- Confounds as in the P2 pilot section; here both seats change, so a Δ mixes how each deck's pilot changed.

Incidents: CawGates vs Wildfire and the Wildfire mirror hit the 10 h cap (`HOURS=10`) at ≈7–7.5 h into their second
config and lost all their games; they were rerun as 100-game halves with `HOURS=14` (4 pods; RTX 4090s were
unavailable in EU-RO-1 for ≈2 h). My cleanup script died with a container restart, so the 4 finished rerun pods idled
≈7–9 h (≈$28) before they were terminated. Cost ≈ $128 (balance ≈ $360 → $232, ≈$85 for the main run and ≈$43 for the
rerun, including the idle pods), against an estimate of $70–110.

Conclusion: under the pick, Burn loses most of its self-play edge (−14.8) and the other decks move by ±6 or less;
Rally stays the strongest deck and CawGates the weakest.

## P2-2-mixed: P2-1 warm-started on all the Pauper data (Nov 19-22)

**Wrong data.** P2-2 was meant to train only on P2-1's games and the offline-vs-offline games (gen0). The runs in this section used P1's split plus P2-1's games instead (now `runpod/splits/P2-2-mixed.txt`): it also holds ver15's 8x8 and self-play and the gen1-gen3 Faeries games of v4, v5, ver7, ver10 and ver13. Everything below, the dropout, Faeries-mirror and Burn results included, is for that mix; the intended P2-2 is retrained on `runpod/splits/P2-2.txt`.

Naming: the P2 line numbers its models. **P2-1** is the pick (`v4-dd-players-legal-noname-h5-d128-sv20-es002-ld02-cos-s1`
e10, trained on gen0 only), which played the two P2 sets above; **P2-2** is trained on them.

Split `runpod/splits/P2-2-mixed.txt` (then named `P2-2.txt`): P1's blocks (gen0-gen3 Faeries data, ver15's 8x8 and self-play) plus P2-1's 8x8 and
self-play, every 5th game held out per block (9.84M training states, 153,720 steps per epoch). PlayerB's P2 Wildfire
mirror shard was left out: its copy was truncated when the volume filled on Nov 17. Warm start from P2-1 e10 (d128,
leaf dropout 0.2, scaled value, legal-only, no player names), lr 3e-5 cosine to 1e-5, `--val-every 10000`, RTX PRO
6000. Epochs 1-3 (`/workspace/runs/pauper-P2-2-gnn/P2-2`) hit the pod's 14 h cap at e3 step 90K; epochs 4-6
(`/workspace/runs/pauper-P2-2b-gnn/P2-2`) resumed from `model-e3-s90000` (`--resume`: optimizer kept, the cosine
spans 6 epochs, so lr went back to 2e-5 and down to 1e-5).

Held-out value MSE (mean of the four Pauper sets, and each) and priority KL (mean):

| point | value mean | p2-offline | p2-selfplay | pauper-offline | pauper-selfplay | prio KL mean |
|---|---|---|---|---|---|---|
| e1 10K | .0506 | .0448 | .0501 | .0458 | .0617 | .0674 |
| e1 150K | .0453 | .0406 | .0450 | .0403 | .0552 | .0612 |
| e2 150K | .0431 | .0394 | .0421 | .0394 | .0513 | .0593 |
| e3 90K | .0426 | .0391 | .0421 | .0385 | .0506 | .0589 |
| e4 150K | .0422 | .0393 | .0414 | .0384 | .0498 | .0583 |
| e5 150K | .0415 | .0380 | .0409 | .0371 | .0499 | .0578 |
| **e6 140K** | **.0412** | .0377 | .0410 | .0370 | .0491 | **.0574** |

On the gen0 sets P2-1 was trained on (the only sets both were tested on), value / priority KL:

| | gen0 held-out | gen0-test (Faeries mirror) |
|---|---|---|
| P2-1 e10 | **.0321** / **.0565** | .0310 / **.0415** |
| P2-2 e6 (mean of its validations) | .0349 / .0647 | **.0295** / .0523 |

- Still improving at the end (−.0004 value per epoch over e4-e6, down from −.005 in e1), with no overfitting: the
  best checkpoint is the last few, and training value loss (.047, leaf dropout on) stays above held-out (.041).
- P2-2 gives up some of P2-1's fit on gen0 (value +.003, priority KL +.008) for the 7 new decks; on the Faeries mirror
  test set its value is better and its priority worse.
- The slow late progress points at lr 3e-5 (ver15's fine-tuning rate) being low for what is mostly new data; P2-1
  itself trained at 1e-4 → 1e-5. Untested: a 1e-4 epoch from P2-1 against e1 above would tell.
- Speed: validation in batches of 64 over 2.55M held-out states took ≈12 of every 18m50s per 10K steps. Validating in
  batches of 512 (`cdf628b`, metrics unchanged) cut it to 7m54s (5.6 min training at 30 steps/s, ≈2.3 min
  validation), ≈2.1 h per epoch instead of 4.7 h.

Pick: **`pauper-P2-2b-gnn/P2-2/model-e6-s140000.pt.gz`** (lowest mean held-out value; e6 150K and the end of e6 are
within .00002). Not yet tested in play.

Incidents: the first launch crashed on the truncated shard; the 14 h cap stopped epoch 3 at 90K and the pod then
idled ≈2 h (≈$5) because my cleanup script dies with my container's restarts; the first continuation pod had 94 GB of
RAM and was OOM-killed while loading the data (≈$4), the relaunch required ≥180 GB (188 GB).
Cost ≈ $72 (RunPod balance $231.97 → $159.84, Nov 19 15:02 → Nov 20 17:06 UTC, including the files pod): epochs 1-3 ≈ $45, the OOM-killed pod ≈ $4, epochs 4-6 ≈ $23.

### Epochs 7-9 (Nov 20-21)

Resumed from the end of epoch 6 (`/workspace/runs/pauper-P2-2c-gnn/P2-2`; cosine over 9 epochs, lr 1.5e-5 → 1e-5),
validation every 10K steps in batches of 512, RTX PRO 6000 (188 GB), 7m34s per 10K steps.

| point | value mean | p2-offline | p2-selfplay | pauper-offline | pauper-selfplay | prio KL mean |
|---|---|---|---|---|---|---|
| e7 150K | .0413 | .0384 | .0407 | .0375 | .0486 | .0572 |
| e8 150K | .0408 | .0375 | .0405 | .0365 | .0489 | .0569 |
| **e9 150K** | **.0407** | .0371 | .0404 | .0367 | .0485 | **.0565** |

gen0 / gen0-test at e9 150K: .0341 / .0633 and .0290 / .0510 (P2-1: .0321 / .0565 and .0310 / .0415).

- Three more epochs gained −.0005 value and −.0009 priority KL over e6 140K: close to a plateau. Epoch 7 first got
  slightly worse (.0417 at 70K) after the lr went back up to 1.5e-5, as epoch 4 did.
- No overfitting, measured directly: e9 70K in eval mode (no dropout) on the training and held-out games of three
  shards (P2 Burn vs Rally, P2 self-play Elves vs Spy, ver15 Affinity vs Faeries; 41K / 11K states) gives value
  .0387 / .0371 and priority loss .0528 / .0560: no train/held-out gap. The loss is set by capacity or label noise,
  not memorisation.

New pick: **`pauper-P2-2c-gnn/P2-2/model-e9-s150000.pt.gz`** (mean held-out value .0407; e9 90K and 140K within
.00003). Cost ≈ $19 (balance $144.42 at 23:34 UTC Nov 20, less the dropout run that overlapped it).

### Leaf dropout 0.05 (Nov 21)

P2-2's recipe with leaf dropout 0.05 instead of P2-1's 0.2 (a warm start can now change it, `7006ceb`): same warm start
from P2-1 e10, split, seed and schedule (3e-5 → 1e-5 over 3 epochs), validation every 20K
(`/workspace/runs/pauper-P2-2-ld005-gnn/P2-2`, RTX PRO 6000, 31 steps/s). Mean held-out value / priority KL over the
four Pauper sets, at matched steps:

| step | dropout 0.05 | dropout 0.2 (P2-2) | Δ value / KL |
|---|---|---|---|
| e1 20K | .0485 / .0651 | .0494 / .0659 | −.0009 / −.0008 |
| e1 140K | .0441 / .0608 | .0452 / .0618 | −.0012 / −.0010 |
| e2 140K | .0423 / .0585 | .0433 / .0596 | −.0010 / −.0011 |
| e3 80K | .0414 / .0576 | .0427 / .0589 | −.0013 / −.0013 |
| **e3 140K** | **.0412** / **.0573** | (e6 140K: .0412 / .0574; e9 150K: .0407 / .0565) | |

- Lower at every one of the 21 matched points, by .0009-.0017 value and .0008-.0015 KL: the same held-out loss as
  P2-2 in half the epochs. It doesn't yet reach P2-2's e9 (.0407).
- Training loss (train mode) is now below held-out: value .039 against .041 at e3, where P2-2 at dropout 0.2 was above it.
- gen0 is unchanged (.035 / .064; gen0-test .0295 / .051).

Conclusion: dropout 0.2 was costing P2-2 fit; 0.05 is better at matched steps. Best checkpoint e3 140K
(`model-e3-s140000.pt.gz`). Cost ≈ $16 (balance $121.05 → $105.53 by 12:40 UTC) plus ≈ $0.75 for a first attempt on
an RTX 4090 with 125 GB that ran out of memory while loading (the data needs more than 125 GB; 188 GB works).

Epochs 4-6 (Nov 21): resumed from the end of epoch 3 (`/workspace/runs/pauper-P2-2-ld005b-gnn/P2-2`; cosine over 6
epochs, 2e-5 → 1e-5), 27 steps/s.

| step | dropout 0.05 | dropout 0.2 (P2-2) | Δ value / KL |
|---|---|---|---|
| e4 20K | .0415 / .0577 | .0432 / .0593 | −.0017 / −.0017 |
| e4 140K | .0412 / .0571 | .0425 / .0585 | −.0013 / −.0013 |
| e5 120K | .0407 / .0564 | .0416 / .0578 | −.0009 / −.0014 |
| **e6 140K** | **.0401** / **.0558** | .0412 / .0574 | −.0011 / −.0016 |

- Still lower at every matched point, and below P2-2's best after 9 epochs (e9 150K: .0407 / .0565).
- Per set at e6 140K: p2-offline .0366, p2-selfplay .0402, pauper-offline .0359, pauper-selfplay .0478. gen0 .0342 /
  .0628, gen0-test .0286 / .0503 (P2-2 e9: .0341 / .0633 and .0290 / .0510).
- Training loss (train mode): value .037 at e6.

**New pick for P2-2: `pauper-P2-2-ld005b-gnn/P2-2/model-e6-s140000.pt.gz`** (mean held-out value .0401). Cost of
epochs 4-6 ≈ $21 (balance $97.50 → $76.78 at 22:51 UTC), plus ≈ $7 for the finished epoch-3 pod idling 2.5 h after my
cleanup script died with a container restart.

Epochs 7-9 (Nov 21-22): resumed from the end of epoch 6 (`/workspace/runs/pauper-P2-2-ld005c-gnn/P2-2`; cosine over
9 epochs, 1.5e-5 → 1e-5), 30 steps/s.

| point | value mean | prio KL mean |
|---|---|---|
| e6 140K | .0401 | .0558 |
| e7 140K | .0403 | .0560 |
| e8 140K | .0400 | .0554 |
| **e9 80K** | **.0399** | .0553 |
| e9 140K | .0399 | **.0551** |

- Value has plateaued: −.0002 over three epochs, within the ±.0003 spread between checkpoints; priority KL still
  creeps down (−.0007). P2-2 at dropout 0.2 after 9 epochs: .0407 / .0565.
- Training loss (train mode) .036 against held-out .040: dropout 0.05 now fits the training games better than new
  ones, but held-out loss isn't rising.
- gen0 .034 / .062, gen0-test .029 / .050 (P2-1: .032 / .057 and .031 / .042).
- By deck (e8 140K against dropout-0.2 e9 150K): priority KL lower in 31 of 32 deck × set groups, value lower or level
  in most. The hardest shards by far are the Spy mirror (value .10-.155 in all four sets, against ≈.04 on average) and
  Rally (mirror value .10, priority KL .12-.13; Rally's ver15 self-play shards ≈.105 KL); the easiest are Elves against
  Faeries and CawGates (value ≈.013).

**Pick: `pauper-P2-2-ld005c-gnn/P2-2/model-e9-s140000.pt.gz`** (value .0399, tied with e9 80K, and the lowest priority
KL). More epochs at d128 won't buy more. Cost of epochs 7-9 ≈ $20 (balance $76.33 → $56.77 at 06:16 UTC).

### P2-2 vs P2-1 in play: Faeries mirror (Nov 22)

200 games, `runpod/configs/h2h-Faeries.yml`: player A P2-2 (`pauper-P2-2-ld005c-gnn/P2-2/model-e9-s140000`), player B
P2-1 (`v4-dd-players-legal-noname-h5-d128-sv20-es002-ld02-cos-s1/gnn/model-e10`), both with priority priors 1.5 and
target priors off (as in the P2 games), 300 sims, `c_puct` 1.0, `max_pending` 4, opponent's hand visible, random first
player, XMage `b6ca7b35`, RTX 4090 / Ryzen 9 7950X (`/workspace/runs/h2h-p22-vs-p21`).

**P2-2 wins 67.5% (135/200, ±3.5)**, 0 failed: ≈5 SE above even, on P2-1's own deck, although P2-1 was trained only
on Faeries data and P2-2 fits gen0 slightly worse by held-out loss. Cost ≈ $2.50 (2.8 h on one RTX 4090).

### P2-2 piloting Burn vs the offline Burn: land skipping (Nov 22)

200 games, `runpod/configs/gen5/Burn-vs-Burn.yml` (the P2 pilot's config): P2-2 (`model-e9-s140000`) with priority
priors 1.5, target off, against the offline search's Burn; 300 sims, `max_pending` 4, XMage `b6ca7b35`, RTX 4090 /
Ryzen 9 7950X (shards `/workspace/runs/gen5-p22-offline/`). "Pass top" (`util/land_pass.py`): share of early land-drop
decisions (fewer than 3 own lands, a land play legal) where the search's most-visited move is Pass.

| pilot | win | Pass top, < 3 lands | < 4 lands |
|---|---|---|---|
| P2-1, priors (P2 pilot) | 61.5% | 4.3% (938) | 4.6% (1205) |
| **P2-2, priors** | **71.5%** (143/200) | **29.4%** (1104) | 27.6% (1401) |
| P2-1, value only (Nov 15) | 58.0% | 24.2% | |

The skips came back with the priors on, though P2-2 wins 10 points more. Their likely source is the training data: in
the Burn-mirror shards P2-2 was trained on, ver15's games skip in 54.8% (vs offline) and 41.3% (self-play) of these
decisions, P2-1's self-play in 5.2%. P2-2's priority head learned Pass from ver15's searches, and the priors now push
the search towards it. Cost ≈ $1.40.

### lr 1e-4 test (Nov 20)

Is P2-2's slow late progress the learning rate? The same run (P2-1 e10 warm start, split, seed and batch order, 3-epoch
cosine to 1e-5) at **1e-4** instead of 3e-5 (P2-1's own starting rate), stopped after epoch 1
(`/workspace/runs/pauper-P2-2-lr1e4-gnn/P2-2`). Mean held-out value / priority KL over the four Pauper sets:

| step (epoch 1) | lr 1e-4 | lr 3e-5 (P2-2) | Δ value |
|---|---|---|---|
| 10K | .0515 / .0672 | .0506 / .0674 | +.0009 |
| 40K | .0492 / .0646 | .0479 / .0645 | +.0013 |
| 70K | .0492 / .0633 | .0469 / .0628 | +.0024 |
| 100K | .0466 / .0624 | .0459 / .0621 | +.0007 |
| 130K | .0453 / .0614 | .0454 / .0615 | −.0001 |
| 150K | .0445 / .0610 | .0453 / .0612 | −.0008 |
| e2 10K | .0453 / .0609 | .0453 / .0612 | .0000 |

On gen0 (P2-1's data) 1e-4 forgets more: e1 130K value .038 / priority KL .072 against .036 / .068 at 3e-5.

Conclusion: 1e-4 is worse for most of the epoch and only level by its end; the learning rate is not what slows P2-2.
The next suspects are capacity (d128, 2.4M parameters for 8 decks) and the data itself. Data loading took ≈3 h on this
pod (EPYC 9655, 283 GB) against ≈70 min before. Cost ≈ $15 (balance $159.71 → $144.65, 17:25 → 22:47 UTC).

## P2-2: P2-1 warm-started on its own games and gen0 (Nov 22-23)

Split `runpod/splits/P2-2.txt`: train only on P2-1's 8x8 vs the offline search, P2-1's self-play and gen0 (offline
search vs offline search, Faeries), every 5th game held out (3.89M training states, 60,852 steps per epoch); ver15's
gen5 games are validation sets only, on the same held-out games as in the mixed runs. Warm start from P2-1 e10, leaf
dropout 0.05, lr 3e-5 cosine to 1e-5 over 6 epochs, `--val-every 20000` (`/workspace/runs/pauper-P2-2v2-gnn/P2-2`,
RTX 4090 with 61 GB, Ryzen 9 7950X, 34 steps/s, ≈35 min per epoch).

Held-out value MSE (P2 sets: mean of p2-offline and p2-selfplay; ver15 sets: mean of pauper-offline and
pauper-selfplay, not trained on) and priority KL over the four:

| point | P2 sets | ver15 sets | prio KL |
|---|---|---|---|
| e1 60K | .0423 | .0536 | .0757 |
| e2 60K | .0412 | .0531 | .0744 |
| e4 60K | .0400 | .0514 | .0728 |
| e5 60K | .0397 | .0512 | .0718 |
| **e6 60K** | .0398 | .0510 | .0715 |
| P2-2-mixed (dropout 0.05, best) | .0379 | .0418 | |

| | gen0 held-out (value / prio KL) | gen0-test (value / prio KL) |
|---|---|---|
| P2-1 e10 | .0321 / .0565 | .0310 / .0415 |
| P2-2-mixed (dropout 0.05) | .034 / .062 | .029 / .050 |
| **P2-2 e6 60K** | **.0305** / **.0545** | **.0285** / **.0400** |

- On gen0 P2-2 is now better than P2-1 on value and policy, where the mixed runs lost about .003 / .007.
- On P2-1's 8-deck games it plateaus at .0397-.0398 from epoch 5, above the mixed run (.0379), which saw 2.5× the
  data; on ver15's games, now unseen, it is far worse (.051 against .042).
- Pick: **`pauper-P2-2v2-gnn/P2-2/model-e6-s60000.pt.gz`**. It ties e5 60K on the P2 sets (.0398 against .0397,
  within noise) and is better on the ver15 sets, priority KL and gen0.
- Cost ≈ $4 (4.4 h on one RTX 4090).

### P2-2 in play (Nov 23)

Burn vs the offline Burn, 200 games (`runpod/configs/gen5/Burn-vs-Burn.yml`, priority priors 1.5, target off, 300
sims, `max_pending` 4, XMage `b6ca7b35`, RTX 4090; shards `/workspace/runs/gen5-p22v2-offline/`). Pass top: early
land-drop decisions (< 3 own lands) where the most-visited move is Pass (`util/land_pass.py`).

| pilot | win | Pass top, < 3 lands | < 4 lands |
|---|---|---|---|
| P2-1 | 61.5% | 4.3% (938) | 4.6% |
| P2-2-mixed (dropout 0.05) | 71.5% | 29.4% (1104) | 27.6% |
| **P2-2** | **73.5%** (147/200) | **8.4%** (953) | 9.2% |

Without ver15's Burn games the skips fall from 29% to 8% (P2-1: 4%) and the win rate is the highest of the three, 12
points over P2-1. Cost ≈ $1.20.

Faeries mirror against P2-1, 200 games (`runpod/configs/h2h-Faeries.yml`, player A P2-2, player B P2-1 e10, priority
priors 1.5 on both seats, target off, same search settings, RTX 4090 / Ryzen 9 7950X): **P2-2 wins 57.0%** (114/200,
±3.5), 0 failed: ≈2 SE above even, where P2-2-mixed won 67.5%. The first attempt was OOM-killed on a pod with a 30 GB
container limit under a 32 GB heap (launched without the host check). Cost ≈ $1.40.

Conclusion: on its intended data P2-2 beats P2-1 in both checks (Burn vs offline 73.5% against 61.5%, Faeries mirror
57.0%) and keeps land skips low (8%); the mixed data played the Faeries mirror better (67.5%) but brought ver15's
skips back (29%).

#### Head-to-head mirrors: P2-2 vs P2-1 (Nov 23-24)

Each deck against itself, player A P2-2 (`model-e6-s60000`), player B P2-1 (`model-e10`)
(`runpod/configs/h2h-<deck>.yml`; CawGates and Wildfire as two 100-game halves), priority priors 1.5 on both seats,
target off, 300 sims, `max_pending` 4, opponent's hand visible, random first player, XMage `b6ca7b35`, RTX 4090s
(`/workspace/runs/h2h-p22v2-vs-p21/`). P2-2's win rate, 0 failed games:

| mirror | P2-2 wins | ± SE |
|---|---|---|
| CawGates | **75.5%** (151/200) | 3.0 |
| Affinity | 68.0% (136/200) | 3.3 |
| Rally | 64.0% (128/200) | 3.4 |
| Spy | 62.0% (124/200) | 3.4 |
| Burn | 61.5% (123/200) | 3.4 |
| Wildfire | 57.5% (115/200) | 3.5 |
| Faeries | 57.0% (114/200) | 3.5 |
| **all 7** | **63.6%** (891/1400) | 1.3 |
| Elves | not measured | |

P2-2 wins all seven by 7-26 points, ≈10 SE over even in total. The Elves mirror hit the pods' 8 h cap twice without
finishing 200 games (the shard is only saved at the end); it would need 100-game halves.

Incidents: the RunPod balance ran out at ≈01:40 UTC Nov 24 with five mirrors running; RunPod terminated every pod,
including the files pod (recreated as `zuzo2y3tuj4f6u`). CawGates (both halves) and Wildfire half 2 had already
finished and were saved; Elves and Wildfire half 1 were rerun. Cost ≈ $40 (≈$26 before the balance ran out, then
≈$7 per rerun pod; the two Elves attempts ≈$13 with nothing to show).

#### Head-to-head 8x8: P2-2 vs P2-1 (Nov 24)

The 56 non-mirror pairings, 100 games each: P2-2 (`model-e6-s60000`) pilots the row deck, P2-1 (`model-e10`) the column
deck (`runpod/configs/h2h/<A>-vs-<B>.yml`), priority priors 1.5 on both seats, target off, 300 sims, `max_pending` 4,
opponent's hand visible, XMage `b6ca7b35`; 16 RTX 4090 pods with 3-4 matchups each
(`/workspace/runs/h2h-p22v2-vs-p21/`). P2-2's win rate; the 200-game mirrors above in italics on the diagonal (Elves'
mirror missing). 2 of 5,600 games failed.

| P2-2 ↓ / P2-1 → | Affinity | Burn | CawGates | Elves | Faeries | Rally | Spy | Wildfire | mean (non-mirror) |
|---|---|---|---|---|---|---|---|---|---|
| **Affinity** | *68* | 63 | 63 | 69 | 69 | 40 | 59 | 81 | **63.4** |
| **Burn** | 67 | *62* | 68 | 66 | 76 | 32 | 61 | 72 | **63.1** |
| **CawGates** | 63 | 63 | *76* | 40 | 80 | 25 | 54 | 82 | **58.1** |
| **Elves** | 68 | 48 | 85 | – | 81 | 50 | 59 | 67 | **65.4** |
| **Faeries** | 55 | 38 | 45 | 33 | *57* | 18 | 40 | 53 | **40.3** |
| **Rally** | 75 | 78 | 92 | 64 | 81 | *64* | 51 | 87 | **75.4** |
| **Spy** | 72 | 61 | 70 | 55 | 52 | 58 | *62* | 61 | **61.2** |
| **Wildfire** | 59 | 52 | 63 | 54 | 67 | 31 | 74 | *58* | **57.1** |
| mean vs this P2-1 deck | 65.6 | 57.6 | 69.5 | 54.4 | 72.3 | 36.3 | 56.9 | 71.9 | |

non-mirror: 56 matchups, 3389/5598 = 60.5% ± 0.7
with mirrors: 4280/6998 = 61.2% ± 0.6

- **P2-2 wins 60.5% ± 0.7 of the non-mirror games, 61.2% ± 0.6 with the mirrors**: a clear gain over P2-1 across the
  format, ≈15 SE above even.
- Every pairing is played in both directions, so deck strength cancels in the overall number; a row mean mixes how
  well P2-2 pilots that deck with how strong the deck is. Rally is the strongest deck either way (P2-2 wins 75% piloting
  it and only 36% against P2-1's Rally), as in self-play.
- Faeries is P2-2's weakest deck (40% piloting it, while P2-1's Faeries is among the easiest to beat at 72%): P2-2's
  gain doesn't reach the deck P2-1 was trained on, which fits the earlier finding that priority priors cost Faeries.

Incidents: the files pod died ≈12:30 UTC (no container running) and was replaced at 13:28 (`2l0dg1dyklek34`); until
then the cleanup script couldn't see two finished pods, which idled ≈50 min (≈$1.50). Cost ≈ $72 (balance $191 at
08:42 UTC → $115.77 at 19:16, less ≈$3 for the overlapping Elves mirror pod and the files pod).

## P2-3: P2-2 warm-started on its own games (Nov 24-25)

Split `runpod/splits/P2-3.txt`: P2-2's data (P2-1's 8x8 and self-play, gen0) plus P2-2's games: its head-to-head
games against P2-1 (7 mirrors and the 56 non-mirror pairings, both seats) and its Burn vs the offline Burn; every 5th
game held out (4.79M training states, 74,867 steps per epoch). Not used: P2-2's policy-only games (no visit counts) and
the mixed P2-2's games. Warm start from P2-2 (`pauper-P2-2v2-gnn/P2-2/model-e6-s60000`), leaf dropout 0.05, lr 3e-5
cosine to 1e-5 over 6 epochs, `--val-every 20000` (`/workspace/runs/pauper-P2-3-gnn/P2-3`, RTX PRO 6000 with 188 GB,
30 steps/s, ≈46 min per epoch).

Held-out value MSE (trained sets: mean of p22-h2h, p22-offline, p2-offline, p2-selfplay; P2 sets: p2-offline and
p2-selfplay, as for P2-2; ver15 sets, not trained on) and priority KL over the four:

| point | mean (4) | p22-h2h | p22-offline | P2 sets | ver15 sets | prio KL | gen0 | gen0-test |
|---|---|---|---|---|---|---|---|---|
| e1 20K | .0458 | .0477 | .0548 | .0404 | .0524 | .0449 | .0323 / .0569 | .0310 / .0425 |
| e2 60K | .0450 | .0474 | .0533 | .0396 | .0507 | .0441 | .0311 / .0555 | .0285 / .0410 |
| e4 60K | .0449 | .0466 | .0542 | .0394 | .0516 | .0431 | .0315 / .0547 | .0295 / .0400 |
| **e6 40K** | **.0437** | .0458 | .0514 | **.0389** | .0501 | .0427 | .0310 / .0540 | .0290 / .0400 |
| e6 60K | .0442 | .0458 | .0527 | .0391 | .0502 | .0427 | .0311 / .0542 | .0285 / .0400 |
| P2-2 (e6 60K) | | | | .0398 | .0510 | | .0305 / .0545 | .0285 / .0400 |

- Against P2-2 on the same held-out games: P2 sets .0389 against .0398, ver15's games .0501 against .0510, gen0
  priority KL .0540 against .0545, gen0 value .0310 against .0305 (within noise), gen0-test level.
- p22-offline is P2-2's single Burn-mirror shard (≈40 held-out games), so its column moves by ±.002 between
  checkpoints and dominates the spread of the four-set mean.
- Still improving slowly at the end (priority KL every epoch); no overfitting: training value .034 (train mode) against
  .039 held out on the P2 sets.

Pick: **`pauper-P2-3-gnn/P2-3/model-e6-s40000.pt.gz`** (lowest four-set mean and P2-set value; e6 20K and e5 20K are
next). Not yet tested in play. Cost ≈ $16 (balance $115.57 → $99.42; a 4090 with enough RAM was unavailable for an hour,
so it ran on a PRO 6000).

### P2-3 vs P2-2 in play (Nov 25)

Head-to-head mirrors, 200 games each (`runpod/configs/h2h-{Faeries,Burn}.yml`): player A P2-3 (`model-e6-s40000`),
player B P2-2 (`model-e6-s60000`), priority priors 1.5 on both seats, target off, 300 sims, `max_pending` 4,
opponent's hand visible, XMage `b6ca7b35`, RTX 4090s on Ryzen 9 7950X hosts (`/workspace/runs/h2h-p23-vs-p22/`).

| mirror | P2-3 vs P2-2 | (P2-2 vs P2-1) |
|---|---|---|
| Faeries | 48.5% (97/200) ± 3.5 | 57.0% |
| Burn | 54.0% (108/200) ± 3.5 | 61.5% |
| both | 51.3% (205/400) ± 2.5 | 59.3% |

No measurable gain in play: Faeries level, Burn +4 (≈1 SE). The held-out gains (≈.001 value) are about a quarter of
the ≈.004 that moved win rate by ≈6 points before, which fits. Cost ≈ $5.50 (two RTX 4090 pods).

Land skips in the Burn mirrors (`util/land_pass.py`, each seat's own shard; Pass top at < 3 own lands, < 4 in
brackets; ≈900-960 decisions per seat):

| model (seat) | opponent | Pass top |
|---|---|---|
| P2-1 | P2-2 | 5.3% (5.0%) |
| P2-2 | P2-1 | 8.5% (10.1%) |
| P2-2 | P2-3 | 9.8% (11.0%) |
| P2-3 | P2-2 | 10.8% (10.8%) |

With the reference points vs the offline Burn (P2-1 4.3%, P2-2 8.4%; the offline search itself 13-14%): the skips creep
up by generation, ≈5% → ≈9% → ≈11%, each step about 1-2 SE. P2-3 trained on P2-2's games, which skip at 8-10%, so the
search's skips feed back into the policy; worth watching over more generations.

### P2-2 self-play (Nov 25-26)

P2-2 (`pauper-P2-2v2-gnn/P2-2/model-e6-s60000`) on both seats, all 36 matchups, **100 games each**
(`runpod/configs/gen5/selfplay-<A>-vs-<B>.yml` with `GAMES=100`, `ceb7b11`), priority priors 1.5 on both seats,
target off, 300 sims, `max_pending` 4, XMage `b6ca7b35`, 12 RTX 4090 pods with 3 matchups each
(`/workspace/runs/gen5-p22-selfplay/`). Win rate of the row deck, P2-2 (P2-1 self-play, 200 games) Δ; mirrors are
the PlayerA seat. 8 of 3,600 games failed.

| row ↓ vs column → | Affinity | Burn | CawGates | Elves | Faeries | Rally | Spy | Wildfire | mean (excl. mirror) |
|---|---|---|---|---|---|---|---|---|---|
| **Affinity** | 47 (45) | 55 (57) | 58 (52) | 49 (36) | 71 (68) | 35 (26) | 50 (48) | 53 (62) | **53.0 (49.6) +3** |
| **Burn** | 45 (43) | 52 (52) | 55 (50) | 71 (63) | 65 (61) | 26 (20) | 58 (43) | 59 (54) | **54.1 (47.8) +6** |
| **CawGates** | 42 (48) | 45 (50) | 49 (53) | 28 (20) | 63 (56) | 20 (15) | 39 (43) | 47 (50) | **40.6 (40.3) 0** |
| **Elves** | 51 (64) | 29 (37) | 72 (80) | 39 (48) | 75 (72) | 28 (42) | 50 (54) | 54 (58) | **51.3 (58.0) −7** |
| **Faeries** | 29 (32) | 35 (39) | 37 (44) | 25 (28) | 55 (48) | 15 (12) | 39 (32) | 41 (48) | **31.6 (33.5) −2** |
| **Rally** | 65 (74) | 74 (80) | 80 (85) | 72 (58) | 85 (88) | 56 (52) | 58 (49) | 81 (85) | **73.6 (74.3) −1** |
| **Spy** | 50 (52) | 42 (57) | 61 (57) | 50 (46) | 61 (68) | 42 (51) | 41 (49) | 47 (43) | **50.4 (53.6) −3** |
| **Wildfire** | 47 (38) | 41 (46) | 53 (50) | 46 (42) | 59 (52) | 19 (15) | 53 (57) | 56 (42) | **45.4 (43.0) +2** |

- P2-1 cells are ±3.5 and P2-2 cells ±5, so a matchup's Δ has an SE of ≈6: the mean |Δ| of 6.2 is about what noise
  alone gives (≈5). The deck balance is close to P2-1's (Rally strongest at 74%, Faeries weakest at 32%; mean distance
  from 50% 12.9 against 13.0).
- The row means that moved most: Burn +6 and Elves −7 (≈2 SE each); the others are within ±3.
- Mirrors average 49.4% for the first seat (P2-1 48.5%).

Cost ≈ $50 (balance $92.16 → $41.77, incl. the files pod and ≈$2 of finished pods idling while the cleanup script was
down). The launchers were killed three times by container restarts before all 12 pods were placed (nothing billed while
waiting); one pod landed on a slow EPYC 7352 and was replaced.

### P2-3 retrained with P2-2's self-play (Nov 26)

Same recipe as the first P2-3 (warm start from P2-2 `model-e6-s60000`, leaf dropout 0.05, lr 3e-5 cosine to 1e-5 over
6 epochs, `--val-every 20000`), on `runpod/splits/P2-3.txt` as of `f199200`: the first P2-3's data plus P2-2's self-play
(`gen5-p22-selfplay`, 36 × 100 games, every 5th game held out); the first P2-3's split is kept as
`runpod/splits/P2-3-noselfplay.txt`. 5.23M training states, 81,728 steps per epoch, ≈55 min per epoch on an RTX PRO
6000 (283 GB) (`/workspace/runs/pauper-P2-3b-gnn/P2-3`).

Held-out value (mean of the five trained Pauper sets; p22-selfplay; p22-h2h; p22-offline; P2 sets = mean of p2-offline
and p2-selfplay; ver15 sets, val only), priority KL (mean of the five), gen0 and gen0-test (value / priority KL):

| point | mean (5) | p22-selfplay | p22-h2h | p22-offline | P2 sets | ver15 | prio KL | gen0 | gen0-test |
|---|---|---|---|---|---|---|---|---|---|
| e1 20K | .0468 | .0519 | .0480 | .0537 | .0401 | .0518 | .0466 | .0317 / .0568 | .0300 / .0430 |
| e2 80K | .0459 | .0496 | .0466 | .0553 | .0391 | .0508 | .0458 | .0311 / .0563 | .0305 / .0425 |
| e4 80K | .0448 | .0477 | .0452 | .0534 | .0389 | .0505 | .0441 | .0315 / .0546 | .0290 / .0405 |
| **e6 80K** | **.0444** | .0480 | .0448 | .0518 | .0386 | .0505 | .0433 | .0309 / .0544 | .0290 / .0410 |
| first P2-3 (e6 40K) | | (not trained) | .0458 | .0514 | .0389 | .0501 | | .0310 / .0540 | .0290 / .0400 |
| P2-2 (e6 60K) | | | | | .0398 | .0510 | | .0305 / .0545 | .0285 / .0400 |

- Level with the first P2-3 on every set they share (P2 sets .0386 vs .0389, ver15 .0505 vs .0501, gen0 and gen0-test
  within .001; on the same four sets the mean is .0435 vs .0437); the head-to-head games are a little better (.0448 vs
  .0458). The self-play block's held-out value falls from .0519 to ≈.048.
- 3,600 games of P2-2 self-play (≈0.45M training states, +9%) don't move the model measurably on held-out loss.

Pick: **`pauper-P2-3b-gnn/P2-3/model-e6-s80000.pt.gz`** (lowest five-set mean; e5 20K and e6 60K within .0001). Not
tested in play. Cost ≈ $20 (balance $41.50 → $21.52 incl. the files pod; no RTX 4090 with enough RAM came free in ≈1 h,
and container restarts kept killing the background launcher, so it ran on a PRO 6000).

## Throughput sweep (Nov 28)

P2-2 (`pauper-P2-2v2-gnn/P2-2/model-e6-s60000`, priority priors 1.5, target off, 300 sims) piloting Affinity vs the
offline Rally search, XMage `3b270a33`, configs `runpod/configs/tune/*.yml` (the gen5 Affinity-vs-Rally config with the
settings below), RTX 4090 pods with a Ryzen 9 7950X (16 cores, 32 threads). Wall time from the first search line to the
result; one pod per round, settings run one after another. Also the smoke test for the cost of P2-2 vs offline games.

| setting | threads | client batching (`server:`) | `max_pending` | games | wall | s/game | P2-2 wins | failed |
|---|---|---|---|---|---|---|---|---|
| round 1 `base` | 16 | defaults (8 / 32 / 32000 µs) | 4 | 32 | 542 s | 16.9 | 21 | 0 |
| round 1 `t32` | 32 | defaults | 4 | 32 | 856 s | 26.8 | 26 | 0 |
| `t16` | 16 | defaults | 4 | 48 | 889 s | 18.5 | 33 | 0 |
| `t12` | 12 | defaults | 4 | 48 | 923 s | 19.2 | 28 | 0 |
| `t8` | 8 | defaults | 4 | 48 | 1287 s | 26.8 | 32 | 0 |
| `t16-batch` | 16 | `max_requests` 16, `max_batch` 64, `batch_timeout_us` 4000 | 4 | 48 | 872 s | 18.2 | 29 | 0 |
| `t16-pend8` | 16 | defaults | 8 | 48 | 632 s | 13.2 | 33 | 0 |

- 16 game threads is the sweet spot on this CPU: 12 is as fast, 8 and 32 are ≈45% slower (32: each search took ≈3.4×
  longer, the CPU is oversubscribed). The gen5 vs-offline configs set `threads: 64`, which may be why ver15's 8x8 cost
  ≈80 s/game (not checked).
- Client batching makes no difference (−2%, within noise).
- `max_pending` 8 is ≈30% faster: the searches wait on evaluations, so more in flight per tree helps. It changes the
  search (more virtual-loss leaves per tree), so its play isn't identical; win rate here 33/48 as with 4.
- Cost at 16 threads ≈ 18 s/game ≈ $0.0046 per game (≈$0.0033 with `max_pending` 8), so 12,800 P2-2-vs-offline games
  ≈ $60 (≈$42), plus longer matchups (Elves). One matchup only, 48 games per setting.
- Cost ≈ $2.40 for both rounds.
