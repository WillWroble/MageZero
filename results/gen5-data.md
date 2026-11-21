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

## P2-2: P2-1 warm-started on all the Pauper data (Nov 19-20)

Naming: the P2 line numbers its models. **P2-1** is the pick (`v4-dd-players-legal-noname-h5-d128-sv20-es002-ld02-cos-s1`
e10, trained on gen0 only), which played the two P2 sets above; **P2-2** is trained on them.

Split `runpod/splits/P2-2.txt`: P1's blocks (gen0-gen3 Faeries data, ver15's 8x8 and self-play) plus P2-1's 8x8 and
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
