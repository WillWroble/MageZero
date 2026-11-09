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
