# gen1 inventory (Pauper-Faeries, v0.3 encoding)

Every gen1 shard in `experiment_data/Pauper-Faeries/ver4|ver5/training/gen1`, probed from the h5s
(`/game_offsets`, final value label per game; "opp hand cards" = cards in the non-deciding player's hand per state,
sampled; 0 means the hidden-information encoding). Faeries WR is from Faeries' side in both seats.

| shard | games | states | states/game | Faeries WR | opp hand cards |
|---|---|---|---|---|---|
| ver4/training/gen1/PlayerA/Pauper-Faeries_vs_Standard-MonoB.hdf5 | 171 | 14388 | 84.1 | 0.316 | 3.64 |
| ver4/training/gen1/PlayerA/Pauper-Faeries_vs_Standard-MonoG.hdf5 | 200 | 21884 | 109.4 | 0.160 | 2.71 |
| ver4/training/gen1/PlayerA/Pauper-Faeries_vs_Standard-MonoR.hdf5 | 200 | 16317 | 81.6 | 0.175 | 2.84 |
| ver4/training/gen1/PlayerA/Pauper-Faeries_vs_Standard-MonoU.hdf5 | 199 | 17490 | 87.9 | 0.754 | 4.39 |
| ver4/training/gen1/PlayerA/Pauper-Faeries_vs_Standard-MonoW.hdf5 | 186 | 20468 | 110.0 | 0.575 | 3.57 |
| ver4/training/gen1/PlayerB/Standard-MonoB_vs_Pauper-Faeries.hdf5 | 171 | 8067 | 47.2 | 0.316 | 3.55 |
| ver4/training/gen1/PlayerB/Standard-MonoG_vs_Pauper-Faeries.hdf5 | 200 | 12274 | 61.4 | 0.160 | 2.70 |
| ver4/training/gen1/PlayerB/Standard-MonoR_vs_Pauper-Faeries.hdf5 | 200 | 10509 | 52.5 | 0.175 | 3.48 |
| ver4/training/gen1/PlayerB/Standard-MonoU_vs_Pauper-Faeries.hdf5 | 199 | 11217 | 56.4 | 0.754 | 4.16 |
| ver4/training/gen1/PlayerB/Standard-MonoW_vs_Pauper-Faeries.hdf5 | 186 | 23699 | 127.4 | 0.575 | 2.39 |
| ver5/training/gen1/PlayerA/Pauper-Faeries_vs_Pauper-Affinity.hdf5 | 200 | 17627 | 88.1 | 0.635 | 4.07 |
| ver5/training/gen1/PlayerA/Pauper-Faeries_vs_Pauper-Burn.hdf5 | 200 | 15727 | 78.6 | 0.665 | 3.11 |
| ver5/training/gen1/PlayerA/Pauper-Faeries_vs_Pauper-CawGates.hdf5 | 180 | 18577 | 103.2 | 0.811 | 3.59 |
| ver5/training/gen1/PlayerA/Pauper-Faeries_vs_Pauper-Elves.hdf5 | 200 | 25713 | 128.6 | 0.650 | 3.92 |
| ver5/training/gen1/PlayerA/Pauper-Faeries_vs_Standard-MonoB.hdf5 | 200 | 17619 | 88.1 | 0.265 | 3.32 |
| ver5/training/gen1/PlayerA/Pauper-Faeries_vs_Standard-MonoB2.hdf5 | 200 | 19992 | 100.0 | 0.315 | 3.48 |
| ver5/training/gen1/PlayerA/Pauper-Faeries_vs_Standard-MonoB_pimc.hdf5 | 200 | 14907 | 74.5 | 0.125 | 0.00 |
| ver5/training/gen1/PlayerA/Pauper-Faeries_vs_Standard-MonoB_pimc2.hdf5 | 200 | 13203 | 66.0 | 0.130 | 0.00 |
| ver5/training/gen1/PlayerA/Pauper-Faeries_vs_Standard-MonoB_pimc_pimc.hdf5 | 200 | 14014 | 70.1 | 0.185 | 0.00 |
| ver5/training/gen1/PlayerA/Pauper-Faeries_vs_Standard-MonoB_policy.hdf5 | 199 | 18355 | 92.2 | 0.286 | 3.37 |
| ver5/training/gen1/PlayerA/Pauper-Faeries_vs_Standard-MonoW.hdf5 | 192 | 20061 | 104.5 | 0.406 | 3.42 |
| ver5/training/gen1/PlayerB/Pauper-Affinity_vs_Pauper-Faeries.hdf5 | 200 | 19755 | 98.8 | 0.635 | 3.14 |
| ver5/training/gen1/PlayerB/Pauper-Burn_vs_Pauper-Faeries.hdf5 | 200 | 22549 | 112.7 | 0.665 | 3.91 |
| ver5/training/gen1/PlayerB/Pauper-CawGates_vs_Pauper-Faeries.hdf5 | 180 | 18637 | 103.5 | 0.811 | 3.67 |
| ver5/training/gen1/PlayerB/Pauper-Elves_vs_Pauper-Faeries.hdf5 | 200 | 25070 | 125.3 | 0.650 | 3.38 |
| ver5/training/gen1/PlayerB/Standard-MonoB_vs_Pauper-Faeries.hdf5 | 200 | 11061 | 55.3 | 0.265 | 3.17 |
| ver5/training/gen1/PlayerB/Standard-MonoB_vs_Pauper-Faeries2.hdf5 | 200 | 11000 | 55.0 | 0.315 | 2.98 |
| ver5/training/gen1/PlayerB/Standard-MonoB_vs_Pauper-Faeries_pimc.hdf5 | 200 | 10412 | 52.1 | 0.125 | 3.04 |
| ver5/training/gen1/PlayerB/Standard-MonoB_vs_Pauper-Faeries_pimc2.hdf5 | 200 | 10322 | 51.6 | 0.130 | 3.19 |
| ver5/training/gen1/PlayerB/Standard-MonoB_vs_Pauper-Faeries_pimc_pimc.hdf5 | 200 | 9807 | 49.0 | 0.185 | 0.00 |
| ver5/training/gen1/PlayerB/Standard-MonoB_vs_Pauper-Faeries_policy.hdf5 | 199 | 10034 | 50.4 | 0.286 | 3.51 |
| ver5/training/gen1/PlayerB/Standard-MonoW_vs_Pauper-Faeries.hdf5 | 192 | 25559 | 133.1 | 0.406 | 2.85 |

Summary by matchup (games; both seats are recorded for every shard and the A/B game counts match):

| opponent | ver4 gen1 (v4 model) | ver5 gen1 |
|---|---|---|
| Standard-MonoB | 171 | 200 + 200 (`2`) perfect info, 199 `_policy`, 200 + 200 `_pimc`, 200 `_pimc_pimc` |
| Standard-MonoG | 200 | – |
| Standard-MonoR | 200 | – |
| Standard-MonoU | 199 | – |
| Standard-MonoW | 186 | 192 |
| Pauper-Affinity | – | 200 |
| Pauper-Burn | – | 200 |
| Pauper-CawGates | – | 180 |
| Pauper-Elves | – | 200 |
| Pauper-Rally, Spy, Wildfire | – | – |
| Pauper-Faeries (mirror) | – | – |

Models: every gen1 shard is from the v4 model except `ver5/.../Pauper-Faeries_vs_Standard-MonoB.hdf5` and
`ver5/.../Pauper-Faeries_vs_Standard-MonoW.hdf5` (and their PlayerB pairs), which are from the Mono-only v5 model.

Notes:
- Not yet at gen1 for the gen0 pool: Rally, Spy, Wildfire and the mirror (Pauper), and MonoG/R/U for ver5.
- Hidden information is as labelled: `_pimc` hides MonoB's hand from Faeries' seat only, `_pimc_pimc` from both.
- Faeries WR, perfect info: Pauper 0.64–0.81 (ver5), MonoU 0.75, MonoW 0.58 (ver4) / 0.41 (ver5), MonoB 0.27–0.32,
  MonoG/R 0.16–0.18. With PIMC against MonoB: 0.13 (one side hidden), 0.19 (both).
