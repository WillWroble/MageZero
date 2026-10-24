# MageZero graph encoder: Pauper Faeries findings (Oct 2026)

Handoff notes for an agent testing the graph-encoder model on the attached HDF5 data. Everything here was measured on the data and logs from one working session. Corrections made during the session are already applied. Where something is an inference rather than a measurement, it says so.

---

## 1. Setup

**Deck and pool.** Pauper Faeries (mono-blue), trained against mipo's Spellbench `pauper-kernel` pool:

- Jund Wildfire
- Mono Red Rally
- Grixis Affinity
- Elves
- Spy Combo
- Mono-Red Burn
- Caw-Gates
- Mono-Blue Faeries (the mirror)

**Holdout.** The mirror (Faeries vs Faeries) is the holdout set.

**Model versions.** Python side is on branch `graph-encoder` of WillWroble/MageZero. The Java encoder is on `graph-encoder` of WillWroble/mage.

| Version | Training data |
|---|---|
| ver1 | gen0 PlayerA (Faeries) states vs the 7 pool decks |
| ver2 | ver1 data + UWTempo data (Standard deck, separate data set) |
| ver3 | ver1 data + gen0 PlayerB (opponent seat) states from the 7 non-mirror matchups |

**Generations.**

- **gen0:** offline heuristic MCTS, no network. Visits per root around 999.
- **gen1:** network-guided MCTS. The inference server was launched with `--version 3`, so ver3 is assumed to be gen1's evaluator. Search budget 300 visits per decision. Policy priors were off; value only.

**Continuations.** Each version was then trained 5 more epochs with gen1's non-mirror data added. Validation for these = gen0 mirror + gen1 mirror (46,091 decision states).

**Checkpoints.**

- `train.py` saves `best.pt.gz` at the lowest summed validation loss and `model.pt.gz` at the last epoch.
- `server.py` and `test.py` load `model.pt.gz`.

**Data files.**

| Set | Files | States |
|---|---|---:|
| gen0 PlayerA | 7 files `Pauper-Faeries_vs_Pauper-<X>.hdf5` (no mirror) | 117,917 |
| gen0 PlayerB | 7 files `Pauper-<X>_vs_Pauper-Faeries.hdf5` | 127,195 |
| gen1 PlayerA | 8 files, including `Pauper-Faeries_vs_Pauper-Faeries.hdf5` | 148,568 |

The gen0 mirror holdout is 22,493 states.

### HDF5 layout (LabeledStateWriter)

| Dataset | Type | Contents |
|---|---|---|
| `/indices` | int32 [nodes] | node feature id (hash). Typed nodes carry the hash of their type name; ROOT is 0 |
| `/values` | int32 [nodes] | numeric value |
| `/offsets` | int64 [N+1] | node offsets per state |
| `/edge_child`, `/edge_parent` | int32 [edges] | local node indices |
| `/edge_label` | int32 [edges] | hashed edge label |
| `/edge_offsets` | int64 [N+1] | edge offsets per state |
| `/policy_node` | int32 | candidate node, local index |
| `/policy_visits` | int32 | MCTS visits for that candidate |
| `/policy_offsets` | int64 [N+1] | candidate offsets per state |
| `/row` | float32 [N, 6] | resultLabel, stateScore, isPlayer, actionType, useFalseVisits, useTrueVisits |
| `/game_offsets` | int64 [G+1] | first row of each game |

`actionType` codes: 0 = PRIORITY, 3 = CHOOSE_TARGET, 5 = CHOOSE_USE.

**Candidate lists.** `getActionMap` writes every root child, including children with 0 visits. Zero-visit candidates are rare at these budgets.

### Identifiers

Ids are produced by `StateEncoder.indexFor(hash64(s))`. The Python port is `vocab.feature_id`, which matches Java.

**Names and labels.**

- A typed node's name leaf hangs off an edge labelled `"name"`.
- Exception: permanents. The static printed-name edge (`"static"`) overwrites `"name"` on the same leaf, so permanent names sit on `"static"`. This is intended.
- **Ability names** are `cleanString(ability.toString())`: "Cast <card>", "Play <land>", or the rule text for activated abilities.
- **Step** is a leaf under ROOT, using the `PhaseStep` enum name (e.g. `PRECOMBAT_MAIN`, `END_TURN`).
- **Seat flags:** `IsDecisionPlayer` and `IsActivePlayer` are leaves under the PLAYER node. Player name leaves are "PlayerA" and "PlayerB" (Faeries is PlayerA in every file).
- **Zone nodes** are named "Hand", "Graveyard", "Battlefield", "Exile", "Stack", "CommandZone".

**Faeries priority ability ids.** Decoded by hashing candidate strings.

| id | ability |
|---:|---|
| 679880620 | Pass ("PassAbility") |
| 1254598873 | Play Island |
| 691519731 | Harrier Strix loot: "{2}{U}: Draw a card, then discard a card." |
| 1758931328 | Cast Humbling Elder (flash) |
| 1775296276 | Cast Faerie Miscreant |
| 913115100 | Cast Spellstutter Sprite (flash) |
| 2082092948 | Cast Faerie Seer |
| 1370559227 | Cast Of One Mind (sorcery) |
| 1211918040 | Cast Moon-Circuit Hacker |
| 10181138 | Cast Snap |
| 235080923 | Cast Ninja of the Deep Hours |
| 653878615 | Cast Counterspell |
| 437330601 | Cast Harrier Strix |
| 612652355 | Cast Force Spike |
| 1564034202 | Cast Saiba Cryptomancer (flash) |
| 1222181873, 758602737 | the two ninjutsu abilities, assigned by elimination (rule strings not matched; which is which is unknown) |
| 356170266 | Cast Dispel |
| 1284084735 | Cast Cryogen Relic (Affinity's card) |

**Card facts** (checked in XMage card files): Faerie Seer and Faerie Miscreant are both {U} 1/1 flyers.

- Seer: ETB scry 2.
- Miscreant: ETB draw a card if you control another creature named Faerie Miscreant.

Humbling Elder and Saiba Cryptomancer have flash. Seer, Miscreant and Harrier Strix do not.

### Tooling notes (Claude's sandbox)

- h5py could not be installed there. HDF5 was read via ctypes against the system `libhdf5_serial.so.103` (H5Fopen / H5Dopen2 / H5Dread with the `H5T_NATIVE_*_g` globals).
- `test.py` reads `data/<deck>/ver<N>/testing`. When it was run on gen1, that directory still contained the gen0 mirror holdout, so its reported numbers are gen1 + gen0 mirror (171,061 decision states = 148,568 + 22,493).
- To get gen1-only numbers:
  - Subtract the epoch-10 mirror confusion matrix of the same checkpoint (from the training logs). This was exact for ver2 and ver3; ver1 was within ±40 states.
  - Remove the mirror's share from the averaged losses, weighted by state counts.

---

## 2. Generalization to the mirror (gen0 training, 10 epochs, gen0 mirror holdout)

### Mirror holdout, per head (means over 10 epochs)

| | ver1 | ver2 | ver3 |
|---|---:|---:|---:|
| value loss | 0.456 | 0.370 | **0.069** |
| choose_target loss | 1.388 | 1.306 | **0.504** |
| choose_target accuracy | 0.569 | 0.581 | **0.726** |
| choose_use loss | 0.111 | 0.095 | 0.083 |
| priority loss | 0.468 | **0.436** | 0.522 |
| non-pass recall (epochs 6–10) | 0.309 | **0.350** | 0.110 |

**Reading the priority numbers.** Pass is about 78% of priority states, so always-pass accuracy is about 0.778. Non-pass recall is the informative metric.

**ver1 on its own.**

- Priority transfers to the unseen mirror opponent.
- Value and target do not:
  - Mirror value loss is lowest at epoch 1, then rises while training value keeps falling.
  - Target accuracy is flat at 0.56–0.58.
- Mirror vocab coverage was not the cause: 185 of 187 mirror leaf ids and all edge labels are in the training vocab.

**ver1 checkpoint behaviour.**

- `best.pt.gz` was epoch 5, chosen by the noisy target loss.
- `model.pt.gz` (epoch 10) had choose_use accuracy drop to 0.680.
- Epoch 5 had the strongest instant-speed recall: Counterspell 91/140, Spellstutter 105/231.

**ver2.** Better value in 9 of 10 epochs, with the same training value loss as ver1, so this is generalization rather than fit. UWTempo shares only Island and Pass labels with Faeries.

**ver3.**

- The mirror's novelty is Faeries objects on the non-deciding side. ver3 trains on exactly that, because the graph treats perspective as a flag (IsDecisionPlayer). Value, and the target choices that involve the other player's permanents, became learnable.
- Mirror graphs show the same structure: about 48 fewer leaves per state at the same edge count (shared leaves across seats).

### Data composition

| Addition | Share of priority states | Share of target states | Kept leaf ids |
|---|---:|---:|---|
| UWTempo (ver2) | 62% | 34% | 814 → 2,016 |
| Opponent seats (ver3) | 55% | 46% | 814 → 923 (new decisions, not new tokens) |

UWTempo states: 144,450 total, 121,403 priority, 20,656 target. Opponent seats: 127,195 states.

In ver3's added data, opponent-seat target states are about 29% PERMANENT, versus 15% in Faeries' own states (removal and burn aimed at Faeries creatures). Counterspell picks up 15.6% more labels from the opponent seats, because some pool decks play it.

---

## 3. Priority: the timing split from UWTempo (ver2)

Recall on the gen0 mirror, epochs 6–10, ver1 → ver2:

| Group | Ability | ver1 → ver2 |
|---|---|---|
| Main-phase casts improved | Faerie Miscreant | 0.60 → 0.96 |
| | Moon-Circuit Hacker | 0.30 → 0.80 |
| | Faerie Seer | 0.33 → 0.78 |
| | Ninja cast | 0.31 → 0.65 |
| | Harrier Strix | 0.46 → 0.66 |
| Instant-speed plays collapsed to Pass | Force Spike | 0.46 → 0.04 |
| | Counterspell | 0.31 → 0.07 |
| | Saiba | 0.29 → 0.06 |
| | Elder | 0.25 → 0.03 |
| | Spellstutter | 0.23 → 0.02 |
| | Dispel | 0.16 → 0.02 |
| | Snap | 0.08 → 0.00 |

The line falls exactly at instant speed. Of One Mind (sorcery) is the one exception, at 0.53 → 0.38.

The same split reappears on gen1 data (section 7) and in the continuations (section 8). One ninjutsu dropped 0.87 → 0.43 on gen1; ninjutsu is a combat-time decision.

**Interpretation (inference).** The step and active-player leaves are shared across decks, and UWTempo dominates ver2's priority data. What transferred is a when-to-act prior that agrees with Faeries at sorcery speed and disagrees at instant speed.

Play Island, the only shared non-pass label, did not improve (0.27 → 0.27).

---

## 4. Perspective and zone ownership at ability nodes (ver3 priority collapse)

**What happened.** ver3's priority collapsed on Faeries' own decisions.

- Mirror non-pass recall about 0.11.
- On gen1, every Faeries ability went mostly to Pass. For example, Of One Mind 0.00 and Snap 0.00.
- Training priority loss was also higher (0.446 vs ver1's 0.249).

**Mechanism (structural reading of the model; the counts are measured).**

- Faeries ability nodes appear in all 165K priority states.
- In the 90K opponent-seat states they belong to the non-deciding player and get zero visits.
- Ability nodes sit at the bottom of PLAYER → ZONE → CARD/PERMANENT → ABILITY.
- IsDecisionPlayer is a leaf under PLAYER, and the local-layers run bottom-up only. So an ability node gets its owner flag, and its card's zone, only through the global-layers.
- The value head needs the perspective once (at CLS). The priority head needs it at about 80 ability nodes per state.

**Will's hypothesis.** The perspective flag still needs to be learned, and 2 global self-attention layers may not be enough for the context shift it imposes.

**Update from the continuation.** ver3's priority recovered during 5 epochs with gen1 data:

- Training priority loss 0.437 → 0.286, with a step between epochs 2 and 3.
- Mirror non-pass recall 0.157 → 0.326.

**Metric note.** The confusion matrix matches predictions by ability name. Picking the opponent's same-named node counts as correct, so low recall here means Faeries abilities were scored down overall, not that the wrong copy was chosen.

---

## 5. Encoder: which ability nodes are emitted

`processCardInZone` filters abilities by the zone they function from (`getActivatedAbilities(z)`), not by legality (mana, timing, targets). A card in hand always has its cast node.

Two other paths emit cast nodes for cards that can't be cast:

- **Battlefield:** the `processPermBattlefield` "dynamic abilities" block iterates `p.getAbilities(game)` with no zone filter, so a permanent's SpellAbility gets a node.
- **Graveyard / exile:** a branch in `processCardInZone` adds the hand-zone SpellAbility/PlayLandAbility for every card there. The comment mentions permission effects, but nothing checks for a permission.

The network therefore has to learn zone legality too, which fits the design of no legality mask in policy training.

**Evidence that this matters.** ver1's 937 gen1 Seer→Miscreant errors were almost all this:

- In 1,405 states where search's top move was Cast Faerie Seer, a castable Miscreant existed in only 47.
- In about 955, the only "Cast Faerie Miscreant" nodes were on Miscreants on the battlefield or in the graveyard.
- ver2 nearly removed this confusion (66), suggesting zone legality transfers across decks via UWTempo data.

---

## 6. Faerie Seer vs Faerie Miscreant

### Search preference when both are castable (priority states with both as candidates)

| | gen0 (offline heuristic) | gen1 (ver3 value, policy off) |
|---|---|---|
| No Miscreant on PlayerA's battlefield | Seer 0.58 / Miscreant 0.38 (254 states) | Seer 0.22 / Miscreant 0.77 (166) |
| A Miscreant already out | Seer 0.21 / Miscreant 0.77 (52) | Seer 0.20 / Miscreant 0.79 (85) |

These are the 7 pool matchups only. Including the gen1 mirror gives 303 states, with Miscreant preferred 0.79.

- **gen0** made the human choice: Seer normally, Miscreant when it would draw. That held in 6 of 7 matchups (Wildfire was the exception).
- **gen1** prefers Miscreant regardless. With policy off, that comes from the value head guiding search. The gen0 labels did not cause it at the decision level.

### Value labels by Faeries' board (unmatched groups; correlational only)

The heuristic cannot distinguish two 1/1 flyers, so label differences between these groups reflect what else differs between the states (timing, board, what follows). They are not a heuristic preference for either card.

**gen0 PlayerA**, 7 pool matchups:

| Board | median | mean | Faeries won |
|---|---:|---:|---:|
| Seer only | +0.035 | −0.015 | 0.46 |
| one Miscreant only | +0.179 | +0.079 | 0.38 |
| 2+ Miscreants | +0.423 | +0.263 | 0.51 |

Restricting to games that never had 2 Miscreants on board (379 of 619 games): Seer −0.002, Miscreant +0.049 (medians); win rates 0.46 vs 0.36.

**gen0 PlayerB**, sign-flipped to Faeries:

| Board | median | mean | Faeries won |
|---|---:|---:|---:|
| Seer only | −0.152 | −0.169 | 0.48 |
| one Miscreant only | −0.165 | −0.142 | 0.44 |

Never-2 games: Seer −0.204, Miscreant −0.328 (medians).

**gen1 PlayerA**, 7 pool matchups:

| Board | median | mean | Faeries won |
|---|---:|---:|---:|
| Seer only | +0.034 | −0.007 | 0.50 |
| one Miscreant only | −0.066 | −0.057 | 0.45 |
| 2+ Miscreants | +0.105 | +0.082 | 0.61 |

Never-2 games: Seer +0.033, Miscreant −0.145 (medians). The gen1 mirror is roughly even.

**Will's reading.** Value is one generation ahead of decisions. gen1's labels point back toward Seer, so this is an oscillation (seen before) that he predicts converges with Seer slightly better on average.

**Context.** Revealed cards are not encoded, so scry's benefit is invisible to the state encoding. Its value can only come through long-horizon outcomes (TD over self-play).

---

## 7. gen0 → gen1 behaviour (PlayerA, 7 pool matchups)

Wins are the sign of each game's final value label (about ±0.99). "Chosen" means top-visited when available.

### Win rate

| Opponent | gen0 | gen1 |
|---|---:|---:|
| Burn | 0.01 | 0.24 |
| Elves | 0.59 | 0.81 |
| Rally | 0.12 | 0.23 |
| Spy | 0.49 | 0.60 |
| CawGates | 0.64 | 0.67 |
| Affinity | 0.57 | 0.55 |
| Wildfire | 0.58 | 0.49 |
| **Total** | **0.41 (619 games)** | **0.49 (636 games)** |

The gen1 mirror: PlayerA won 0.63.

### Other changes

- **Game length:** 190 → 196 states per game; 122 → 133 priority decisions per game.
- **Earlier deployment:**
  - Play Island in postcombat main 18% → 13%.
  - Miscreant in postcombat main 16% → 12%.
  - Seer in postcombat main 20% → 17%.
- **Own-turn beginning-of-combat instant plays roughly tripled:**
  - Humbling Elder 34 → 152 (own-turn share 0.83 → 0.92; its −2/−0 shrinks a blocker before attacks).
  - Strix loot 114 → 313.
  - Spellstutter 41 → 118.
  - Snap 17 → 34.
  - Saiba 4 → 21.
- **More counters on its own turn** (protecting its own spells):
  - Force Spike own-turn share 0.14 → 0.31.
  - Counterspell 0.11 → 0.19.
  - Dispel 0.30 → 0.39.
- **Less ninjutsu:** Ninja cast 0.83 → 0.70 when available; ninjutsu 0.74 → 0.58 and 0.61 → 0.48.
- **Chosen more often when available:** Snap 0.24 → 0.34, Miscreant 0.64 → 0.73, Harrier Strix 0.48 → 0.55.
- **Unchanged:** pass rate when an action is available (own turn 0.34 / 0.33; opponent's turn 0.54 / 0.53).
- **Missing in both gens:** almost no plays in END_TURN, the classic Faeries end-of-turn flash. Flash creatures are cast in its own upkeep or draw step instead (Spellstutter: 441 / 347 upkeep casts vs 20 / 13 end-step casts). Not checked whether XMage gives priority in END_TURN in the sims.

---

## 8. Models tested on gen1 (gen1-only, epoch-10 checkpoints)

| | ver1 | ver2 | ver3 |
|---|---:|---:|---:|
| value loss | 0.119 | 0.119 | **0.064*** |
| priority loss | 0.386 | **0.311** | 0.445 |
| priority accuracy | 0.820 | **0.847** | 0.805 |
| Pass recall | 0.92 | 0.96 | 0.99 |
| non-pass recall | **0.466** | 0.449 | 0.156 |
| choose_target loss | **0.602** | 0.767 | 0.641 |
| choose_target accuracy | 0.628 | 0.628 | 0.633 |
| choose_use accuracy | 0.701 | 0.698 | 0.701 |

\* **In-sample caveat for ver3's value.** gen1 labels blend each state's search mean score (`stateScore`, computed with ver3 as the evaluator) into the TD target, so ver3 is partly predicting its own outputs. The size depends on `tdDiscount`.

- **ver1 vs gen0 mirror.** The instant-speed plays were much better on gen1 pool games than on the gen0 mirror: Dispel 0.00 → 0.83, Counterspell 0.19 → 0.75, Spellstutter 0.01 → 0.55. The mirror gap is concentrated in reactive plays.
- **ver2 vs ver1 on gen1.** Same timing split as section 3. ver2's lower priority loss comes from Pass dominance (0.96 vs 0.92), not from better non-pass choices.
- **Never learned in any version:** Harrier Strix loot (gen1 recall 0.02–0.09; about 85% predicted Pass).
- **Play Island:** about 0.29–0.42; mostly predicted Pass.

---

## 9. Continuation training (5 epochs + gen1 non-mirror; validation = both mirrors)

| | ver1 | ver2 | ver3 |
|---|---|---|---|
| value | 0.433–0.517 | 0.309–0.379 | **0.054–0.066** |
| target loss | 1.058–1.174 | 0.887–1.115 | **0.508–0.600** |
| target accuracy | 0.518–0.527 | 0.542–0.560 | **0.682–0.693** |
| priority loss | 0.374–0.446 | 0.380–0.430 | 0.374–0.515 |
| non-pass recall | **0.232–0.380** | 0.220–0.334 | 0.110 → 0.326 |
| best total (epoch) | 2.001 (2) | 1.687 (5) | **1.046 (4)** |

- **ver1:** epoch 5 is its worst epoch on value (0.517), non-pass recall (0.232) and choose_use accuracy (0.753), and it is the saved `model.pt.gz`.
- **ver2:** keeps main-phase casts. Every counterspell is gone on the mirrors (Counterspell 0.00–0.03, Force Spike 0.05–0.07, Dispel 0).
- **ver3 mirror recall, epochs 3–5:**
  - Miscreant 0.82–1.00
  - Harrier Strix 0.62–0.85
  - Hacker 0.49–0.79
  - ninjutsu 0.58–0.69
  - Seer 0.39–0.68
  - Force Spike 0.19–0.56
  - Counterspell 0.06–0.22
  - Spellstutter, Dispel and Strix loot ≈ 0
- **All three versions:** Spellstutter and Dispel are at about 0 on the mirrors.

---

## 10. Engineering findings (inference server and data generator)

### Inference latency

- **Measured:** graph server requests take 41–77 ms. The flat server was 5.6–8 ms.
- **Server code** (`server.py`, waitress inside `init`, single worker thread draining the queue without waiting to fill batches) is the same structure as main's flat server. The per-request CPU work is under 1 ms.
- **Cost is in `NetGraph.forward` (code reading, not profiled).**
  - About 750 small ops per forward versus about 50 for the flat net.
  - 14 sequential LocalLayer calls (7 types × 2 passes), plus `index_copy` of the full `h` per stage.
  - About 20 host-device syncs: `nonzero`, boolean masks, `int(counts.max())`. Stage index lists are rebuilt on GPU every call.
  - Forward time is roughly flat in batch size: a 4-state request costs about the same as a 1-state request.
- **Search speed:** network-guided PlayerA searched about 31 evaluations/s versus about 99/s for the offline opponent.

### Java-side batching

- **One shared evaluator per JVM.** In the data generator, all PlayerA instances share `remoteModelEvaluatorA`, so its queue mixes states from concurrent games.
- **Per tree:** `MAX_PENDING = 4` pending evaluations. Selection spins in `awaitEvaluation()` on pending nodes; virtual loss −1.
- **Evaluator batching:** flush at `maxBatchSize = 4` or every `batchInterval = 2500` µs. `MAX_CONCURRENT_CALLS = 16` applies JVM-wide.
- **Waitress:** `threads` caps concurrent requests across all games; the original value was 6.
- **Result of more concurrent games plus more waitress threads:**

| | Before | After |
|---|---:|---:|
| States per server batch | 4–6 | 10–13 |
| Per-request latency | 41–77 ms | unchanged |
| Throughput | about 150 states/s | about 330 states/s |

### Memory

**FeatureGraph lifecycle.**

- LabeledState copies of FeatureGraph are shallow (shared Node objects plus cached graphArrays).
- They are held in `encoder.labeledStates` for the whole game, then go through `LSQueue` (capacity 32) to the writer thread.
- HDF5 is flushed once per game. The GameResult becomes collectable after its write.

**Leak: old trees kept reachable.**

- `ComputerPlayerMCTS2` declares its own `MCTSNode2 root`, shadowing `ComputerPlayerMCTS.root`.
- The superclass decision methods (`priority`, `chooseTarget`, `makeChoice`, `chooseNum`, `chooseUse`) store the chosen child in the superclass field. Its parent chain keeps the previous tree alive.
- The `ComputerPlayerMCTS` copy constructor copied `root`. XMage's real-game state copies (`saveRollBackGameState` keeps the last 4 turns; the priority-start bookmark) copy players, pinning up to about 5 extra trees per player.
- **Will's fix:** set `this.root = null` in the copy constructor. This removes the copies' references.
- **Remaining:** the live superclass `root` still holds the previous tree until the next decision. That is at most one extra tree per player.

`emancipate()` itself works: it detaches the node from its parent.

---

## 11. Open items (as stated during the session)

- Whether the perspective flag gets learned at the ability nodes with 2 global layers (Will's hypothesis), or needs more capacity.
- Whether Seer vs Miscreant converges with Seer slightly better (Will's prediction), and whether TD eventually credits scry through self-play.
- Harrier Strix loot has never been learned by any version.
- Absence of END_TURN flash plays in both gens.
- Testing ver1 on gen1's mirror games alone, to check whether the mirror gap persists on gen1 labels.
