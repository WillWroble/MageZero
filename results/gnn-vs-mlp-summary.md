# GNN vs MLP: summary of the Oct 27 – Nov 2 experiments

Pauper-Faeries, ver4 data line (gen0 heuristic search → gen1 v4 → gen2 ver7/ver10 → gen3 ver13 → gen4 candidates
ver14/ver15). Details, logs and per-block tables are in `mlp-vs-gnn.md`, `branches.md`, `generations.md` and
`gen3-training.md`; this file only condenses them. All play numbers are Faeries' win rate, 200 games, value-only
search (priors off) unless noted; ±3.5 points is one standard error per deck.

**Models.** GNN = `NetGraph` d256 (9.1M params, 0.4M of them embeddings). MLP = `NetMLP` d64 (0.2M), a
reconstruction of Dama's gorge `MLPNet`: each object is an MLP over the mean of its leaf embeddings, plus the sum of
its ancestors' vectors, pooled per node type. No attention, no learned message passing. Since branch D, both use the
legality mask and drop the player names (§5).

**Verdict.** The GNN is the better architecture at every data size we have, on held-out metrics and in play. Its
lead grows with data. The MLP's only advantages are robustness to unseen opponents at gen0 scale and cheaper
compute per state.

## 1. Held-out metrics by generation

Gap = MLP − GNN loss (positive favours the GNN). In distribution unless noted.

| training data | states | value gap | priority | targets | source |
|---|---|---|---|---|---|
| gen0 | 0.42M | ≈0 (.039 vs .038) | level | GNN +3 pts top-1 | `mlp-vs-gnn.md` |
| gen0 → held-out gen1 | 0.42M | GNN .077 vs .096–.105 | GNN | GNN +4 pts | `mlp-vs-gnn.md` |
| gen1 (3 seeds) | 0.31M | .004–.020 | KL .005 | +2–3 pts | `mlp-vs-gnn.md` |
| gen0 + gen1 (branch D) | 2.34M | .001–.011 | KL .002–.006 | +2.5–4 pts | `branches.md` |
| gen0–gen2 (E) | gen0 + gen1 + gen2 | .002–.009 on gen2 blocks | +1–2 pts top-1 | +3–5 pts | `gen3-training.md` |
| gen0–gen3 (F2, best checkpoints) | 4.37M | .003–.007 on every block | loss .006–.011 | loss .008–.015 | `gen3-training.md` |

- The value gap went from zero at gen0 to .003–.007 on every block at gen0–gen3. The policy and target gaps were
  there from the start.
- The MLP's value fails on PIMC (hidden-hand) states: .118–.122 against the GNN's .055–.065, from branch D through
  F2, at either width.
- The one block where the MLPs are consistently better is `ub` (the unseen UWTempo/Standard16 games): .042–.048
  against .046–.054.
- **Capacity.** A d256 MLP (2.3M params, F2 split) beats the d64 by ≈.0015 value and ≈.002 policy. It closes about
  a third of the value gap and little of the policy gap, and doesn't fix PIMC. Most of the GNN's lead is the
  architecture, not its size. At gen0 scale (0.42M states), d256 ≈ d64.
- **Convergence.** On F2, the d64 MLP's training and val value losses stay within ≈.003 of each other and barely move
  over 6 epochs, so it is capacity-bound. The GNN fits faster than val can follow: its training value keeps falling
  (.035 → .032) while val flattens after about one epoch from a warm start.

## 2. Play

### Against the offline heuristic search

| generation | GNN | MLP (same data) | decks | gap |
|---|---|---|---|---|
| gen1: v4 vs MLP gen1 (gen0 data) | 54.4% | 47.8% | 13 | +6.6 (GNN ahead on 11 of 13) |
| gen2: ver7 vs D-mlp (branch D) | 50.3% | 42.2% | 3 (mirror, MonoB, MonoW) | +8.1 |
| gen2: ver12 vs MLP twin (all-node loss) | 49.7% | 38.3% | 3 | +11.4 |
| gen4: F-gnn (ver14) vs F-mlp | 42.6% | 42.0% | 6 | +0.6 |
| gen4: F2-gnn (ver15) vs F2-mlp | 48.8% | 41.2% | 6 | +7.6 (GNN ahead on 5 of 6, mirror level) |

Six-deck set: mirror, Burn, MonoB, MonoW, Rally, MonoG.

- The MLP has never been ahead on average. Its two deck wins at gen1 were Affinity (68.5 vs 63.5) and MonoR (24.0
  vs 17.5).
- F is the one near-tie, and it is a data-order artifact (§3).
- **Not tested:** MLPs against model-driven opponents (v4, ver10 playing the other deck). Every MLP game is against
  the offline search. GNN-only results against model opponents: ver13 beats v4 head to head (mirror 57.5%) and ver10
  (65.0%); see `generations.md`.

### How they differ in play

- **Value distributions are indistinguishable.** v4 against the gen0 MLP on the gen1 self-play holdout: same
  percentiles (p5–p95 −0.84…+0.90 vs −0.85…+0.91), AUC .895 vs .893, same calibration, per-state correlation .964.
  The win-rate gap does not show in how value is spread over positions.
- **The difference is behaviour.** The MLP's search spends its mana on its own turn: threats and sorcery-speed plays
  are chosen 6–15 points more often. It reaches the opponent's turn with about a third fewer flash or counter
  windows. The GNN keeps mana open. The same shift appears in gen2 and PIMC games and loses each time.

## 3. Small value-loss differences carry large win-rate differences

The value-loss gap is the mean over the five model-vs-model blocks (gen2-vsgen1, gen2-selfplay,
gen3-vsgen1/vsgen2/selfplay):

| pair | value-loss gap | win-rate gap (6 decks) |
|---|---|---|
| F-mlp − F-gnn | .0026 | +0.6 |
| F-gnn (ver14) − F2-gnn (ver15) | .0036 | +6.2 |
| F2-mlp − F2-gnn | .0058 | +7.6 |
| gen1: MLP − v4 (gen0 mirror value) | .006 | +6.6 (13 decks) |

- **The ver15 discovery.**
  - F and F2 are the same recipe: ver13 warm start, the same split, lr 3e-5, 3 epochs, seed 0.
  - F2 only added `--val-every 10000`. Every DataLoader iteration draws from torch's global RNG, including the val
    loaders with `shuffle=False`, so the extra validations changed the shuffle order of epochs 2–3.
  - The two runs match through epoch 1 (±.001) and then diverge: by epoch 3, value loss .0422 (ver14) against .0386
    (ver15).
  - That ≈.004 difference came with 6.2 points of win rate. ver15 (48.8%) is the best model on the six decks: +2.3
    over ver13 and +6.2 over ver14.
- **Why it's plausible.** Value-only search barely separates its options: the median best−worst child gap is
  0.04–0.05, and 35–38% of decisions are within 0.02 between the top two (`results/mcts_gaps.py`). A small
  systematic error in value is enough to flip many choices.
- **Consequences.**
  - Run-to-run noise from data order alone (≈.004 value, ≈6 points of win rate) is as large as most architecture
    and recipe effects we measured. Any single-run comparison, including every A–F comparison, carries that noise.
  - A held-out value-loss difference of ≈.003 is worth testing in play. It isn't negligible.
  - Per-state calibration doesn't predict play; the value-loss mean on model-vs-model blocks has, so far.
  - Fix for reproducibility: give the training DataLoader its own `torch.Generator`. Not implemented yet.
- **One unexplained outlier.** F2-mlp ≈ F-mlp in value loss (.0444 vs .0448) and in play (41.2 vs 42.0), except
  Rally: 17.0% against 26.5% (≈2.7 SE).

## 4. Priority priors

Priors = the priority head's output used as MCTS priors.

| model | buffer | value only | priors on |
|---|---|---|---|
| ver7 (gen2) | gen0 + gen1 (gen0 ≈65% of states) | MonoB 32.0%, MonoW 48.2% | MonoB 25.0%, MonoW **16.5%** |
| ver9 (gen2) | gen1 only (no gen0) | MonoB 22.0% | MonoB 28.5% |
| ver10 (gen2) | gen1, no PIMC or `_policy` | MonoB 26.0% | MonoB 29.5% |
| legality-head GNN | branch D | 49.2% (3 decks) | 45.9% |
| ver13 (gen3), ver15 (gen4) | gen0–gen2 / gen0–gen3 | – | neither hurts nor helps (Will's tests) |

- **The priors carry the buffer's old play style.** With priors on, ver7 reverts to gen0's act-first play: Pass
  chosen 0.48 → 0.38, Ninja 0.24 → 0.51. That undoes the patience that value-only search found at gen1. On held-out
  gen1 games, dropping gen0 improves priority KL (0.069 → 0.058) and leaves the GNN's value about the same.
- **But cutting the buffer hurts value and play.** ver9 and ver10 (no gen0) are worse value-only players than ver7
  (22–26% vs 32% on MonoB). Against the offline search, ver10 averages 58.5% against v4's 63.6% over 8 Pauper decks.
  With priors, they only recover to about ver7's value-only level.
- **Interpretation (Will's).** The priority targets are each generation's search visits, i.e. its own play style, so
  old generations' targets go stale. Value labels stay useful across generations. So priority wants a smaller buffer
  than value, and the two are at odds in one buffer. As gen0 is diluted, the priors have gone from harmful (gen2) to
  neutral (gen3–gen4). Consistent with it: on F2 the GNN's priority loss kept improving after value had plateaued,
  and its priority loss is highest on gen0 (.062 vs ≈.05 on gen3).
- **Untested fix:** a per-section policy-loss weight in `split.txt`. It would train the policy on the latest
  generations only and keep value on the full buffer.

## 5. Masking: legal candidates and player names

### Legality mask (`policy_domain: "legal"`)

The original GNN put its priority and target softmax over every node of the head's type, legal or not. The mask
restricts the softmax (and the loss) to the legal candidates, the same set MCTS uses priors for.

- **It fixed a Pass bias, and with it most of Dama's MLP-vs-GNN priority gap.**
  - gen0 mirror, non-pass top-1: 0.452 → 0.864 (Dama's MLP: 0.837). Pass accuracy moved 0.913 → 0.695, as the
    model stopped defaulting to Pass.
  - Priority KL 0.190 → 0.052, target KL 0.074 → 0.063.
  - Masking an all-node checkpoint only at test time doesn't fix it (non-pass 0.432 → 0.440). The training
    objective is what matters.
  - On held-out gen1 games the masked GNN is better on value (.077 vs .094), priority (non-pass .749 vs .284) and
    choose_use. Unmasked is better on pass accuracy (.915 vs .719) and target top-1 (.700 vs .685).
- **Value isn't affected.**
  - v4 and its legal twin agree per state as closely as two retrains do (corr .965–.975), with the same calibration.
  - The mask costs .002 value on the gen0 mirror and gains .002–.005 on gen1.
- **Play isn't affected.**
  - ver7 (masked) 50.3% vs ver12 (all-node twin) 49.7% over three decks, each within noise.
  - Among the MLPs the masked one is ahead: 42.2% vs 38.3%.
  - An earlier suspicion that the mask caused gen2's regression came from ver10 vs ver12 on Rally (17.0 vs 29.0).
    That pair also differs in data; the clean ver7/ver12 pair rejected it. The mask is kept.
- **Legality heads add nothing.** These are auxiliary BCE heads predicting legality, shared or split, at weight 0.1
  or 1.0. They give priority and target within ±.01; the split heads cost value. In play they're at 49.2%, or 45.9%
  with their priors on. Their main heads had plateaued by epoch 2, though the legality BCE was still falling.

### Player-name masking (`--drop-player-names`)

Faeries is always PlayerA in the training files, so the "PlayerA"/"PlayerB" leaves identify the deck.

- **The full-size GNN learned that shortcut.** On an unseen mirror it predicted about +0.76 for every PlayerA state
  and −0.68 for every PlayerB state: value MSE 0.86–0.95, against the MLP's 0.05–0.06.
- **Dropping the leaves from the vocab removes it.** Pauper-trained GNNs come back to 0.040–0.044 on the unseen
  mirror. With names dropped, the GNN matches or beats the MLP on unseen opponents.
- **Exception:** Mono-only training (no deck shares Faeries' cards) still finds a weaker cue, value 0.09–0.22.
- **In distribution, dropping names changes nothing** (v4 vs its no-names twin). Every model since branch D drops
  them.

## 6. Other findings from this period

- **Self-play games are the most useful data.** Self-play-trained models predict heuristic-opponent games about as
  well as models trained on them; the reverse transfers poorly. Keeping the whole buffer costs nothing on the newest
  generation's value (A–E).
- **Generation progress.**
  - gen2 (ver10) regressed from gen1 because data was cut.
  - ver13 (gen3, v4 warm start, lr 3e-5) is the first model to beat v4: +4.5 points against the offline search over
    13 decks, and ahead head to head.
  - ver15 is the best model so far on the six-deck set.
- **TD labels.** At TD 0.97/0.99 the GNN–MLP value gap shrinks: harder targets don't widen it.
- **Inference.**
  - The MLP is ≈3× cheaper per state (0.5 vs 1.5 ms in served batches).
  - Served, the GNN is faster: `NodeCache` reuses unchanged subtrees, so its forward pass is ≈3 ms against the MLP's
    7 ms. MLP test games take ≈1.5× longer.
- **Training-time tooling added.**
  - `--val-every N` validates all val sets every N steps and keeps each checkpoint.
  - Per-epoch checkpoints (`model-e<N>.pt.gz`).
  - The per-section holdout `split.txt` format, with per-shard val lines.

## Open questions

- MLP against model-driven opponents (never run).
- Policy on the latest generations only with value on the full buffer (§4).
- A fixed training-order generator, so that recipe comparisons stop carrying data-order noise.
- F2-mlp's Rally drop.
