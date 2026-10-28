# Handoff: NetGraph vs Dama's MLP on new gen1 data

Will's goal for this work: keep testing the architecture as new gen1 data is uploaded, to see whether Dama's MLP
claim continues to hold. This note lists what exists and what is known. It is not a plan.

## The claim

Source: https://adams-shaun.github.io/gorge/bots/magezero-replication/ (gorge repo `adams-shaun/gorge`, read at
`5ee887a`).

He scored both nets on "the same 63,518 ver4 mirror-test states, the same candidates and the same metric". At
evaluation, NetGraph's per-node scores were restricted to the legal candidates and renormalised.

| net | params | value MSE | target top-1 | priority non-pass top-1 |
|---|---:|---:|---:|---:|
| NetGraph d256 (this repo's `train.py`) | 8.8M | 0.0366 | 0.720 | 0.457 |
| d64 MLP ("via a bag-of-nodes bridge") | 166K | 0.0373 | 0.738 | 0.837 |

- **His reading:** value and target are the fair comparison, and they're equal. He attributes the priority gap
  mostly to the training objective: NetGraph trains a softmax over every ability node, uncastable ones included,
  while the MLP sees only legal candidates.
- **Cost:** CPU batch 1 takes 4.9 ms vs 0.18 ms. At GPU batch 512 it's 31K vs 1.05M states/s.
- **On his own engine's data:** value MSE 0.0591 vs 0.0589, priority KL 0.0287 vs 0.0272, target top-1 0.473 vs
  0.546 (NetGraph via a Go port of `StateEncoder`).

## The MLP: what is known

The d64 net is `MLPNet` in his Python package `dzg` (`gorge/dzg/models.py`). **That source is not in the gorge
repo.** gorge has only a Go port of its forward pass, documented at the top of `bots/gonet/gonet.go`:

```
card    c = relu(card_phi(relu(card_raw(raw) + card_rows(bag(identity rows)))))
pools   per group (0..3) the 0.25-scaled sum and the max of c (0 for an empty group)
state   s = dense_in(dense) + sparse_proj(bag(sparse)) + pool_proj(pools),
        then x + fc2(relu(fc1(LN(x)))) per block, then LN
value   sigmoid(value MLP(s))
option  o = LN(bag(slots) + hproj(bag(hashed)) + dense(opt dense) + bot·bot_vec
              + ea_proj(c_ea or none_a) + eb_proj(c_eb or none_b))
score   policy MLP([s ‖ o]), a logit
```

- **Size:** the page gives "1 residual block, d=64, 166K parameters". The hashed table has 16,384 rows
  (`policynet.TableRows`), and there are 4 card pool groups.
- **Candidates:** each candidate is scored with its own features, including up to two cards' entity vectors
  (`ea`/`eb`). Only legal candidates are scored.
- **Features** in gorge come from its own engine view (`bots/policynet/features.go`, e.g. `FeaturesMZ`:
  "MageZero-style per-card properties"), not from XMage graphs.
- **Not published:** the "bag-of-nodes bridge" that turned this repo's h5 graphs into MLP inputs, and what the MLP
  was trained on for the ver4 table (data, split, epochs).
- **Other code:** `bots/gonet` also loads a second architecture, `"gnn"` (`bots/gonet/gnn.go`).

## Related results in this repo

- **Bag of tokens** (`n_passes: 0, global_tokens: all`, run mz-s1-edges): collapsed. Priority stayed at uniform
  from epoch 1, because without edges every ABILITY node has the same embedding. `results/experiments.md`.
- **Flat bag of objects** (`local: flat`, mz-s9-flat): each node keeps its own leaves but loses the hierarchy.
  Val total 1.472 vs 0.905 for the full graph; priority 0.913 vs 0.542, target 0.399 vs 0.275, value 0.102 vs
  0.039. `results/experiments.md`.
- **Inference cost after the subtree cache:** `results/caching.md`.
- **Gen0 to gen1 behaviour, classic test, PIMC shards:** `results/generations.md`.

## Code and commands (branch `graph-encoder-test`)

```
python src/magezero/train.py --train DIR --test DIR --out DIR [--epochs N] [--arch JSON] [--hide-opponent-hand] [--checkpoint]
python src/magezero/test.py --data DIR --model DIR
```

- **DIR:** any folder of `.h5`/`.hdf5` shards, searched recursively.
- **Logs:** each run appends to `train.log` / `test.log` in the model directory.
- **Metrics:** `test.py` prints per-head loss, priority accuracy (by ability name, over all ability nodes),
  non-pass accuracy, target and use accuracy, and `*_legal_accuracy` / `*_legal_mass` (top-scored candidate,
  restricted to legal candidates). It doesn't print a legal-restricted non-pass priority accuracy, which is the
  metric in Dama's table.
- **Hidden hand:** `--hide-opponent-hand` is stored in the checkpoint, and `test.py` applies it.
- **NetGraph arch options:** in the `NetGraph` docstring in `src/magezero/model.py`.

## Data in the repo (git LFS)

Fetch only what you need, e.g. `git lfs pull --include="experiment_data/Pauper-Faeries/ver5/**"`.

- **`experiment_data/Pauper-Faeries/ver4/`:**
  - `training/gen0/PlayerA|PlayerB`: 13 matchups (8 Pauper including the mirror, 5 Standard-Mono).
  - `training/gen1/PlayerA|PlayerB`: the five Standard-Mono matchups.
  - `testing/gen0/PlayerA|PlayerB`: the mirror test set. This is the "ver4 mirror test" in Dama's table.
- **`experiment_data/Pauper-Faeries/ver5/`:**
  - `training/gen1`: Affinity, Burn, CawGates, Elves, MonoB, MonoW.
  - Extra MonoB shards. A trailing number (`MonoB2`, `_pimc2`) means another seed of the same run.
    - `_pimc`: Faeries ran PIMC with hidden information; MonoB kept perfect information. Faeries' seat already
      encodes the opponent's hand as hidden.
    - `_pimc_pimc`: hidden information on both seats. So far only a PlayerA shard exists.
    - `_policy`: settings not recorded here; ask Will.
  - `testing/gen0`: the mirror test set.
- **Checkpoint:** `experiment_models/Pauper-Faeries/ver4/{model,best}.pt.gz`.
- **Will's machine** uses the `data/{deck}/ver{N}/...` layout from `CLAUDE.md`. In this repo it's
  `experiment_data/`.

## Working alongside this session

- **Branch:** this session pushes to `graph-encoder-test`. Use a separate branch, or pull before every push.
- **Compute:** `CLAUDE.md` describes Kaggle only (the `kaggle/launch.py` launcher). RunPod isn't covered there.
- **Records:** experiments go in `results/` (config, run, metrics, one-line conclusion), per `CLAUDE.md`.
