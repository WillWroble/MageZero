# MageZero: notes for Claude

MageZero trains deck-specific AlphaZero-style agents for Magic: The Gathering, using XMage as the game engine.
This repo holds the Python side: dataset, model, training and the inference server. The Java side (state encoding, MCTS,
`RemoteModelEvaluator`) is in `WillWroble/mage`.

## How to work
- Write clean, minimal code. Reuse existing code where it's reasonable, and follow the conventions already in the repo.
- If a design decision is unclear, ask rather than guess.
- The datasets are small. Prefer models that generalize over models that fit the training data hard, and judge
  every change by its test metrics.
- Develop on `graph-encoder-test`. Commit and push experiments there.
- Will runs his own experiments in parallel on his machine. Work autonomously, then report back with results.

## Compute
- Cloud sessions have no GPU. GPU work runs on **Kaggle** through its API (`KAGGLE_API_KEY`).
  Runs have no live logs, and each one is capped at 12 h.
- You may use Will's whole weekly Kaggle GPU quota (about 30 h). Plan runs so each one answers a specific question.
- Kaggle GPUs (T4/P100) are not Will's RTX 4070 (12 GB). Compare timings against each other on the same GPU,
  never against Will's absolute numbers.
- Record every experiment in `results/`: its config, the Kaggle run, metrics, and a one-line conclusion.

## Code map
- `src/magezero/model.py`: `NetGraph`. d=256, 4 heads, FF 512, no dropout (chosen from the sweep in `results/experiments.md`). Two bottom-up local passes, each one a `LocalLayer`
  per node type, run in order ABILITY -> CARD -> PERMANENT -> STACK_OBJECT -> ZONE -> PLAYER -> ROOT. Then a
  2-layer global TransformerEncoder over [CLS, internal nodes], padded to the batch's largest state.
  Heads: per-node priority and target, CLS choose_use and value.
- `src/magezero/dataset.py`: `H5Graphs` and `collate_graphs`. `train.py` / `test.py`: training and evaluation.
  `train.py --train DIR --test DIR --out DIR`, `test.py --data DIR --model DIR`; each appends its output to
  `train.log` / `test.log` in the model directory. `--hide-opponent-hand` is saved in the checkpoint and
  `test.py` applies it.
- `src/magezero/server.py`: Flask on waitress, with one worker thread that batches queued requests (msgpack).
- `src/magezero/cache.py`: `NodeCache`, the server's persistent cache of local-layer outputs keyed by XMage's
  subtree hashes (`node_hash` in requests). Results: `results/experiments.md`.
- The flat embedding-bag (EB) model from v0.1 is not on this branch; flat-only code was removed in `42afeab`.

## Data
Layout: `data/{deck}/ver{N}/training/*.h5` and `data/{deck}/ver{N}/testing/*.h5`.

**UWTempo** (vs the Standard16 pool)
- `ver30`: v0.1 era, before the GNN, flat embedding bag. 100 games per shard. Useful for a rough EB vs GNN comparison.
- `ver31`: v0.2 era, the first GNN. 100 games per shard. Four matchups are corrupted; they were kept in case games
  can be salvaged from their shards. There is no separate test set. The only MonoW matchup was used as the test set,
  which makes it an out-of-distribution opponent.

**Pauper-Faeries** (vs the Pauper pool)
- `ver1`-`ver3`: v0.2 GNN era. 100 games per shard. There is no in-distribution test set; the mirror was the holdout.
  The three versions are permutations of the same data:
  - `ver1`: PlayerA perspectives only.
  - `ver2`: PlayerA, plus UWTempo games.
  - `ver3`: PlayerA and PlayerB perspectives together.
- `ver4`: v0.3 era. 100-400 games per shard. Differences from v0.2:
  - Creatures already decided in factorized combat carry a "passed" flag.
  - Only playable activated abilities are encoded, instead of every activated ability of every card in a zone.
  - Many engine changes that may have changed behavior.
  - Sim budget 300 instead of 1000, with a 60 s timeout.
  - It has a real test set: 100-game mirror shards, separate from the 400-game mirror shard in training.

Results of earlier experiments on this data: `magezero-faeries-findings.md`. Read it before starting. More data will
be added over time. As data is added from later generations, probe the h5s directly to see how the agent's play evolves over time.  

## Current goals
1. **Architecture sweeps.** Embedding width, number of layers, heads, FF size. Keep the dataset size in mind.
2. **Inference speed.** The GNN takes about 50 ms per evaluation, mostly from the sequential message passing.
   - Ablation: ignore all edge data and treat each state as a bag of tokens. Does accuracy collapse?
   - Try 1 bottom-up pass, and multiple passes run in parallel instead of in sequence.
3. **EB v0.1 vs GNN v0.2 vs GNN v0.3.** The comparison has heavy confounds and mixed data, but there should be
   enough to make ballpark observations about each architecture's strengths and weaknesses.

## Inference throughput (read from the code, not measured)
The goal is for GNN evaluation to keep up with the game generator: 16 game threads at 300 sims/s each,
4,800 sims/s in total.

What currently caps a batch:
| setting | where | value | effect |
|---|---|---|---|
| waitress threads | `server.py` `init` | `threads=6` | each `/evaluate` call holds a thread until its batch is done, so at most 6 requests are ever queued |
| client states per call | `RemoteModelEvaluator.java` | `maxBatchSize = 4` | combined with the above: at most about 24 states per forward pass |
| client concurrent calls | `RemoteModelEvaluator.java` | `MAX_CONCURRENT_CALLS = 16` | doesn't help past the 6 server threads |
| server batch cap | `server.py` | `MAX_BATCH = 64` requests | unreachable with 6 threads |
| pending evals per tree | `Mage.MageZero/config/config.yml` | `max_pending: 4` | evaluations in flight per game |

Other costs:
- `NetGraph.forward` forces about 8 GPU syncs: `nonzero()` and `len()` per stage, plus `int(counts.max())`. The 14 local
  layers each launch many small kernels, so small batches are probably bound by launch overhead.
  CUDA graphs or `torch.compile` are candidates once batches are larger.
- The server runs `collate_graphs` on the CPU in the worker thread, so the GPU sits idle meanwhile.
- The server splits states in a Python loop (`request_states`) and calls `.tolist()` for every state on the way out.
- The server prints three lines per request and one per batch.

Estimates from reading the model:
- About 15-20 MB of activations per state at about 1,500 nodes (300 internal) with fp16 autocast, so roughly 600
  states fit in one batch.
- About 0.8 ms of GPU time per state.
- If the 50 ms is mostly fixed overhead, throughput levels off at around 64-128 states per batch.

First measurement: time one forward pass at batch sizes 1, 8, 32, 128 and 512, replicating a real state, and record
ms and peak memory (run each size twice and keep the second). A flat curve means batching is the lever; a linear
curve means compute per state is.

Open questions:
- Where does the 50 ms go: HTTP and msgpack, Python, or the forward pass?
- Once batches are large, is it worth sorting states by size to cut padding in the global layers?
