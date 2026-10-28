# Subtree cache of local-layer outputs

Inference optimization for the GNN server. It cuts the local layers' cost by reusing outputs for subtrees the server
has already seen. The network's results are unchanged; the outputs only differ by floating-point noise.

- MageZero `graph-encoder-test`: `1ab2848` (cache), `fc02fb1` (in-place writes)
- XMage `graph-encoder`: `caa4d43d` (`RemoteModelEvaluator` sends the hashes)
- Detailed runs: `results/experiments.md`, sections "subtree cache", "CUDA graphs on the cached path" and
  "local-layer rows written in place"; logs in `results/kaggle/bench-cache`, `bench-cache-graphs` and `prof-local`.

## Idea

A node's local-layer outputs depend only on the subgraph below it: its type, its leaves, its edge labels and its
children's outputs. The stage order of the local passes is fixed by node type, so this holds for every pass.
Consecutive states of a game, and states across MCTS trees, share most of their subtrees: the same cards, abilities
and permanents. XMage already computes a hash per subtree (`FeatureGraph.Node.hash`, 16 WL rounds in
`getStateHash`), so the cache key comes free.

Reuse measured on recorded games before building it (gen1 data, a Python stand-in for the hash), share of internal
nodes already seen:

| reuse from | all | ABILITY | CARD | PERMANENT | ZONE | PLAYER | STACK_OBJECT | ROOT |
|---|---|---|---|---|---|---|---|---|
| every earlier state of the game | 93% | 99% | 97% | 91% | 84% | 41% | 36% | 0% |
| the previous state only | 86% | | | | | | | |

Deduplicating within one batch alone (4 states × 8 games) would leave 13% of nodes unique. So the cache has to
persist across batches.

## Design

- **Client** (`RemoteModelEvaluator.inferAsync`): calls `getStateHash()` (a no-op when the search already did), reads
  each node's `hash` in `GraphArrays.order`, and sends the batch's hashes as one raw little-endian int64 array,
  `node_hash`. A server without the cache ignores the key.
- **Server** (`cache.py`, `NodeCache`):
  - **Table:** a GPU table `[passes, 2^19 slots, d]` in fp32, about 1 GiB, shared by all games, plus a dict from
    hash to slot.
  - **Eviction:** least recently used, freeing the oldest eighth at a time, never a slot the current batch uses.
  - **Per batch:** each hash the cache doesn't hold is computed once (at its first node). Every other internal node
    reads each pass's output from its slot.
  - **Order:** a node is read from the cache at its own stage, so later stages see it exactly as if computed, and
    the sequential semantics hold. Stages with nothing to compute are skipped.
- **Fallbacks:** requests without `node_hash` take the old path. A failed forward pass clears the cache, so slots
  that were assigned but never written are never read. The cache needs `WORKERS = 1`.
- **In-place writes** (`model.py`): each local stage used to copy the whole node table, leaves included, to update a
  few rows. With the cache that was the main leftover local-layer cost, about 6.7 GB of memory traffic per
  512-state pass. Inference now writes rows in place; training and the "parallel" mode keep the copies.

## Correctness

| check | result |
|---|---|
| CPU fp32, 1,280 states streamed (8 games × 4 states per batch), cached vs plain | max diff 2e-5; 0 top-1 changes except exact float ties |
| same, 800-slot cache evicting constantly | same |
| server, old-format requests (no hashes) | byte-identical to the original server |
| server, hashed vs unhashed requests | max diff 1.3e-5 |
| Java round trip, 200 states, one at a time and batched | all 86,898 numbers within 1.9e-5 of uncached, 0 top-1 changes |
| T4 fp16, cached vs plain | max diff 0.023 (fp16 noise); 2 top-1 changes in ~1,400 states, both near-ties within the noise |
| in-place writes vs the previous commit, CPU | bit-identical, plain and cached |

The 16 hash rounds cover the local layers' whole dependency depth (at most 14 hops over 2 passes). The 64-bit hash
makes collisions negligible.

## Speed

**T4, server path** (lookup + plan + forward + copy back, fp16), stream benchmark (`bench.py`), ms per batch:

| states per batch | before | cache | cache + in-place | nodes computed |
|---|---|---|---|---|
| 1 | 20.6 | 13.4 | 12.4 | 9.7% |
| 8 | 23.6 | 20.1 | 19.7 | 5.6% |
| 32 | 30.9 | 27.2 | 23.4 | 4.7% |
| 128 | 36.8 | 32.0 | 30.1 | 4.6% |
| 512 | 135.7 | 88.4 | **64.8** (2.1×) | 4.6% |

At 512 states that is about 7,900 states/s, up from about 3,800.

**Will's RTX 4070, live games** (server logs; batch sizes vary with load):

| | states per batch | forward | request latency | nodes computed |
|---|---|---|---|---|
| before | 32–50 | 69–110 ms | 84–159 ms | – |
| after | 13–66 | 32–38 ms | 48–79 ms | 1.3–1.5% |

The states in the later log are larger: about 310 nodes each, against about 205.

## What's left

- **Large batches are compute-bound in the global layers.** At 512 states the GPU is busy 48 of 52.8 ms. The top
  kernels are layer norm, elementwise ops, attention and fp16 GEMMs, mostly in the 2-layer global transformer.
- **Mid-size batches wait on the CPU.** At 128 states the GPU is busy 13 of 24 ms; the rest is Python and kernel
  launches.
- **CUDA graphs:** tried on the cached path and removed. They cut batch 1–8 latency to 8–10 ms on the T4, but
  padding made them slower above about 32 states, where throughput comes from.

## Not tested

- A load test of the full server on a T4.
- Win rates in games played with the cache on, against games without it.
