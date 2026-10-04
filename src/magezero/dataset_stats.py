# dataset_stats.py

import os

import numpy as np

from dataset import H5Graphs, states_of
from model import ActionType, NodeType, load_model
from vocab import FeatureVocab, feature_id, feature_state_counts, kept_feature_ids



SHOW_PLOTS = False          # headless? set False
SAVE_PLOTS = True           # save PNGs to OUT_DIR
TOP_K = 50                  # show first K bars for each head
HIST_BINS = 21              # value histogram bins
PREVIEW_N = 30              # preview count

if not SHOW_PLOTS:
    import matplotlib
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = None
MODEL_DIR = None
OUT_DIR = None

NAME_EDGE = feature_id("name")   # edge label from a typed node to its name leaf



def load_vocabs():
    """The checkpoint's leaf and edge label vocabs (the global kept sets). gen 0 has no checkpoint."""
    checkpoint_path = os.path.join(MODEL_DIR, "model.pt.gz")
    if not os.path.exists(checkpoint_path):
        return None, None
    try:
        checkpoint = load_model(checkpoint_path)
        vocab = FeatureVocab.from_state_dict(checkpoint["feature_vocab"])
        edge_vocab = FeatureVocab.from_state_dict(checkpoint["edge_vocab"])
        print(f"[info] loaded vocabs from checkpoint: {len(vocab)} leaf rows, {len(edge_vocab)} edge label rows")
        return vocab, edge_vocab
    except Exception as e:
        print(f"[warn] failed to load vocabs: {e}")
        return None, None

def first_node(ds: H5Graphs, offsets):
    """For each item of an offsets array (edges, candidates), the first node of its state."""
    return ds.offsets[states_of(offsets)]

def candidate_names(ds: H5Graphs):
    """Name leaf id of each policy candidate: the leaf on its "name" edge, -1 if it has none.
    Ids can be looked up in FeatureTable.txt."""
    named = ds.edge_label == NAME_EDGE
    base = first_node(ds, ds.edge_offsets)[named]
    name_of = np.full(len(ds.ids), -1, dtype=np.int64)
    name_of[ds.edge_parent[named] + base] = ds.ids[ds.edge_child[named] + base]
    return name_of[ds.cand_node + first_node(ds, ds.cand_offsets)]

def mean_visit_share(ds: H5Graphs, action_type: ActionType, keys):
    """(distinct keys, mean share of MCTS visits each key's candidates get, decision states averaged
    over). Averages over this decision type's states with visits; keys gives one key per candidate."""
    state = states_of(ds.cand_offsets)
    visits = ds.cand_visits.astype(np.float64)
    totals = np.bincount(state, weights=visits, minlength=len(ds))
    decided = (ds.row[:, 3] == action_type.value) & (totals > 0)
    mask = decided[state]
    ids, inverse = np.unique(keys[mask], return_inverse=True)
    shares = np.bincount(inverse, weights=visits[mask] / totals[state[mask]], minlength=len(ids))
    n = int(decided.sum())
    return ids, shares / max(n, 1), n



def finish(out: str | None):
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def plot_value_hist(vals: np.ndarray, bins: int, title: str, out: str | None):
    edges = np.linspace(-1.0, 1.0, bins + 1)
    plt.figure()
    plt.hist(vals, bins=edges)
    plt.xlabel("Value label")
    plt.ylabel("Count")
    plt.title(title)
    finish(out)

def plot_bars(labels, values: np.ndarray, title: str, ylabel: str, out: str | None):
    if len(labels) == 0:
        print(f"[skip] {title} (no decisions)")
        return
    plt.figure()
    plt.bar(np.arange(len(labels)), values, tick_label=labels)
    plt.ylabel(ylabel)
    plt.title(title)
    finish(out)

def plot_ranked(ids: np.ndarray, values: np.ndarray, title: str, ylabel: str, out: str | None,
                top_print: int = 50, max_bars: int = 10000):
    """Bars for ids sorted by value, largest first, and the top ones printed (look ids up in FeatureTable.txt)."""
    if ids.size == 0:
        print(f"[skip] {title} (nothing to plot)")
        return
    order = np.argsort(-values, kind="mergesort")  # stable
    sorted_ids, sorted_values = ids[order], values[order]

    n_plot = int(min(max_bars, sorted_ids.size))
    xs = np.arange(n_plot)
    plt.figure()
    plt.bar(xs, sorted_values[:n_plot])
    plt.xlabel("Feature id, ranked")
    plt.ylabel(ylabel)
    plt.title(f"{title} | ids={ids.size:,} | plotted={n_plot:,}")
    if n_plot <= 100:  #annotate sparse plots with feature ids
        plt.xticks(xs, [str(i) for i in sorted_ids[:n_plot]], rotation=90, fontsize=8)
    finish(out)

    k = int(min(top_print, sorted_ids.size))
    print(f"\n{title}, top {k}:")
    for r, (fid, v) in enumerate(zip(sorted_ids[:k].tolist(), sorted_values[:k].tolist()), 1):
        print(f"{r:>3}. idx={fid:<11d}  {ylabel.lower()}={v:.4g}")



def preview(ds: H5Graphs, names: np.ndarray, n=PREVIEW_N, max_cands=24) -> str:
    lines = []
    for k in range(min(n, len(ds))):
        nodes = slice(ds.offsets[k], ds.offsets[k + 1])
        cands = slice(ds.cand_offsets[k], ds.cand_offsets[k + 1])
        value, _, is_player, action_type, use_false, use_true = ds.row[k].tolist()
        leaves = int((ds.node_type[nodes] == NodeType.LEAF).sum())
        cand = " ".join(f"{c}:{nm}:{v}" for c, nm, v in zip(ds.cand_node[cands][:max_cands].tolist(),
                                                           names[cands][:max_cands].tolist(),
                                                           ds.cand_visits[cands][:max_cands].tolist()))
        more = " ..." if cands.stop - cands.start > max_cands else ""
        lines.append(
            f"State[{k}]: actionType={int(action_type)} isPlayer={is_player > 0.5}  value={value:+.4f}\n"
            f"  nodes={nodes.stop - nodes.start} (leaves={leaves})  edges={ds.edge_offsets[k + 1] - ds.edge_offsets[k]}"
            f"  use_visits=[{use_false:g}, {use_true:g}]\n"
            f"  candidates (node:name:visits)={cand}{more}\n"
        )
    return "\n".join(lines)


def main(deck, version, split):
    global  DATA_DIR, MODEL_DIR, OUT_DIR
    DATA_DIR = f"data/{deck}/ver{version}/{split}"
    MODEL_DIR = f"models/{deck}/ver{version}"
    OUT_DIR = f"models/{deck}/ver{version}"

    def out(name):
        return os.path.join(OUT_DIR, f"{name}_{deck}_v{version}_{split}.png") if SAVE_PLOTS else None

    # everything below runs on raw ids (not vocab rows) so the printed feature ids
    # can be looked up in FeatureTable.txt
    print(f"[load] {DATA_DIR}")
    ds = H5Graphs(DATA_DIR)
    print(f"[stats] samples={len(ds)}")
    if len(ds) == 0:
        return
    print(f"[stats] per state: nodes={len(ds.ids) / len(ds):.1f}  edges={len(ds.edge_child) / len(ds):.1f}  "
          f"candidates={len(ds.cand_node) / len(ds):.1f}")
    per_type = np.bincount(ds.node_type, minlength=len(NodeType)) / len(ds)
    print("[stats] nodes per state by type: " + "  ".join(f"{t.name}={per_type[t]:.1f}" for t in NodeType))

    # leaves and edge labels: occurring here, kept by the >10-states rule over this split alone,
    # and covered by the checkpoint's vocabs (the global kept sets)
    leaf = ds.node_type == NodeType.LEAF
    leaf_ids, leaf_counts = feature_state_counts(ds.ids[leaf], states_of(ds.offsets)[leaf])
    label_ids = np.unique(ds.edge_label)
    local_leaves = kept_feature_ids(ds.ids[leaf], states_of(ds.offsets)[leaf])
    local_labels = kept_feature_ids(ds.edge_label, states_of(ds.edge_offsets))
    vocab, edge_vocab = load_vocabs()
    print(f"[stats] leaf ids: occurring={leaf_ids.size}  kept locally={local_leaves.size}"
          + (f"  in checkpoint vocab={int((vocab.lookup(leaf_ids) >= 0).sum())}" if vocab is not None else ""))
    print(f"[stats] edge labels: occurring={label_ids.size}  kept locally={local_labels.size}"
          + (f"  in checkpoint vocab={int((edge_vocab.lookup(label_ids) >= 0).sum())}" if edge_vocab is not None else ""))

    # policy targets: priority by ability name, targets by node type, choose_use as false/true
    names = candidate_names(ds)
    cand_type = ds.node_type[ds.cand_node + first_node(ds, ds.cand_offsets)]
    priority_names, priority_share, n_priority = mean_visit_share(ds, ActionType.PRIORITY, names)
    target_types, target_share, n_target = mean_visit_share(ds, ActionType.CHOOSE_TARGET, cand_type)
    use = ds.row[:, 4:6]
    used = (ds.row[:, 3] == ActionType.CHOOSE_USE.value) & (use.sum(1) > 0)
    use_share = (use[used] / use[used].sum(1, keepdims=True)).mean(0) if used.any() else np.zeros(2)
    print(f"[stats] decision states with visits: priority={n_priority}  target={n_target}  use={int(used.sum())}")

    # value histogram
    plot_value_hist(ds.row[:, 0], HIST_BINS, f"Value labels ({deck} v{version} {split})", out("value_hist"))

    # per-head mean visit share
    plot_ranked(priority_names, priority_share, "PRIORITY visit share by ability name (-1 = no name edge)",
                "Mean visit share", out("avg_policy_priority"), top_print=TOP_K, max_bars=TOP_K)
    plot_bars([NodeType(int(t)).name for t in target_types], target_share,
              "CHOOSE_TARGET visit share by node type", "Mean visit share", out("avg_policy_target"))
    plot_bars(["false", "true"] if used.any() else [], use_share,
              "CHOOSE_USE visit share", "Mean visit share", out("avg_policy_binary"))
    print(f"[stats] CHOOSE_TARGET share by type: "
          + "  ".join(f"{NodeType(int(t)).name}={s:.3f}" for t, s in zip(target_types, target_share)))
    print(f"[stats] CHOOSE_USE share: false={use_share[0]:.3f}  true={use_share[1]:.3f}")

    # leaf distribution
    plot_ranked(leaf_ids, leaf_counts, "Leaf id distribution", "States", out("idx_dist"))


    print("\n=== Preview ===")
    print(preview(ds, names, PREVIEW_N))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--split", default="testing")
    args = parser.parse_args()
    main(args.deck, args.version, args.split)
