# mz_dataset_stats_simple.py

import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import H5Indexed, collate_batch
from vocab import FeatureVocab, kept_feature_ids
from model import load_model, PRIORITY_A_MAX, PRIORITY_B_MAX, TARGETS_MAX, BINARY_MAX, ActionType



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



def dataloader(ds, bs=512):
    return DataLoader(
        ds, batch_size=bs, shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_batch
    )

def load_vocab() -> FeatureVocab | None:
    """the checkpoint's vocab is the global kept set (what ignore.roar used to be). gen 0 has no checkpoint."""
    checkpoint_path = os.path.join(MODEL_DIR, "model.pt.gz")
    if not os.path.exists(checkpoint_path):
        return None
    try:
        checkpoint = load_model(checkpoint_path)
        vocab = FeatureVocab.from_state_dict(checkpoint["feature_vocab"])
        print(f"[info] loaded feature vocab from checkpoint: {len(vocab)} rows")
        return vocab
    except Exception as e:
        print(f"[warn] failed to load feature vocab: {e}")
        return None

def feature_counts(ds: H5Indexed):
    """(observed feature ids, activation count of each). H5Indexed already holds the flat id array,
    so this replaces the 2M-wide scatter_add: only ids that occur get a slot."""
    ids, counts = np.unique(ds.indices_t.numpy(), return_counts=True)
    return ids, counts



def stream_stats(ds: H5Indexed):
    """Stream once and collect per-head average policy + value histogram data."""
    dl = dataloader(ds, bs=512)

    pA_sum = torch.zeros(PRIORITY_A_MAX, dtype=torch.float32)
    pB_sum = torch.zeros(PRIORITY_B_MAX, dtype=torch.float32)
    t_sum  = torch.zeros(TARGETS_MAX,   dtype=torch.float32)
    b_sum  = torch.zeros(BINARY_MAX,    dtype=torch.float32)

    npA = npB = nT = nB = 0
    vals = []

    with torch.no_grad():
        for batch in dl:
            idx, off, policy, value, is_player, action_type = batch
            # decision masks (same semantics as train.py)
            nonzero = (policy > 0).sum(dim=1)
            decision_mask = nonzero > 1
            if decision_mask.any():
                is_p = is_player.squeeze(-1) > 0
                a_t  = action_type.squeeze(-1).to(torch.long)
                mask_pA = (a_t == ActionType.PRIORITY.value) & is_p & decision_mask
                mask_pB = (a_t == ActionType.PRIORITY.value) & (~is_p) & decision_mask
                mask_t  = (a_t == ActionType.CHOOSE_TARGET.value) & decision_mask
                mask_b  = (a_t == ActionType.CHOOSE_USE.value) & decision_mask

                if mask_pA.any():
                    pA_sum += policy[mask_pA, :PRIORITY_A_MAX].sum(dim=0)
                    npA += int(mask_pA.sum().item())
                if mask_pB.any():
                    pB_sum += policy[mask_pB, :PRIORITY_B_MAX].sum(dim=0)
                    npB += int(mask_pB.sum().item())
                if mask_t.any():
                    t_sum  += policy[mask_t, :TARGETS_MAX].sum(dim=0)
                    nT += int(mask_t.sum().item())
                if mask_b.any():
                    b_sum  += policy[mask_b, :BINARY_MAX].sum(dim=0)
                    nB += int(mask_b.sum().item())

            vals.extend(torch.atleast_1d(value).squeeze(-1).tolist())

    ids, counts = feature_counts(ds)

    return {
        "avg_player_priority": (pA_sum / max(npA, 1)).cpu().numpy(),
        "avg_opponent_priority": (pB_sum / max(npB, 1)).cpu().numpy(),
        "avg_targets": (t_sum / max(nT, 1)).cpu().numpy(),
        "avg_binary": (b_sum / max(nB, 1)).cpu().numpy(),
        "values": np.array(vals, dtype=np.float32),
        "feature_ids": ids,
        "feature_counts": counts,
        "num_samples": len(ds),
        "counts": {"pA": npA, "pB": npB, "t": nT, "b": nB},
    }


def plot_value_hist(vals: np.ndarray, bins: int, title: str, out: str | None):
    edges = np.linspace(-1.0, 1.0, bins + 1)
    plt.figure()
    plt.hist(vals, bins=edges)
    plt.xlabel("Value label")
    plt.ylabel("Count")
    plt.title(title)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def plot_avg_bar(arr: np.ndarray, k: int, title: str, out: str | None):
    A = len(arr)
    if A == 0:
        print(f"[skip] {title} (no actions present)")
        return
    kk = max(1, min(A, k))
    xs = np.arange(kk); ys = arr[:kk]
    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Action index (0..K-1)")
    plt.ylabel("Mean policy label (train target)")
    plt.title(f"{title} | A={A}, first {kk}")
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

def plot_idx_dist(ids: np.ndarray,
                  counts: np.ndarray,
                  title: str,
                  out: str | None,
                  top_print: int = 50,
                  max_bars: int = 10000):

    order = np.argsort(-counts, kind="mergesort")  # stable
    sorted_idx = ids[order]
    sorted_counts = counts[order]

    n_plot = int(min(max_bars, sorted_counts.size))
    xs = np.arange(n_plot)
    ys = sorted_counts[:n_plot]

    plt.figure()
    plt.bar(xs, ys)
    plt.xlabel("Occurring features sorted by frequency (rank)")
    plt.ylabel("Activation count")
    plt.title(f"{title} | occurring={ids.size:,} | plotted={n_plot:,}")
    if n_plot <= 100:  #annotate sparse plots with feature ids
        plt.xticks(xs, [str(i) for i in sorted_idx[:n_plot]], rotation=90, fontsize=8)
    if out:
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, bbox_inches="tight")
    if SHOW_PLOTS:
        plt.show()
    plt.close()

    k = int(min(top_print, sorted_idx.size))
    top_pairs = list(zip(sorted_idx[:k].tolist(), sorted_counts[:k].tolist()))

    print(f"\nTop {k} most occurring feature indices:")
    for r, (fi, cnt) in enumerate(top_pairs, 1):
        print(f"{r:>3}. idx={fi:<10d}  count={cnt}")


    return top_pairs


def preview(ds: H5Indexed, n=PREVIEW_N, max_idx=96) -> str:
    dl = dataloader(ds, bs=1)
    lines = []
    for i, batch in enumerate(dl):
        if i >= n: break
        idxs, off, pol, val, is_p, a_t = batch
        idxs = idxs.tolist()
        idx_str = " ".join(map(str, idxs[:max_idx])) + (" ..." if len(idxs) > max_idx else "")
        valf = float(torch.atleast_1d(val).squeeze(-1)[0].item())
        ispf = bool(int(is_p.squeeze(-1)[0].item()))
        aty  = int(a_t.squeeze(-1)[0].item())
        lines.append(
            f"State[{i}]: {idx_str}\n"
            f"  actionType={aty} isPlayer={ispf}  value={valf:+.4f}\n"
            f"  policy[:16]={pol[0, :16].tolist()}\n"
        )
    return "\n".join(lines)


def main(deck, version, split):
    global  DATA_DIR, MODEL_DIR, OUT_DIR
    DATA_DIR = f"data/{deck}/ver{version}/{split}"
    MODEL_DIR = f"models/{deck}/ver{version}"
    OUT_DIR = f"models/{deck}/ver{version}"

    # everything below runs on raw ids (not vocab rows) so the printed feature ids
    # can be looked up in FeatureTable.txt; policy/value stats don't depend on indices anyway
    print(f"[load] {DATA_DIR}")
    ds = H5Indexed(DATA_DIR)
    print(f"[stats] samples={len(ds)}")

    ids, counts = feature_counts(ds)
    print(f"[stats] unique active raw feature indices ={ids.size}")

    # global kept set = the checkpoint's vocab (was ignore.roar)
    vocab = load_vocab()
    if vocab is not None:
        in_vocab = vocab.lookup(ids) >= 0
        print(f"[stats] unique active feature indices in checkpoint vocab ={int(in_vocab.sum())}")

    # local kept set = ignore rule over this split alone (was create_redundancy_ignore_list)
    local_kept = kept_feature_ids(ds.indices_t.numpy(), ds.idxptr_t.numpy())
    print(f"[stats] unique active feature indices after local ignore ={local_kept.size}")

    sv = stream_stats(ds)
    print(f"[stats] aggregated samples={sv['num_samples']} "
          f"(pA={sv['counts']['pA']}, pB={sv['counts']['pB']}, "
          f"t={sv['counts']['t']}, b={sv['counts']['b']})")

    if SAVE_PLOTS and not os.path.exists(OUT_DIR):
        os.makedirs(OUT_DIR, exist_ok=True)

    # value histogram
    plot_value_hist(
        sv["values"], HIST_BINS, f"Value labels ({deck} v{version} {split})",
        os.path.join(OUT_DIR, f"value_hist_{deck}_v{version}_{split}.png") if SAVE_PLOTS else None
    )

    # per-head avg policy bars
    plot_avg_bar(
        sv["avg_player_priority"], TOP_K, "Avg policy – Player PRIORITY",
        os.path.join(OUT_DIR, f"avg_policy_pA_{deck}_v{version}_{split}.png") if SAVE_PLOTS else None
    )
    plot_avg_bar(
        sv["avg_opponent_priority"], TOP_K, "Avg policy – Opponent PRIORITY",
        os.path.join(OUT_DIR, f"avg_policy_pB_{deck}_v{version}_{split}.png") if SAVE_PLOTS else None
    )
    plot_avg_bar(
        sv["avg_targets"], TOP_K, "Avg policy – CHOOSE_TARGET",
        os.path.join(OUT_DIR, f"avg_policy_target_{deck}_v{version}_{split}.png") if SAVE_PLOTS else None
    )
    plot_avg_bar(
        sv["avg_binary"], TOP_K, "Avg policy – CHOOSE_USE",
        os.path.join(OUT_DIR, f"avg_policy_binary_{deck}_v{version}_{split}.png") if SAVE_PLOTS else None
    )
    plot_idx_dist(
        sv["feature_ids"], sv["feature_counts"], "Idx distribution",
        os.path.join(OUT_DIR, f"idx_dist_{deck}_v{version}_{split}.png") if SAVE_PLOTS else None
    )


    print("\n=== Preview ===")
    print(preview(ds, PREVIEW_N))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    parser.add_argument("--split", default="testing")
    args = parser.parse_args()
    main(args.deck, args.version, args.split)