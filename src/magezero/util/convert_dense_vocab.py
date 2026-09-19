"""
Convert a full-table checkpoint (one embedding row per possible feature id) into a dense-vocab
checkpoint (one row per feature the model actually uses). Lossless: the kept rows are copied
unchanged and the rows that are dropped belong to ignored ids, which the model never reads.

  python src/magezero/util/convert_dense_vocab.py --deck UWTempo --version 3 --in-place
  python src/magezero/util/convert_dense_vocab.py --model m.pt.gz --ignore ignore.roar --out dense.pt.gz

Continue training the converted model with `train.py --dense-vocab --checkpoint` (or
`training.dense_vocab: true` in run.yml). The optimizer state is not carried over: its tensors
are shaped like the full table, and train.py starts a fresh optimizer each generation anyway.
"""
import argparse
import gzip
import os
import shutil
import sys

import torch
from pyroaring import BitMap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from model import load_model  # noqa: E402
from vocab import FeatureVocab  # noqa: E402


def convert(checkpoint: dict, ignore: BitMap) -> dict:
    if "feature_vocab" in checkpoint:
        raise ValueError("checkpoint already has a feature vocab")
    state = dict(checkpoint["model_state_dict"])
    weight = state["embedding.weight"]
    table_rows = weight.shape[0]
    kept = torch.tensor(list(BitMap(range(table_rows)) - ignore), dtype=torch.long)  # ascending
    vocab = FeatureVocab(kept.numpy(), feature_hash_bins=table_rows)
    state["embedding.weight"] = weight.index_select(0, kept).clone()
    out = {k: v for k, v in checkpoint.items() if k not in ("model_state_dict", "optimizer_dense_state_dict")}
    out["model_state_dict"] = state
    out["feature_vocab"] = vocab.state_dict()
    return out


def save_gz(obj: dict, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    with open(tmp, "rb") as f_in, gzip.open(path, "wb", compresslevel=1) as f_out:
        shutil.copyfileobj(f_in, f_out, length=16 * 1024 * 1024)
    os.remove(tmp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--deck")
    ap.add_argument("--version", type=int)
    ap.add_argument("--model", help="checkpoint path (default models/<deck>/ver<version>/model.pt.gz)")
    ap.add_argument("--ignore", help="ignore list path (default next to the checkpoint)")
    ap.add_argument("--out", help="output checkpoint path")
    ap.add_argument("--in-place", action="store_true", help="overwrite the input checkpoint")
    args = ap.parse_args()

    model_path = args.model or f"models/{args.deck}/ver{args.version}/model.pt.gz"
    ignore_path = args.ignore or os.path.join(os.path.dirname(model_path), "ignore.roar")
    if not args.out and not args.in_place:
        ap.error("pass --out PATH or --in-place")
    out_path = model_path if args.in_place else args.out

    with open(ignore_path, "rb") as f:
        ignore = BitMap.deserialize(f.read())
    checkpoint = load_model(model_path)
    rows_before = checkpoint["model_state_dict"]["embedding.weight"].shape[0]
    converted = convert(checkpoint, ignore)
    rows_after = converted["model_state_dict"]["embedding.weight"].shape[0]
    save_gz(converted, out_path)
    print(f"embedding rows {rows_before:,} -> {rows_after:,}; wrote {out_path} "
          f"({os.path.getsize(out_path) / 2**20:.1f} MB)")


if __name__ == "__main__":
    main()
