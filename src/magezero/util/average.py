"""Averages the weights of checkpoints (e.g. a run's last few validated ones) into one; everything else comes from the last.
  PYTHONPATH=. python util/average.py OUT.pt.gz CKPT ..."""
import gzip, sys

import torch

from model import load_model

out, paths = sys.argv[1], sys.argv[2:]
cps = [load_model(p) for p in paths]
avg = cps[-1]
avg["model_state_dict"] = {k: (sum(c["model_state_dict"][k].double() for c in cps) / len(cps)).to(v.dtype)
                           for k, v in avg["model_state_dict"].items()}
avg["averaged"] = paths
with gzip.open(out, "wb", compresslevel=1) as f:
    torch.save(avg, f)
