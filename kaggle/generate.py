"""
Generate games in a Kaggle GPU kernel (T4, internet on): kaggle/generate.sh serves the model from --dataset and plays
--config with XMage at --mage-commit. Outputs (shards, logs, result.txt, xmage.tar.gz on a fresh build) land in the
kernel's output. The log, with a progress line every 5 minutes, streams live: kaggle kernels logs -f willwroble/mz-<name>

  python kaggle/generate.py --name onpolicy-monob --config kaggle/configs/onpolicy-MonoB.yml \
      --dataset mz-faeries-ver15-model --mage-commit 7296f0e1
  kaggle kernels output willwroble/mz-onpolicy-monob -p results/kaggle/onpolicy-monob

--xmage mz-<name> reuses the XMage build saved by an earlier kernel instead of building it again.
"""
import argparse
import base64
import io
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path

USER = "willwroble"
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src" / "magezero"

KERNEL = '''
import base64, io, os, subprocess, tarfile

tarfile.open(fileobj=io.BytesIO(base64.b64decode({code!r})), mode="r:gz").extractall("/tmp/mz", filter="data")
env = {{**os.environ, "SRC": "/tmp/mz/src", "CONFIG": "/tmp/mz/config.yml", "MAGE_COMMIT": {commit!r},
       "GEN_SECONDS": "{gen_seconds}"}}
# the log goes to stdout as it happens (kaggle kernels logs -f) and to run.log
p = subprocess.Popen(["bash", "/tmp/mz/generate.sh"], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
with open("/kaggle/working/run.log", "w") as log:
    for line in p.stdout:
        log.write(line)
        log.flush()
        print(line, end="", flush=True)
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="kernel name: mz-<name>")
    parser.add_argument("--config", required=True, help="XMage config in the repo")
    parser.add_argument("--dataset", required=True, help="Kaggle dataset holding the model's model.pt.gz")
    parser.add_argument("--mage-commit", required=True, help="WillWroble/mage commit to build")
    parser.add_argument("--xmage", help="earlier kernel (mz-<name>) whose xmage.tar.gz to reuse")
    parser.add_argument("--hours", type=float, default=11.5, help="kernel timeout; the games get 0.5 h less")
    args = parser.parse_args()

    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for f in SRC.rglob("*.py"):  # util/ included: dataset.py imports from it
            tar.add(f, arcname=f"src/{f.relative_to(SRC)}")
        tar.add(ROOT / "kaggle" / "generate.sh", arcname="generate.sh")
        tar.add(ROOT / args.config, arcname="config.yml")
    code = base64.b64encode(buf.getvalue()).decode()

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "run.py").write_text(KERNEL.format(code=code, commit=args.mage_commit,
                                                      gen_seconds=int((args.hours - 0.5) * 3600)))
        (Path(d) / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/mz-{args.name}", "title": f"mz-{args.name}", "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": True, "dataset_sources": [f"{USER}/{args.dataset}"],
            "kernel_sources": [f"{USER}/{args.xmage}"] if args.xmage else [],
        }))
        subprocess.run(["kaggle", "kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4",
                        "-t", str(int(args.hours * 3600))], check=True)


if __name__ == "__main__":
    main()
