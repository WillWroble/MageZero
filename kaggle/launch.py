"""
Push an experiment to Kaggle as a GPU script kernel (T4 x2). The kernel unpacks src/magezero, mounts the
dataset as data/<run>/ver<version>, and runs one training run per GPU, each followed by bench.py.
Logs land in the kernel's output as <run>.log.

  python kaggle/launch.py --name passes --runs '[{"name": "seq1", "args": "--epochs 6", "arch": {"n_passes": 1}}]'
Optional run keys: "ref" trains with src/magezero from a git ref instead of the working tree; "data" picks a
subdirectory of the dataset (the one holding training/ and testing/); "setup" is a shell command run first
(it turns on the kernel's internet access, e.g. for pip).
  kaggle kernels status willwroble/mz-baseline
  kaggle kernels output willwroble/mz-baseline -p results/kaggle/baseline
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
SRC = Path(__file__).resolve().parents[1] / "src" / "magezero"

KERNEL = '''
import base64, glob, io, json, os, shlex, subprocess, tarfile, time
from pathlib import Path

CODE = {code!r}
RUNS = {runs!r}
VERSION = {version}

for ref, code in CODE.items():
    tarfile.open(fileobj=io.BytesIO(base64.b64decode(code))).extractall(f"/tmp/mz/{{ref}}", filter="data")
os.chdir("/kaggle/working")
procs = []
for gpu, run in enumerate(RUNS):
    # one run loads the dataset at a time: loading peaks at about twice the resident size
    while procs and procs[-1].poll() is None and "arch:" not in open(f"{{RUNS[gpu - 1]['name']}}.log").read():
        time.sleep(10)
    Path(f"data/{{run['name']}}").mkdir(parents=True, exist_ok=True)
    data = glob.glob(f"/kaggle/input/**/{{run.get('data', '')}}/training".replace("//", "/"), recursive=True)[0]
    os.symlink(Path(data).parent, f"data/{{run['name']}}/ver{{VERSION}}")
    src = f"/tmp/mz/{{run.get('ref', 'HEAD')}}"
    deck = f"--deck {{run['name']}} --version {{VERSION}}"
    arch = f"--arch {{shlex.quote(json.dumps(run['arch']))}}" if "arch" in run else ""
    cmd = (f"{{run.get('setup', 'true')}} && python -u {{src}}/train.py {{deck}} {{arch}} {{run['args']}} && "
           f"(test ! -f {{src}}/bench.py || python -u {{src}}/bench.py {{deck}}); "
           f"rm -rf models/{{run['name']}}/ver{{VERSION}}/best.pt.gz")
    log = open(f"{{run['name']}}.log", "w")
    procs.append(subprocess.Popen(cmd, shell=True, stdout=log, stderr=subprocess.STDOUT,
                                  env={{**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}}))
for p in procs:
    p.wait()
for run in RUNS:
    print(open(f"{{run['name']}}.log").read())
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="kernel name: mz-<name>")
    parser.add_argument("--runs", type=json.loads, required=True,
                        help='at most 2 (one per GPU): [{"name": ..., "args": "train.py args", "arch": {NetGraph options}}]')
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--dataset", default="magezero-pauper-faeries-ver4")
    args = parser.parse_args()
    assert len(args.runs) <= 2, "one run per GPU"

    code = {}
    for ref in {run.get("ref", "HEAD") for run in args.runs}:
        if ref == "HEAD":  # the working tree, uncommitted changes included
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w:gz") as tar:
                for f in SRC.glob("*.py"):
                    tar.add(f, arcname=f.name)
            code[ref] = base64.b64encode(buf.getvalue()).decode()
        else:
            tar = subprocess.run(["git", "archive", "--format=tar.gz", f"{ref}:src/magezero"],
                                 cwd=SRC, capture_output=True, check=True).stdout
            code[ref] = base64.b64encode(tar).decode()

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "run.py").write_text(KERNEL.format(code=code, runs=args.runs, version=args.version))
        internet = any("setup" in run for run in args.runs)
        (Path(d) / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/mz-{args.name}", "title": f"mz-{args.name}", "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": internet, "dataset_sources": [f"{USER}/{args.dataset}"],
        }))
        subprocess.run(["kaggle", "kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4"], check=True)


if __name__ == "__main__":
    main()
