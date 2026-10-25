"""
Push an experiment to Kaggle as a GPU script kernel (T4 x2). The kernel unpacks src/magezero, mounts the
dataset as data/<run>/ver<version>, and runs the training runs alternately on the two GPUs (one after another
on each, or all at once with --parallel), each followed by bench.py. Logs land in the kernel's output as <run>.log.

  python kaggle/launch.py --name passes --runs '[{"name": "seq1", "args": "--epochs 6", "arch": {"n_passes": 1}}]'
  kaggle kernels status willwroble/mz-passes
  kaggle kernels output willwroble/mz-passes -p results/kaggle/passes

Optional run keys: "ref" trains with src/magezero from a git ref instead of the working tree; "data" picks a
subdirectory of the dataset (the one holding training/ and testing/); "setup" is a shell command run first
(see --internet for pip); "post" is a shell command run after training, with $SRC (the code) and $DECK
(train.py's --deck/--version) set.
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
PARALLEL = {parallel}

for ref, code in CODE.items():
    tarfile.open(fileobj=io.BytesIO(base64.b64decode(code))).extractall(f"/tmp/mz/{{ref}}", filter="data")
os.chdir("/kaggle/working")
def command(run):
    Path(f"data/{{run['name']}}").mkdir(parents=True, exist_ok=True)
    data = glob.glob(f"/kaggle/input/**/{{run.get('data', '')}}/training".replace("//", "/"), recursive=True)[0]
    os.symlink(Path(data).parent, f"data/{{run['name']}}/ver{{VERSION}}")
    src = f"/tmp/mz/{{run.get('ref', 'HEAD')}}"
    deck = f"--deck {{run['name']}} --version {{VERSION}}"
    arch = f"--arch {{shlex.quote(json.dumps(run['arch']))}}" if "arch" in run else ""
    post = f"(SRC={{src}}; DECK={{shlex.quote(deck)}}; {{run['post']}})" if "post" in run else "true"
    return (f"{{run.get('setup', 'true')}} && python -u {{src}}/train.py {{deck}} {{arch}} {{run['args']}} && {{post}} && "
            f"(test ! -f {{src}}/bench.py || python -u {{src}}/bench.py {{deck}}); "
            f"rm -rf models/{{run['name']}}/ver{{VERSION}}/best.pt.gz")


# runs alternate between the two GPUs; each GPU runs its share one after another, or all at once with PARALLEL
chains = [[run] for run in RUNS] if PARALLEL else [RUNS[0::2], RUNS[1::2]]
procs = []
for i, chain in enumerate(chains):
    if not chain:
        continue
    previous = Path(f"{{chains[i - 1][0]['name']}}.log")
    # one run loads the dataset at a time: loading peaks at about twice the resident size
    while procs and procs[-1].poll() is None and not (previous.exists() and "arch:" in previous.read_text()):
        time.sleep(10)
    cmd = "; ".join(f"({{command(run)}}) > {{run['name']}}.log 2>&1" for run in chain)
    procs.append(subprocess.Popen(cmd, shell=True, env={{**os.environ, "CUDA_VISIBLE_DEVICES": str(i % 2)}}))
for p in procs:
    p.wait()
for run in RUNS:
    print(open(f"{{run['name']}}.log").read())
'''


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="kernel name: mz-<name>")
    parser.add_argument("--runs", type=json.loads, required=True,
                        help='[{"name": ..., "args": "train.py args", "arch": {NetGraph options}}], alternating between 2 GPUs')
    parser.add_argument("--version", type=int, default=4)
    parser.add_argument("--dataset", default="magezero-pauper-faeries-ver4")
    parser.add_argument("--sources", nargs="*", default=[], help="earlier kernels whose output to mount (mz-<name>)")
    parser.add_argument("--internet", action="store_true", help="e.g. for pip in a run's setup")
    parser.add_argument("--timeout", type=int, help="stop the kernel after this many seconds")
    parser.add_argument("--parallel", action="store_true", help="start every run at once (several per GPU)")
    args = parser.parse_args()

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
                                 cwd=SRC.parents[1], capture_output=True, check=True).stdout
            code[ref] = base64.b64encode(tar).decode()

    with tempfile.TemporaryDirectory() as d:
        (Path(d) / "run.py").write_text(KERNEL.format(code=code, runs=args.runs, version=args.version, parallel=args.parallel))
        (Path(d) / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{USER}/mz-{args.name}", "title": f"mz-{args.name}", "code_file": "run.py",
            "language": "python", "kernel_type": "script", "is_private": True, "enable_gpu": True,
            "enable_internet": args.internet, "dataset_sources": [f"{USER}/{args.dataset}"],
            "kernel_sources": [f"{USER}/{k}" for k in args.sources],
        }))
        timeout = ["-t", str(args.timeout)] if args.timeout else []
        subprocess.run(["kaggle", "kernels", "push", "-p", d, "--accelerator", "NvidiaTeslaT4", *timeout], check=True)


if __name__ == "__main__":
    main()
