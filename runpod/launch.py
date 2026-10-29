"""
Run experiments on a RunPod GPU pod. The pod clones this repo at the current pushed commit (code only), reads the
data sets that files.sh builds on the network volume (/workspace/data), runs the runs `--concurrent` at a time on
its one GPU, then writes DONE there (`--reap` deletes pods that are done). Each run trains, then tests on each val set of its split.txt; its model directory (with
train.log and test.log) is /workspace/runs/<name>/<run>, and its console output /workspace/runs/<name>/<run>.log.

  python runpod/launch.py --name s1 --runs '[{"name": "mlp", "split": "holdout 5\\ntrain:\\nexperiment_data/.*/gen0/.*",
      "arch": {"net": "mlp"}, "args": "--epochs 6 --lr 1e-3"}]'
  python runpod/launch.py --files      # (re)start the files pod: fetches data, builds data sets, serves /workspace/runs
  python runpod/launch.py --reap       # delete the experiment pods whose runs are all done
  python runpod/launch.py --name selfplay-monob --script runpod/generate.sh --env MAGE_COMMIT=<sha> CONFIG=<yml> MODEL=<dir>
      # any repo script, run from the repo root with $OUT=/workspace/runs/<name>; its output goes to $OUT/script.log

Logs: https://<files pod id>-8000.proxy.runpod.net/<name>/<run>.log (the files pod's id: python runpod/launch.py --list)
Run keys: name, split (the model's split.txt, dataset.read_split; regexes match paths from the repo root), arch (JSON
for --arch), args (more train.py args); model (an existing model directory under /workspace/runs) skips training and
only tests it on its split.txt.
"""
import argparse
import email.utils
import json
import os
import shlex
import subprocess
import urllib.request
from datetime import datetime
from pathlib import Path

API = "https://rest.runpod.io/v1"
VOLUME = "0ksqf5sep2"   # magezero-data, EU-RO-1
DATA_CENTER = "EU-RO-1"
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
REPO = "https://github.com/WillWroble/MageZero.git"
REPO_DIR = "/workspace/repo"  # the files pod's checkout (with LFS data): split.txt regexes match paths from its root
HERE = Path(__file__).resolve().parent

POD = '''set -x
apt-get update -qq && apt-get install -y -qq git >/dev/null; pip install -q h5py
OUT=/workspace/runs/{name}; mkdir -p $OUT; rm -f $OUT/DONE
GIT_LFS_SKIP_SMUDGE=1 git clone -q {repo} /tmp/code && cd /tmp/code && git checkout -q {commit}
nvidia-smi > $OUT/gpu.txt
{setup}
export OMP_NUM_THREADS=$(( $(nproc) / {concurrent} ))  # concurrent runs otherwise oversubscribe the CPU
cat > /tmp/runs.txt <<'RUNS'
{commands}
RUNS
timeout {hours}h xargs -P {concurrent} -d '\\n' -I CMD bash -c CMD < /tmp/runs.txt
echo done > $OUT/DONE
sleep infinity
'''

SCRIPT_POD = '''set -x
apt-get update -qq && apt-get install -y -qq git >/dev/null
export OUT=/workspace/runs/{name}; mkdir -p $OUT; rm -f $OUT/DONE
GIT_LFS_SKIP_SMUDGE=1 git clone -q {repo} /tmp/code && cd /tmp/code && git checkout -q {commit}
{env} timeout {hours}h bash {script} > $OUT/script.log 2>&1
echo $? > $OUT/DONE
sleep infinity
'''


def command(name, run):
    log = f"/workspace/runs/{name}/{run['name']}.log"
    out = f"/workspace/runs/{run['model']}" if "model" in run else log[:-len(".log")]
    steps = [f"cd {REPO_DIR}", f"python -u /tmp/code/src/magezero/test.py --model {out}"]
    if "model" not in run:
        arch = f"--arch {shlex.quote(json.dumps(run['arch']))}" if "arch" in run else ""
        lines = " ".join(shlex.quote(line) for line in run["split"].splitlines())
        steps[1:1] = [f"mkdir -p {out}", f"printf '%s\\n' {lines} > {out}/split.txt",
                      f"python -u /tmp/code/src/magezero/train.py --model {out} {arch} {run.get('args', '')}"]
    return f"({' && '.join(steps)}) > {log} 2>&1"


def api(method, path, body=None):
    req = urllib.request.Request(f"{API}/{path}", method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
                                          "Content-Type": "application/json", "User-Agent": "magezero"})
    try:
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read() or "null")
    except urllib.error.HTTPError as e:
        raise SystemExit(f"{method} {path}: {e.code} {e.read().decode()}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="experiment name: its directory under /workspace/runs")
    parser.add_argument("--runs", type=json.loads)
    parser.add_argument("--gpu", nargs="+", default=["NVIDIA RTX PRO 6000 Blackwell Server Edition"],
                        help="acceptable GPU types, in order of preference")
    parser.add_argument("--concurrent", type=int, default=4, help="runs at once on the GPU")
    parser.add_argument("--hours", type=float, default=10, help="time limit for all runs")
    parser.add_argument("--setup", default="", help="shell command run on the pod before the runs")
    parser.add_argument("--script", help="run this repo script instead of training runs")
    parser.add_argument("--env", nargs="*", default=[], help="KEY=VALUE pairs for --script")
    parser.add_argument("--cpu", type=int, help="a CPU pod with this many vCPUs instead of a GPU pod")
    parser.add_argument("--files", action="store_true", help="(re)start the files pod")
    parser.add_argument("--list", action="store_true", help="list pods")
    parser.add_argument("--reap", action="store_true", help="delete experiment pods that wrote DONE")
    args = parser.parse_args()

    if args.list:
        for p in api("GET", "pods"):
            print(p["id"], p["name"], p["desiredStatus"], p.get("costPerHr"), p.get("gpu", {}).get("displayName"))
        return
    if args.reap:
        pods = api("GET", "pods")
        files = next((p["id"] for p in pods if p["name"] == "mz-files"), None)
        if files is None:
            return
        for p in pods:
            if p["name"].startswith("mz-") and p["name"] != "mz-files":
                done = f"https://{files}-8000.proxy.runpod.net/{p['name'][3:]}/DONE"
                try:
                    with urllib.request.urlopen(urllib.request.Request(done, headers={"User-Agent": "magezero"})) as r:
                        written = email.utils.parsedate_to_datetime(r.headers["Last-Modified"])
                except urllib.error.HTTPError:
                    continue
                started = datetime.fromisoformat(p["lastStartedAt"].replace(" +0000 UTC", "+00:00"))
                if written < started:  # left by an earlier pod of the same name
                    continue
                api("DELETE", f"pods/{p['id']}")
                print("deleted", p["name"])
        return
    if args.files:
        for p in api("GET", "pods"):
            if p["name"] == "mz-files":
                api("DELETE", f"pods/{p['id']}")
        body = {"name": "mz-files", "computeType": "CPU", "cpuFlavorIds": ["cpu3c"], "vcpuCount": 2,
                "imageName": "ubuntu:22.04", "ports": ["8000/http"], "containerDiskInGb": 10,
                "dockerStartCmd": ["bash", "-c", (HERE / "files.sh").read_text()]}
    else:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        if subprocess.run(["git", "branch", "-r", "--contains", commit], capture_output=True, text=True).stdout == "":
            raise SystemExit(f"push {commit} first: the pod clones it from GitHub")
        if args.script:
            script = SCRIPT_POD.format(name=args.name, repo=REPO, commit=commit, hours=args.hours, script=args.script,
                                       env=" ".join(k + "=" + shlex.quote(v) for k, v in (e.split("=", 1) for e in args.env)))
        else:
            commands = "\n".join(command(args.name, run) for run in args.runs)
            script = POD.format(name=args.name, repo=REPO, commit=commit, commands=commands, hours=args.hours,
                                concurrent=args.concurrent, setup=args.setup)
        if args.cpu:
            body = {"name": f"mz-{args.name}", "computeType": "CPU", "cpuFlavorIds": ["cpu5c", "cpu3c"],
                    "vcpuCount": args.cpu, "imageName": "ubuntu:22.04", "containerDiskInGb": 30,
                    "dockerStartCmd": ["bash", "-c", script]}
        else:
            body = {"name": f"mz-{args.name}", "computeType": "GPU", "gpuTypeIds": args.gpu, "gpuCount": 1,
                    "gpuTypePriority": "custom", "cloudType": "SECURE", "imageName": IMAGE, "containerDiskInGb": 60,
                    "dockerStartCmd": ["bash", "-c", script]}
    body.update(networkVolumeId=VOLUME, volumeMountPath="/workspace", dataCenterIds=[DATA_CENTER])
    pod = api("POST", "pods", body)
    print(pod["id"], pod.get("costPerHr"), pod.get("machine", {}).get("gpuTypeId"))


if __name__ == "__main__":
    main()
