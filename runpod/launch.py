"""
Run experiments on a RunPod GPU pod. The pod clones this repo at the current pushed commit (code only), reads the
data sets that files.sh builds on the network volume (/workspace/data), runs the runs `--concurrent` at a time on
its one GPU, then deletes itself. Each run trains, then tests on each of its test sets; its model directory (with
train.log and test.log) is /workspace/runs/<name>/<run>, and its console output /workspace/runs/<name>/<run>.log.

  python runpod/launch.py --name s1 --runs '[{"name": "mlp", "train": "gen0/train", "tests": ["gen0/test"],
      "arch": {"net": "mlp"}, "args": "--epochs 6 --lr 1e-3"}]'
  python runpod/launch.py --files      # (re)start the files pod: fetches data, builds data sets, serves /workspace/runs

Logs: https://<files pod id>-8000.proxy.runpod.net/<name>/<run>.log (the files pod's id: python runpod/launch.py --list)
Run keys: name, train and tests (data sets under /workspace/data), arch (JSON for --arch), args (more train.py args).
"""
import argparse
import json
import os
import shlex
import subprocess
import urllib.request
from pathlib import Path

API = "https://rest.runpod.io/v1"
VOLUME = "0ksqf5sep2"   # magezero-data, EU-RO-1
DATA_CENTER = "EU-RO-1"
IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
REPO = "https://github.com/WillWroble/MageZero.git"
HERE = Path(__file__).resolve().parent

POD = '''set -x
apt-get update -qq && apt-get install -y -qq git >/dev/null; pip install -q h5py
OUT=/workspace/runs/{name}; mkdir -p $OUT
GIT_LFS_SKIP_SMUDGE=1 git clone -q {repo} /tmp/code && cd /tmp/code && git checkout -q {commit}
cd /tmp/code/src/magezero && nvidia-smi > $OUT/gpu.txt
export OMP_NUM_THREADS=$(( $(nproc) / {concurrent} ))  # concurrent runs otherwise oversubscribe the CPU
cat > /tmp/runs.txt <<'RUNS'
{commands}
RUNS
timeout {hours}h xargs -P {concurrent} -d '\\n' -I CMD bash -c CMD < /tmp/runs.txt
echo done > $OUT/DONE
python -c "import os, urllib.request as u; u.urlopen(u.Request('{api}/pods/' + os.environ['RUNPOD_POD_ID'], \
method='DELETE', headers={{'Authorization': 'Bearer ' + os.environ['RUNPOD_API_KEY'], 'User-Agent': 'magezero'}}))"
sleep infinity
'''


def command(name, run):
    out, data = f"/workspace/runs/{name}/{run['name']}", "/workspace/data"
    arch = f"--arch {shlex.quote(json.dumps(run['arch']))}" if "arch" in run else ""
    tests = " && ".join(f"python -u test.py --data {data}/{t} --model {out}" for t in run["tests"])
    return (f"(python -u train.py --train {data}/{run['train']} --test {data}/{run['tests'][0]} --out {out} {arch} "
            f"{run.get('args', '')} && {tests}) > {out}.log 2>&1")


def api(method, path, body=None):
    req = urllib.request.Request(f"{API}/{path}", method=method, data=json.dumps(body).encode() if body else None,
                                 headers={"Authorization": f"Bearer {os.environ['RUNPOD_API_KEY']}",
                                          "Content-Type": "application/json", "User-Agent": "magezero"})
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read() or "null")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", help="experiment name: its directory under /workspace/runs")
    parser.add_argument("--runs", type=json.loads)
    parser.add_argument("--gpu", nargs="+", default=["NVIDIA RTX PRO 6000 Blackwell Server Edition"],
                        help="acceptable GPU types, in order of preference")
    parser.add_argument("--concurrent", type=int, default=4, help="runs at once on the GPU")
    parser.add_argument("--hours", type=float, default=10, help="time limit for all runs")
    parser.add_argument("--files", action="store_true", help="(re)start the files pod")
    parser.add_argument("--list", action="store_true", help="list pods")
    args = parser.parse_args()

    if args.list:
        for p in api("GET", "pods"):
            print(p["id"], p["name"], p["desiredStatus"], p.get("costPerHr"), p.get("gpu", {}).get("displayName"))
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
        commands = "\n".join(command(args.name, run) for run in args.runs)
        script = POD.format(name=args.name, repo=REPO, commit=commit, commands=commands, hours=args.hours,
                            concurrent=args.concurrent, api=API)
        body = {"name": f"mz-{args.name}", "computeType": "GPU", "gpuTypeIds": args.gpu, "gpuCount": 1,
                "gpuTypePriority": "custom", "cloudType": "SECURE", "imageName": IMAGE, "containerDiskInGb": 20,
                "dockerStartCmd": ["bash", "-c", script]}
    body.update(networkVolumeId=VOLUME, volumeMountPath="/workspace", dataCenterIds=[DATA_CENTER])
    pod = api("POST", "pods", body)
    print(pod["id"], pod.get("costPerHr"), pod.get("machine", {}).get("gpuTypeId"))


if __name__ == "__main__":
    main()
