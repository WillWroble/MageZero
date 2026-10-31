"""
Keeps an eye on the RunPod experiment pods: every --every minutes, reaps the pods that are done (launch.py --reap)
and checks the rest. A pod is STALLED when nothing in its run directory changed for --stale minutes (generation pods
update their logs every minute, training pods every few hundred steps) or when both its GPU and CPU sit idle; the
volume is STALLED when the files pod's own progress file stops changing. Each check prints one line per pod;
problems start with ALERT. --kill-stalled deletes stalled pods.
  python runpod/watchdog.py [--every 10] [--stale 20] [--kill-stalled]
"""
import argparse
import email.utils
import json
import os
import re
import time
import urllib.request
from datetime import datetime, timezone

import launch


def get(url, method="GET"):
    req = urllib.request.Request(url, method=method, headers={"User-Agent": "magezero"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.read().decode(), r.headers
    except Exception:
        return None, None


def newest_change(base, name):
    """Latest Last-Modified among the files of a pod's run directory, or None."""
    listing, _ = get(f"{base}/{name}/")
    if not listing:
        return None
    times = []
    for f in set(re.findall(r'href="([^"/]+)"', listing)):
        _, headers = get(f"{base}/{name}/{f}", "HEAD")
        if headers and headers.get("Last-Modified"):
            times.append(email.utils.parsedate_to_datetime(headers["Last-Modified"]))
    return max(times, default=None)


def runtimes():
    """pod name -> runtime (uptime, GPU and CPU use) from RunPod's GraphQL API."""
    q = '{"query":"query { myself { pods { name runtime { uptimeInSeconds gpus { gpuUtilPercent } container { cpuPercent } } } } }"}'
    req = urllib.request.Request(f"https://api.runpod.io/graphql?api_key={os.environ['RUNPOD_API_KEY']}", data=q.encode(),
                                 headers={"Content-Type": "application/json", "User-Agent": "magezero"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return {p["name"]: p["runtime"] or {} for p in json.loads(r.read())["data"]["myself"]["pods"]}


def check(args):
    now = datetime.now(timezone.utc)
    pods = launch.api("GET", "pods")
    files = next((p for p in pods if p["name"] == "mz-files"), None)
    if files is None:
        print(f"{now:%H:%M} ALERT no files pod")
        return
    base = f"https://{files['id']}-8000.proxy.runpod.net"
    _, headers = get(f"{base}/lfs-progress.txt", "HEAD")
    volume = email.utils.parsedate_to_datetime(headers["Last-Modified"]) if headers else None
    if volume is None or (now - volume).total_seconds() > 300:
        print(f"{now:%H:%M} ALERT volume: the files pod's progress file last changed {volume}")
    run = runtimes()
    for p in pods:
        if not p["name"].startswith("mz-") or p["name"] == "mz-files":
            continue
        name, rt = p["name"][3:], run.get(p["name"], {})
        uptime = rt.get("uptimeInSeconds") or 0
        gpu = max((g.get("gpuUtilPercent") or 0 for g in rt.get("gpus") or []), default=0)
        cpu = (rt.get("container") or {}).get("cpuPercent") or 0
        changed = newest_change(base, name)
        idle_min = (now - changed).total_seconds() / 60 if changed else None
        stalled = uptime > 20 * 60 and ((idle_min is not None and idle_min > args.stale) or (gpu == 0 and cpu < 2))
        print(f"{now:%H:%M} {'ALERT stalled' if stalled else 'ok'} {p['name']} up {uptime // 60} min, gpu {gpu}%, "
              f"cpu {cpu}%, last write {f'{idle_min:.0f} min ago' if idle_min is not None else 'none'}", flush=True)
        if stalled and args.kill_stalled:
            launch.api("DELETE", f"pods/{p['id']}")
            print(f"{now:%H:%M} deleted {p['name']}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--every", type=float, default=10, help="minutes between checks")
    parser.add_argument("--stale", type=float, default=30, help="minutes without a write before a pod counts as stalled")
    parser.add_argument("--kill-stalled", action="store_true")
    args = parser.parse_args()
    while True:
        os.system("python3 runpod/launch.py --reap")
        check(args)
        if not any(p["name"].startswith("mz-") and p["name"] != "mz-files" for p in launch.api("GET", "pods")):
            print("no experiment pods left")
            return
        time.sleep(args.every * 60)


if __name__ == "__main__":
    main()
