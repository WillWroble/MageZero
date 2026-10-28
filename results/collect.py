"""Per-test-set metrics from RunPod run logs (runpod/launch.py): the test.py runs after training, one row each.
  python results/collect.py <files pod id> <experiment> <run> [<run> ...]"""
import re
import sys
import urllib.request

COLUMNS = {
    "value": r"value_loss=([\d.]+)",
    "prio KL": r"priority_legal_kl=([\d.]+)",
    "prio acc": r"priority legal: accuracy=([\d.]+)",
    "non-pass": r"priority legal: accuracy=[\d.]+  non-pass=([\d.]+)",
    "pass": r"  pass=([\d.]+)",
    "tgt KL": r"choose_target_legal_kl=([\d.]+)",
    "tgt acc": r"choose_target_legal_accuracy=([\d.]+)",
    "use acc": r"choose_use_accuracy=([\d.]+)",
    "states": r"decision_states=(\d+)",
}


def rows(text):
    """(test set, {column: value}) for each test.py section of a run log."""
    for section in text.split("Testing on ")[1:]:
        name = "/".join(section.split()[0].split("/")[-2:])
        yield name, {c: (m.group(1) if (m := re.search(p, section)) else "") for c, p in COLUMNS.items()}


def main():
    pod, experiment, runs = sys.argv[1], sys.argv[2], sys.argv[3:]
    print("| run | test | " + " | ".join(COLUMNS) + " |")
    print("|---|---|" + "---|" * len(COLUMNS))
    for run in runs:
        url = f"https://{pod}-8000.proxy.runpod.net/{experiment}/{run}.log"
        text = urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "magezero"})).read().decode()
        for test, values in rows(text):
            print(f"| {run} | {test} | " + " | ".join(values.values()) + " |")


if __name__ == "__main__":
    main()
