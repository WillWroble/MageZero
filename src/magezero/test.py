import sys
from collections import Counter
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import H5Graphs, collate_batch, read_split, BATCH_SIZE
from model import (Graphs, Targets, ActionType, NodeType, load_checkpoint, load_model, losses, accumulate, averages,
                   decision_states, graph_index, segment_logsumexp, PRIORITY_TYPES, TARGET_TYPES, AMP, DEVICE)
from vocab import feature_id

SHOW_CONFUSION_MATRIX = True

PASS_ID = feature_id("PassAbility")
NAME_LABEL = feature_id("name")   # edge label from a typed node to its name leaf


class Tee:
    """A stream that writes to several streams."""

    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for stream in self.streams:
            stream.write(s)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def log_to(path: str) -> None:
    """Copies everything printed from here on (stdout and stderr) to the end of `path`, after a header line."""
    f = open(path, "a", buffering=1)
    f.write(f"\n=== {datetime.now().isoformat(timespec='seconds')}  {' '.join(sys.argv)} ===\n")
    sys.stdout, sys.stderr = Tee(sys.stdout, f), Tee(sys.stderr, f)


def priority_pairs(priority, g: Graphs, t: Targets, name_label: int):
    """(MCTS's action, network's action, network's legal action) for each priority state with visits, all keyed by
    ability name so identical abilities count as one action (like the flat encoder's slots). MCTS's action is
    the name whose candidates got the most visits; the network's is the name of its top-scored ABILITY
    node, legal or not; its legal action the name of its top-scored candidate. Names are leaf vocab rows, -1 when a node has no name edge or its name is
    outside the vocab. All inputs are numpy (CPU) arrays."""
    named = g.edge_label == name_label
    name = np.full(len(g.node_type), -1, dtype=np.int64)
    name[g.edge_parent[named]] = g.node_row[g.edge_child[named]]

    pairs = []
    for b in range(len(g.node_offsets) - 1):
        if t.action_type[b] != ActionType.PRIORITY.value:
            continue
        visits_by_name = {}
        cands = t.cand_node[t.cand_graph == b]
        for node, visits in zip(cands, t.cand_visits[t.cand_graph == b]):
            visits_by_name[int(name[node])] = visits_by_name.get(int(name[node]), 0) + visits
        if sum(visits_by_name.values()) == 0:
            continue
        nodes = np.arange(g.node_offsets[b], g.node_offsets[b + 1])
        abilities = nodes[g.node_type[nodes] == NodeType.ABILITY]
        predicted = int(name[abilities[np.argmax(priority[abilities])]])
        pairs.append((max(visits_by_name, key=visits_by_name.get), predicted, int(name[cands[np.argmax(priority[cands])]])))
    return pairs


def legal_stats(scores, g: Graphs, t: Targets, action: ActionType, types) -> torch.Tensor:
    """[states, legal hits, legal mass, legal KL] for one policy head over the states of its decision type that
    have visits. Legal hit: the top-scored candidate (MCTS's root children, i.e. the legal actions) is one
    of the most visited. Legal mass: the probability the head's softmax (over every node of its types, as
    in training) puts on the candidates. Legal KL: from the normalized visits to the softmax over the candidates
    only, comparable between models trained with either policy domain."""
    B = t.value.shape[0]
    node_graph = graph_index(g.node_offsets)
    visits = torch.zeros(B, device=scores.device).index_add(0, t.cand_graph, t.cand_visits)
    states = (t.action_type == action.value) & (visits > 0)
    domain = torch.isin(g.node_type, torch.tensor(types, device=scores.device))
    domain[t.cand_node] = True
    idx = (domain & states[node_graph]).nonzero().squeeze(1)
    lse = segment_logsumexp(scores[idx].float(), node_graph[idx], B)
    keep = states[t.cand_graph]
    s, v, b = scores[t.cand_node[keep]].float(), t.cand_visits[keep], t.cand_graph[keep]
    best_s = torch.full((B,), float("-inf"), device=s.device).scatter_reduce(0, b, s, "amax")
    best_v = torch.zeros(B, device=s.device).scatter_reduce(0, b, v, "amax")
    hit = torch.zeros(B, device=s.device).index_add(0, b, ((s == best_s[b]) & (v == best_v[b])).float()) > 0
    mass = torch.zeros(B, device=s.device).index_add(0, b, (s - lse[b]).exp())
    p = v / torch.zeros(B, device=s.device).index_add(0, b, v)[b]
    kl = (torch.xlogy(p, p) - p * (s - segment_logsumexp(s, b, B)[b])).sum()
    return torch.stack([states.sum(), hit[states].sum(), mass[states].sum(), kl]).float().cpu()


def print_matrix(pairs, vocab):
    """Priority confusion matrix (True \\ Predicted) over ability names, most frequent true action
    first, with a legend from row index to the name's feature id (look it up in FeatureTable.txt)."""
    true_counts = Counter(true for true, _ in pairs)
    names = sorted(true_counts, key=true_counts.get, reverse=True)
    for _, predicted in pairs:
        if predicted not in names:
            names.append(predicted)
    index = {n: i for i, n in enumerate(names)}
    matrix = np.zeros((len(names), len(names)), dtype=np.int64)
    for true, predicted in pairs:
        matrix[index[true], index[predicted]] += 1

    print("--- Priority Confusion Matrix (True \\ Predicted) ---")
    header = "True |" + "".join(f"{j: >5}" for j in range(len(names)))
    print(header)
    print("-" * len(header))
    for r in range(len(names)):
        print(f"{r: >4} |" + "".join(f"{matrix[r, c]: >5}" for c in range(len(names))))
    print("-" * len(header))
    for i, n in enumerate(names):
        fid = int(vocab.ids[n]) if n >= 0 else None
        label = "Pass" if fid == PASS_ID else (f"idx={fid}" if fid is not None else "no name edge / name outside vocab")
        print(f"{i: >4} = {label}")
    print("-" * 60)


def validate(model, dl, vocab, edge_vocab):
    """Prints per-head validation loss and accuracy, and the priority confusion matrix. Accuracy:
    the network's top-scored node among all nodes of the head's types (no legality mask) is one of
    MCTS's most visited; for priority, compared by ability name. Legal accuracy and mass: see
    legal_stats. Returns the summed per-head losses."""
    row = int(edge_vocab.lookup([NAME_LABEL])[0])
    name_label = row + 1 if row >= 0 else -1   # edge rows are vocab row + 1
    stats = {}
    pairs = []
    legal = {name: torch.zeros(4) for name in ("priority", "choose_target")}
    legal_only = model.arch.get("policy_domain") == "legal"
    model.eval()
    with torch.no_grad():
        for graphs, targets in dl:
            g, t = graphs.to(DEVICE), targets.to(DEVICE)
            with torch.amp.autocast(DEVICE.type, enabled=AMP):
                out = model(g)
                accumulate(stats, losses(out, g, t, legal_only))
            legal["priority"] += legal_stats(out[0], g, t, ActionType.PRIORITY, PRIORITY_TYPES)
            legal["choose_target"] += legal_stats(out[1], g, t, ActionType.CHOOSE_TARGET, TARGET_TYPES)
            pairs += priority_pairs(out[0].float().cpu().numpy(), Graphs(*(np.asarray(x) for x in graphs)),
                                    Targets(*(np.asarray(x) for x in targets)), name_label)

    avg = averages(stats)
    avg_combined_loss = sum(avg.values())
    print(f"Validation loss:  priority_loss={avg['priority']:.3f}  choose_target_loss={avg['choose_target']:.3f}  "
          f"choose_use_loss={avg['choose_use']:.3f}  value_loss={avg['value']:.3f}  "
          f"avg_total_loss={avg_combined_loss:.3f}  decision_states={decision_states(stats)}")

    pass_row = int(vocab.lookup([PASS_ID])[0])
    non_pass = [pair for pair in pairs if pair[0] != pass_row]
    if pairs:
        def accuracy(group, k):
            return sum(pair[0] == pair[k] for pair in group) / max(len(group), 1)
        passes = [pair for pair in pairs if pair[0] == pass_row]
        print(f"Test priority_accuracy={accuracy(pairs, 1):.3f}  non-pass={accuracy(non_pass, 1):.3f} "
              f"({len(non_pass)} of {len(pairs)} states)")
        print(f"Test priority legal: accuracy={accuracy(pairs, 2):.3f}  non-pass={accuracy(non_pass, 2):.3f}  "
              f"pass={accuracy(passes, 2):.3f}")
        if SHOW_CONFUSION_MATRIX:
            print_matrix([pair[:2] for pair in pairs], vocab)
    else:
        print("No priority samples in test set to calculate accuracy.")
    for name in ("choose_target", "choose_use"):
        _, n, correct = stats.get(name, (0.0, 0, 0))
        if n > 0:
            print(f"Test {name}_accuracy={correct / n:.3f}")
        else:
            print(f"No {name} samples in test set to calculate accuracy.")
    for name, (n, hits, mass, kl) in legal.items():
        if n > 0:
            print(f"Test {name}_legal_accuracy={hits / n:.3f}  {name}_legal_mass={mass / n:.3f}  "
                  f"{name}_legal_kl={kl / n:.3f}")

    return avg_combined_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True,
                        help="model directory (model.pt.gz): tests on each val set of its split.txt; the log goes to its test.log")
    parser.add_argument("--td", type=float, help="test against value labels rebuilt with this TD discount "
                                                 "(dataset.td_labels; 1 = the game result) instead of the stored ones")
    args = parser.parse_args()
    log_to(f"{args.model}/test.log")

    checkpoint_path = f"{args.model}/model.pt.gz"
    model, vocab, edge_vocab = load_checkpoint(checkpoint_path)
    hide_opponent_hand = load_model(checkpoint_path).get("hide_opponent_hand", False)
    print(f"Loaded checkpoint from {checkpoint_path}: {len(vocab)} leaf rows, {len(edge_vocab)} edge label rows, "
          f"hide_opponent_hand={hide_opponent_hand}")

    model = model.to(DEVICE)
    for name, shards in read_split(f"{args.model}/split.txt")[1].items():
        print(f"Testing on {name}" + ("" if args.td is None else f" (TD {args.td})"))
        ds = H5Graphs(shards, vocab, edge_vocab, hide_opponent_hand, args.td)
        dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                        collate_fn=collate_batch, pin_memory=AMP, persistent_workers=False)
        validate(model, dl, vocab, edge_vocab)
