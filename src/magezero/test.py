from collections import Counter

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import H5Graphs, collate_batch, BATCH_SIZE
from model import (Graphs, Targets, ActionType, NodeType, load_checkpoint, losses, accumulate, averages,
                   decision_states, AMP, DEVICE)
from vocab import feature_id

SHOW_CONFUSION_MATRIX = True

PASS_ID = feature_id("PassAbility")
NAME_LABEL = feature_id("name")   # edge label from a typed node to its name leaf


def priority_pairs(priority, g: Graphs, t: Targets, name_label: int):
    """(MCTS's action, network's action) for each priority state with visits, both keyed by ability
    name so identical abilities count as one action (like the flat encoder's slots). MCTS's action is
    the name whose candidates got the most visits; the network's is the name of its top-scored ABILITY
    node, legal or not. Names are leaf vocab rows, -1 when a node has no name edge or its name is
    outside the vocab. All inputs are numpy (CPU) arrays."""
    named = g.edge_label == name_label
    name = np.full(len(g.node_type), -1, dtype=np.int64)
    name[g.edge_parent[named]] = g.node_row[g.edge_child[named]]

    pairs = []
    for b in range(len(g.node_offsets) - 1):
        if t.action_type[b] != ActionType.PRIORITY.value:
            continue
        visits_by_name = {}
        for node, visits in zip(t.cand_node[t.cand_graph == b], t.cand_visits[t.cand_graph == b]):
            visits_by_name[int(name[node])] = visits_by_name.get(int(name[node]), 0) + visits
        if sum(visits_by_name.values()) == 0:
            continue
        nodes = np.arange(g.node_offsets[b], g.node_offsets[b + 1])
        abilities = nodes[g.node_type[nodes] == NodeType.ABILITY]
        predicted = int(name[abilities[np.argmax(priority[abilities])]])
        pairs.append((max(visits_by_name, key=visits_by_name.get), predicted))
    return pairs


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
    MCTS's most visited; for priority, compared by ability name. Returns the summed per-head losses."""
    row = int(edge_vocab.lookup([NAME_LABEL])[0])
    name_label = row + 1 if row >= 0 else -1   # edge rows are vocab row + 1
    stats = {}
    pairs = []
    model.eval()
    with torch.no_grad():
        for graphs, targets in dl:
            g, t = graphs.to(DEVICE), targets.to(DEVICE)
            with torch.amp.autocast(DEVICE.type, enabled=AMP):
                out = model(g)
                accumulate(stats, losses(out, g, t))
            pairs += priority_pairs(out[0].float().cpu().numpy(), Graphs(*(np.asarray(x) for x in graphs)),
                                    Targets(*(np.asarray(x) for x in targets)), name_label)

    avg = averages(stats)
    avg_combined_loss = sum(avg.values())
    print(f"Validation loss:  priority_loss={avg['priority']:.3f}  choose_target_loss={avg['choose_target']:.3f}  "
          f"choose_use_loss={avg['choose_use']:.3f}  value_loss={avg['value']:.3f}  "
          f"avg_total_loss={avg_combined_loss:.3f}  decision_states={decision_states(stats)}")

    pass_row = int(vocab.lookup([PASS_ID])[0])
    non_pass = [(true, predicted) for true, predicted in pairs if true != pass_row]
    if pairs:
        print(f"Test priority_accuracy={sum(true == predicted for true, predicted in pairs) / len(pairs):.3f}  "
              f"non-pass={sum(true == predicted for true, predicted in non_pass) / max(len(non_pass), 1):.3f} "
              f"({len(non_pass)} of {len(pairs)} states)")
        if SHOW_CONFUSION_MATRIX:
            print_matrix(pairs, vocab)
    else:
        print("No priority samples in test set to calculate accuracy.")
    for name in ("choose_target", "choose_use"):
        _, n, correct = stats.get(name, (0.0, 0, 0))
        if n > 0:
            print(f"Test {name}_accuracy={correct / n:.3f}")
        else:
            print(f"No {name} samples in test set to calculate accuracy.")

    return avg_combined_loss

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--deck", required=True)
    parser.add_argument("--version", type=int, required=True)
    args = parser.parse_args()

    checkpoint_path = f"models/{args.deck}/ver{args.version}/model.pt.gz"
    model, vocab, edge_vocab = load_checkpoint(checkpoint_path)
    print(f"Loaded checkpoint from {checkpoint_path}: {len(vocab)} leaf rows, {len(edge_vocab)} edge label rows")

    ds = H5Graphs(f"data/{args.deck}/ver{args.version}/testing", vocab, edge_vocab)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                    collate_fn=collate_batch, pin_memory=AMP, persistent_workers=False)

    validate(model.to(DEVICE), dl, vocab, edge_vocab)
