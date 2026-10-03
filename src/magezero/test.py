import torch
from torch.utils.data import DataLoader

from dataset import H5Graphs, collate_batch, BATCH_SIZE
from model import load_checkpoint, losses, accumulate, averages, decision_states, AMP, DEVICE, POLICY_HEADS


def validate(model, dl):
    """Prints per-head validation loss and, for the policy heads, how often the network's top
    candidate is MCTS's most visited one. Returns the summed per-head losses."""
    stats = {}
    model.eval()
    with torch.no_grad():
        for graphs, targets in dl:
            graphs, targets = graphs.to(DEVICE), targets.to(DEVICE)
            with torch.amp.autocast(DEVICE.type, enabled=AMP):
                accumulate(stats, losses(model(graphs), graphs, targets))

    avg = averages(stats)
    avg_combined_loss = sum(avg.values())
    print(f"Validation loss:  priority_loss={avg['priority']:.3f}  choose_target_loss={avg['choose_target']:.3f}  "
          f"choose_use_loss={avg['choose_use']:.3f}  value_loss={avg['value']:.3f}  "
          f"avg_total_loss={avg_combined_loss:.3f}  decision_states={decision_states(stats)}")

    for name in POLICY_HEADS:
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

    validate(model.to(DEVICE), dl)
