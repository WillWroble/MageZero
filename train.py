import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import LabeledStateDataset,collate_batch
ACTIONS_MAX = 128
GLOBAL_MAX = 100000


class Net(nn.Module):
    def __init__(self, num_embeddings, policy_size_A):
        super().__init__()

        embedding_dim = 512  # Output of EmbeddingBag, matches your original fc1 output
        hidden_dim_mlp = 256  # Output of the main MLP block, matches your original fc2 output

        self.embedding_bag = nn.EmbeddingBag(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            mode='sum',
            sparse=True
        )

        self.fc_after_embedding = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim_mlp),  # From 512 to 256
            nn.ReLU(),
        )

        self.policy_head = nn.Linear(hidden_dim_mlp, policy_size_A)  # From 256 to A
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim_mlp, 1),  # From 256 to 1
            nn.Tanh()
        )

    def forward(self, indices, offsets):
        embedded_sum = self.embedding_bag(indices, offsets)
        h = self.fc_after_embedding(embedded_sum)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


def train():
    ds = LabeledStateDataset("data/UWTempo2/ver1/training.bin")
    #ds.states = ds.states.mul(2.0).sub(1.0) #fix activations
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4, collate_fn=collate_batch)
    model = Net(GLOBAL_MAX, ACTIONS_MAX).cuda()

    sparse_params = model.embedding_bag.parameters()
    opt_sparse = optim.SparseAdam(sparse_params, lr=1e-3)

    # Optimizer for all other (dense) parameters
    dense_params = []
    for name, param in model.named_parameters():
        if "embedding_bag" not in name:  # Exclude embedding_bag parameters
            dense_params.append(param)
    opt_dense = optim.Adam(dense_params, lr=1e-3)

    ce    = nn.CrossEntropyLoss()
    mse   = nn.MSELoss()

    for epoch in range(1, 21):
        total_p_loss, total_v_loss = 0.0, 0.0  # Initialize as floats
        model.train()

        # 4. Update the loop to unpack all parts returned by collate_batch
        for batch_indices, batch_offsets, batch_policy_labels, batch_value_labels in dl:
            # 5. Move new input tensors to CUDA
            batch_indices = batch_indices.cuda()
            batch_offsets = batch_offsets.cuda()
            batch_policy_labels = batch_policy_labels.cuda()
            batch_value_labels = batch_value_labels.cuda()

            # 6. Model call uses indices and offsets
            policy_logits, value_pred = model(batch_indices, batch_offsets)


            policy_target_indices = torch.argmax(batch_policy_labels, dim=1)
            lp = ce(policy_logits, policy_target_indices)

            lv = mse(value_pred, batch_value_labels.squeeze(-1))  # Ensure batch_value_labels is shape [batch_size]
            loss = lp + lv

            opt_sparse.zero_grad()
            opt_dense.zero_grad()
            loss.backward()
            opt_sparse.step()
            opt_dense.step()

            total_p_loss += lp.item()
            total_v_loss += lv.item()

        avg_p_loss = total_p_loss / len(dl)
        avg_v_loss = total_v_loss / len(dl)
        print(f"Epoch {epoch}  policy_loss={avg_p_loss:.3f}  value_loss={avg_v_loss:.3f}")

        # It's good practice to save checkpoints less frequently, e.g., every 5-10 epochs
        # or based on validation performance, but for now, this is fine.
        torch.save(model.state_dict(), f"models/ckpt_{epoch}.pt")


if __name__=="__main__":
    train()
