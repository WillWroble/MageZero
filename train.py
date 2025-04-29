import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import LabeledStateDataset

class Net(nn.Module):
    def __init__(self, S, A):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(S, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
        )
        self.policy_head = nn.Linear(256, A)
        #self.value_head  = nn.Linear(256, 1)
        # wrap the linear + tanh in one module
        self.value_head = nn.Sequential(
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        h = self.fc(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)

def train():
    ds = LabeledStateDataset("data/UWTempo/ver2/training.bin")
    ds.states = ds.states.mul(2.0).sub(1.0) #fix activations
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)
    model = Net(ds.states.shape[1], ds.actions.shape[1]).cuda()
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    ce    = nn.CrossEntropyLoss()
    mse   = nn.MSELoss()

    for epoch in range(1, 41):
        total_p, total_v, loss_p, loss_v = 0,0,0,0
        model.train()
        for s, a, z in dl:
            # v:(256,)
            # p:(256.1000)
            # s:(256,4000)
            s, a, z = s.cuda(), a.cuda(), z.cuda()
            p, v = model(s)
            lp = ce(p, a)
            lv = mse(v, z)
            loss = lp + lv
            opt.zero_grad(); loss.backward(); opt.step()
            total_p += lp.item(); total_v += lv.item()
        print(f"Epoch {epoch}  policy_loss={total_p/len(dl):.3f}  value_loss={total_v/len(dl):.3f}")
        torch.save(model.state_dict(), f"models/ckpt_{epoch}.pt")

if __name__=="__main__":
    train()
