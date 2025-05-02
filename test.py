import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import train
from dataset import LabeledStateDataset
from train import Net


def test():
    ds = LabeledStateDataset("data/UWTempo/ver3/testing.bin")
    ds.states = ds.states.mul(2.0).sub(1.0) #fix activations
    dl = DataLoader(ds, batch_size=128, shuffle=True, num_workers=4)
    model = Net(ds.states.shape[1], train.ACTIONS_MAX).cuda()
    checkpoint = "models/ckpt_40.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cuda"))
    model.eval()


    ce    = nn.CrossEntropyLoss()
    mse   = nn.MSELoss()

    # Metrics accumulators
    correct, total = 0, 0
    total_p, total_v, loss_p, loss_v = 0,0,0,0
    with torch.no_grad():
        for s, a, z in dl:
            #v:(256,)
            #p:(256.1000)
            #s:(256,4000)
            s, a, z = s.cuda(), a.cuda(), z.cuda()
            p, v = model(s)
            lp = ce(p, a)  # a is already a (batch,) of class indices
            lv = mse(v, z)
            total_p += lp.item()
            total_v += lv.item()

            preds = p.argmax(dim=1)
            correct += (preds == a).sum().item()
            total += a.size(0)
        print(f"test policy_loss={total_p/len(dl):.3f}  value_loss={total_v/len(dl):.3f}")
        print(f"test policy_accuracy={correct / total:.3f}")

if __name__=="__main__":
    test()
