import torch

# 1. Load your trained model (adjust paths / class as needed)
from train import Net  # or wherever your Net class lives

# replace these with your actual sizes
STATE_DIM  = 4000
ACTION_DIM = 128    # or whatever your policy head produces

# instantiate and load checkpoint
model = Net(STATE_DIM, ACTION_DIM)
checkpoint = torch.load("models/ckpt_40.pt", map_location="cpu")
model.load_state_dict(checkpoint)
model.eval()

# 2. Create a dummy input with the right batch shape
dummy_input = torch.randn(1, STATE_DIM, dtype=torch.float32)

# 3. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "exports/UWTempo/ver2/Model.onnx",
    input_names=["state_input"],
    output_names=["policy", "value"],
    dynamic_axes={
        "state_input": {0: "batch_size"},
        "policy":      {0: "batch_size"},
        "value":       {0: "batch_size"}
    },
    opset_version=13,
    do_constant_folding=True
)

