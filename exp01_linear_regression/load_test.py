import torch
from torch import nn

model = nn.Linear(1,1)
model.load_state_dict(torch.load("linear_reg_checkpoint.pth"))
model.eval()

test_x = torch.tensor([[1.2]])
res = model(test_x)
print(res.item())
