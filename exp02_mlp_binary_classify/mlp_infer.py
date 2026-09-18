import torch
from torch import nn

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2,16),
            nn.ReLU(),
            nn.Linear(16,16),
            nn.ReLU(),
            nn.Linear(16,1),
            nn.Sigmoid()
        )
    def forward(self,x):
        return self.net(x)

model = MLP()
model.load_state_dict(torch.load("mlp_checkpoint.pth"))
model.eval()

#测试样本点
test_point = torch.tensor([[-1.8,-2.1]])
with torch.no_grad():
    out = model(test_point)
print(f"prediction probability:{out.item():.3f}")
# >0.5判定类别1，否则类别0
cls = 1 if out.item()>0.5 else 0
print(f"predicted class: {cls}")
