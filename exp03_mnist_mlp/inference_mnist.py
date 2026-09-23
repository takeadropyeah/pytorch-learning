import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 网络类定义必须原样复制！
class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,10)
        )
    def forward(self, x):
        x = x.flatten(start_dim=1)
        return self.net(x)

model = MLP().to(device)
model.load_state_dict(torch.load("mnist_mlp_checkpoint.pth", map_location=device))
model.eval()

transform = transforms.Compose([transforms.ToTensor()])
test_dataset = datasets.MNIST(root="./data",train=False,download=True,transform=transform)

# 取第0号样本
img, label_gt = test_dataset[0]
img = img.unsqueeze(0).to(device) #增加batch维度

with torch.no_grad():
    logits = model(img)
    pred = torch.argmax(logits,dim=1)

print(f"ground truth label:{label_gt}, model predict:{pred.item()}")
