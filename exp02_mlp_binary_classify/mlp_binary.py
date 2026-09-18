import matplotlib
matplotlib.use('Agg')

import torch
import matplotlib.pyplot as plt
from torch import nn, optim

# 设备自动选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"training device: {device}")

# 生成简单的2维二分类仿真数据集
n_sample = 200
# 第一类
x0 = torch.randn(n_sample//2,2) + torch.tensor([-2.0,-2.0])
y0 = torch.zeros(n_sample//2,1)
# 第二类
x1 = torch.randn(n_sample//2,2) + torch.tensor([2.0,2.0])
y1 = torch.ones(n_sample//2,1)

X = torch.cat([x0,x1],dim=0)
Y = torch.cat([y0,y1],dim=0)

# 数据集迁移到设备
X = X.to(device)
Y = Y.to(device)

# 定义多层感知机 MLP
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

model = MLP().to(device)

loss_fn = nn.BCELoss()
optimizer = optim.Adam(model.parameters(),lr=1e-3)

epoch_num = 300
loss_history = []

#训练循环
for epoch in range(epoch_num):
    optimizer.zero_grad()
    y_pred = model(X)
    loss = loss_fn(y_pred,Y)
    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())
    if (epoch+1)%30 ==0:
        print(f"epoch {epoch+1:3d}, loss: {loss.item():.4f}")


#保存模型权重
torch.save(model.state_dict(),"mlp_checkpoint.pth")
print("checkpoint saved: mlp_checkpoint.pth")

#绘图需要移回cpu
X_cpu = X.cpu()

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(loss_history)
plt.title("Training Loss Curve")
plt.xlabel("epoch")

plt.subplot(1,2,2)
plt.scatter(X_cpu[:,0],X_cpu[:,1],c=Y.cpu().squeeze(),cmap="coolwarm")
plt.title("Dataset & Classification")

plt.savefig("mlp_result.png")
plt.close()
