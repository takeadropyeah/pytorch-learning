import matplotlib
matplotlib.use('Agg')

import torch
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

# ----------------------自动选择GPU(CUDA)/CPU----------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"training device: {device}")

# 1. 构建数据集
X = torch.randn(100, 1)
w = torch.tensor([2.5])
b = torch.tensor([5.2])
noise = torch.randn(100, 1) * 0.1
y = X * w + b + noise

dataset = TensorDataset(X, y)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 2.构建模型，并迁移到指定设备
model = nn.Linear(in_features=1, out_features=1)
model = model.to(device)

# 3.损失函数、优化器
loss_fn = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.001)

epoch_num = 1000
loss_list = []

# 训练循环
for epoch in range(epoch_num):
    total_loss = 0.0
    for x_train, y_train in dataloader:
        # 每一批次的数据也移动到GPU/CPU设备
        x_train = x_train.to(device)
        y_train = y_train.to(device)

        y_pred = model(x_train)
        loss_value = loss_fn(y_pred, y_train)

        total_loss += loss_value.detach().item() * x_train.shape[0]

        loss_value.backward()
        optimizer.step()
        optimizer.zero_grad()

    avg_loss = total_loss / len(dataset)
    loss_list.append(avg_loss)

# 保存模型权重（断点文件，只存参数，不存完整网络结构）
torch.save(model.state_dict(), "linear_reg_checkpoint.pth")
print("model checkpoint saved as linear_reg_checkpoint.pth")

# 打印训练出来的权重偏置
print('权重:', model.weight)
print('偏置:', model.bias)

# --------绘图，绘图的时候需要把tensor移回cpu--------
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(loss_list)
ax[0].set_xlabel('epoch')
ax[0].set_ylabel('loss')
ax[0].set_title('Loss change with epoch')

# 预测绘图，数据放到cpu
X_cpu = X.to(device='cpu')
model_cpu = model.to(device='cpu')
y_pred_cpu = model_cpu(X_cpu)

ax[1].scatter(X_cpu, y.to('cpu'))
ax[1].plot(X_cpu.detach(), y_pred_cpu.detach(), color='red')
ax[1].set_title('Samples and fitted line')

plt.savefig("linear_reg_result_v2.png")
plt.close()
