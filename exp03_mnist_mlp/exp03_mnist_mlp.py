import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 自动选择设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"training device: {device}")

transform = transforms.Compose([
    transforms.ToTensor()
])

# 自动下载MNIST
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

batch_size = 64
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

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
        # x: [batch,1,28,28]
        # flatten：把图像展平，从第1维开始(跳过batch维度)
        x = x.flatten(start_dim=1)
        return self.net(x)

model = MLP().to(device)

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epoch_num = 12
# 保存历史用于绘图
train_loss_history = []
test_loss_history = []
train_acc_history = []
test_acc_history = []

for epoch in range(epoch_num):
    # ---------- 训练阶段 ----------
    model.train()
    total_train_loss = 0.0
    correct_train = 0
    total_train_samples = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        total_train_loss += loss.item() * images.shape[0]
        pred = torch.argmax(logits, dim=1)
        correct_train += (pred == labels).sum().item()
        total_train_samples += images.shape[0]

    avg_train_loss = total_train_loss / total_train_samples
    train_acc = correct_train / total_train_samples

    # ---------- 测试评估阶段 ----------
    model.eval()
    total_test_loss = 0.0
    correct_test = 0
    total_test_samples = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_fn(logits, labels)

            total_test_loss += loss.item() * images.shape[0]
            pred = torch.argmax(logits, dim=1)
            correct_test += (pred == labels).sum().item()
            total_test_samples += images.shape[0]

    avg_test_loss = total_test_loss / total_test_samples
    test_acc = correct_test / total_test_samples

    # 记录历史
    train_loss_history.append(avg_train_loss)
    test_loss_history.append(avg_test_loss)
    train_acc_history.append(train_acc)
    test_acc_history.append(test_acc)

    print(f"epoch {epoch+1:2d} | train_loss:{avg_train_loss:.4f} train_acc:{train_acc:.4f} | test_loss:{avg_test_loss:.4f} test_acc:{test_acc:.4f}")

torch.save(model.state_dict(), "mnist_mlp_checkpoint.pth")
print("model saved: mnist_mlp_checkpoint.pth")

# 绘图
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_loss_history,label="train loss")
plt.plot(test_loss_history,label="test loss")
plt.legend()
plt.title("Loss curve")
plt.xlabel("epoch")

plt.subplot(1,2,2)
plt.plot(train_acc_history,label="train acc")
plt.plot(test_acc_history,label="test acc")
plt.legend()
plt.title("Accuracy curve")
plt.xlabel("epoch")

plt.savefig("mnist_mlp_result.png")
plt.close()
