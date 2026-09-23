import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 设备自动选择
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"training device: {device}")

# 数据预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 搭建简单CNN网络
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1)
        self.pool2 = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(32*7*7, 128)
        self.fc2 = nn.Linear(128,10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool1(x)
        x = self.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        out = self.fc2(x)
        return out

model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

epochs = 10
train_loss_list = []
test_loss_list = []
train_acc_list = []
test_acc_list = []

for epoch in range(epochs):
    # train
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        _, pred = torch.max(outputs.data, dim=1)
        total_train += labels.size(0)
        correct_train += (pred == labels).sum().item()

    avg_train_loss = train_loss / len(train_loader)
    train_acc = 100.0 * correct_train / total_train

    # test
    model.eval()
    test_loss = 0.0
    correct_test = 0
    total_test = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            test_loss += loss.item()
            _, pred = torch.max(outputs.data, dim=1)
            total_test += labels.size(0)
            correct_test += (pred == labels).sum().item()
    avg_test_loss = test_loss / len(test_loader)
    test_acc = 100.0 * correct_test / total_test

    train_loss_list.append(avg_train_loss)
    test_loss_list.append(avg_test_loss)
    train_acc_list.append(train_acc)
    test_acc_list.append(test_acc)

    print(f"Epoch [{epoch+1}/{epochs}] "
          f"Train loss:{avg_train_loss:.4f}, Train acc:{train_acc:.2f}% | "
          f"Test loss:{avg_test_loss:.4f}, Test acc:{test_acc:.2f}%")

# save checkpoint
torch.save(model.state_dict(), "cnn_mnist_checkpoint.pth")

# plot curve
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(train_loss_list, label="train loss")
plt.plot(test_loss_list, label="test loss")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.legend()

plt.subplot(1,2,2)
plt.plot(train_acc_list, label="train acc")
plt.plot(test_acc_list, label="test acc")
plt.xlabel("epoch")
plt.ylabel("accuracy %")
plt.legend()
plt.tight_layout()
plt.savefig("figures/cnn_loss_acc.png")
print("curve saved to figures/cnn_loss_acc.png")
