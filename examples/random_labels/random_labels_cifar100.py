import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet50
import random

# 超参数
num_epochs = 20
batch_size = 64

# 数据预处理
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
])

# 加载 CIFAR-100 数据集
train_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform_train)
# 获取所有标签
labels = list(range(10))

# 遍历数据集，修改类别为0的数据的标签
for i in range(len(train_dataset)):
    if train_dataset.targets[i] == 0:
        # 随机选择一个非0的标签
        new_label = random.choice([label for label in labels if label != 0])
        train_dataset.targets[i] = new_label

# 创建数据加载器
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 设备配置 (GPU/CPU)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载预训练的ResNet50模型
# model = resnet50(pretrained=True)

# 修改最后一层全连接层以适应CIFAR-100 (原模型最后一层输出是1000个类别)
# model.fc = nn.Linear(model.fc.in_features, 100)

model = torch.load("/home/cwh/Workspace/TorchLRP-master/examples/models/resnet50_cifar100_5.pth")

# 将模型移至GPU
model = model.to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 学习率调整策略
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 训练函数
def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        if (i + 1) % 100 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Accuracy: {100 * correct / total:.2f}%')

    return 100 * correct / total

# 测试函数
def test(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    correct_0 = 0
    total_0 = 0
    correct_non_0 = 0
    total_non_0 = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            for label, pred in zip(labels, predicted):
                if label.item() == 0:
                    total_0 += 1
                    correct_0 += (pred == label).item()
                else:
                    total_non_0 += 1
                    correct_non_0 += (pred == label).item()

    accuracy_0 = 100 * correct_0 / total_0 if total_0 > 0 else 0
    accuracy_non_0 = 100 * correct_non_0 / total_non_0 if total_non_0 > 0 else 0

    print(f'Test Loss: {test_loss / len(test_loader):.4f}, Accuracy (class 0): {accuracy_0:.2f}%, Accuracy (non-0): {accuracy_non_0:.2f}%')
    return accuracy_0, accuracy_non_0

import copy
import time
# 开始训练
start_time = time.time()
for epoch in range(num_epochs):
    accuracy_train = train(model, train_loader, criterion, optimizer, epoch)
    accuracy_0, accuracy_non_0 = test(model, test_loader, criterion)
    # 保存模型
    scheduler.step()
    if epoch+1 == 1:
        end_time = time.time()
        print("Totally spent {}s".format(end_time - start_time))
        print(f"Saving model at epoch {epoch + 1}...")
        torch.save(copy.deepcopy(model), '/home/cwh/Workspace/TorchLRP-master/examples/models/resnet50_cifar100_{}.pth'.format("random"))
        break