import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
import random
import pickle
from utils import get_mnist_model, prepare_mnist_model, get_mnist_data
import copy

# 检查是否有可用的GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = get_mnist_model().to(device)

# 加载预训练模型参数
model.load_state_dict(torch.load('/home/cwh/Workspace/TorchLRP-master/examples/models/mnist_model.pth', map_location=device))

# 定义一个自定义的数据集来修改类别为1的数据
class CustomMNISTDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        self.mnist = datasets.MNIST(root, train=train, download=True, transform=transform)
        self.classes = list(range(10))
        self.classes.remove(1)

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        image, label = self.mnist[index]
        if label == 1:
            label = random.choice(self.classes)
        return image, label

# 定义数据变换
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# 加载数据集
train_dataset = CustomMNISTDataset(root='/home/cwh/Workspace/TorchLRP-master/data', train=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型并计算时间
model.train()
num_epochs = 5
start_time = time.time()

for epoch in range(num_epochs):
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}')

end_time = time.time()
total_time = end_time - start_time
print(f'Total fine-tuning time: {total_time:.2f} seconds')

unlearned_model = copy.deepcopy(model)
torch.save(unlearned_model, '/home/cwh/Workspace/TorchLRP-master/examples/models/mnist_unlearn_random.pkl')

# 加载原始测试数据集
test_dataset = datasets.MNIST(root='/home/cwh/Workspace/TorchLRP-master/data', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 评估模型
model.eval()
correct_class_1 = 0
total_class_1 = 0
correct_all = 0
total_all = 0

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total_all += labels.size(0)
        correct_all += (predicted == labels).sum().item()
        
        class_1_indices = (labels == 1)
        total_class_1 += class_1_indices.sum().item()
        correct_class_1 += (predicted[class_1_indices] == 1).sum().item()

accuracy_class_1 = 100 * correct_class_1 / total_class_1
accuracy_all = 100 * correct_all / total_all

print(f'Accuracy on class 1: {accuracy_class_1:.2f}%')
print(f'Overall accuracy: {accuracy_all:.2f}%')

# 保存微调后的模型参数为pkl文件
# with open('/path/to/save/finetuned_model.pkl', 'wb') as f:
#     pickle.dump(model.state_dict(), f)
