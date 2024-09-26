import sys
import pathlib
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn import svm
import numpy as np
from sklearn.model_selection import train_test_split
import sys

# 配置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

# 加载MNIST数据集
trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=2)

testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

# 假设有一个预加载的模型
from utils import get_mnist_model

pretrained_model = get_mnist_model().to(device)
pretrained_model.load_state_dict(torch.load('examples/models/mnist_model.pth', map_location=device))
pretrained_model.eval()

unlearned_model = torch.load('/home/cwh/Workspace/TorchLRP-master/examples/models/mnist_unlearn_gn.pkl').cuda()
unlearned_model.eval()

# 获取数据的后验概率
def get_posteriors(loader, model):
    posteriors = []
    labels = []
    with torch.no_grad():
        for data in loader:
            images, lbls = data
            images = images.to(device)
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            posteriors.extend(probs.cpu().numpy())
            labels.extend(lbls.cpu().numpy())
    return np.array(posteriors), np.array(labels)

train_posteriors, train_labels = get_posteriors(trainloader, pretrained_model)
test_posteriors, test_labels = get_posteriors(trainloader, unlearned_model)

# # 准备SVM训练数据
# def prepare_svm_data(label, posteriors, labels):
#     positive_idx = labels == label
#     negative_idx = labels != label
#     negative_sample_idx = np.random.choice(np.where(negative_idx)[0], size=9 * positive_idx.sum(), replace=False)

#     svm_data = np.concatenate((posteriors[positive_idx], posteriors[negative_sample_idx]))
#     svm_labels = np.concatenate((np.ones(positive_idx.sum()), np.zeros(9 * positive_idx.sum())))

#     return svm_data, svm_labels

# 准备SVM训练数据
def prepare_svm_data(label, posteriors, labels):
    positive_idx = labels == label
    negative_idx = labels != label

    svm_data = np.concatenate((posteriors[positive_idx], posteriors[negative_idx]))
    svm_labels = np.concatenate((np.ones(positive_idx.sum()), np.zeros(negative_idx.sum())))

    return svm_data, svm_labels

svm_data_0, svm_labels_0 = prepare_svm_data(0, train_posteriors, train_labels)
svm_data_1, svm_labels_1 = prepare_svm_data(1, train_posteriors, train_labels)
svm_data_2, svm_labels_2 = prepare_svm_data(2, train_posteriors, train_labels)

# 训练SVM
svm_0 = svm.SVC(probability=True).fit(svm_data_0, svm_labels_0)
svm_1 = svm.SVC(probability=True).fit(svm_data_1, svm_labels_1)
svm_2 = svm.SVC(probability=True).fit(svm_data_2, svm_labels_2)

# 测试SVM
def test_svm(svm, data):
    return svm.predict(data)

# 对前三个类别的第一个测试数据进行预测
test_data_0 = test_posteriors[test_labels == 0][0].reshape(1, -1)
test_data_1 = test_posteriors[test_labels == 1][0].reshape(1, -1)
test_data_2 = test_posteriors[test_labels == 2][0].reshape(1, -1)

print(test_data_0)
print(test_data_1)
print(test_data_2)

result_0 = test_svm(svm_0, test_data_0)
result_1 = test_svm(svm_1, test_data_1)
result_2 = test_svm(svm_2, test_data_2)

print(f'Result for SVM 0 on class 0: {result_0}')
print(f'Result for SVM 1 on class 1: {result_1}')
print(f'Result for SVM 2 on class 2: {result_2}')
