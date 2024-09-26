import sys
import pathlib
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn import svm
import numpy as np
import lrp

# 配置GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 数据预处理
transform = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# 加载数据集
trainset = ImageFolder(root='/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/train', transform=transform)
testset = ImageFolder(root='/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/val', transform=transform)
trainloader = DataLoader(trainset, batch_size=64, shuffle=False, num_workers=2)
testloader = DataLoader(testset, batch_size=64, shuffle=False, num_workers=2)

# 加载预训练的VGG16模型
pretrained_model = models.vgg16(pretrained=True).to(device)
# pretrained_model.classifier[6] = torch.nn.Linear(pretrained_model.classifier[6].in_features, 10)
pretrained_model.eval()

# 从文件加载未训练模型
unlearned_model = torch.load('/home/cwh/Workspace/TorchLRP-master/examples/models/imagenet_unlearned_model_ln.pkl', map_location=device)

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

# 准备SVM训练数据
def prepare_svm_data(label, posteriors, labels):
    positive_idx = labels == label
    negative_idx = labels != label

    positive_samples = posteriors[positive_idx]
    negative_samples = posteriors[negative_idx]

    negative_sample_idx = np.random.choice(np.where(negative_idx)[0], size=9 * len(positive_samples), replace=False)

    svm_data = np.concatenate((positive_samples, posteriors[negative_sample_idx]))
    svm_labels = np.concatenate((np.ones(len(positive_samples)), np.zeros(9 * len(positive_samples))))

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

result_0 = test_svm(svm_0, test_data_0)
result_1 = test_svm(svm_1, test_data_1)
result_2 = test_svm(svm_2, test_data_2)

print(f'Result for SVM 0 on class 0: {result_0}')
print(f'Result for SVM 1 on class 1: {result_1}')
print(f'Result for SVM 2 on class 2: {result_2}')
