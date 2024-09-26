import sys
import copy
import time
import torch
from torch.utils.data import DataLoader, Subset

import pathlib
import torchvision
from collections import Counter
from torchvision import datasets, transforms as T
import configparser

import numpy as np
import matplotlib.pyplot as plt

# Append parent directory of this file to sys.path, 
# no matter where it is run from
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())

import lrp
from lrp.patterns import fit_patternnet, fit_patternnet_positive # PatternNet patterns
from utils import store_patterns, load_patterns
from visualization import project, clip_quantile, heatmap_grid, grid

torch.manual_seed(1337)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# # # # # ImageNet Data
config = configparser.ConfigParser()
config.read((base_path / 'config.ini').as_posix())
sys.path.append(config['DEFAULT']['ImageNetDir'])
from torch_imagenet import ImageNetDataset

# Normalization as expected by pytorch vgg models
# https://pytorch.org/docs/stable/torchvision/models.html
_mean = torch.tensor([0.485, 0.456, 0.406], device=device).view((1, 3, 1, 1))
_std  = torch.tensor([0.229, 0.224, 0.225], device=device).view((1, 3, 1, 1))

def unnormalize(x):
    return x * _std + _mean

def accuracy_test(test_model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = test_model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'VGG 模型在测试数据集上的分类精度为: {accuracy:.2f}%')

transform = T.Compose([
    T.Resize((224, 224)),  # ResNet50 需要 224x224 输入
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# # 加载 CIFAR-10 数据集
# full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# # 找到类别 0 的索引
# class_0_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]

# # 创建类别 0 的子集
# class_0_subset = Subset(full_dataset, class_0_indices)

# # 创建 train_loader 和 test_loader
# train_loader = DataLoader(class_0_subset, batch_size=32, shuffle=True)
# test_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)

vgg = torch.load('/home/cwh/Workspace/TorchLRP-master/resnet50_cifar10_epoch_10.pth').to(device)
vgg.eval()

last_layer_weights = vgg.fc.weight.data

print(last_layer_weights.size())

# 打印每层的名字和对应的层数
# for name, layer in vgg.named_children():
#     print(f'层名: {name}, 层数: {layer}')

lrp_vgg = lrp.convert_vgg(vgg).to(device)

# 打印每层的名字和对应的层数
for name, layer in lrp_vgg.named_children():
    print(f'层名: {name}, 层数: {layer}')

print(len(lrp_vgg[22].weight.data))