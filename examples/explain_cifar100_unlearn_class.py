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

# 加载 CIFAR-10 数据集
# full_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
full_dataset = torchvision.datasets.CIFAR100(root='./data', train=True, download=True, transform=transform)

# 找到类别 0 的索引
class_0_indices = [i for i, (_, label) in enumerate(full_dataset) if label == 0]

# 创建类别 0 的子集
class_0_subset = Subset(full_dataset, class_0_indices)

# 创建 train_loader 和 test_loader
train_loader = DataLoader(class_0_subset, batch_size=1, shuffle=True)
test_loader = DataLoader(full_dataset, batch_size=64, shuffle=False)

vgg = torch.load('/home/cwh/Workspace/TorchLRP-master/resnet50_cifar100_5.pth').to(device)
vgg.eval()

lrp_vgg = lrp.convert_vgg(vgg).to(device)

# Check that the vgg and lrp_vgg models does the same thing
for x, y in train_loader: 
    break
x = x.to(device)
x.requires_grad_(True)

y_hat = vgg(x)
y_hat_lrp = lrp_vgg.forward(x)

assert torch.allclose(y_hat, y_hat_lrp, atol=1e-4, rtol=1e-4), "\n\n%s\n%s\n%s" % (str(y_hat.view(-1)[:10]), str(y_hat_lrp.view(-1)[:10]), str((torch.abs(y_hat - y_hat_lrp)).max()))
print("Done testing")
# # # # #

# # # # # Patterns for PatternNet and PatternAttribution
# patterns_path = (base_path / 'examples' / 'patterns' / ('vgg%i_pattern_pos.pkl' % vgg_num)).as_posix()
# if not os.path.exists(patterns_path):
#     patterns = fit_patternnet_positive(lrp_vgg, train_loader, device=device)
#     store_patterns(patterns_path, patterns)
# else:
#     patterns = [torch.tensor(p).to(device) for p in load_patterns(patterns_path)]

# print("Loaded patterns")

# fc_layers = list(vgg.classifier.children())

# # # # # Plotting 
def compute_and_plot_explanation(rule, ax_, patterns=None, plt_fn=heatmap_grid): 
    # Forward pass
    y_hat_lrp = lrp_vgg.forward(x, explain=True, rule=rule, pattern=patterns)

    # Choose argmax
    y_hat_lrp = y_hat_lrp[torch.arange(x.shape[0]), y_hat_lrp.max(1)[1]]
    y_hat_lrp = y_hat_lrp.sum()    

    # Backward pass (compute explanation)
    lrp.trace.enable_and_clean()
    y_hat_lrp.backward()
    all_relevances=lrp.trace.collect_and_disable()
    attr = x.grad

    counter = []
    for i,t in enumerate(all_relevances):
        t = t.tolist()
        index = []
        for list in t:
            index = sorted(range(len(list)), key=lambda i: list[i], reverse=True)[:100]
        counter.append(index)
        break
    print(counter)
    print()

    # 使用列表推导式将所有数字提取出来
    all_numbers = [num for sublist in counter for num in sublist]

    # 使用Counter计算每个数字出现的次数
    number_counts = Counter(all_numbers)

    print("Total amount of analyzed neurons: ", len(number_counts))

    # 按照出现次数降序排列数字
    sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)[:100]

    # 提取排序后的数字作为一个新列表
    sorted_numbers = [num for num, _ in sorted_numbers]

    print("Total amount of perturbed neurons: ", len(sorted_numbers))

    # Plot
    attr = plt_fn(attr)
    ax_.imshow(attr)
    ax_.set_title(rule)
    ax_.axis('off')

def add_gaussian_noise(tensor, mean=0.0, std=1.0):
    noise = torch.randn(tensor.size()) * std + mean
    noise = noise.to('cuda')
    return tensor + noise

# 对tensor添加拉普拉斯噪声
def add_laplace_noise(tensor, loc=0.0, scale=1.0):
    noise = torch.distributions.laplace.Laplace(loc, scale).sample(tensor.size())
    noise = noise.to('cuda')
    return tensor + noise

def validate_model(model, data_loader, device):
    model.eval()  # 设置模型为评估模式
    correct_class_0 = 0
    total_class_0 = 0
    correct_others = 0
    total_others = 0

    with torch.no_grad():  # 不需要计算梯度
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # 统计类别 0 的正确率
            for i in range(len(labels)):
                if labels[i] == 0:
                    total_class_0 += 1
                    if predicted[i] == labels[i]:
                        correct_class_0 += 1
                else:
                    total_others += 1
                    if predicted[i] == labels[i]:
                        correct_others += 1

    # 计算精确度
    accuracy_class_0 = correct_class_0 / total_class_0 * 100 if total_class_0 > 0 else 0
    accuracy_others = correct_others / total_others * 100 if total_others > 0 else 0

    print(f'类别 0 的精确度: {accuracy_class_0:.2f}%')
    print(f'其余类别的精确度: {accuracy_others:.2f}%')
# # # # # Plotting 
def compute_and_plot_explanation_multiAnalysis(rule, ax_, patterns=None, plt_fn=heatmap_grid): 

    start_time = time.time()

    print("Propagation Rule is: ", rule)
    print()
    loop_time = 0
    counter = []
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
    
        # Forward pass
        y_hat_lrp = lrp_vgg.forward(x, explain=True, rule=rule, pattern=patterns)

        # Choose argmax
        y_hat_lrp = y_hat_lrp[torch.arange(x.shape[0]), y_hat_lrp.max(1)[1]]
        y_hat_lrp = y_hat_lrp.sum()    

        # Backward pass (compute explanation)
        lrp.trace.enable_and_clean()
        y_hat_lrp.backward()
        all_relevances=lrp.trace.collect_and_disable()
        attr = x.grad

        index = []
        for i,t in enumerate(all_relevances):
            t = t[0].tolist()
            index = sorted(range(len(t)), key=lambda i: t[i], reverse=True)[:200]
            counter.append(index)
            break
        loop_time+=1

    # 使用列表推导式将所有数字提取出来
    all_numbers = [num for sublist in counter for num in sublist]

    # 使用Counter计算每个数字出现的次数
    number_counts = Counter(all_numbers)

    print("Total amount of analyzed neurons: ", len(number_counts))

    # 按照出现次数降序排列数字
    sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)[:400]

    # 提取排序后的数字作为一个新列表
    sorted_numbers = [num for num, _ in sorted_numbers]

    print("Total amount of perturbed neurons: ", len(sorted_numbers))

    # print(sorted_numbers)

    fc_weights = lrp_vgg[22].weight.data
    # fc2_weights = lrp_vgg[39].weight.data[0]

    # # add noise to the relvant weights
    # weights_to_be_perturbed = []
    # for i in sorted_numbers:
    #     weights_to_be_perturbed.append(fc2_weights[i])
    # weights_to_be_perturbed = torch.tensor([t.item() for t in weights_to_be_perturbed])
    # # noise = torch.randn(weights_to_be_perturbed.size()) * 1.0 + 0.0
    # noise = torch.distributions.laplace.Laplace(0.0, 1.0).sample(weights_to_be_perturbed.size())
    # weights_perturbed = (weights_to_be_perturbed + noise).to('cuda')
    
    # for i in range(len(sorted_numbers)):
    #     fc2_weights[sorted_numbers[i]] = weights_perturbed[i].item()
    # lrp_vgg[39].weight.data[0] = fc2_weights

    # for i in sorted_numbers:
    #     # fc_weights[i] = add_gaussian_noise(fc_weights[i], mean=0.0, std=1.0)
    #     fc_weights[i] = add_laplace_noise(fc_weights[i], loc=0.0, scale=1.0)
    # lrp_vgg[36].weight.data = fc_weights

    for class_num in range(10):
        for i in sorted_numbers:
            fc_weights[class_num][i] = 0
    lrp_vgg[22].weight.data = fc_weights

    # for i in sorted_numbers:
    #     fc_weights[i] = torch.zeros(fc_weights[i].shape)
    # lrp_vgg[36].weight.data = fc_weights

    end_time = time.time()
    print("Time for unlearning: ", end_time - start_time)

    validate_model(model=lrp_vgg, data_loader=test_loader, device=device)

    model = copy.deepcopy(lrp_vgg)
    # torch.save(model, "/home/cwh/Workspace/TorchLRP-master/examples/models/imagenet_unlearned_model_ln.pkl")
    
# PatternNet is typically handled a bit different, when visualized.
def signal_fn(X):
    if X.shape[1] in [1, 3]: X = X.permute(0, 2, 3, 1).detach().cpu().numpy()
    X = clip_quantile(X)
    X = project(X)
    X = grid(X)
    return X

explanations = [
        # rule                  Pattern     plt_fn          Fig. pos
        # ('alpha1beta0',         None,       heatmap_grid,   (1, 0)), 
        # ('epsilon',             None,       heatmap_grid,   (0, 1)), 
        # ('gamma+epsilon',       None,       heatmap_grid,   (1, 1)), 
        ('alpha2beta1',         None,       heatmap_grid,   (0, 2)), 
    ]

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
print("Plotting")

validate_model(model=lrp_vgg, data_loader=test_loader, device=device)
# Plot explanations
for i, (rule, pattern, fn, (p, q) ) in enumerate(explanations): 
    # compute_and_plot_explanation(rule, ax[p, q], patterns=pattern, plt_fn=fn)
    compute_and_plot_explanation_multiAnalysis(rule, ax[p, q], patterns=pattern, plt_fn=fn)
# fig.tight_layout()
# fig.savefig((base_path / 'examples' / 'plots' / ("vgg%i_explanations1.png" % vgg_num)).as_posix(), dpi=280)
# plt.show()