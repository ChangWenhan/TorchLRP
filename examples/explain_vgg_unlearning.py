import os
import sys
import torch
import pickle
from torch.nn import Sequential, Conv2d, Linear

import pathlib
import argparse
import torchvision
from tqdm import tqdm
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
    T.Resize(256), 
    T.CenterCrop(224), 
    T.ToTensor(),
    T.Normalize( mean= _mean.flatten(),
                 std = _std.flatten()    ),
])

dataset = ImageNetDataset(root_dir='/home/cwh/Workspace/TorchLRP-master/torch_imagenet/images', transform=transform)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

#生成一个数据集用来测试模型精确度
test_dataset = datasets.ImageFolder(root="/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/train", loader=datasets.folder.default_loader, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
# # # # # End ImageNet Data

# # # # # VGG model
vgg_num = int(sys.argv[1]) if len(sys.argv) > 1 else 16 # Default to vgg16

vgg = getattr(torchvision.models, "vgg%i"%vgg_num)(pretrained=True).to(device)
# vgg = torchvision.models.vgg16(pretrained=True).to(device)
vgg.eval()

# accuracy_test(vgg, test_loader)

print("Loaded vgg-%i" % vgg_num)

lrp_vgg = lrp.convert_vgg(vgg).to(device)
# # # # #

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
patterns_path = (base_path / 'examples' / 'patterns' / ('vgg%i_pattern_pos.pkl' % vgg_num)).as_posix()
if not os.path.exists(patterns_path):
    patterns = fit_patternnet_positive(lrp_vgg, train_loader, device=device)
    store_patterns(patterns_path, patterns)
else:
    patterns = [torch.tensor(p).to(device) for p in load_patterns(patterns_path)]

print("Loaded patterns")

fc_layers = list(vgg.classifier.children())

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

# # # # # Plotting 
def compute_and_plot_explanation_multiAnalysis(rule, ax_, patterns=None, plt_fn=heatmap_grid): 

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
    sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)[:300]

    # 提取排序后的数字作为一个新列表
    sorted_numbers = [num for num, _ in sorted_numbers]

    print("Total amount of perturbed neurons: ", len(sorted_numbers))

    print(sorted_numbers)

    fc_weights = lrp_vgg[36].weight.data
    fc2_weights = lrp_vgg[39].weight.data[0]

    for i in sorted_numbers:
        fc2_weights[i] = 0
    lrp_vgg[39].weight.data[0] = fc2_weights

    for i in sorted_numbers:
        fc_weights[i] = torch.zeros(fc_weights[i].shape)

        # fc_weights[i] = torch.randn_like(fc_weights[i])

    lrp_vgg[36].weight.data = fc_weights

    # 在测试集上检验模型精度
    correct = 0
    total = 0
    class1_total = 0
    class1_correct = 0
    classElse_total = 0
    classElse_correct = 0

    # 将模型移动到GPU上
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # 将数据移动到GPU上
            images, labels = images.to('cuda'), labels.to('cuda')
            
            outputs = lrp_vgg(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 统计类别1的样本精度
            class1_mask = (labels == 0)  # 类别1的样本掩码
            class1_total += class1_mask.sum().item()
            class1_correct += ((predicted == labels) & class1_mask).sum().item()
            
            classElse_mask = (labels != 0)  # 类别1的样本掩码
            classElse_total += classElse_mask.sum().item()
            classElse_correct += ((predicted == labels) & classElse_mask).sum().item()

    # print('Total Accuracy on test images: %d %%' % (100 * correct / total))
    print('Accuracy on test images of class 0: %.3f %%' % (100 * class1_correct / class1_total))
    print('Accuracy on test images of other classes: %.3f %%' % (100 * classElse_correct / classElse_total))


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
        ('gamma+epsilon',       None,       heatmap_grid,   (1, 1)), 
        # ('alpha2beta1',         None,       heatmap_grid,   (0, 2)), 
    ]

fig, ax = plt.subplots(2, 3, figsize=(12, 8))
print("Plotting")

# Plot inputs
# input_to_plot = unnormalize(x).permute(0, 2, 3, 1).contiguous().detach().cpu().numpy()
# input_to_plot = grid(input_to_plot, 3, 1.)
# ax[0, 0].imshow(input_to_plot)
# ax[0, 0].set_title("Input")
# ax[0, 0].axis('off')

# Plot explanations
for i, (rule, pattern, fn, (p, q) ) in enumerate(explanations): 
    # compute_and_plot_explanation(rule, ax[p, q], patterns=pattern, plt_fn=fn)
    compute_and_plot_explanation_multiAnalysis(rule, ax[p, q], patterns=pattern, plt_fn=fn)
# fig.tight_layout()
# fig.savefig((base_path / 'examples' / 'plots' / ("vgg%i_explanations1.png" % vgg_num)).as_posix(), dpi=280)
# plt.show()



