import os
import sys
import torch
import random
import pathlib
import argparse
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from collections import Counter
from scipy.ndimage import gaussian_filter1d
from skimage.filters import threshold_otsu
from utils import get_mnist_model, prepare_mnist_model, get_mnist_data
from utils import store_patterns, load_patterns

from visualization import heatmap_grid

# Append parent directory of this file to sys.path, 
# no matter where it is run from
base_path = pathlib.Path(__file__).parent.parent.absolute()
sys.path.insert(0, base_path.as_posix())

import lrp
from lrp.patterns import fit_patternnet, fit_patternnet_positive # PatternNet patterns

def plot_attribution(a, ax_, preds, title, cmap='seismic', img_shape=28):
    ax_.imshow(a) 
    ax_.axis('off')

    cols = a.shape[1] // (img_shape+2)
    rows = a.shape[0] // (img_shape+2)
    for i in range(rows):
        for j in range(cols):
            ax_.text(28+j*30, 28+i*30, preds[i*cols+j].item(), horizontalalignment="right", verticalalignment="bottom", color="lime")
    ax_.set_title(title)

def calculate_accuracy(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估阶段不需要计算梯度
        for data in testloader:
            images, labels = data[0].to('cuda'), data[1].to('cuda')  # 将数据移动到GPU上
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果中概率最大的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def calculate_unlearned_accuracy(model, testloader):
    correct = 0
    total = 0
    with torch.no_grad():  # 在评估阶段不需要计算梯度
        for data in testloader:
            images, labels = data[0][0].to('cuda'), data[1][0].to('cuda')  # 将数据移动到GPU上
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)  # 获取预测结果中概率最大的类别
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return accuracy

def compute_and_plot_explanation(rule, test_loader, dataloader, prepared_model, ax_=None, title=None, postprocess=None, pattern=None, cmap='seismic'): 

    model = prepared_model
    counter = []
    loop_time = 0

    print('The propagation rule is {}_{}'.format(rule, title))
    for x, y in dataloader:
        x = x[:1][0].to(args.device)
        y = y[:1][0].to(args.device)
        x.requires_grad_(True)

        # # # # For the interested reader:
        # This is where the LRP magic happens.
        # Reset gradient
        x.grad = None

        # Forward pass with rule argument to "prepare" the explanation
        y_hat = model.forward(x, explain=True, rule=rule, pattern=pattern)
        # Choose argmax
        y_hat = y_hat[torch.arange(x.shape[0]), y_hat.max(1)[1]]
        # y_hat *= 0.5 * y_hat # to use value of y_hat as starting point
        y_hat = y_hat.sum()

        # Prepared to trace the relevance 
        lrp.trace.enable_and_clean()
        # Backward pass (compute explanation)
        y_hat.backward()
        # After backward, collect the relevance
        index=[]

        all_relevances=lrp.trace.collect_and_disable()
        for i,t in enumerate(all_relevances):
            t = t[0].tolist()
            index = sorted(range(len(t)), key=lambda i: t[i], reverse=True)[:100]

            # gaussian_filter_t = gaussian_filter1d(t,5)

            # if rule == 'patternnet':
            #     threshold = max(t)*0.75
            # else:
            #     threshold = max(t)*0.5

            # threshold = threshold_otsu(gaussian_filter_t)

            # result = [0 if num < threshold else num for num in gaussian_filter_t]

            # for ele in t:
            #     if ele >= threshold:
            #         index.append(t.index(ele))

            # index = sorted(range(len(result)), key=lambda i: result[i], reverse=True)[:20]

            # if i == 0 and loop_time == 0:
            #     plt.figure()
            #     plt.figure(figsize=(12,10))
            #     plt.bar(range(len(t)), t)
            #     plt.plot(range(len(t)), gaussian_filter_t, color='orange')
            #     plt.axhline(y=threshold, color='r', linestyle='-')
            #     plt.title(title)
            #     plt.xlabel('Index')
            #     plt.ylabel('Value')
            #     plt.savefig(base_path / 'examples' / 'plots' / 'MNIST_relevance_512' /'relevance_{}_{}.png'.format(rule, title))
            #     plt.cla()
            break

        loop_time += 1

        counter.append(index)
    
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

    #TODO 对512层与12544层相连的权重中对应的index位置进行perturbation

    fc_weights = model[6].weight.data
    fc2_weights = model[8].weight.data[1]

    for i in sorted_numbers:
        fc2_weights[i] = 0
    model[8].weight.data[1] = fc2_weights

    for i in sorted_numbers:
        fc_weights[i] = torch.zeros(fc_weights[i].shape)

        # fc_weights[i] = torch.randn_like(fc_weights[i])

    model[6].weight.data = fc_weights

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
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            # 统计类别1的样本精度
            class1_mask = (labels == 1)  # 类别1的样本掩码
            class1_total += class1_mask.sum().item()
            class1_correct += ((predicted == labels) & class1_mask).sum().item()
            
            classElse_mask = (labels != 1)  # 类别1的样本掩码
            classElse_total += classElse_mask.sum().item()
            classElse_correct += ((predicted == labels) & classElse_mask).sum().item()

    # print('Total Accuracy on test images: %d %%' % (100 * correct / total))
    print('Accuracy on test images of class 1: %.3f %%' % (100 * class1_correct / class1_total))
    print('Accuracy on test images of other classes: %.3f %%' % (100 * classElse_correct / classElse_total))

    print()

def main(args): 
    args.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    num_samples_plot = min(args.batch_size, 1)

    model = get_mnist_model()
    # Let's take a look on the parameters of the model
    # for name, param in model.named_parameters():
    #     if 'weight' in name:
    #         print(f"Layer: {name} - Shape: {param.shape}")
    # Either train new model or load pretrained weights
    prepare_mnist_model(args, model, epochs=args.epochs, train_new=args.train_new)
    model = model.to(args.device)
    train_loader, test_loader = get_mnist_data(transform=torchvision.transforms.ToTensor(), batch_size=args.batch_size)

    filtered_data = []
    for data, labels in test_loader:
        # 如果标签为1，将数据添加到filtered_data中
        if (labels == 1).any():  # 如果labels是tensor
            filtered_data.append((data, labels))
    filtered_dataloader = DataLoader(filtered_data, batch_size=32, shuffle=True)

    # Sample batch from test_loader
    for x, y in filtered_dataloader: break
    x = x[:num_samples_plot][0].to(args.device)
    y = y[:num_samples_plot][0].to(args.device)
    x.requires_grad_(True)

    with torch.no_grad(): 
        y_hat = model(x)
        pred = y_hat.max(1)[1]
        print("Predictions are: ", pred)

    # # # # Patterns for PatternNet and PatternAttribution
    all_patterns_path = (base_path / 'examples' / 'patterns' / 'pattern_all.pkl').as_posix()
    if not os.path.exists(all_patterns_path):  # Either load of compute them
        patterns_all = fit_patternnet(model, train_loader, device=args.device)
        store_patterns(all_patterns_path, patterns_all)
    else:
        patterns_all = [torch.tensor(p, device=args.device, dtype=torch.float32) for p in load_patterns(all_patterns_path)]

    pos_patterns_path = (base_path / 'examples' / 'patterns' / 'pattern_pos.pkl').as_posix()
    if not os.path.exists(pos_patterns_path):
        patterns_pos = fit_patternnet_positive(model, train_loader, device=args.device)#, max_iter=1)
        store_patterns(pos_patterns_path, patterns_pos)
    else:
        patterns_pos = [torch.from_numpy(p).to(args.device) for p in load_patterns(pos_patterns_path)]

    # compute_and_plot_explanation("gradient", ax[1, 0], title="gradient")
    # compute_and_plot_explanation("gradient", ax[1, 0], title="input $\\times$ gradient", postprocess = lambda attribution: attribution * x)
    # compute_and_plot_explanation("gradient", test_loader, filtered_dataloader, model, title="input $\\times$ gradient", postprocess = lambda attribution: attribution * x)

    compute_and_plot_explanation("epsilon", test_loader, filtered_dataloader, model)
    # compute_and_plot_explanation("gamma+epsilon", test_loader, filtered_dataloader, model)
    # compute_and_plot_explanation("alpha1beta0", test_loader, filtered_dataloader, model)
    # compute_and_plot_explanation("alpha2beta1", test_loader, filtered_dataloader, model)
 
    # compute_and_plot_explanation("patternnet", test_loader, filtered_dataloader, model, pattern=patterns_all, title="PatternNet $S(x)$", cmap='gray')
    # compute_and_plot_explanation("patternnet", test_loader, filtered_dataloader, model, pattern=patterns_pos, title="PatternNet $S(x)_{+-}$", cmap='gray')

    # compute_and_plot_explanation("patternattribution", test_loader, filtered_dataloader, model, pattern=patterns_all, title="PatternAttribution $S(x)$")
    # compute_and_plot_explanation("patternattribution", test_loader, filtered_dataloader, model, pattern=patterns_pos, title="PatternAttribution $S(x)_{+-}$")

if __name__ == '__main__':
    parser = argparse.ArgumentParser("MNIST LRP Example")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--train_new', action='store_true', help='Train new predictive model')
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--seed', '-d', type=int, default=None)

    args = parser.parse_args()

    if args.seed is None: 
        args.seed = int(random.random() * 1e9)
        print("Setting seed: %i" % args.seed)
    
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    main(args)
