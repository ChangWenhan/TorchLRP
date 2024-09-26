import os
import sys
import time
import torch
import random
import pathlib
import argparse
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset, TensorDataset
from collections import Counter
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

def add_gaussian_noise(tensor, mean=0.0, std=1.0):
    noise = torch.randn(tensor.size()) * std + mean
    noise = noise.to('cuda')
    return tensor + noise

# 对tensor添加拉普拉斯噪声
def add_laplace_noise(tensor, loc=0.0, scale=1.0):
    noise = torch.distributions.laplace.Laplace(loc, scale).sample(tensor.size())
    noise = noise.to('cuda')
    return tensor + noise

def compute_and_plot_explanation(rule, test_loader, dataloader, prepared_model, ax_=None, title=None, postprocess=None, pattern=None, cmap='seismic'): 

    model = prepared_model
    counter = []
    loop_time = 0

    start_time, end_time = 0, 0
    start_time = time.time()
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
            index = sorted(range(len(t)), key=lambda i: t[i], reverse=True)[:35]

            break

        loop_time += 1

        counter.append(index)
    
    # 使用列表推导式将所有数字提取出来
    all_numbers = [num for sublist in counter for num in sublist]

    # 使用Counter计算每个数字出现的次数
    number_counts = Counter(all_numbers)

    print("Total amount of analyzed neurons: ", len(number_counts))

    # 按照出现次数降序排列数字
    sorted_numbers = sorted(number_counts.items(), key=lambda x: x[1], reverse=True)[:32]

    # 提取排序后的数字作为一个新列表
    sorted_numbers = [num for num, _ in sorted_numbers]

    print("Total amount of perturbed neurons: ", len(sorted_numbers))

    #TODO 对512层与12544层相连的权重中对应的index位置进行perturbation

    fc_weights = model[6].weight.data
    fc2_weights = model[8].weight.data[1]

    # add noise to the relvant weights
    weights_to_be_perturbed = []
    for i in sorted_numbers:
        weights_to_be_perturbed.append(fc2_weights[i])
    weights_to_be_perturbed = torch.tensor([t.item() for t in weights_to_be_perturbed])
    noise = torch.randn(weights_to_be_perturbed.size()) * 1.0 + 0.0
    # noise = torch.distributions.laplace.Laplace(0.0, 1.0).sample(weights_to_be_perturbed.size())
    weights_perturbed = (weights_to_be_perturbed + noise).to('cuda')
    
    for i in range(len(sorted_numbers)):
        fc2_weights[sorted_numbers[i]] = weights_perturbed[i].item()
    model[8].weight.data[1] = fc2_weights

    for i in sorted_numbers:
        fc_weights[i] = add_gaussian_noise(fc_weights[i], mean=0.0, std=1.0)
        # fc_weights[i] = add_laplace_noise(fc_weights[i], loc=0.0, scale=1.0)
    model[6].weight.data = fc_weights

    # for i in sorted_numbers:
    #     fc2_weights[i] = 0
    # model[8].weight.data[1] = fc2_weights

    # for i in sorted_numbers:
    #     fc_weights[i] = torch.zeros(fc_weights[i].shape)
    # model[6].weight.data = fc_weights

    end_time = time.time()
    print(f'Time for {loop_time} times: {end_time - start_time}')

    import copy
    torch.save(copy.deepcopy(model), "/home/cwh/Workspace/TorchLRP-master/examples/models/mnist_unlearn_gn.pkl")

    # 在测试集上检验模型精度
    correct = 0
    total = 0

    # 将模型移动到GPU上
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            images = torch.squeeze(images, dim=2)
            # 将数据移动到GPU上
            images, labels = images.to('cuda'), labels.to('cuda')
            labels = torch.squeeze(labels, 1)
            
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += len(labels)

            correct += torch.sum(predicted == labels).item()
    print('Total Accuracy on filter images: %d %%' % (100*correct / total))

    correct = 0
    total = 0

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

    print('Total Accuracy on test images: %d %%' % (100 * correct / total))

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

    unlearn_times = 0
    filtered_data = []
    for data, labels in train_loader:
        if unlearn_times > 2:
            break
        else:
            unlearn_times += 1
            filtered_data = []
        
        #将数据添加到filtered_data中
        if (labels != 10).any() and len(filtered_data) < 1:  # 如果labels是tensor
            filtered_data.append((data, labels))
            print(len(filtered_data))
            filtered_dataloader = DataLoader(filtered_data, batch_size=32, shuffle=True)
            print(len(filtered_dataloader))
            print("label is: ", labels)

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

            compute_and_plot_explanation("epsilon", train_loader, filtered_dataloader, model)
            # compute_and_plot_explanation("gamma+epsilon", test_loader, filtered_dataloader, model)
            # compute_and_plot_explanation("alpha1beta0", test_loader, filtered_dataloader, model)
            # compute_and_plot_explanation("alpha2beta1", test_loader, filtered_dataloader, model)
        
            # compute_and_plot_explanation("patternnet", test_loader, filtered_dataloader, model, pattern=patterns_all, title="PatternNet $S(x)$", cmap='gray')
            # compute_and_plot_explanation("patternnet", test_loader, filtered_dataloader, model, pattern=patterns_pos, title="PatternNet $S(x)_{+-}$", cmap='gray')

            # compute_and_plot_explanation("patternattribution", test_loader, filtered_dataloader, model, pattern=patterns_all, title="PatternAttribution $S(x)$")
            # compute_and_plot_explanation("patternattribution", test_loader, filtered_dataloader, model, pattern=patterns_pos, title="PatternAttribution $S(x)_{+-}$")
        print(unlearn_times)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("MNIST LRP Example")
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--train_new', action='store_true', help='Train new predictive model')
    parser.add_argument('--epochs', '-e', type=int, default=5)
    parser.add_argument('--seed', '-d', type=int, default=None)

    args = parser.parse_args()

    if args.seed is None: 
        args.seed = int(random.random() * 1e9)
        # args.seed = 735537353
        print("Setting seed: %i" % args.seed)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    main(args)
