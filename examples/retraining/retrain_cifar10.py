import torch
# from torch.utils.tensorboard.summary import image
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

# from torch.utils.tensorboard import SummaryWriter
# myWriter = SummaryWriter('./tensorboard/log/')

myTransforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])


#  load
# train_dataset = CustomCIFAR10(root='/home/cwh/Workspace/TorchLRP-master/data', train=True, download=True, transform=myTransforms)
train_dataset = torchvision.datasets.CIFAR10(root='/home/cwh/Workspace/TorchLRP-master/data', train=True, download=True, transform=myTransforms)
train_dataset = torch.utils.data.Subset(train_dataset, [i for i in range(len(train_dataset)) if train_dataset.targets[i] != 0])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

test_dataset = torchvision.datasets.CIFAR10(root='/home/cwh/Workspace/TorchLRP-master/data', train=False, download=True, transform=myTransforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=0)

# 定义模型
myModel = torchvision.models.resnet50(pretrained=True)
# 将原来的ResNet18的最后两层全连接层拿掉,替换成一个输出单元为10的全连接层
inchannel = myModel.fc.in_features
myModel.fc = nn.Linear(inchannel, 10)

# 损失函数及优化器
# GPU加速
myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
myModel = myModel.to(myDevice)


learning_rate=0.001
myOptimzier = optim.SGD(myModel.parameters(), lr = learning_rate, momentum=0.9)
myLoss = torch.nn.CrossEntropyLoss()

import time
total_time = 0.0

for _epoch in range(10):
    training_loss = 0.0
    start_time = time.time()
    for _step, input_data in enumerate(train_loader):
        images, label = input_data[0].to(myDevice), input_data[1].to(myDevice)   # GPU加速
        predict_label = myModel.forward(images)
       
        loss = myLoss(predict_label, label)

        # myWriter.add_scalar('training loss', loss, global_step = _epoch*len(train_loader) + _step)

        myOptimzier.zero_grad()
        loss.backward()
        myOptimzier.step()

        training_loss = training_loss + loss.item()
        if _step % 10 == 0 :
            print('[iteration - %3d] training loss: %.3f' % (_epoch*len(train_loader) + _step, training_loss/10))
            training_loss = 0.0
            print()
    end_time = time.time()

    total_time += (end_time - start_time)

    print("Epoch spent {}".format(end_time - start_time))
    print("Total time is {}".format(total_time))

    correct_0 = 0
    total_0 = 0
    correct_non_0 = 0
    total_non_0 = 0

    myModel.eval()
    for images, labels in test_loader:
        images = images.to(myDevice)
        labels = labels.to(myDevice)
        outputs = myModel(images)

        numbers, predicted = torch.max(outputs.data, 1)

        # 计算类别为0的数据精度
        mask_0 = (labels == 0)
        correct_0 += (predicted[mask_0] == labels[mask_0]).sum().item()
        total_0 += mask_0.sum().item()

        # 计算其他类别的数据精度
        mask_non_0 = (labels != 0)
        correct_non_0 += (predicted[mask_non_0] == labels[mask_non_0]).sum().item()
        total_non_0 += mask_non_0.sum().item()

    # 计算精度
    if total_0 > 0:
        accuracy_0 = 100 * correct_0 / total_0
    else:
        accuracy_0 = 0

    if total_non_0 > 0:
        accuracy_non_0 = 100 * correct_non_0 / total_non_0
    else:
        accuracy_non_0 = 0

    print('Testing Accuracy on Class 0: %.3f %%' % accuracy_0)
    print('Testing Accuracy on Other Classes: %.3f %%' % accuracy_non_0)
    # myWriter.add_scalar('test_Accuracy',100 * correct / total)

    if accuracy_non_0 >= 95.5:
        print("Retrain process spent {}".format(total_time))
        torch.save(myModel, "/home/cwh/Workspace/TorchLRP-master/examples/models/resnet50_cifar10_retrain.pth")
        break
