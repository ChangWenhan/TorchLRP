import copy
import time
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import random
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transformations for the data
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load the dataset
train_dataset = datasets.ImageFolder(
    root="/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/train",
    transform=transform
)

# Load the test dataset
test_dataset = datasets.ImageFolder(root="/home/cwh/Workspace/TorchLRP-master/torch_imagenet/imagenet-mini/train", loader=datasets.folder.default_loader, transform=transform)
# test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False)

def evaluate_model(model, dataloader):
    model.eval()
    total, correct_0, total_0, correct_other, total_other = 0, 0, 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct_0 += (predicted[labels == 0] == 0).sum().item()
            total_0 += (labels == 0).sum().item()
            correct_other += (predicted[labels != 0] == labels[labels != 0]).sum().item()
            total_other += (labels != 0).sum().item()
    
    accuracy_0 = correct_0 / total_0 if total_0 > 0 else 0
    accuracy_other = correct_other / total_other if total_other > 0 else 0
    return accuracy_0, accuracy_other

# Modify class 0 to a random other class
num_classes = len(train_dataset.classes)
for i in range(len(train_dataset.targets)):
    if train_dataset.targets[i] == 0:
        train_dataset.targets[i] = random.choice([x for x in range(num_classes) if x != 0])

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

vgg_num = int(sys.argv[1]) if len(sys.argv) > 1 else 16 # Default to vgg16

vgg = getattr(torchvision.models, "vgg%i"%vgg_num)(pretrained=True).to(device)
vgg.eval()

# Copy the original VGG16 model for evaluation purposes
# original_vgg = copy.deepcopy(vgg)

# Evaluate the model
# accuracy_0, accuracy_other = evaluate_model(vgg, test_loader)
# print(f"Accuracy on class 0: {accuracy_0 * 100:.2f}%")
# print(f"Accuracy on other classes: {accuracy_other * 100:.2f}%")

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(vgg.parameters(), lr=0.001, momentum=0.9)

# Training loop
num_epochs = 1
start_time = time.time()
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = vgg(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")
end_time = time.time()
print(f"Training time: {end_time - start_time} seconds")
print("Finished Training")

torch.save(copy.deepcopy(vgg), '/home/cwh/Workspace/TorchLRP-master/examples/models/imagenet_unlearn_random.pkl')

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    total, correct_0, total_0, correct_other, total_other = 0, 0, 0, 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct_0 += (predicted[labels == 0] == 0).sum().item()
            total_0 += (labels == 0).sum().item()
            correct_other += (predicted[labels != 0] == labels[labels != 0]).sum().item()
            total_other += (labels != 0).sum().item()
    
    accuracy_0 = correct_0 / total_0 if total_0 > 0 else 0
    accuracy_other = correct_other / total_other if total_other > 0 else 0
    return accuracy_0, accuracy_other

# Evaluate the model
accuracy_0, accuracy_other = evaluate_model(vgg, train_loader)
print(f"Accuracy on class 0: {accuracy_0 * 100:.2f}%")
print(f"Accuracy on other classes: {accuracy_other * 100:.2f}%")
