import os
import sys
import copy
import time
import random
import pathlib
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST
from torch.nn import Sequential, Conv2d, MaxPool2d, Linear
from collections import Counter
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
# from utils import get_mnist_model, prepare_mnist_model, get_mnist_data
from utils import store_patterns, load_patterns

from visualization import heatmap_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the model
def get_mnist_model():
    model = Sequential(
        Conv2d(1, 32, 3, 1, 1),
        nn.ReLU(),
        Conv2d(32, 64, 3, 1, 1),
        nn.ReLU(),
        MaxPool2d(2, 2),
        nn.Flatten(),
        Linear(14*14*64, 512),
        nn.ReLU(),
        Linear(512, 10)
    )
    return model

# Transformations for the data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Load the MNIST dataset
train_dataset = MNIST(root='/home/cwh/Workspace/TorchLRP-master/data', train=True, download=True, transform=transform)
test_dataset = MNIST(root='/home/cwh/Workspace/TorchLRP-master/data', train=False, download=True, transform=transform)

# Remove the first three data samples from the training dataset
removed_samples = [train_dataset[i] for i in range(3)]
train_dataset = Subset(train_dataset, list(range(3, len(train_dataset))))

# Data loaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
removed_loader = DataLoader(removed_samples, batch_size=3, shuffle=False)

# Initialize the model, loss function and optimizer
model = get_mnist_model().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# Training loop
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        print(inputs.shape)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}")

print("Finished Training")

# Evaluation function
def evaluate_model(model, dataloader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy

# Evaluate the model on the training dataset
train_accuracy = evaluate_model(model, test_loader)
print(f"Training Set Accuracy: {train_accuracy * 100:.2f}%")

# Evaluate the model on the removed samples
removed_accuracy = evaluate_model(model, removed_loader)
print(f"Removed Samples Accuracy: {removed_accuracy * 100:.2f}%")