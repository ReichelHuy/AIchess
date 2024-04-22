# Pytorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
from keras.applications import EfficientNetB7
from efficientnet_pytorch import EfficientNet
import torch.optim as optim


import time
# The usual
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt     # For plotting graphs
import os
from PIL import Image   # For importing images

root = "Data"

# Transformation for all the images in the training set
train_transform = transforms.Compose([
    # Play with the data a bit
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    # Structure the data
    transforms.Resize((224)),     # Resizing to roughly the mean of the dataset / 4
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])       # Common normalisation values
    #transforms.Normalize(mean=[0.6772, 0.6613, 0.6426], std=[0.2391, 0.2456, 0.2503])   # Data set specific values
])

# For evaluating the model
test_transform = transforms.Compose([
    transforms.Resize(224),     # Resizing to roughly the mean of the dataset
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])       # Common normalisation values
    transforms.Normalize(mean=[0.5232, 0.5575, 0.5076], std=[0.1557, 0.1391, 0.1305])   # Data set specific values
])


# Load training data
train_data = datasets.ImageFolder(os.path.join(root, 'train'), train_transform)
train_loader = DataLoader(train_data, batch_size=10, shuffle=True)

# Load test data
test_data = datasets.ImageFolder(os.path.join(root, 'test'), test_transform)
test_loader = DataLoader(train_data, batch_size=10, shuffle=False)

class_names = train_data.classes

print("Classes: ", class_names)
print("Training dataset size: ", len(train_data))
print("Test dataset size: ", len(test_data))


# Have a peek at the first batch
for images, labels in train_loader:
    break

print("Label:\t", labels.numpy())
print("Class:\t", *np.array([class_names[i] for i in labels]))
print("Image data dimensions: ", images.shape)  # 10(batch size) x3 (channels)x 224(w)x 224(h). torch.Size([10, 3, 224, 224])

# im = make_grid(images, nrow=5)
# plt.figure(figsize=(12, 4))
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Chọn thiết bị GPU nếu có

model = EfficientNet.from_name('efficientnet-b7', num_classes=len(class_names))
model = model.to(device)
# Định nghĩa hàm mất mát và bộ tối ưu hóa
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Số epoch huấn luyện
num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs.data, 1)
        total_predictions += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
        # Tính toán accuracy cho từng batch
        batch_accuracy = (predicted == labels).sum().item() / labels.size(0)
        
        print(f"Batch [{batch_idx+1}/{len(train_loader)}] | Loss: {loss.item():.4f} | Accuracy: {batch_accuracy*100:.2f}%")
    
    epoch_loss = running_loss / len(train_data)
    epoch_accuracy = correct_predictions / total_predictions
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Loss: {epoch_loss:.4f} | Accuracy: {epoch_accuracy*100:.2f}%")

# Lưu trọng số của mô hình
torch.save(model.state_dict(), "chessModelEfficientNetB7.pt")