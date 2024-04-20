# Pytorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid

# The usual
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt     # For plotting graphs
import os
from PIL import Image   # For importing images

'''
# Let's have a look at the data
# Get file names
path = "Data"
img_names = []
for folder, subfolders, filenames in os.walk(path):
    for img in filenames:
        img_names.append(folder + '/' + img)

# Get image dimensions
img_sizes = []
for item in img_names:
    with Image.open(item) as img:
        img_sizes.append(img.size)

df = pd.DataFrame(img_sizes)

print(f'Number of images: {len(img_names)}')
print(df[0].describe())
print(df[1].describe())


# Check out one of the images
knight = Image.open("Data/Knight/00000032.jpg")     # Open image

print(knight.size)
print(knight.getpixel((0, 0)))

# Transform into a tensor
transform = transforms.Compose(
    [transforms.ToTensor()]
)

im = transform(knight)
print(type(im))
print(im.shape)
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
plt.show()
'''

root = "Data"

# Transformation for all the images in the training set
train_transform = transforms.Compose([
    # Play with the data a bit
    #transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    # Structure the data
    transforms.Resize((224)),     # Resizing to roughly the mean of the dataset / 4
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])       # Common normalisation values
    transforms.Normalize(mean=[0.6772, 0.6613, 0.6426], std=[0.2391, 0.2456, 0.2503])   # Data set specific values
])

# For evaluating the model
test_transform = transforms.Compose([
    transforms.Resize(224),     # Resizing to roughly the mean of the dataset
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    #transforms.Normalize(mean=[.485, .456, .406], std=[.229, .224, .225])       # Common normalisation values
    transforms.Normalize(mean=[0.6772, 0.6613, 0.6426], std=[0.2391, 0.2456, 0.2503])   # Data set specific values
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

print("Image data dimensions: ", images.shape)  # 10(batch size)x3(channels)x800(w)x800(h)

im = make_grid(images, nrow=5)
plt.figure(figsize=(12, 4))
plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
#plt.show()

# Make the CNN
'''
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, 1, 1)  # 3 input channels, 4 output filters, 3x3 kernel, stride of 1
        self.conv2 = nn.Conv2d(4, 8, 3, 1, 1)
        self.conv3 = nn.Conv2d(8, 16, 3, 1, 1)
        self.conv4 = nn.Conv2d(16, 32, 3, 1, 1)
        self.fc1 = nn.Linear(12*12*32, 128) # Will received the flattened image
        self.fc2 = nn.Linear(128, 16)
        self.fc3 = nn.Linear(16, 6)
    
    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv3(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv4(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 12*12*32)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)
'''
class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 6)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.max_pool2d(X, 2, 2)
        X = F.relu(self.conv2(X))
        X = F.max_pool2d(X, 2, 2)
        X = X.view(-1, 54*54*16)
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = self.fc3(X)
        return F.log_softmax(X, dim=1)

CNNmodel = ConvolutionalNetwork()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(CNNmodel.parameters(), lr=0.0001)

print(CNNmodel)
print("Parameter numbers:")
for p in CNNmodel.parameters():
    print(p.numel())

import time
start_time = time.time()

epochs =20

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    print("Epoch: ", i)
    train_correct_epoch = 0
    test_correct_epoch = 0

    for b, (X_train, y_train) in enumerate(train_loader):
        b += 1
        y_pred = CNNmodel(X_train)
        loss = criterion(y_pred, y_train)

        predicted = torch.max(y_pred.data, 1)[1]
        batch_correct = (predicted.numpy() == y_train.numpy()).sum()

        train_correct_epoch += batch_correct

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if b % 10 == 0:
            print("Batch: ", b, "\tLoss: ", loss.item())
    
    train_losses.append(loss)
    train_correct.append(train_correct_epoch)

    # Test set check during training

    with torch.no_grad():
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = CNNmodel(X_test)
            predicted = torch.max(y_val.data, 1)[1]
            test_correct_epoch += (predicted.numpy() == y_test.numpy()).sum()

    # Update loss
    loss = criterion(y_val, y_test)
    test_losses.append(loss)
    test_correct.append(test_correct_epoch)

total_time = time.time() - start_time

print("Finished in: ", total_time / 60, " minutes")
print(test_correct)
print("Test correct: ", (test_correct[-1].item() /3475) * 100, "%")

torch.save(CNNmodel.state_dict(), "chessModel.pt")
"""
plt.clf()
plt.plot(train_losses, label="Training loss")
plt.plot(test_losses, label="Validation loss")
plt.title("Loss over batches")
plt.legend()
plt.show()

plt.clf()
plt.plot([t/80 for t in train_correct], label='training accuracy')
plt.plot([t/30 for t in test_correct], label='validation accuracy')
plt.title('Accuracy at the end of each epoch')
plt.legend();
"""