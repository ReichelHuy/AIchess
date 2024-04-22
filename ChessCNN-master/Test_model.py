# Pytorch stuff
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
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

# print("Classes: ", class_names)
# print("Training dataset size: ", len(train_data))
# print("Test dataset size: ", len(test_data))


# Have a peek at the first batch
for images, labels in train_loader:
    break

# print("Label:\t", labels.numpy())
# print("Class:\t", *np.array([class_names[i] for i in labels]))
# print("Image data dimensions: ", images.shape)  # 10(batch size) x3 (channels)x 224(w)x 224(h)

#im = make_grid(images, nrow=5)
#plt.figure(figsize=(12, 4))
#plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
#plt.show()

class ConvolutionalNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, 1)  # 3 input channels, 6 output filters, 3x3 kernel, stride of 1
        self.conv2 = nn.Conv2d(6, 16, 3, 1)
        self.fc1 = nn.Linear(54*54*16, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 13) # output 13

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


CNNmodel = ConvolutionalNetwork()  # Create an instance of the model
CNNmodel.load_state_dict(torch.load("chessModelCNN.pt"))
num_classes = CNNmodel.fc3.out_features
class_names = train_data.classes
# print("Number of classes: ", num_classes)
# print("Class names: ", class_names)

image_path = 'Data/test/wk/0761_60.jpg'
def predict_picture(image):
    # Preprocess the image
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    input_image = transform(image)
    input_image = torch.unsqueeze(input_image, 0)
    # Make a forward pass
    CNNmodel.eval()
    with torch.no_grad():
        outputs = CNNmodel(input_image)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
     
    # Top 1
    label = class_names[predicted.item()]
    probability = probabilities[predicted.item()].item()
    return label, probability
    
    # all
    '''
    results = []
    for i, probability in enumerate(probabilities):
        label = class_names[i]
        results.append((label, probability.item()))
    return results
    '''


'''
#Top 1
image = Image.open(image_path)
label, probability = predict_picture(image)
print("Label: {}, Probability: {:.2f}".format(label, probability))
result = predict_picture(image)
'''


'''
for label, probability in result:
    print("Label: {}, Probability: {:.2f}".format(label, probability))
'''
    
   
