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


# Let's have a look at the data
# Get file names
path = "Data"
img_names = []
for folder, subfolders, filenames in os.walk(path):
    for img in filenames:
        img_names.append(folder + '/' + img)
        
valid_img_names = []
for item in img_names:
    if item.lower().endswith(('.jpg', '.jpeg')):
        valid_img_names.append(item)


img_names = valid_img_names.copy() 

# Get image dimensions
img_sizes = []
for item in img_names:
    with Image.open(item) as img:
        img_sizes.append(img.size)

df = pd.DataFrame(img_sizes)

print()
print(f'Number of images: {len(img_names)}')
print(df[0].describe())
print(df[1].describe())
print(df.head())


# Check out one of the images
knight = Image.open("Data/test/wk/0760_60.jpg")     # Open image

print(knight.size)
print(knight.getpixel((0, 0)))

# Transform into a tensor
transform = transforms.Compose(
    [transforms.ToTensor()]
)

im = transform(knight)
print(type(im))
print(im.shape)
# plt.imshow(np.transpose(im.numpy(), (1, 2, 0)))
# plt.show()
