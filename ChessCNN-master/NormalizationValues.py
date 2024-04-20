# Modified code for getting the mean/std of a dataset original code by ptrblock on https://discuss.pytorch.org/t/about-normalization-using-pre-trained-vgg16-networks/23560/6

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid
import numpy as np
import os

if __name__ == '__main__':    
    root = "Data"

    train_transform = transforms.Compose([
        # Play with the data a bit
        #transforms.RandomRotation(15),
        transforms.RandomHorizontalFlip(),
        # Structure the data
        transforms.Resize((200, 200)),     # Resizing to roughly the mean of the dataset / 4
        transforms.CenterCrop(200),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(os.path.join(root, 'train'), train_transform)
    loader = DataLoader(
        dataset,
        batch_size=10,
        num_workers=1,
        shuffle=False
    )


    mean = 0.
    std = 0.
    nb_samples = 0.
    for data, _ in loader:
        batch_samples = data.size(0)
        data = data.view(batch_samples, data.size(1), -1)
        mean += data.mean(2).sum(0)
        std += data.std(2).sum(0)
        nb_samples += batch_samples

    mean /= nb_samples
    std /= nb_samples

    print(mean, " ", std)