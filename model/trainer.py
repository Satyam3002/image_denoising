import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os
import numpy as np
import json
from torch.utils.data import random_split
from model.dataset import DenoisingDataset
from model.utils import Args, run

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 8
transform = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),  # Normalize between [-1,1]
    ]
)

full_dataset = DenoisingDataset(
    noisy_dir="/kaggle/input/image-dataset-collection/HR",
    clean_dir="/kaggle/input/image-dataset-collection/train",
    transform=transform,
    json_path="/kaggle/input/image-mapping/train_X4.json",
)

train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size

train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
train_loader_x = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader_x = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(len(train_loader_x))
print(len(val_loader_x))

if __name__ == "__main__":
    args = Args()
    print(args)
    run(args)
