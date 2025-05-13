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

import torchvision.transforms as transforms


class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, transform=None, json_path: str = None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.transform = transform
        with open(file=json_path, mode="r") as fp:
            self.image_files = json.load(fp=fp)
        self.resize_transform = transforms.Resize((256, 256))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # some minor tweaks
        noisy_image_path = os.path.join(
            self.noisy_dir, self.image_files[idx]["path_gt"]
        )
        clean_image_path = os.path.join(
            self.clean_dir, self.image_files[idx]["path_lq"].replace("X4/", "")
        )

        noisy_image = Image.open(noisy_image_path).convert("L")
        clean_image = Image.open(clean_image_path).convert("L")

        noisy_image = noisy_image.convert("RGB")
        clean_image = clean_image.convert("RGB")

        noisy_image = self.resize_transform(noisy_image)
        clean_image = self.resize_transform(clean_image)

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)
        return noisy_image, clean_image
