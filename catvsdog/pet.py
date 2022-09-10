import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

import pandas as pd
import numpy as np


class Dataset(Dataset):
    def __init__(self, img_dir, files, mode="train", transform=None):
        self.files = files
        self.img_dir = img_dir
        self.mode = mode
        self.transform = transform

        self.label = 0  # 0: cat
        if "dog" in files[0]:
            self.label = 1

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.files[idx]}"
        img = read_image(img_path)

        if self.transform:
            img = self.transform(img)

        if self.mode == "train":
            return img, torch.tensor([self.label])
        else:
            return img, self.files[idx]
