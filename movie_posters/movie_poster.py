import torch
from torchvision.io import read_image
from torch.utils.data import Dataset

import pandas as pd
import numpy as np


class Dataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.csv = pd.read_csv(annotations_file, header=0)
        self.img_dir = img_dir

        self.transform = transform
        self.labels = self.csv.drop(["Id", "Genre"], axis="columns")

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        img_path = f"{self.img_dir}/{self.csv.iloc[idx, 0]}.jpg"
        image = read_image(img_path)

        labels = torch.tensor(self.labels.iloc[idx].astype(float))

        if self.transform:
            image = self.transform(image)

        # result = {'image': image,'label':labels}
        # return result

        return image, labels
