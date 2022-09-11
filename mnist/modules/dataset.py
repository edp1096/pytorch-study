import torch
from torchvision.io import read_image
from torch.utils.data import Dataset


class MNIST(Dataset):
    def __init__(self, label, files, transform=None):
        self.files = files
        self.label = label
        self.transform = transform

        # print(f"MNIST: {label} {len(files)}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image = read_image(f"{self.files[idx]}")
        # label = torch.tensor([self.label])
        label = self.label

        return image, label
