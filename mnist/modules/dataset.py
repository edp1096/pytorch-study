import torch
from torchvision.io import read_image
from torchvision import transforms, datasets
from torch.utils.data import Dataset, ConcatDataset

from glob import glob


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


def prepareCustomDatasets(last_num: int, train_transform=None, valid_transform=None):
    train_files = []
    valid_files = []
    for i in range(0, last_num):
        train_files.append(glob(f"datas/train/{i}/*.jpg"))
        valid_files.append(glob(f"datas/test/{i}/*.jpg"))

    train_sets = []
    valid_sets = []
    for i in range(0, last_num):
        train_sets.append(MNIST(i, train_files[i], transform=train_transform))
        valid_sets.append(MNIST(i, valid_files[i], transform=valid_transform))

    train_set = ConcatDataset(train_sets)
    valid_set = ConcatDataset(valid_sets)

    return train_set, valid_set


def prepareTorchvisionDatasets(train_transform=None, valid_transform=None):
    train_set = datasets.MNIST(root="data/MNIST_data/", train=True, transform=train_transform, download=True)
    valid_set = datasets.MNIST(root="data/MNIST_data/", train=False, transform=valid_transform, download=True)

    return train_set, valid_set


def getDataLoaders(train_set, test_set, batch_size_train, batch_size_test):
    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size_train, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size_test, shuffle=False)

    return train_loader, valid_loader


def getTestData(last_num=9, transform=None):
    test_files = []
    for i in range(0, last_num):
        test_files.append(glob(f"datas/test/{i}/*.jpg"))

    test_sets = []
    for i in range(0, last_num):
        test_sets.append(MNIST(i, test_files[i], transform=transform))

    test_set = ConcatDataset(test_sets)

    return test_set
