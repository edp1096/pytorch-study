import torch
from torch import nn
import torchvision
from torchvision import models, datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader, ConcatDataset
from torchinfo import summary

import nn_model.network as network
import nn_model.train as train
import nn_model.test as test
import pet
import fit

import pandas as pd
import matplotlib.pyplot as plt


img_train_dir = "datas/train"
img_valid_dir = "datas/valid"
img_test_dir = "datas/test"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


# batch_size = 32
batch_size = 64

dog_train_files = [f"dog.{i}.jpg" for i in range(10000)]
cat_train_files = [f"cat.{i}.jpg" for i in range(10000)]
dog_valid_files = [f"dog.{i+11250}.jpg" for i in range(1249)]
cat_valid_files = [f"cat.{i+11250}.jpg" for i in range(1249)]
test_files = [f"{i+1}.jpg" for i in range(2500)]

train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)


train_dog_dataset = pet.Dataset(img_train_dir, dog_train_files, "train", train_transform)
train_cat_dataset = pet.Dataset(img_train_dir, cat_train_files, "train", train_transform)
valid_dog_dataset = pet.Dataset(img_valid_dir, dog_valid_files, "train", train_transform)
valid_cat_dataset = pet.Dataset(img_valid_dir, cat_valid_files, "train", train_transform)

train_dataset = ConcatDataset([train_dog_dataset, train_cat_dataset])
valid_dataset = ConcatDataset([valid_dog_dataset, valid_cat_dataset])
test_dataset = pet.Dataset(img_test_dir, test_files, "", test_transform)


print(f"number of train dataset : {len(train_dataset)}")
print(f"number of valid dataset : {len(valid_dataset)}")
print(f"number of test dataset : {len(test_dataset)}")


train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)

model = torchvision.models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2, progress=True)

num_ftrs = model.fc.in_features
print(num_ftrs)
model.fc = nn.Sequential(
    nn.Dropout(0.5),
    nn.Linear(num_ftrs, 1024),
    nn.Dropout(0.2),
    nn.Linear(1024, 512),
    nn.Dropout(0.1),
    nn.Linear(512, 1),
    nn.Sigmoid(),
)

model.to(device)
# print(model)

# model.cuda()
# summary(model, input_size=(3,224,224))

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

trained_model = fit.run(device, model, criterion, optimizer, 10, train_dataloader, valid_dataloader)

# torch.save(trained_model.state_dict(), "model.pth")
torch.save(trained_model, "model.pth")
