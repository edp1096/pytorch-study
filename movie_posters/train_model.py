import torch
from torch import nn
import torchvision
from torchvision import models, datasets
from torchvision.transforms import ToTensor
import torchvision.transforms as transforms
from torchvision.io import read_image
from torch.utils.data import DataLoader

import nn_model.network as network
import nn_model.fit as fit
import nn_model.test as test
import movie_poster

import pandas as pd
import matplotlib.pyplot as plt
import gc


gc.collect()
torch.cuda.empty_cache()


data_dir = "datas"
img_dir = f"{data_dir}/images"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


batch_size = 32

train_csv_fpath = f"{data_dir}/train_data.csv"
train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
    ]
)

train_dataset = movie_poster.Dataset(train_csv_fpath, img_dir, train_transform)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# model = network.NeuralNetwork().to(device)
model = torchvision.models.vgg16().to(device)

print(model)

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")

    fit.run(device, train_dataloader, model, loss_fn, optimizer)

print("Training done!")


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
