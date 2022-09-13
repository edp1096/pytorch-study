import torch
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn
from torchsummary import summary

import modules.dataset as dset
import modules.network as net
import modules.fit as fit
import modules.valid as valid

import matplotlib.pyplot as plt
import random
from glob import glob


use_torchvision_dataset = False
model_fname = "model_mnist.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

random.seed(777)
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)


epochs = 15
batch_size = 100
learning_rate = 0.001

train_transform = transforms.Compose([transforms.ToTensor()])
valid_transform = train_transform

if use_torchvision_dataset:
    train_set, valid_set = dset.prepareTorchvisionDataset(train_transform, valid_transform)  # Torchvision Dataset
else:
    train_set = dset.prepareCustomDataset(9, "datas/train1", train_transform, valid_transform)  # Custom Dataset
    valid_set = dset.prepareCustomDataset(9, "datas/test", train_transform, valid_transform)

train_loader, valid_loader = dset.getDataLoaders(train_set, valid_set, batch_size, batch_size)


# criterion = nn.Linear(784, 10, bias=True)
# criterion = nn.NLLLoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

model = net.CNN2().to(device)
criterion = nn.CrossEntropyLoss().to(device)  # 비용 함수에 소프트맥스 함수 포함되어져 있음
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(train_loader)

summary(model, input_size=(1, 28, 28))  # 모델 정보 출력 (channels, height, width)
print("총 배치의 수 : {}".format(total_batch))

# 훈련 시작
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")

    fit.run(device, train_loader, model, criterion, optimizer)
    valid.run(device, valid_loader, model, criterion)

print("Training done!")

torch.save(model.state_dict(), model_fname)
print(f"Saved PyTorch Model State to {model_fname}")
