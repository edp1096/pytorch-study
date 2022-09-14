import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchinfo import summary
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor

import modules.dataset as dset
import modules.fit as fit
import modules.network as net
import modules.valid as valid

import random


use_torchvision_dataset = False
model_fname = "model_mnist.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

random.seed(777)
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)


# epochs = 5
epochs = 10
batch_size = 100
learning_rate = 0.001

train_transform = transforms.Compose([transforms.ToTensor()])
valid_transform = train_transform

if use_torchvision_dataset:
    train_set, valid_set = dset.prepareTorchvisionDataset(train_transform, valid_transform)  # Torchvision Dataset
else:
    train_set = dset.prepareCustomDataset(9, "datas/train", train_transform)  # Custom Dataset
    valid_set = dset.prepareCustomDataset(9, "datas/test", valid_transform)

train_loader, valid_loader = dset.getDataLoaders(train_set, valid_set, batch_size, batch_size)


# model = nn.Linear(784, 10, bias=True)  # linear
# model = net.CNN2()  # cnn

# vgg
# model = models.vgg11()
# model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False)
# model.classifier[6] = nn.Linear(4096, 10)

# resnet
# model = models.resnet18()
# model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
# model.fc = nn.Linear(512, 10)

# mobilenet
model = models.mobilenet_v2()
model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
model.classifier[1] = nn.Linear(1280, 10)

# # rnn, lstm
# input_size = 28
# hidden_size = 128
# layer_count = 4
# output_class_count = 10

# # model = net.RNN(device, input_size, hidden_size, layer_count, output_class_count)
# model = net.LSTM(device, input_size, hidden_size, layer_count, output_class_count)

model.to(device)
print(model)


# summary(model, input_size=(batch_size, 1, 28 * 28))  # linear. 모델 정보 출력 (channels, height, width)
summary(model, input_size=(batch_size, 1, 28, 28))  # cnn
# summary(model, input_size=(batch_size, 1, input_size))  # rnn, lstm

total_batch = len(train_loader)
print("총 배치의 수 : {}".format(total_batch))

criterion = nn.CrossEntropyLoss().to(device)  # 비용 함수에 소프트맥스 함수 포함되어져 있음
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


# 훈련 시작
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")

    # linear
    # fit.run(device, train_loader, model, criterion, optimizer)
    # valid.run(device, valid_loader, model, criterion)

    # cnn
    fit.runCNN(device, train_loader, model, criterion, optimizer)
    valid.runCNN(device, valid_loader, model, criterion)

    # rnn, lstm
    # fit.runRNN(device, train_loader, model, criterion, optimizer)
    # valid.runRNN(device, valid_loader, model, criterion)

print("Training done!")

torch.save(model.state_dict(), model_fname)
print(f"Saved PyTorch Model State to {model_fname}")
