import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset
from torchinfo import summary
from torchvision import datasets, models, transforms
from torchvision.transforms import ToTensor

import modules.fit as fit
import modules.network as net
import modules.valid as valid

import matplotlib.pyplot as plt
import random


cudnn.benchmark = True
plt.ion()  # 대화형 모드

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

random.seed(777)
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)


epochs = 5
batch_size = 100
learning_rate = 0.001

workers = 4

transform = {
    "train": transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
    "val": transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    ),
}


dataset = {x: datasets.ImageFolder(f"datas/{x}", transform[x]) for x in ["train", "val"]}
loaders = {
    x: DataLoader(dataset[x], batch_size=batch_size, shuffle=True, num_workers=workers)
    for x in ["train", "val"]
}

dataset_sizes = {x: len(dataset[x]) for x in ["train", "val"]}
class_names = dataset["train"].classes


# resnet
model = models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
model.fc = nn.Linear(512, 10)

model.to(device)
print(model)

exit()

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
