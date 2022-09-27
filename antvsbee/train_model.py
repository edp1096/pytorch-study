import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.utils
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, models, transforms

import modules.fit as fit
import modules.util as util
import modules.valid as valid

import copy
import matplotlib.pyplot as plt
import numpy as np
import random


model_fname = "model_resnet.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

random.seed(777)
torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)


cudnn.benchmark = True

epochs = 24
batch_size = 64
learning_rate = 0.002
sgd_momentum = 0.9

transform = {}
transform["train"] = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)
transform["valid"] = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

dataset = {x: datasets.ImageFolder(f"datas/{x}", transform[x]) for x in ["train", "valid"]}
loaders = {x: DataLoader(dataset[x], batch_size=batch_size, shuffle=True) for x in ["train", "valid"]}

dataset_sizes = {x: len(dataset[x]) for x in ["train", "valid"]}
class_names = dataset["train"].classes


def imshow(images, classes, class_names):
    for i in range(4):
        inp = images[i].numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)

        sp = plt.subplot(1, 4, i + 1)
        sp.axis("Off")

        plt.title(class_names[classes[i]])

        sp.imshow(inp)

    plt.show()


# images, classes = next(iter(loaders["train"]))
# imshow(images, classes, class_names)

# resnet
# model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

# mobilenet
model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)


for param in model.parameters():
    param.requires_grad = False


# resnet
# features_count = model.fc.in_features
# model.fc = nn.Linear(features_count, 2)

# mobilenet
features_count = model.classifier[1].in_features
model.classifier[1] = nn.Linear(features_count, 2)

model.to(device)
util.printModelInfo(model, batch_size, loaders["train"])

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=sgd_momentum)
# optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Decay Learning-Rate by a factor of 0.1 every 7 epochs
learning_rate_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# 훈련 시작
best_model_wts = copy.deepcopy(model.state_dict())
best_acc = 0.0

for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")

    # cnn
    fit.run(device, loaders["train"], model, criterion, optimizer, learning_rate_scheduler)
    model, best_model_wts, best_acc = valid.run(device, loaders["valid"], model, criterion, optimizer, best_model_wts, best_acc)

    # 가장 나은 모델 가중치를 불러옴
    model.load_state_dict(best_model_wts)

print("Training done")

torch.save(model.state_dict(), model_fname)
print(f"Saved PyTorch Model State to {model_fname}")
