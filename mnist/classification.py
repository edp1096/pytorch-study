import torch
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, ConcatDataset
import torch.nn as nn

import modules.dataset as dset
import modules.network as net

import matplotlib.pyplot as plt
import random
from glob import glob


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# random.seed(777)
# torch.manual_seed(777)
# if device == "cuda":
#     torch.cuda.manual_seed_all(777)


train_files = []
test_files = []
for i in range(0, 9):
    train_files.append(glob(f"datas/train/{i}/*.jpg"))
    test_files.append(glob(f"datas/test/{i}/*.jpg"))

learning_rate = 0.001
# epochs = 15
epochs = 3
batch_size = 100

# Custom Dataset
# train_transform = transforms.Compose([transforms.ToTensor()])
# test_transform = train_transform

# train_sets = []
# test_sets = []
# for i in range(0, 9):
#     train_sets.append(dset.MNIST(i, train_files[i], transform=train_transform))
#     test_sets.append(dset.MNIST(i, test_files[i], transform=test_transform))

# train_set = ConcatDataset(train_sets)
# test_set = ConcatDataset(test_sets)

# Torchvision Dataset
train_set = datasets.MNIST(root="data/MNIST_data/", train=True, transform=transforms.ToTensor(), download=True)
test_set = datasets.MNIST(root="data/MNIST_data/", train=False, transform=transforms.ToTensor(), download=True)

train_loader = DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True, drop_last=True)

# Hidden layer 1 : 32
# Hidden layer 2 : 32
# model = net.CNN1().to(device)
model = net.CNN2().to(device)
# model = net.CNN3().to(device)

# from torchsummary import summary
# summary(model, input_size=(1, 28, 28))

# criterion = nn.NLLLoss().to(device)
criterion = nn.CrossEntropyLoss().to(device)  # 비용 함수에 소프트맥스 함수 포함되어져 있음
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_batch = len(train_loader)
print("총 배치의 수 : {}".format(total_batch))

# 훈련 시작
for epoch in range(epochs):
    avg_cost = 0

    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()
        # hyp_x = model(X)  # CNN1
        hyp_x = model(X.float())  # CNN1, CNN2
        cost = criterion(hyp_x, Y.squeeze())
        cost.backward()
        optimizer.step()

        avg_cost += cost / total_batch

    print("[Epoch: {:>4}] cost = {:>.9}".format(epoch + 1, avg_cost))


# MNIST data image of shape 28 * 28 = 784
# Input : 28 * 28 = 784
# Output : 10 (0 ~ 9 digits)
# hypothesis = nn.Softmax()
# hypothesis = nn.Sigmoid()
hypothesis = nn.Linear(784, 10, bias=True)
hypothesis = hypothesis.to(device)


with torch.no_grad():
    for X_test, Y_test in test_loader:
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)

        # pred = model(X_test)
        pred = model(X_test.float())
        correct_prediction = torch.argmax(pred, 1) == Y_test
        accuracy = correct_prediction.float().mean()

    print("Accuracy:", accuracy.item())

    r = random.randint(0, len(test_loader.dataset) - 1)
    print("Random no:", r)

    X_single_data = test_loader.dataset[r][0].view(-1, 28 * 28).float().to(device)
    Y_single_data = test_loader.dataset[r][1]
    if type(Y_single_data) == "torch.Tensor":
        Y_single_data = Y_single_data.item()[0]
    print("Label: ", Y_single_data)

    single_prediction = hypothesis(X_single_data)
    print(single_prediction)
    print("Prediction: ", torch.argmax(single_prediction, 1).item())

    plt.imshow(test_loader.dataset[r][0].view(28, 28), cmap="Greys", interpolation="nearest")
    plt.show()
