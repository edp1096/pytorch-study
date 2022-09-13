import torch
from torchvision import transforms
import torch.nn as nn
from torchsummary import summary

import modules.dataset as dset
import modules.network as net
import modules.fit as fit
import modules.valid as valid

import random


model_fname = "model_mnist.pt"
model_fname_new = "model_mnist.pt"

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

train_set = dset.prepareCustomDataset(9, "datas/train2", train_transform, valid_transform)  # Custom Dataset
valid_set = dset.prepareCustomDataset(9, "datas/test", train_transform, valid_transform)
train_loader, valid_loader = dset.getDataLoaders(train_set, valid_set, batch_size, batch_size)

model = net.CNN2().to(device)
model.load_state_dict(torch.load(model_fname))
model.train()
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
total_batch = len(train_loader)

summary(model, input_size=(1, 28, 28))
print("총 배치의 수 : {}".format(total_batch))

# 훈련 시작
for epoch in range(epochs):
    print(f"Epoch {epoch+1}\n-------------------------------")

    fit.run(device, train_loader, model, criterion, optimizer)
    valid.run(device, valid_loader, model, criterion)

print("Training done!")

torch.save(model.state_dict(), model_fname_new)
print(f"Saved PyTorch Model State to {model_fname_new}")
