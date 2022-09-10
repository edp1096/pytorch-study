import torch
from torch import nn
from torchvision import datasets
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

import nn_model.network as network
import nn_model.train as train
import nn_model.test as test


device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")


batch_size = 64

training_data = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=4)


model = network.NeuralNetwork().to(device)
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")

    train.run(device, train_dataloader, model, loss_fn, optimizer)
    test.run(device, test_dataloader, model, loss_fn)

print("Training done!")


torch.save(model.state_dict(), "model.pth")
print("Saved PyTorch Model State to model.pth")
