import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor

import modules.dataset as dset
import modules.network as net

import matplotlib.pyplot as plt
import random


use_torchvision_dataset = False
model_fname = "model_mnist.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)


if use_torchvision_dataset:
    test_data = datasets.MNIST(root="data/MNIST_data/", train=False, download=True, transform=ToTensor())
else:
    test_data = dset.getTestDataset(9, transform=ToTensor())
    test_data = dset.prepareCustomDataset(9, "datas/test", transforms.ToTensor(), transforms.ToTensor())

model = nn.Linear(784, 10, bias=True).to(device)
# model = net.CNN2().to(device)
model.load_state_dict(torch.load(model_fname))
model.eval()

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
r = random.randint(0, len(test_data) - 1)
image, label = test_data[r][0].to(device), test_data[r][1]

# with torch.no_grad():
if use_torchvision_dataset:
    pred = model(image.unsqueeze(dim=0))
else:
    pred = model(image.float().unsqueeze(dim=0))

predicted, actual = classes[pred[0].argmax(0)], classes[label]

print(pred)
print(f'Index: "{r}", Predicted: "{predicted}", Actual: "{actual}"')

plt.imshow(image.view(28, 28).cpu().numpy(), cmap="Greys", interpolation="nearest")
plt.show()
