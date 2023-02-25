import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torchvision.io import ImageReadMode
from torchvision.transforms import ToTensor

import modules.dataset as dset
import modules.network as net

import matplotlib.pyplot as plt
import random


use_torchvision_dataset = False
model_fname = "model_mnist.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# random.seed(777)
# torch.manual_seed(777)
# if device == "cuda":
#     torch.cuda.manual_seed_all(777)


test_transform = transforms.Compose(
    [
        transforms.Normalize(mean=(0.5), std=(0.5)),
        transforms.ToTensor(),
    ]
)

if use_torchvision_dataset:
    test_data = datasets.MNIST(root="data/MNIST_data/", train=False, download=True, transform=test_transform)
else:
    test_data = dset.prepareCustomDataset(9, "datas/test", test_transform)

# model = nn.Linear(784, 10, bias=True) # linear
# model = net.CNN2()  # cnn

# vgg
# model = models.vgg11()
# model.features[0] = nn.Conv2d(1, 64, kernel_size=(3, 3), stride=(1, 1), padding=(3, 3), bias=False)
# model.classifier[6] = nn.Linear(4096, 10)

# resnet
model = models.resnet18()
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
model.fc = nn.Linear(512, 10)

# # mobilenet
# model = models.mobilenet_v2()
# model.features[0][0] = nn.Conv2d(1, 32, kernel_size=(7, 7), stride=(1, 1), padding=(3, 3), bias=False)
# model.classifier[1] = nn.Linear(1280, 10)

# # rnn, lstm
# input_size = 28
# hidden_size = 128
# layer_count = 4
# output_class_count = 10

# # model = net.RNN(device, input_size, hidden_size, layer_count, output_class_count)
# model = net.LSTM(device, input_size, hidden_size, layer_count, output_class_count)

model.to(device)
model.load_state_dict(torch.load(model_fname))
model.eval()

classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
r = random.randint(0, len(test_data) - 1)

# image, label = test_data[r][0].view(-1, 28 * 28), test_data[r][1]  # linear
image, label = test_data[r][0], test_data[r][1]


# 임의 숫자
from torchvision.io import read_image
fname, label = "5.jpg", 5
# fname, label = "6.jpg", 6
image = read_image(fname, ImageReadMode.GRAY)
image = transforms.Resize((28, 28))(image)


with torch.no_grad():
    if use_torchvision_dataset:
        pred = model(image.unsqueeze(dim=0)).to(device)
    else:
        # pred = model(image.float().to(device)) # linear
        pred = model(image.float().unsqueeze(dim=0).to(device)) # cnn
        # pred = model(image.float().to(device)) # rnn, lstm

predicted, actual = classes[pred[0].argmax(0).cpu().numpy()], classes[label]
probability = (255 - pred[0].cpu().numpy()[predicted]) / 255

print(pred, classes, pred[0].argmax(0).cpu().numpy())
print(
    f"Index: {r}, Probability(%): {probability * 100:.2f}%, Tensor: {pred[0].cpu().numpy()[predicted]:.2f}, Predicted: {predicted}, Actual: {actual}"
)

plt.imshow(image.view(28, 28).cpu().numpy(), cmap="Greys", interpolation="nearest")
plt.show()
