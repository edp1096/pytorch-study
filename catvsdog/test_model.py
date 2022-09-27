import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.io import ImageReadMode
from torchvision.transforms import ToTensor

import modules.util as util


use_torchvision_dataset = False
model_fname = "model_resnet.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

batch_size = 64


dataset_test = datasets.ImageFolder("datas/test", transform=ToTensor())
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)


model = models.resnet18()
features_count = model.fc.in_features
model.fc = nn.Linear(features_count, 2)

weights = torch.load(model_fname)
model.load_state_dict(weights)


classes = ["ant", "bee"]
images, labels = [], []

model.eval()

for image, label in loader_test.dataset:
    pred = model(image.unsqueeze(0))
    predicted, actual = classes[pred[0].argmax(0)], classes[label]

    images += [image]
    labels += [actual + "/" + predicted]

    print(f'Predicted: "{predicted}", Actual: "{actual}"')

util.imshow(images, labels)
