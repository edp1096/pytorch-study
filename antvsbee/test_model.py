import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.io import ImageReadMode
from torchvision.transforms import ToTensor

import modules.util as util


use_torchvision_dataset = False
model_fname = "model_resnet.pt"

transform = transforms.Compose(
    [
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cpu")
print("Device:", device)

batch_size = 64


dataset_test = datasets.ImageFolder("datas/test", transform=transform)
loader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)


# resnet
model = models.resnet18()
features_count = model.fc.in_features
model.fc = nn.Linear(features_count, 2)

# mobilenet
# model = models.mobilenet_v2()
# features_count = model.classifier[1].in_features
# model.classifier[1] = nn.Linear(features_count, 2)

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
