import torch
from torch import nn
import torchvision
from torchvision import models, datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, ConcatDataset

import pet

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def eval(device, model, criterion, test_loader):
    with torch.no_grad():
        model.eval()
        correct = 0
        losses = 0
        for test_x, test_y in test_loader:
            test_x, test_y = test_x.to(device), test_y.to(device).float()
            pred = model(test_x)
            loss = criterion(pred, test_y)

            y_pred = pred.cpu()
            y_pred[y_pred >= 0.5] = 1
            y_pred[y_pred < 0.5] = 0

            losses += loss.item()
            correct += y_pred.eq(test_y.cpu()).int().sum()
    print(f"eval loss: {losses/len(test_loader):.4f}, eval acc: {correct/len(test_loader.dataset)*100:.3f}%")


img_test_dir = "datas/test"
img_valid_dir = "datas/valid"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

batch_size = 64

dog_valid_files = [f"dog.{i+11250}.jpg" for i in range(1250)]
cat_valid_files = [f"cat.{i+11250}.jpg" for i in range(1250)]
test_files = [f"{i+1}.jpg" for i in range(2500)]

train_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        torchvision.transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
)

valid_dog_dataset = pet.Dataset(img_valid_dir, dog_valid_files, "train", train_transform)
valid_cat_dataset = pet.Dataset(img_valid_dir, cat_valid_files, "train", train_transform)

valid_dataset = ConcatDataset([valid_dog_dataset, valid_cat_dataset])
test_dataset = pet.Dataset(img_test_dir, test_files, "", test_transform)

valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

print(f"number of test dataset : {len(test_dataset)}")
print(f"number of valid dataset : {len(valid_dataset)}")


model = torch.load("model.pth")
model.to(device)

criterion = nn.BCELoss()

eval(device, model, criterion, valid_dataloader)


submit_files = [f"{i+1}.jpg" for i in range(12500)]
submit_dataset = pet.Dataset(submit_files, img_test_dir, mode="test", transform=test_transform)
submit_loader = DataLoader(submit_dataset, batch_size=32, shuffle=False)

samples, files = iter(test_dataloader).next()

fig = plt.figure()
for i in range(24):
    sp = fig.add_subplot(4, 6, i + 1)
    sp.set_title(files[i])
    sp.axis("off")
    sp.imshow(samples[i].numpy().transpose((1, 2, 0)))

plt.subplots_adjust(bottom=0.05, top=0.9, hspace=0)
plt.show()

def predict(model, data_loader):
    with torch.no_grad():
        model.eval()
        ret = None
        for img, files in data_loader:
            img = img.to(device)
            pred = model(img)

            if ret is None:
                ret = pred.cpu().numpy()
            else:
                ret = np.vstack([ret, pred.cpu().numpy()])
    return ret


pred = predict(model, test_dataloader)

sample_pred = pred[:24]
sample_pred[sample_pred >= 0.5] = 1
sample_pred[sample_pred < 0.5] = 0

imgs, files = iter(test_dataloader).next()
classes = {0: "cat", 1: "dog"}

fig = plt.figure()
for i in range(24):
    sp = fig.add_subplot(4, 6, i + 1)
    sp.set_title(classes[sample_pred[i][0]])
    sp.axis("off")
    sp.imshow(imgs[i].numpy().transpose((1, 2, 0)))

plt.subplots_adjust(bottom=0.05, top=0.9, hspace=0)
plt.show()

submission = pd.DataFrame(np.clip(pred, 1e-6, 1 - 1e-6), columns=["label"])
submission["id"] = submission.index + 1
submission = submission[["id", "label"]]
submission.head(10)
submission.to_csv("submission.csv", index=False)
