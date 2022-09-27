import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import pet

import pandas as pd
import matplotlib.pyplot as plt


data_dir = "datas"
train_dir = f"{data_dir}/train"
test_dir = f"{data_dir}/test"

batch_size = 32
test_size = 0.15

dog_train_files = [f'dog.{i}.jpg' for i in range(12500)]
cat_train_files = [f'cat.{i}.jpg' for i in range(12500)]
dog_valid_files = [f'dog.{i}.jpg' for i in range(2500)]
cat_valid_files = [f'cat.{i}.jpg' for i in range(2500)]
dog_test_files = [f'dog.{i+2500}.jpg' for i in range(2500)]
cat_test_files = [f'cat.{i+2500}.jpg' for i in range(2500)]


test_set_fpath = f"{data_dir}/test_data.csv"

test_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
    ]
)

test_dataset = pet.Dataset(test_set_fpath, train_dir, test_transform)
test_dataloader = DataLoader(test_dataset, batch_size=4, shuffle=True)


images, labels = next(iter(test_dataloader))
output = torchvision.utils.make_grid(images)
print("labels", labels)

def imshow(inp):
    inp = inp.transpose((1, 2, 0))
    plt.imshow(inp)

# imshow(output.numpy())
# plt.show()

print("Done")
