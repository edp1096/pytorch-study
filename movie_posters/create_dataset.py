import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import movie_poster

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


data_dir = "datas"
img_dir = f"{data_dir}/images"

batch_size = 32
test_size = 0.15

all_dataset = pd.read_csv(f"{data_dir}/all_data.csv")
train_set, test_set = train_test_split(all_dataset, test_size=0.15)

pd.DataFrame(train_set).to_csv(f"{data_dir}/train_data.csv", header=True, index=False)
pd.DataFrame(test_set).to_csv(f"{data_dir}/test_data.csv", header=True, index=False)

test_set_fpath = f"{data_dir}/test_data.csv"

test_transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
    ]
)

test_dataset = movie_poster.Dataset(test_set_fpath, img_dir, test_transform)
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
