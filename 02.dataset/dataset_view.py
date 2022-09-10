import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt


test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())

labels_map = {
    0: "T-Shirt",
    1: "Trouser",
    2: "Pullover",
    3: "Dress",
    4: "Coat",
    5: "Sandal",
    6: "Shirt",
    7: "Sneaker",
    8: "Bag",
    9: "Ankle Boot",
}

figure = plt.figure()
cols, rows = 3, 3
range_end = cols * rows + 1

for i in range(1, range_end):
    sample_idx = torch.randint(len(test_data), size=(1,)).item()
    print(sample_idx)
    img, label = test_data[sample_idx]
    figure.add_subplot(rows, cols, i)

    plt.title(labels_map[label])
    plt.axis("off")
    plt.imshow(img.squeeze(), cmap="gray")

plt.show()
