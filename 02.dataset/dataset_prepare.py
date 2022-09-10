from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


batch_size = 64

test_data = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
test_dataloader = DataLoader(test_data, batch_size, shuffle=True)

# 이미지와 정답(label)을 표시합니다.
# train_features, train_labels = next(iter(test_dataloader))
test_features, test_labels = test_dataloader.__iter__().__next__()
print(f"Feature batch shape: {test_features.size()}")
print(f"Labels batch shape: {test_labels.size()}")

img = test_features[0].squeeze()
label = test_labels[0]
print(f"Label: {label}")
# plt.imshow(img, cmap="gray")
plt.imshow(img)
plt.show()
