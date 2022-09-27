from torchinfo import summary

import matplotlib.pyplot as plt
import numpy as np

def printModelInfo(model, batch_size, loader):
    print(model)

    summary(model, input_size=(batch_size, 3, 7, 7))

    total_batch = len(loader)
    print("Batch count : {}".format(total_batch))


def imshow(images, predicts):
    count = len(images)

    for i in range(count):
        inp = images[i].numpy().transpose((1, 2, 0))

        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        inp = std * inp + mean
        inp = np.clip(inp, 0, 1)

        sp = plt.subplot(1, count, i + 1)
        sp.axis("Off")

        plt.title(predicts[i])

        sp.imshow(inp)

    plt.show()

