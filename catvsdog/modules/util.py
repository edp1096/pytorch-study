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

    fig = plt.figure(figsize=(8, 7))

    for i in range(count):
        inp = images[i].numpy().transpose((1, 2, 0))
        inp = np.clip(inp, 0, 1)

        col = 5
        row = int(np.ceil(count / col))

        # sp = plt.subplot(row, col, i + 1)
        sp = fig.add_subplot(row, col, i + 1)
        sp.axis("Off")

        plt.title(predicts[i])

        sp.imshow(inp)

    plt.show()
