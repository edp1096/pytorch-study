from torchinfo import summary

def printModelInfo(model, batch_size, loader):
    print(model)

    summary(model, input_size=(batch_size, 3, 7, 7))

    total_batch = len(loader)
    print("총 배치 수 : {}".format(total_batch))
