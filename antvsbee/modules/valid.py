import torch

def run(device, dataloader, model, loss_fn):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    valid_loss, correct = 0, 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)

            pred = model(X.float())
            valid_loss += loss_fn(pred, y).item()

            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    valid_loss /= num_batches
    correct /= size
    print(f"Test error: \n Accuracy: {(100*correct):>.1f}%, Avg loss: {valid_loss:>8f} \n")
