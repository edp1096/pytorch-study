import torch
import copy


def run(device, dataloader, model, loss_fn, optimizer, best_model_wts, best_acc):
    model.eval()

    dataset_size = len(dataloader.dataset)

    running_loss = 0.0
    running_corrects = 0

    optimizer.zero_grad()

    for image, label in dataloader:
        with torch.no_grad():
            image, label = image.to(device), label.to(device)

            output = model(image)
            _, pred = torch.max(output, 1)
            loss = loss_fn(output, label)

        # 통계
        running_loss += loss.item() * image.size(0)
        running_corrects += torch.sum(pred == label.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    print(f"Valid Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

    # 모델을 deep copy
    if epoch_acc > best_acc:
        best_acc = epoch_acc
        best_model_wts = copy.deepcopy(model.state_dict())

    return model, best_model_wts, best_acc
