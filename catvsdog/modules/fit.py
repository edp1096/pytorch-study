import torch

def run(device, loader, model, loss_fn, optimizer, learning_rate_scheduler):
    model.train()

    dataset_size = len(loader.dataset)

    running_loss = 0.0
    running_corrects = 0

    for batch, (image, label) in enumerate(loader):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            # 예측 오류 계산
            output = model(image)
            _, pred = torch.max(output, 1)
            loss = loss_fn(output, label)

            # 역전파
            loss.backward()
            optimizer.step()
        
        # 통계
        running_loss += loss.item() * image.size(0)
        running_corrects += torch.sum(pred == label.data)

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size

    print(f'Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    learning_rate_scheduler.step()