def run(device, loader, model, loss_fn, optimizer, learning_rate_scheduler):
    model.train()

    size = len(loader.dataset)

    for batch, (image, label) in enumerate(loader):
        image, label = image.to(device), label.to(device)

        optimizer.zero_grad()

        # 예측 오류 계산
        pred = model(image.float())
        loss = loss_fn(pred, label)

        # 역전파
        loss.backward()
        optimizer.step()

        learning_rate_scheduler.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(image)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
