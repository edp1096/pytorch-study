def run(device, loader, model, loss_fn, optimizer):
    model.train()

    size = len(loader.dataset)

    for batch, (X, y) in enumerate(loader):
        X, y = X.view(-1, 28 * 28).to(device), y.to(device)  # linear

        # 예측 오류 계산
        pred = model(X.float())
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def runCNN(device, loader, model, loss_fn, optimizer):
    model.train()

    size = len(loader.dataset)

    for batch, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        # 예측 오류 계산
        pred = model(X.float())
        loss = loss_fn(pred, y)

        # 역전파
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def runRNN(device, loader, model, loss_fn, optimizer):
    model.train()

    size = len(loader.dataset)

    for batch, (X, y) in enumerate(loader):
        # X = X.view(-1, 28, 28).float().requires_grad_().to(device) # batch, seq_dim, input_dim
        X = X.reshape(-1, 28, 28).float().to(device)  # batch, seq_len, input_size
        y = y.to(device)


        # 예측 오류 계산
        pred = model(X)
        loss = loss_fn(pred, y)

        # 역전파
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
