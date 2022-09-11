# 이동 시간 이동 거리
# (H) (Km)
# 1	70
# 2	140
# 3	210
# 4	?

from unicodedata import numeric
import torch
import torch.nn as nn
import torch.optim as optim

import copy

# 데이터
X = torch.FloatTensor([[1], [2], [3]])  # 이동 시간
Y = torch.FloatTensor([[70], [140], [210]])  # 이동 거리

# 가중치, 편향 초기화
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)


# 손실 함수, 최적화 기법
loss_fn = nn.MSELoss()
optimizer = optim.SGD([W, b], lr=0.01)

# 모델 훈련
num_epochs = 3000
# num_epochs = 10

i = 0
for epoch in range(num_epochs):
    model_pred = W * X + b
    loss = loss_fn(model_pred, Y)

    optimizer.zero_grad()  # Gradient 초기화
    loss.backward()  # 역전파를 통한 기울기 계산
    optimizer.step()  # Parameter Update

    if i == 0:
        Wold = copy.deepcopy(W)
        bold = copy.deepcopy(b)

    i = i + 1

print("Wb 1:", Wold, bold)
print(f"Wb {num_epochs}:", W, b)

x_test = 4  # 4시간 이동

model_pred = W * x_test + b
print("Predict:", model_pred)
