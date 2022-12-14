import torch
import torch.nn as nn
import torch.nn.functional as F

import copy

torch.manual_seed(1)

# 데이터
x_train = torch.FloatTensor(
    [
        [73, 80, 75],
        [93, 88, 93],
        [89, 91, 90],
        [96, 98, 100],
        [73, 66, 70],
    ]
)
y_train = torch.FloatTensor(
    [
        [152],
        [185],
        [180],
        [196],
        [142],
    ]
)

# 모델을 선언 및 초기화. 다중 선형 회귀이므로 input_dim=3, output_dim=1.
model = nn.Linear(3, 1)

params_before = copy.deepcopy(list(model.parameters()))

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 2000
for epoch in range(nb_epochs + 1):

    # H(x) 계산
    prediction = model(x_train)
    # model(x_train)은 model.forward(x_train)와 동일함.

    # cost 계산
    cost = F.mse_loss(prediction, y_train)  # <== 파이토치에서 제공하는 평균 제곱 오차 함수

    # cost로 H(x) 개선하는 부분
    # gradient를 0으로 초기화
    optimizer.zero_grad()
    # 비용 함수를 미분하여 gradient 계산
    cost.backward()
    # W와 b를 업데이트
    optimizer.step()

    if epoch % 100 == 0:
        # 100번마다 로그 출력
        print("Epoch {:4d}/{} Cost: {:.6f}".format(epoch, nb_epochs, cost.item()))


# 사실 3개의 값 73, 80, 75는 훈련 데이터로 사용되었던 값입니다.
# 당시 y의 값은 152였는데, 현재 예측값이 151이 나온 것으로 보아 어느정도는 3개의 w와 b의 값이 최적화 된것으로 보입니다.

# 임의의 입력 [73, 80, 75]를 선언
new_var = torch.FloatTensor([[73, 80, 75]])
# 입력한 값 [73, 80, 75]에 대해서 예측값 y를 리턴받아서 pred_y에 저장
pred_y = model(new_var)
print("훈련 후 입력이 73, 80, 75일 때의 예측값 :", pred_y)


# 이제 학습 후의 3개의 w와 b의 값을 출력해보겠습니다.
params_after = list(model.parameters())

print("W, b 3 set before : ", params_before)
print("W, b 3 set after : ", params_after)
