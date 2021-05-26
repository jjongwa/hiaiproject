# _*_coding:utf-8_*_#

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import time

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random

from sklearn.metrics import confusion_matrix
from torch.utils.data import TensorDataset, DataLoader


np.random.seed(0)  # numpy random seed 설정
torch.manual_seed(0)  # pytorch seed 설정
torch.backends.cudnn.deterministic = True  # Cuda seed 설정(총 2줄)
torch.backends.cudnn.benchmark = False  # CuDNN -> 딥러닝에 특화된 CUDA library
random.seed(0)  # python seed 설정

# 데이터셋 불러오기
train = pd.read_csv('sign_mnist_train.csv')
torch_train = torch.tensor(train.values)

test = pd.read_csv('sign_mnist_test.csv')
torch_test = torch.tensor(test.values)

# 데이터셋의 개수만큼 0으로 초가화된 x, y 텐서 생성
# y는 레이블값, x는 28*28의 각 픽셀에 해당하는 값
train_x = torch.zeros(27455, 28, 28)
train_y = torch.zeros(27455, )
test_x = torch.zeros(7172, 28, 28)
test_y = torch.zeros((7172,))

# --------- processing training data set ---------

# train_x, train_y 텐서에 각각 train 값 삽입
for idx, data in enumerate(torch_train):
    train_y[idx] += data[0]
    data = data[1:]
    x_train = data.reshape(28, 28)
    train_x[idx] += x_train

# 훈련 모델에 맞춰서 (x, y) tensor 로 ds 생성
train_y = train_y.long()
train_ds = TensorDataset(train_x, train_y)

# 배치 사이즈 95 로 training set 미니배치
# 그래프 개형이 우는거 해결하기위해 배치 사이즈 늘려볼것
train_loader = DataLoader(train_ds, batch_size=95, shuffle=True)


# --------- processing test data set ---------

# test_x, test_y 텐서에 각각 test 값 삽입
for idx, data in enumerate(torch_test):
    test_y[idx] += data[0]
    data = data[1:]
    x_test = data.reshape(28, 28)
    test_x[idx] += x_test

# 훈련 모델에 맞춰서 (x, y) tensor 로 ds 생성
test_y = test_y.long()
test_ds = TensorDataset(test_x, test_y)

# 배치 사이즈 652 로 test set 미니배치
test_loader = DataLoader(test_ds, batch_size=652, shuffle=False)


class MultilayerPerceptron(nn.Module):
    def __init__(self, in_sz, out_sz, layers=[120, 60]):
        super().__init__()
        # linear regression
        self.fc1 = nn.Linear(in_sz, layers[0])
        self.fc2 = nn.Linear(layers[0], layers[1])
        self.fc3 = nn.Linear(layers[1], out_sz)

    def forward(self, X):
        # ReLU로 Forward propogation 진행
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))

        # 마지막 layer는 softmax
        X = self.fc3(X)

        return F.log_softmax(X, dim=1)


model = MultilayerPerceptron(784, 26)  # input feature 784, output feature 26으로 객체(학습 모델) 생성


# loss = np.log(sum(np.exp(output))) - output[target[0]]  # CrossEntropyLoss 원형
# np.exp(y_pred[j]) -> 계산된 아웃풋의 각 원소에 exp 적용 (e^x)
# sum(np.exp(y-pred[j]) -> 시그마
# output[target[0]] -> 실제로 0번클래스에 속하는 데이터 셋과 현재 계산된 output 에 대해 에러 계산
criterion = nn.CrossEntropyLoss()

# SGD(stochastic gradient descent) : 미분값이 0이되도록 음수(-)방향으로 최적화
# Momentum : SGD의 단점인 지역 최소값에서 더이상 loss를 감소하지 않는것을 막기위해 관성을 주어 탈출하도록 보완
# Adagrad : 최적의 해에 가까워질수록 learning rate를 감소시키는 최적화 함수
# adam 은 SGD, Momentum, Adagrad 를 모두 합친 것
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   # adam 가지고 parameters 를 학습하는데 learing rate는 1
# 이 optimizer 가 parameter 들을 update
# 출처 : https://velog.io/@reversesky/Optimizer%EC%9D%98-%EC%A2%85%EB%A5%98%EC%99%80-%EA%B0%84%EB%8B%A8%ED%95%9C-%EC%A0%95%EB%A6%AC


start_time = time.time()

# training
epochs = 10
# 27455개 training data 에서
# 배치사이즈가 95이면 27455/95 = 289번 가중치 업데이트
# 95개를 묶어서 학습하고 가중치 업데이트 -> 289 번 반복
# 에포크가 10이면 289번 업데이트가 10번 반복

train_losses = []
test_losses = []
train_correct = []
test_correct = []

for i in range(epochs):
    trn_corr = 0
    tst_corr = 0

    for b, (X_train, y_train) in enumerate(train_loader):  # train_lodaer ( 95개를 묶은 1개의 미니 배치한 데이터셋 ) 에 대해서 반복
        b += 1  # index 값

        y_pred = model(X_train.view(95, -1))  # output 을 예측
        # model.forward(X_train.view(95,-1)) 를 한것 (계산결과가 같음)

        loss = criterion(y_pred, y_train)  # CrossEntropyLoss 로 에러 게산

        predicted = torch.max(y_pred.data, 1)[1]  # 학습 모델이 95(배치 사이즈)개의 각 데이터가 속한다고 예측한 클래스를 tensor 로 저장
        # predicted[0] 은 예측한 raw value, predicted[1] 은 실제 26개의 클래스 중 하나로 표현

        batch_corr = (predicted == y_train).sum()
        # 95(미니 배치 사이즈)개중 예측이 맞은 데이터들의 개수

        trn_corr += batch_corr
        # 누적해서 더함

        optimizer.zero_grad()  # 혹시나 기존에 gradient 를 구한게 있으면 0으로 초기화
        loss.backward()  # 앞에서 구한 loss 에 대해 back propagation 을 수행, parameter 의 gradient 를 구함
        # grad 가 붙은 함수는 대부분 gradient 계산에 사용한다고 보면 된다.
        # 역전파를 계산하기 위해 자동미분(loss.backward() 사용)
        # 출처 : https://green-late7.tistory.com/48
        optimizer.step()  # gradient 를 사용하여 loss 를 줄이는 방향으로 parameter 들을 update
        # 이 세 친구는 항상 붙어다님
        # 출처 : https://www.youtube.com/watch?v=HgPWRqtg254&list=PLQ28Nx3M4JrhkqBVIXg-i5_CVVoS1UzAv&index=8

        if b % 190 == 0:  # 정해진 index 마다 결과 출력
            acc = trn_corr.item() * 95 / (95 * b)
            print(f'epoch {i} batch {b} loss : {loss.item()} acc : {acc}')

    train_losses.append(loss)  # 95개의 미니배치한 각 data set 별 loss 를 저장
    train_correct.append(trn_corr)  # 95개의 미니배치한 각 data set 의 95개 중 예측을 맞춘 것의 개수를 누적해서 저장
    # 결론적으로 27455개의 모든 input data 들에 대해 loss 와 올바르게 에측한 것의 개수를 저장. epoch마다 반복

    # 위에서 95개씩 데이터셋을 다 돌고나서,
    # 훈련을 통해 얻어진 파라미터값들을 test data set 에 적용해 얼마만큼의 정확도가 나오는지 확인하는 코드
    with torch.no_grad():  # 파라미터를 학습 할 필요가 없으니 추적을 할 필요가 없고, 메모리 사용량을 절약하기위해 no_grad()로 wrapping
        for b, (X_test, y_test) in enumerate(test_loader):
            y_val = model(X_test.view(652, -1))
            predicted = torch.max(y_val.data, 1)[1]
            tst_corr += (predicted == y_test).sum()

        loss = criterion(y_val, y_test)
        test_losses.append(loss)
        test_correct.append(tst_corr)
        print("tst_corr", tst_corr)

# 총 걸린시간 계산
total_time = time.time() - start_time
print(f'Duration : {total_time / 60} mins')

# --------- 학습 할 때와 실제 테스트 할때에 대해 에러 추이 출력 ---------
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='test/validation loss')
plt.legend()
plt.show()


# --------- 학습 할 때와 과 실제 테스트를 할 때에 대해 정확도 출력 ---------

train_acc = [t / 275 for t in train_correct]
# train_correct 는 27455개 중 에측을 맞춘 data 의 개수가 epoch 만큼 저장이니까
# (1~10까지 epoch 별로 27455개 데이터 중 맞춘 데이터 개수 / 27455) * 100

test_acc = [t / 72 for t in test_correct]
# 위와 같은 원리로 test set에 대해서는
# ( 7172 개 중 에측을 맞춘 데이터의 개수 / 7172 ) * 100

# 정확도 추이 출력
plt.plot(train_acc, label='train acc')
plt.plot(test_acc, label='test acc')
plt.legend()
plt.show()


