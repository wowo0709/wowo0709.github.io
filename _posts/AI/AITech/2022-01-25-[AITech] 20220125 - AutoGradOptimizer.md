---
layout: single
title: "[AITech] 20220125 - AutoGrad&Optimizer"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['torch.nn.Module']
---



<br>

## 학습 내용 정리

### AutoGrad & Optimizer

딥러닝 모델의 구조는 **블록들의 연속**이다. 해당 블록은 하나의 연산을 수행하는 단일 층일 수도 있고, 여러 단일 층들이 모인 하나의 블록일 수도 있다. 

모델의 반복 구조를 설계하기 쉽게 하기 위하여, 파이토치에서는 여러 모듈을 제공한다. 

#### torch.nn.Module

* 딥러닝을 구성하는 Layer의 base class
* **Input, Output, Forward, Backward** 정의
* 학습의 대상이 되는 **parameter** 정의

**nn.Parameter**

* nn.Module 내에 attribute가 될 때는 **required_grad=True**로 지정하여 학습 대상으로 설정
* 하지만 이를 직접 지정해 줄 일은 거의 없다. 
  * 대부분의 layer에는 weights 값들이 지정되어 있음

```python
class MyLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features # 입력 피쳐 개수(입력 노드 개수)
        self.out_features = out_features # 출력 피쳐 개수(출력 노드 개수)
        
        self.weights = nn.Parameter( # 학습 parameter 설정(가중치)
                torch.randn(in_features, out_features))
        
        self.bias = nn.Parameter(torch.randn(out_features)) # 학습 parameter 설정(편향)
        
    def forward(self, x: Tensor):
        return x @ self.weights + self.bias # linear 연산
```

**Backward**

* **Layer에 있는 parameter들의 미분을 수행(그래디언트 값 전달)**
* Loss를 미분한 값으로 parameter 갱신

```python
for epoch in range(epochs):
    ...
    # 이전 그래디언트 값 초기화
    optimizer.zero_grad()
    # 모델 예측 값
    output = model(inputs)
    # 손실 함수 값 계산
    loss = criterion(outputs, labels)
    # 손실 함수 미분, 그래디언트 값 계산
    loss.backward()
    # 파라미터 갱신
    optimizer.step()
```

* backward 함수와 optimizer는 Module 레벨에서 직접 오버라이딩 할 수 있고, 직접 미분 수식을 써야 하므로 실제로 할 일은 거의 없지만 순서를 이해할 필요는 있다. 

```python
class LogisticRegression(nn.Module):
    def __init__(self, dim, lr=torch.scalar_tensor(0.01)):
        super(LR, self).__init__()
        # initialize parameters
        self.w = torch.zeros(dim, 1, dtype=torch.float).to(device)
        self.b = torch.scalar_tensor(0).to(device)
        self.grads = {"dw": torch.zeros(dim, 1, dtype=torch.float).to(device), 
                     "db": torch.scalar_tensor(0).to(device)}
        self.lr = lr.to(device)
        
    def forward(self, x):
        # compute forward
        z = torch.mm(self.w.T, x)
        a = self.sigmoid(z)
        return a
    
    def sigmoid(self, z):
        return 1/(1+torch.exp(-z))
    
    def backward(self, x, yhat, y):
        # 미분 수식을 직접 작성
        self.grads["dw"] = (1/x.shape[1])*torch.mm(x, (yhat-y).T)
        self.grads["db"] = (1/x.shape[1])(torch.sum(yhat-y))
        
    def optimize(self):
        # 파라미터 업데이트
        self.w = self.w - self.lr*self.grads["dw"]
        self.b = self.b - self.lr*self.grads["db"]
```

<br>
