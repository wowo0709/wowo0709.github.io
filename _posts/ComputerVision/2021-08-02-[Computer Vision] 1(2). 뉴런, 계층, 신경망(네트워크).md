---
layout: single
title: "[Computer Vision] 1(2). 뉴런, 계층, 신경망(네트워크)"
---



# 뉴런, 계층, 신경망(네트워크)

### 뉴런 복제하기

- 하나의 뉴런에서는 다음 과정을 수행한다. 
    - 입력 -> 가중합 -> 활성화 함수 -> 출력
- 활성화 함수로는 일반적으로 시그모이드 함수, 하이퍼볼릭 탄젠트 함수, ReLU 함수를 많이 사용한다. 

##### 뉴런 클래스 구현


```python
import numpy as np
```


```python
class Neuron(object):
    """
    간단한 전방 전달 인공 뉴런.
    Args:
        num_inputs (int): 입력 값의 크기
        activation_function (callable): 활성화 함수
    Attributes:
        W (ndarray): 각 입력에 대한 가중치
        b (float): 편향값, 가중합에 더해짐. 
        activation_function (callable): 활성화 함수. 
    """

    def __init__(self, num_inputs, activation_function):
        super().__init__()

        # 랜덤값으로 가중치 벡터와 편향값을 초기화(1과 -1 사이)
        self.W = np.random.uniform(size=num_inputs, low=-1., high=1.)
        self.b = np.random.uniform(size=1, low=-1., high=1.)

        self.activation_function = activation_function

    def forward(self, x):
        # 뉴런을 통해 입력 신호를 전달(입력 -> 가중합 -> 활성화 함수 -> 출력)
        z = np.dot(x, self.W) + self.b
        return self.activation_function(z)
```

##### 퍼셉트론 인스턴스 만들기


```python
# 결과를 복제할 수 있도록 랜덤 숫자 생성기의 시드 값을 고정
np.random.seed(42)
# 3개의 랜덤 입력을 칼럼으로 받을 수 있는 배열 (shape = '(1,3)')
x = np.random.rand(3).reshape(1,3)
print(x)
```

    [[0.37454012 0.95071431 0.73199394]]



```python
# 퍼셉트론을 인스턴스화(step function을 활성화 함수로 사용)
step_fn = lambda y: 0 if y <= 0 else 1

perceptron = Neuron(num_inputs=x.size, activation_function=step_fn)
print(perceptron.W)
print(perceptron.b)
```

    [0.73235229 0.20223002 0.41614516]
    [-0.95883101]



```python
out = perceptron.forward(x)
print(out)
```

    0

<br>

### 뉴런을 계층화하기

- 이 부분에서는 **완전 연결 계층(밀집 계층)**을 하나의 클래스로 구현해본다. 
- 파라미터로 입력 값의 개수, (은닉) 계층의 층수, 활성화 함수를 전달한다. 
- 신경망 인스턴스는 가중치, 편향, 활성화 함수와 더불어 두번째 파라미터인 **은닉 계층의 층수**를 프로퍼티로 가지고 있어야 한다. 

##### 완전연결신경망 구현

- 한 계층에 여러 뉴런을 반영할 수 있게 일부 변수의 '차원'을 바꾸기만 하면 된다. 이렇게 구현하면 계층이 한 번에 여러 개의 입력을 처리할 수 있다. 


```python
# import numpy as np
```


```python
# 하나의 계층을 정의
class FullyConnectedLayer(object):
    """A simple fully-connected NN layer.
    Args:
        num_inputs (int): 입력 백터의 크기
        layer_size (int): 출력 벡터의 크기
        activation_function (callable): 활성화 함수
    Attributes:
        W (ndarray): 입력값에 대한 가중치
        b (ndarray): 가중합에 더해질 편향값
        size (int): 은닉층의 개수
        activation_function (callable): 활성화 함수
    """

    def __init__(self, num_inputs, layer_size, activation_function, derivated_activation_function=None):
        super().__init__()

        # 임의로 가중치와 편향을 초기화 (여기서는 정규 분포 사용)
        self.W = np.random.standard_normal((num_inputs, layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        self.activation_function = activation_function

    def forward(self, x):
        """계층을 통해 입력 신호를 전달"""
        z = np.dot(x, self.W) + self.b
        return self.activation_function(z)
```

##### 신경망 인스턴스 만들기


```python
np.random.seed(42)      

x1 = np.random.uniform(-1, 1, 2).reshape(1, 2) # [[값1 값2 값3]]
x2 = np.random.uniform(-1, 1, 2).reshape(1, 2) # [[값1 값2 값3]]

relu_fn = lambda y: np.maximum(y, 0)    # 활성화 함수 정의: ReLU

layer = FullyConnectedLayer(2, 3, relu_fn)
print(layer.W)
print(layer.b)
```

    [[-0.23415337 -0.23413696  1.57921282]
     [ 0.76743473 -0.46947439  0.54256004]]
    [-0.46341769 -0.46572975  0.24196227]



```python
# 2개의 입력을 개별적으로 처리하기
out1 = layer.forward(x1)
out2 = layer.forward(x2)

print(out1, out2)
```

    [[0.28712364 0.         0.33478571]] [[0.         0.         1.08175419]]



```python
# 동시에 처리할 수도 있음
x1_2 = np.concatenate((x1,x2)) # 입력 벡터의 스택 (shape = '(2,2)')
out1_2 = layer.forward(x1_2)
print(out1_2)
```

    [[0.28712364 0.         0.33478571]
     [0.         0.         1.08175419]]


- 일반적으로 입력 데이터의 스택을 **배치(batch)**라고 한다. 

<br>

### 예제 신경망을 분류에 적용하기

- 계층을 컴퓨터 비전에 활용하기 위해서는 이 계층을 초기화하고 네트워크에 연결해야 한다. 
- 여기서는 유명한 이미지 분류 데이터셋인 mnist(손글씨 이미지) 데이터셋을 이용한다. 
- 입력 벡터는 28x28=784 개의 값을 가지며 출력은 10개(0~9)의 값을 갖는다. 출력으로는 각 클래스에 대한 '확신 점수'를 반환하며 이 확신 점수는 보통 추가 계산이나 해석을 단순화하기 위해 확률로 변환된다. 
- 우리가 해야 할 것은 은닉 계층의 수와 그 크기를 정의하는 것이다. 

##### MNIST 데이터 준비

- 훈련과 테스트 데이터를 나누고 feature와 label로 나눈다. 
- 입력 데이터를 NN의 입력 모양에 맞게 reshape한다. 
- 다중 분류이기 때문에 레이블을 원-핫 벡터(NN의 결과)로 변환한다. 


```python
import numpy as np
import mnist
np.random.seed(42)

# 훈련 및 테스트 데이터 로딩
# 훈련과 테스트 데이터에 대해 feature와 label로 나눈다. 
X_train, y_train = mnist.train_images(), mnist.train_labels()
X_test, y_test = mnist.test_images(), mnist.test_labels()
num_classes = 10 # 분류 클래스(0~9)

print(X_train, y_train)
```


```python
# 이미지를 칼럼 벡터(NN의 입력모양으로)로 변환
X_train, X_test = X_train.reshape(-1, 28*28), X_test.reshape(-1, 28*28)
# 레이블을 원-핫 벡터(NN의 결과)로 변환
y_train = np.eye(num_classes)[y_train]
```

- np.eye(n, k=m, dtype=int)
    - n은 nxn 크기의 결과 행렬을, k는 정방단위행렬을 기준으로 어느 부분에 대각 행렬을 나타낼 것인지 결정한다. 

##### 네트워크 구현하기

- 신경망을 구현하기 위해서는 계층을 함께 감싸고 전체 네트워크를 통해 전달하고 출력 벡터에 따라 클래스를 예측하기 위한 몇 가지 기법을 추가해야 한다. 


```python
import numpy as np
from fully_connected_layer import FullyConnectedLayer # 완전연결계층(입력크기, 은닉층 개수)

def sigmoid(x): # x의 요소에 시그모이드 함수를 적용
    return 1 / (1 + np.exp(-x)) # y


class SimpleNetwork(object):
    """간단한 완전 연결 신경망
    Args:
        num_inputs (int): 입력 벡터 크기 / 입력값 개수
        num_outputs (int): 출력 벡터 크기
        hidden_layers_sizes (list): 네트워크에 추가될 은닉 계층의 크기를 포함한 리스트
    Attributes:
        layers (list): 네트워크를 구성하는 계층 리스트
    """

    def __init__(self, num_inputs, num_outputs, hidden_layers_sizes=(64, 32)):
        super().__init__()
        # 네트워크를 구성하는 계층 리스트를 구성
        # 입력층, 은닉층([64, 32]), 출력층
        layer_sizes = [num_inputs, *hidden_layers_sizes, num_outputs]
        self.layers = [
            FullyConnectedLayer(layer_sizes[i], layer_sizes[i + 1], sigmoid) # 각 계층을 완전연결계층으로 연결(입력, 출력, 활성화)
            for i in range(len(layer_sizes) - 1)]

    def forward(self, x):
        """입력벡터 'x'를 계층을 통해 전달"""
        for layer in self.layers: # from the input layer to the output one
            x = layer.forward(x)
        return x

    def predict(self, x):
        """'x'에 대응하는 출력을 계산하고 출력값이 가장 큰 인덱스를 반환"""
        estimations = self.forward(x)
        best_class = np.argmax(estimations)
        return best_class

    def evaluate_accuracy(self, X_val, y_val):
        """검증 데이터셋을 사용해 네트워크의 정확도를 평가"""
        num_corrects = 0
        for i in range(len(X_val)):
            pred_class = self.predict(X_val[i])
            if pred_class == y_val[i]:
                num_corrects += 1
        return num_corrects / len(X_val)
```

##### 예측하기


```python
# MNIST images 분류를 위한 네트워크, 그 안에 크기가 64, 32 인 2개의 은닉 계층이 있음
mnist_classifier = SimpleNetwork(X_train.shape[1], num_classes, [64,32]) # 신경망(입력크기, 출력크기, 은닉층)

# 네트워크 정확도 평가
accuracy = mnist_classifier.evaluate_accuracy(X_test, y_test)
print("accuracy = {:.2f}%".format(accuracy * 100))
# out: accuracy = 12.06%
```

- 지금 만든 신경망의 매개변수는 임의의 값으로 정했기 때문에 당연한 결과이다. 
- 다음 포스팅에서 이 네트워크를 훈련(학습)시켜볼 것이다. 
