---
layout: single
title: "[Computer Vision] 1(3). 신경망 훈련시키기"
categories: ['AI', 'ComputerVision']
---



# 신경망 훈련시키기

- 신경망이 새로운 데이터에 대해 좋은 예측 결과를 내려면 그와 비슷한 데이터들로 학습이 먼저 이루어져야 한다. 
- 여기서는 먼저 이론적 배경에 대해 살펴보고 학습 전략을 안아본 다음 실제 훈련이 어떻게 이루어지는 지 알아본다. 그런 다음 MNIST 데이터 셋에 이 개념을 직접 적용해 단순한 네트워크가 인식과제를 해결하도록 한다. 

<br>

### 학습 전략

##### 지도 학습

- 지도학습은 입력 데이터와 그 입력 데이터의 레이블 값이 함께 있을 때 사용한다. 
- 학습 과정은 다음과 같다. 
    - 네트워크에 이미지를 제공하고 결과(예측된 레이블)을 수집한다. 
    - 네트워크의 손실을 평가한다. 즉, 예측을 실제 레이블과 비교해 얼마나 잘못했는지 평가한다. 
    - 이 손실을 줄이기 위해 네트워크 매개변수를 조정한다. 
    - 네트워크가 수렴할 때까지, 즉 이 훈련 데이터로는 더 개션할 수 없을 때까지 반복한다. 

<br>

##### 비지도 학습

- 비지도 학습은 입력 데이터에 레이블 값이 없을 때 사용한다.
- 이 때에는 네트워크의 입력과 그에 대응하는 출력에만 기반하여 네트워크의 손실을 계산한다. 
- 이 전략은 **클러스터링**이나 **압축**에서 많이 쓰이며, 두 경우에 손실은 각각 '하나의 이미지가 다른 클러스터와 얼마나 다른지'와 '압축 데이터가 원본 데이터와 비교해 중요한 속성을 잘 보존하고 있는지'를 평가한다. 
- 이와 같이 비지도 학습은 의미있는 손실 함수를 구하기 위해 용도에 맞는 '전문성'이 필요하다. 

<br>

##### 강화학습

- 강화 학습은 **상호 작용에 기반한 전략**이다. 
- 매 순간 환경으로부터 '관측 데이터'와 '보상'만 신경망에 제공되고, 신경망은 이로부터 어떻게 해야 더 높은 보상을 받을 수 있는 지 배우고 그에 따라 에이전트가 취할 '단기 혹은 장기 정책으로 무엇이 최선인지 추정'한다. 
- 컴퓨터 비전에는 강화학습이 일반적으로 사용되지 않지만, 머신러닝의 팬이라면 강화학습에 대해 더 배우기를 추천한다. 

<br>

<br>



### 네트워크 훈련

- 학습 전략과 상관없이 훈련 데이터가 주어지면 네트워크는 예측을 하고 피드백을 받아 이를 네트워크 매개변수를 업데이트하는 과정은 동일하다. 그리고 이는 네트워크가 더 이상 최적화될 수 없을 때까지 반복한다. 

<br>

##### 손실 평가

- 손실 함수는 **네트워크 매개변수의 함수로 예측의 품질을 나타낸다. **
- 손실 함수는 네트워크의 목표룰 나타내기 때문에 주어진 과제만큼 다양한 함수가 존재한다. 
    - 지도학습에서 흔히 쓰이는 **L2 손실**은 출력 벡터와 실제 레이블의 각 요소간 차이를 제곱하여 모두 더한다. 
    - 벡터 간 차이의 절댓값을 계산하는 **L1손실**이나, 기댓값과 비교하기 전에 예측 확률을 로그 척도로 변환하는 **교차 엔트로피 손실** 같은 서로 다른 속성을 갖는 다양한 손실 함수가 있다. 
    - 이외에도 L2손실의 평균을 구하는 **평균제곱오차(MSE)**, L2손실의 평균을 구하는 **평균 절댓값 오차(MAE)** 등이 있다. 

<br>

##### 손실의 역전파

- 훈련을 반복할 때마다 네트워크의 각 매개변수에 관한 손실 도함수가 계산되고, 이 도함수는 조정된 매개변수 값 중 어느것을 적용해야 할 지 알려준다. 이 반복 절차를 **경사 하강**이라 한다. 
- 이 도함수는 연쇄 법칙에 의해 구해진다. 이것은 의미론적으로, 각 매개변수가 계층마다 손실에 얼마나 영향을 미치는지 재귀적으로 되돌아가면서 계산하는 것이다. 이 개념은 신경망을 **계산 그래프**로 표현해 낼 수 있다. 
- 위에서 도함수가 되돌아가며 매개변수를 조정하는 과정을 **역전파**라 한다. 
- 손실 도함수는 매개변수 업데이트에 사용되기 전에 계수 '앱실론'으로 곱하는데, 이 계수를 **학습률**이라 한다. 이 계수는 매회 얼마의 강도로 업데이트 되어야 하는지를 제어한다. 

<br>

- 전체 훈련 프로세스를 요약해본다. <br>
    1. n개의 다음 훈련 이미지를 선택(배치)해 네트워크에 제공한다. <br>
    2. 연쇄 법칙을 사용해 손실을 계산하고 역전파해서 계층 매개변수와 관련한 미분값을 얻는다. <br>
    3. 해당 미분값으로 매개변수를 업데이트 한다. (학습률로 조정)<br>
    4. 전체 훈련 집합에 대해 1~3 단계를 반복한다. <br>
    5. 수렴하거나 정해진 반복 횟수에 도달할 때까지 1~4 단계를 반복한다. 
- 전체 훈련 집합을 1회 반복하는 것(1~4 단계)을 **에포크**라 한다. 

<br>

##### 신경망에 분류하는 방법 가르치기


```python
# 먼저 FullyConnetedLayer 클래스에 역전파와 최적화 메서드를 추가한다. 

class FullyConnectedLayer(object):
    """A simple fully-connected NN layer.
    Args:
        num_inputs (int): The input vector size / number of input values.
        layer_size (int): The output vector size / number of neurons in the layer.
        activation_function (callable): The activation function for this layer.
    Attributes:
        W (ndarray): The weight values for each input.
        b (ndarray): The bias value, added to the weighted sum.
        size (int): The layer size / number of neurons.
        activation_function (callable): The activation function computing the neuron's output.
        x (ndarray): The last provided input vector, stored for backpropagation.
        y (ndarray): The corresponding output, also stored for backpropagation.
        derivated_activation_function (callable): The corresponding derivated function for backpropagation.
        dL_dW (ndarray): The derivative of the loss, with respect to the weights W.
        dL_db (ndarray): The derivative of the loss, with respect to the bias b.
    """

    def __init__(self, num_inputs, layer_size, activation_function, derivated_activation_function=None):
        super().__init__()

        # Randomly initializing the weight vector and the bias value (using a normal distribution this time):
        self.W = np.random.standard_normal((num_inputs, layer_size))
        self.b = np.random.standard_normal(layer_size)
        self.size = layer_size
        # 활성화 함수, 활성화 도함수
        # 입력, 출력 데이터(역전파를 위해 가지고 있어야 함)
        # 매개변수에 대한 손실 도함수
        self.activation_function = activation_function
        self.derivated_activation_function = derivated_activation_function
        self.x, self.y = None, None
        self.dL_dW, self.dL_db = None, None

    def forward(self, x):
        """
        Forward the input vector through the layer, returning its activation vector.
        Args:
            x (ndarray): The input vector, of shape `(batch_size, num_inputs)`
        Returns:
            activation (ndarray): The activation value, of shape `(batch_size, layer_size)`.
        """
        z = np.dot(x, self.W) + self.b
        self.y = self.activation_function(z)
        self.x = x  # (we store the input and output values for back-propagation)
        return self.y

    def backward(self, dL_dy):
        """
        Back-propagate the loss, computing all the derivatives, storing those w.r.t. the layer parameters,
        and returning the loss w.r.t. its inputs for further propagation.
        Args:
            dL_dy (ndarray): The loss derivative w.r.t. the layer's output (dL/dy = l'_{k+1}).
        Returns:
            dL_dx (ndarray): The loss derivative w.r.t. the layer's input (dL/dx).
        """
        dy_dz = self.derivated_activation_function(self.y)  # = f'
        dL_dz = (dL_dy * dy_dz) # dL/dz = dL/dy * dy/dz = l'_{k+1} * f'
        dz_dw = self.x.T
        dz_dx = self.W.T
        dz_db = np.ones(dL_dy.shape[0]) # dz/db = d(W.x + b)/db = 0 + db/db = "ones"-vector

        # 계층 매개변수 저장
        self.dL_dW = np.dot(dz_dw, dL_dz)
        self.dL_db = np.dot(dz_db, dL_dz)

        # 이전 계층의 x 미분값 계산
        dL_dx = np.dot(dL_dz, dz_dx)
        return dL_dx

    def optimize(self, epsilon):
        """
        Optimize the layer's parameters, using the stored derivative values.
        Args:
            epsilon (float): The learning rate.
        """
        self.W -= epsilon * self.dL_dW
        self.b -= epsilon * self.dL_db
```


```python
# 이제 계층별로 역전파하고 최적화하는 메서드와 마지막으로 전체 훈련(1~5단계)을 
# 다루는 메서드를 추가해 SimpleNetwork 클래스를 수정한다. (코드 수정이 없는 부분은 주석 처리)

def sigmoid(x):     # 시그모이드 함수
    return 1 / (1 + np.exp(-x)) # y


def derivated_sigmoid(y):   # 시그모이드 도함수
    return y * (1 - y)


def loss_L2(pred, target):    # L2 손실함수
    return np.sum(np.square(pred - target)) / pred.shape[0] # opt. we divide by the batch size


def derivated_loss_L2(pred, target):    # L2 손실 도함수
    return 2 * (pred - target)


#==============================================================================
# Class Definition
#==============================================================================

class SimpleNetwork(object):
    """A simple fully-connected NN.
    Args:
        num_inputs (int): The input vector size / number of input values.
        num_outputs (int): The output vector size.
        hidden_layers_sizes (list): A list of sizes for each hidden layer to add to the network
        activation_function (callable): The activation function for all the layers
        derivated_activation_function (callable): The derivated activation function
        loss_function (callable): The loss function to train this network
        derivated_loss_function (callable): The derivative of the loss function, for back-propagation
    Attributes:
        layers (list): The list of layers forming this simple network.
        loss_function (callable): The loss function to train this network.
        derivated_loss_function (callable): The derivative of the loss function, for back-propagation.
    """
    
    # 입력 벡터 크기, 출력 벡터 크기, 은닉층, 활성화 함수와 도함수, 손실 함수와 도함수
    def __init__(self, num_inputs, num_outputs, hidden_layers_sizes=(64, 32),
                 activation_function=sigmoid, derivated_activation_function=derivated_sigmoid,
                 loss_function=loss_L2, derivated_loss_function=derivated_loss_L2):
        super().__init__()
        
#         layer_sizes = [num_inputs, *hidden_layers_sizes, num_outputs]
#         self.layers = [
#             FullyConnectedLayer(layer_sizes[i], layer_sizes[i + 1], activation_function, derivated_activation_function)
#             for i in range(len(layer_sizes) - 1)]

        self.loss_function = loss_function
        self.derivated_loss_function = derivated_loss_function

#     def forward(self, x):
#         """
#         Forward the input vector through the layers, returning the output vector.
#         Args:
#             x (ndarray): The input vector, of shape `(batch_size, num_inputs)`.
#         Returns:
#             activation (ndarray): The output activation value, of shape `(batch_size, layer_size)`.
#         """
#         for layer in self.layers: # from the input layer to the output one
#             x = layer.forward(x)
#         return x

#     def predict(self, x):
#         """
#         Compute the output corresponding to input `x`, and return the index of the largest output value.
#         Args:
#             x (ndarray): The input vector, of shape `(1, num_inputs)`.
#         Returns:
#             best_class (int): The predicted class ID.
#         """
#         estimations = self.forward(x)
#         best_class = np.argmax(estimations)
#         return best_class

    def backward(self, dL_dy):
        """
        마지막 계층에서 처음 계층까지 손실 미분값을 역전파(forward() 메서드 이후에 호출되어야 함).
        Args:
            dL_dy (ndarray): The loss derivative w.r.t. the network's output (dL/dy).
        Returns:
            dL_dx (ndarray): The loss derivative w.r.t. the network's input (dL/dx).
        """
        for layer in reversed(self.layers): # from the output layer to the input one
            dL_dy = layer.backward(dL_dy)
        return dL_dy

    def optimize(self, epsilon): # 학습률 전달
        """
        저장된 경삿값에 따라 매개변수를 최적화(backward() 메서드가 먼저 호출되어야 함)
        Args:
            epsilon (float): The learning rate.
        """
        for layer in self.layers:             # the order doesn't matter here
            layer.optimize(epsilon)

#     def evaluate_accuracy(self, X_val, y_val):
#         """
#         Given a dataset and its ground-truth labels, evaluate the current accuracy of the network.
#         Args:
#             X_val (ndarray): The input validation dataset.
#             y_val (ndarray): The corresponding ground-truth validation dataset.
#         Returns:
#             accuracy (float): The accuracy of the network (= number of correct predictions / dataset size).
#         """
#         num_corrects = 0
#         for i in range(len(X_val)):
#             pred_class = self.predict(X_val[i])
#             if pred_class == y_val[i]:
#                 num_corrects += 1
#         return num_corrects / len(X_val)

    def train(self, X_train, y_train, X_val=None, y_val=None, batch_size=32, num_epochs=5, learning_rate=1e-3):
        """
        제공된 데이터셋에서 네트워크를 훈련하고 평가
        Args:
            X_train (ndarray): The input training dataset.
            y_train (ndarray): The corresponding ground-truth training dataset.
            X_val (ndarray): The input validation dataset.
            y_val (ndarray): The corresponding ground-truth validation dataset.
            batch_size (int): The mini-batch size.
            num_epochs (int): The number of training epochs i.e. iterations over the whole dataset.
            learning_rate (float): The learning rate to scale the derivatives.
        Returns:
            losses (list): The list of training losses for each epoch.
            accuracies (list): The list of validation accuracy values for each epoch.
        """
        num_batches_per_epoch = len(X_train) // batch_size
        do_validation = X_val is not None and y_val is not None
        losses, accuracies = [], []
        for i in range(num_epochs): # for each training epoch
            epoch_loss = 0
            for b in range(num_batches_per_epoch):  # for each batch composing the dataset
                # Get batch:
                batch_index_begin = b * batch_size
                batch_index_end = batch_index_begin + batch_size
                x = X_train[batch_index_begin: batch_index_end]
                targets = y_train[batch_index_begin: batch_index_end]
                # Optimize on batch:
                predictions = y = self.forward(x)  # forward pass
                L = self.loss_function(predictions, targets)  # loss computation
                dL_dy = self.derivated_loss_function(predictions, targets)  # loss derivation
                self.backward(dL_dy)  # back-propagation pass
                self.optimize(learning_rate)  # optimization of the NN
                epoch_loss += L

            # Logging training loss and validation accuracy, to follow the training:
            epoch_loss /= num_batches_per_epoch
            losses.append(epoch_loss)
            if do_validation:
                accuracy = self.evaluate_accuracy(X_val, y_val)
                accuracies.append(accuracy)
            else:
                accuracy = np.NaN
            print("Epoch {:4d}: training loss = {:.6f} | val accuracy = {:.2f}%".format(i, epoch_loss, accuracy * 100))
        return losses, accuracies
```

> - 네트워크의 backward 메서드에서는 끝부터 처음까지 계층의 backward 메서드를 호출하여 역전파를 수행한다. 
> - 네트워크의 optimize 메서드에서는 학습률과 함께 계층의 optimize 메서드를 호출한다. 
> - 네트워크의 train 메서드는 다음의 과정으로 이루어진다. 
>     - 훈련데이터와 라벨 배치 가져오기(인덱스 계산)
>     - forward 메서드 호출, 손실과 손실 미분값(dL/dy) 계산(각 에포크의 손실은 각 배치의 손실을 모두 더해서 만들어짐)
>     - backward 메서드 호출
>     - optimize 메서드 호출
>     - 각 에포크에 대해 위의 과정을 반복

<br>


```python
# 이제 네트워크를 훈련시켜 성능을 확인한다. 
losses, accuracies = mnist_classifier.train(
                        X_train, y_train, X_test, y_test, batch_size=30, num_epochs=500)

# >> Epoch 499: training loss = 0.045963 | val accuracy = 94.83%
```

<br>

##### 훈련 시 고려사항 - 과소적합과 과대적합

- **과소적합**
    - 네트워크 계층이 너무 적거나 너무 작은 계층으로 구성되어 정확도가 향상되지 않는 것. 즉, 과제의 복잡도를 다룰 만큼 충분한 매개변수를 갖지 못했음을 말한다. 
    - 유일한 해결책은 애플리케이션에 더 적합한 새로운 아키텍쳐를 선택하는 것
- **과대적합**
    - 네트워크가 너무 복잡하거나 훈련 데이터셋이 너무 작아 훈련 분포에만 너무 잘 적합되게 학습하고, 새로운 샘플에 적용될만큼은 일반화되지 않는 것. 
    - 더 많고 다양한 데이터셋을 확보하거나, 네트워크가 얼마나 자세히 학습할지 제한하기 위해 훈련 과정을 조정하는 것. 
