---
layout: single
title: "[Computer Vision] 2(1). 텐서플로 2와 케라스 시작하기"
categories: ['AI', 'ComputerVision']
---



<br>

# 텐서플로 2와 케라스 시작하기

<br>

### 텐서플로 소개

---

- 처음에 텐서플로는 '머신러닝 알고리즘을 표현하기 위한 인터페이스와 이러한 알고리즘을 실행하기 위한 구현'으로 정의되어 구글에서 연구원과 개발자의 머신러닝 연구 지원을 위해 만들어졌다. 
- 텐서플로의 주요 목적은 머신러닝 솔루션을 다양한 플랫폼(CPU, GPU, 모바일 기기, 브라우저 등)에 간단하게 배포하는 것이다. 
- 2019년에는 성능은 유지하면서 사용 편의성에 중점을 둔 텐서플로 2가 출시되었다. 
- 텐서플로 API는 다양한 수준의 복잡도를 지원하고 있어 초심자도 간단한 API로 머신러닝을 시작할 수 있고 동시에 전문가는 매우 복잡한 모델을 생성할 수도 있다. 

<br>

##### 텐서플로 주요 아키텍쳐
<img src="https://user-images.githubusercontent.com/70505378/128350096-ccca987d-2f2b-4c9a-aef9-26442b09f051.png" alt="KakaoTalk_20210804_083155376" style="zoom:50%;" />

- 대부분의 딥러닝 계산은 C++로 코딩돼 있다. GPU에서 이 계산을 실행하기 위해 텐서플로는 NVIDIA 에서 개발한 CUDA 라이브러리를 사용하고, 이것이 GPU를 사용하기 위해 CUDA를 설치해야 하고 타사의 GPU를 이용할 수 없는 이유이다. 
- 다음으로 파이썬의 **저수준 API**가 C++ 소스를 감싸고 있다. 텐서플로의 파이썬 메서드를 호출하면 일반적으로 그 내부에 있는 C++코드를 불러온다. 
- 가장 위에 케라스와 Estimator API로 구성된 **고수준 API**가 있다. 
    - **케라스**는 사용자 친화적이며 모듈로 구성되고 확장성 있는 텐서플로용 래퍼다. 
    - **에스티메이터 API**는 머신러닝 모델을 쉽게 구성할 수 있게 해주는 사전 제작된 구성 요소들을 포함하고 있다. 이는 기본 구성 요소 혹은 템플릿으로 이해하면 된다. 

<br>

##### 케라스 소개

- 2015년 발표된 초기 케라스는 딥러닝 작업을 실행하기 위해 텐서플로 또는 씨아노에 의존했다. 사용자 친화적인 것으로 유명한 케라스는 초심자들이 쓰기 적합한 라이브러리다. 
- 2017년 이후로 텐서플로는 케라스를 완전히 통합해 텐서플로를 설치하면 케라스를 함께 사용할 수 있도록 하였다. 앞으로는 독립형 케라스 대신 tf.keras를 사용한다. 

<br>

<br>

### 케라스를 사용한 간단한 컴퓨터 비전 모델

---

- 케라스를 이용해 MNIST 문제를 풀어보자.

<br>

##### 데이터 준비

- tf,keras,datasets는 고전적인 데이터셋을 내려받고 빠르게 인스턴스화 하기 위해 접근할 수 있도록 도와준다. 
- [0, 255]의 값을 [0,1]로 변환한다. 일반적으로 데이터를 정규화할 때는 [0, 1] 또는 [-1, 1] 의 범위로 변환한다. 


```python
import tensorflow as tf
```


```python
num_classes = 10
img_rows, img_cols = 28, 28
num_channels = 1
input_shape = (img_rows, img_cols, num_channels)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 0s 0us/step

<br>

##### 모델 구성

- 두 개의 완전 연결 계층으로 구성된 아키텍쳐를 사용해보자. 
- 이 모델은 계층을 선형으로 쌓기 때문에 Sequential 함수를 먼저 호출하고, 각 계층을 하나씩 추가한다. 


```python
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))
model.add(tf.keras.layers.Dense(num_classes, activation='softmax'))
```

- **평명화(Flatten) 계층**: 이 계층은 이미지 픽셀을 표현하는 2차원 행렬을 취해서 1차원 배열로 전환한다(reshape 역할). 이 작업은 완전 연결계층을 추가하기 전에 이루어져야 한다. 
- **크기가 128인 밀집(Dense) 계층**: 이 계층은 784 픽셀값을 128x784 크기의 가중치 행렬과 128 크기의 편향치 행렬을 사용해 128개의 활성화값으로 전환한다. 
- **크기가 10인 밀집(Dense) 계층**: 이 계층은 128개의 활성화 값을 최종 예측 값으로 전환한다. 확률의 합이 1이 되도록 '소프트맥스' 활성화 함수를 사용한다. 

<br>

- 먼저 model.build()를 사용하여 신경망을 구축한다. 파라미터로 input_shape을 전달해야 한다. 
- 이후에 model.summary() 를 사용하면 모델의 설명, 출력, 가중치를 확인할 수 있다. 


```python
model.build((None, 28*28))
model.summary()
```

    Model: "sequential_15"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    flatten_14 (Flatten)         (None, 784)               0         
    _________________________________________________________________
    dense_21 (Dense)             (None, 128)               100480    
    _________________________________________________________________
    dense_22 (Dense)             (None, 10)                1290      
    =================================================================
    Total params: 101,770
    Trainable params: 101,770
    Non-trainable params: 0
    _________________________________________________________________

<br>

##### 모델 훈련


```python
model.compile(optimizer='sgd', 
             loss='sparse_categorical_crossentropy', 
             metrics=['accuracy'])

# verbose를 1로 지정하면 학습을 진행하며 앞에서 선택한 메트릭인 손실과 
# ETA(Estimated Time of Arrival)를 진행 표시줄로 확인할 수 있다. 
model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_test, y_test))

```

    Epoch 1/10
    WARNING:tensorflow:Model was constructed with shape (None, 784) for input KerasTensor(type_spec=TensorSpec(shape=(None, 784), dtype=tf.float32, name='flatten_14_input'), name='flatten_14_input', description="created by layer 'flatten_14_input'"), but it was called on an input with incompatible shape (32, 28, 28).
    WARNING:tensorflow:Model was constructed with shape (None, 784) for input KerasTensor(type_spec=TensorSpec(shape=(None, 784), dtype=tf.float32, name='flatten_14_input'), name='flatten_14_input', description="created by layer 'flatten_14_input'"), but it was called on an input with incompatible shape (32, 28, 28).
    1870/1875 [============================>.] - ETA: 0s - loss: 0.2167 - accuracy: 0.9395WARNING:tensorflow:Model was constructed with shape (None, 784) for input KerasTensor(type_spec=TensorSpec(shape=(None, 784), dtype=tf.float32, name='flatten_14_input'), name='flatten_14_input', description="created by layer 'flatten_14_input'"), but it was called on an input with incompatible shape (None, 28, 28).
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.2168 - accuracy: 0.9395 - val_loss: 0.2043 - val_accuracy: 0.9426
    Epoch 2/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.2006 - accuracy: 0.9442 - val_loss: 0.1937 - val_accuracy: 0.9469
    Epoch 3/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1869 - accuracy: 0.9481 - val_loss: 0.1786 - val_accuracy: 0.9504
    Epoch 4/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1749 - accuracy: 0.9508 - val_loss: 0.1709 - val_accuracy: 0.9526
    Epoch 5/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1643 - accuracy: 0.9542 - val_loss: 0.1615 - val_accuracy: 0.9547
    Epoch 6/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1551 - accuracy: 0.9566 - val_loss: 0.1529 - val_accuracy: 0.9562
    Epoch 7/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1468 - accuracy: 0.9597 - val_loss: 0.1465 - val_accuracy: 0.9587
    Epoch 8/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1395 - accuracy: 0.9612 - val_loss: 0.1415 - val_accuracy: 0.9597
    Epoch 9/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1328 - accuracy: 0.9634 - val_loss: 0.1360 - val_accuracy: 0.9613
    Epoch 10/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1271 - accuracy: 0.9651 - val_loss: 0.1303 - val_accuracy: 0.9628

    <tensorflow.python.keras.callbacks.History at 0x2a6a2e66d90>

<br>

✋ **compile의 파라미터**
- **optimizer**: 여기서는 경사 하강법을 수행하도록 한다. 
- **loss**: 손실 함수를 전달한다. 여기서는 이전 포스팅과 마찬가지로 교차-엔트로피 손실 함수를 선택한다. 
- **metrics**: 훈련하는 동안 모델 성능(최적화 프로세스에서 사용되지 않는 손실과 달리) 더 시각적으로 보여주기 위해 평가되는 추가적인 거리 함수이다. 

✋ **sgd**
- 'sgd'를 케라스에 전달하는 것은 tf.keras.optimizers.SGD()를 전달하는 것과 동일하다. 전자는 읽기 쉽지만 후자는 학습률 같은 매개변수를 직접 지정할 수 있다는 장점이 있다. 케라스 메서드에 전달되는 손실, 메트릭 등 다른 대부분의 인자도 마찬가지다. 

✋ **sparse_categorical_crossentropy**
- sparse_categorical_crossentropy라고 하는 케라스 손실은 categorical_crossentropy와 동일한 교차 엔트로피 연산을 수행하지만, 전자는 실제 레이블을 입력으로 직접 받는 반면 후자는 그 전에 실제 레이블을 원-핫 레이블로 인코딩되어 받는다. 따라서 수작업으로 레이블을 변환하는 과정이 필요없다. 
