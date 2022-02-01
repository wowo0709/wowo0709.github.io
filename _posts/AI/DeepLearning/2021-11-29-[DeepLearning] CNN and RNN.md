---
layout: single
title: "[Deep Learning] CNN and RNN"
categories: ['AI', 'DeepLearning']
toc: true
toc_sticky: true
tag: []
---

## Convolutional Neural Nets (CNN)

- from Deep Learning with Python, by Francois Chollet

- 컨브넷이 (image_height, image_width, image_channels) 크기의 입력 텐서를 사용한다는 점이 중요합니다(배치 차원은 포함하지 않습니다). 이 예제에서는 MNIST 이미지 포맷인 (28, 28, 1) 크기의 입력을 처리하도록 컨브넷을 설정해야 합니다. 이 때문에 첫 번째 층의 매개변수로 input_shape=(28, 28, 1)을 전달합니다.
- Conv2D와 MaxPooling2D 층의 출력은 (height, width, channels) 크기의 3D 텐서입니다. 높이와 넓이 차원은 네트워크가 깊어질수록 작아지는 경향이 있습니다. 채널의 수는 Conv2D 층에 전달된 첫 번째 매개변수에 의해 조절됩니다(32개 또는 64개).
- 다음 단계에서 마지막 층의 ((3, 3, 64) 크기인) 출력 텐서를 완전 연결 네트워크에 주입합니다. 이 네트워크는 이미 익숙하게 보았던 Dense 층을 쌓은 분류기입니다. 이 분류기는 1D 벡터를 처리하는데 이전 층의 출력이 3D 텐서입니다. 그래서 먼저 3D 출력을 1D 텐서로 펼쳐야 합니다. 그다음 몇 개의 Dense 층을 추가합니다.


```python
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers
from tensorflow.keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d (Conv2D)              (None, 26, 26, 32)        320       
    _________________________________________________________________
    max_pooling2d (MaxPooling2D) (None, 13, 13, 32)        0         
    _________________________________________________________________
    conv2d_1 (Conv2D)            (None, 11, 11, 64)        18496     
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 5, 5, 64)          0         
    _________________________________________________________________
    conv2d_2 (Conv2D)            (None, 3, 3, 64)          36928     
    _________________________________________________________________
    flatten (Flatten)            (None, 576)               0         
    _________________________________________________________________
    dense (Dense)                (None, 64)                36928     
    _________________________________________________________________
    dense_1 (Dense)              (None, 10)                650       
    =================================================================
    Total params: 93,322
    Trainable params: 93,322
    Non-trainable params: 0
    _________________________________________________________________

<br>

```python
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 0s 0us/step
    11501568/11490434 [==============================] - 0s 0us/step

```python
train_images.shape, train_labels.shape
```


    ((60000, 28, 28, 1), (60000, 10))

<br>


```python
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
history = model.fit(train_images, train_labels, epochs=5, batch_size=64)
```

    Epoch 1/5
    938/938 [==============================] - 34s 9ms/step - loss: 0.1740 - accuracy: 0.9460
    Epoch 2/5
    938/938 [==============================] - 8s 9ms/step - loss: 0.0462 - accuracy: 0.9855
    Epoch 3/5
    938/938 [==============================] - 8s 9ms/step - loss: 0.0325 - accuracy: 0.9898
    Epoch 4/5
    938/938 [==============================] - 8s 9ms/step - loss: 0.0247 - accuracy: 0.9926
    Epoch 5/5
    938/938 [==============================] - 8s 9ms/step - loss: 0.0188 - accuracy: 0.9941

<br>

```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
```

    313/313 [==============================] - 2s 4ms/step - loss: 0.0391 - accuracy: 0.9896


```python
import matplotlib.pyplot as plt
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'])
```




![output_9_1](https://user-images.githubusercontent.com/70505378/143826415-62328c50-85cd-43e9-8ad7-88a42a9da34b.png)
    

<br>

<br>

## Recurrent Neural Network (RNN)

- from https://github.com/gilbutITbook/006975

- SimpleRNN: 
  - SimpleRNN이 한 가지 다른 점은 넘파이 예제처럼 하나의 시퀀스가 아니라 다른 케라스 층과 마찬가지로 시퀀스 배치를 처리한다는 것입니다. 즉, (timesteps, input_features) 크기가 아니라 (batch_size, timesteps, input_features) 크기의 입력을 받습니다.
  - 케라스에 있는 모든 순환 층과 동일하게 SimpleRNN은 두 가지 모드로 실행할 수 있습니다. 각 타임스텝의 출력을 모은 전체 시퀀스를 반환하거나(크기가 (batch_size, timesteps, output_features)인 3D 텐서), 입력 시퀀스에 대한 마지막 출력만 반환할 수 있습니다(크기가 (batch_size, output_features)인 2D 텐서). 이 모드는 객체를 생성할 때 return_sequences 매개변수로 선택할 수 있습니다. 


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, SimpleRNN

model = Sequential()
model.add(Embedding(10000, 32)) # 문장 길이(단어 개수) 10000, 단어벡터 차원 32
model.add(SimpleRNN(32))
model.summary()
model.input_shape, model.output_shape
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, None, 32)          320000    
    _________________________________________________________________
    simple_rnn (SimpleRNN)       (None, 32)                2080      
    =================================================================
    Total params: 322,080
    Trainable params: 322,080
    Non-trainable params: 0
    _________________________________________________________________

    ((None, None), (None, 32))

- embedding(in_dim, out_dim): 10000 * 32 = 320000
- simpleRNN: (32 + 32)*32 + 32 = 2080

<br>


```python
# 뒤에 flatten 이나 Dense layer 에 연결하기위해서는 length 가 고정되어야 함
model = Sequential()
model.add(Embedding(10000, 32, input_length=20)) # 인풋 20개
model.add(SimpleRNN(32))
model.summary()
model.input_shape, model.output_shape
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 20, 32)            320000    
    _________________________________________________________________
    simple_rnn_1 (SimpleRNN)     (None, 32)                2080      
    =================================================================
    Total params: 322,080
    Trainable params: 322,080
    Non-trainable params: 0
    _________________________________________________________________

    ((None, 20), (None, 32))

<br>


```python
# RNN layer의 중간 결과들을 저장
model = Sequential()
model.add(Embedding(10000, 32, input_length=20))
model.add(SimpleRNN(32, return_sequences=True)) # 중간 결과 반환
model.summary()
```

    Model: "sequential_5"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_4 (Embedding)      (None, 20, 32)            320000    
    _________________________________________________________________
    simple_rnn_4 (SimpleRNN)     (None, 20, 32)            2080      
    =================================================================
    Total params: 322,080
    Trainable params: 322,080
    Non-trainable params: 0
    _________________________________________________________________

<br>


- 네트워크의 표현력을 증가시키기 위해 여러 개의 순환 층을 차례대로 쌓는 것이 유용할 때가 있다. 이런 설정에서는 중간 층들이 전체 출력 시퀀스를 반환하도록 설정해야 한다:


```python
model = Sequential()
model.add(Embedding(10000, 32, input_length=20))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))  # 맨 위 층만 마지막 출력을 반환합니다.
model.summary()
```

    Model: "sequential_12"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_12 (Embedding)     (None, 20, 32)            320000    
    _________________________________________________________________
    simple_rnn_12 (SimpleRNN)    (None, 20, 32)            2080      
    _________________________________________________________________
    simple_rnn_13 (SimpleRNN)    (None, 20, 32)            2080      
    _________________________________________________________________
    simple_rnn_14 (SimpleRNN)    (None, 20, 32)            2080      
    _________________________________________________________________
    simple_rnn_15 (SimpleRNN)    (None, 32)                2080      
    =================================================================
    Total params: 328,320
    Trainable params: 328,320
    Non-trainable params: 0
    _________________________________________________________________

<br>


- input shape and output shape 확인


```python
import numpy as np
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
model.summary()
print(model.input_shape, model.output_shape)
input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)

assert output_array.shape == (32, 10, 64)
```

    Model: "sequential_16"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_16 (Embedding)     (None, 10, 64)            64000     
    =================================================================
    Total params: 64,000
    Trainable params: 64,000
    Non-trainable params: 0
    _________________________________________________________________
    (None, 10) (None, 10, 64)

