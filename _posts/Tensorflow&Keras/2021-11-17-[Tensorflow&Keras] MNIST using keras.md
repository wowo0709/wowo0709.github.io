---
layout: single
title: "[Deep Learning] MNIST using keras"
categories: ['AI', 'TensorflowKeras']
toc: true
toc_sticky: true
tag: ['MNIST','Keras']
---



## Keras with MNIST dataset


```python
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical
```

### Data load


```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
print(train_images.shape, test_images.shape)
print(train_labels.shape, test_labels.shape)
```

    (60000, 28, 28) (10000, 28, 28)
    (60000,) (10000,)


```python
type(train_images), type(train_images[0]), train_images[0].shape
```


    (numpy.ndarray, numpy.ndarray, (28, 28))


```python
type(train_labels), type(train_labels[0]), train_labels[0]
```


    (numpy.ndarray, numpy.uint8, 5)


```python
train_images.shape, train_labels.dtype
```


    ((60000, 28, 28), dtype('uint8'))

<br>

### MNIST image and Preprocessing


```python
import matplotlib.pyplot as plt
%matplotlib inline
plt.subplot(1,2,1)
plt.imshow(train_images[0], cmap=plt.cm.binary)
plt.subplot(1,2,2)
plt.imshow(train_images[0]/255, cmap=plt.cm.binary)
plt.show
```




![output_10_1](https://user-images.githubusercontent.com/70505378/142135074-5a4215f8-eeb6-40fc-98ab-dcfd9266266a.png)
    



```python
# just for checking
print(type(train_labels), train_labels[:10])
train_images.shape, train_labels.shape
```

    <class 'numpy.ndarray'> [5 0 4 1 9 2 1 3 1 4]
    ((60000, 28, 28), (60000,))

<br>


```python
train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32')/255         # scaling

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)   # one-hot encoding
test_labels = to_categorical(test_labels)
```


```python
train_images.shape, train_labels.shape
```


    ((60000, 28, 28, 1), (60000, 10))


```python
print(train_labels[:10], type(train_labels), )
```

    [[0. 0. 0. 0. 0. 1. 0. 0. 0. 0.]
     [1. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 0. 0. 0. 0. 0. 1.]
     [0. 0. 1. 0. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 1. 0. 0. 0. 0. 0. 0.]
     [0. 1. 0. 0. 0. 0. 0. 0. 0. 0.]
     [0. 0. 0. 0. 1. 0. 0. 0. 0. 0.]] <class 'numpy.ndarray'>

<br>

### MLP

입력 이미지를 하나로 쭉 펴서 모델 학습. 좋은 성능이 나오지 않음. 

여기서는 숫자가 가운데에 정렬되어 있기 때문에 비교적 좋은 성능을 보임. 

#### Single flattened layer


```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28*28*1))
train_images = train_images.astype('float32')/255         # scaling

test_images = test_images.reshape((10000, 28*28*1))
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)   # one-hot encoding
test_labels = to_categorical(test_labels)
```


```python
# a single MLP layer

model = models.Sequential()
# vectorize
model.add(layers.Dense(10, activation='softmax', input_shape = (28*28*1, )))
model.summary()

```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 10)                7850      
    =================================================================
    Total params: 7,850
    Trainable params: 7,850
    Non-trainable params: 0
    _________________________________________________________________

<br>

```python
model.compile(optimizer= 'rmsprop',             # set up hyperparamers
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])

history = model.fit(train_images, train_labels, epochs=30, batch_size=128, verbose=2)

test_loss, test_acc = model.evaluate(test_images, test_labels) # default batch size=32
print('test_acc = ',test_acc)
```

    Epoch 1/30
    469/469 - 2s - loss: 0.6063 - accuracy: 0.8486
    Epoch 2/30
    469/469 - 1s - loss: 0.3314 - accuracy: 0.9079
    ...
    Epoch 30/30
    469/469 - 1s - loss: 0.2482 - accuracy: 0.9336
    313/313 [==============================] - 1s 2ms/step - loss: 0.2729 - accuracy: 0.9275
    test_acc =  0.9275000095367432

```python
plt.subplot(1,2,1)
plt.plot(history.history['loss'])
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'])
```




![output_20_1](https://user-images.githubusercontent.com/70505378/142135076-0375317f-7b9f-4c81-9b92-4036e2c5f255.png)
    

```python
history.history.keys()
```


    dict_keys(['loss', 'accuracy'])

<br>

#### MLP with hidden layers


```python
model = models.Sequential()
model.add(layers.Dense(100, activation='relu', input_shape = (28*28*1, ))) # fully-connected
model.add(layers.Dense(100, activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_1 (Dense)              (None, 100)               78500     
    _________________________________________________________________
    dense_2 (Dense)              (None, 100)               10100     
    _________________________________________________________________
    flatten (Flatten)            (None, 100)               0         
    _________________________________________________________________
    dense_3 (Dense)              (None, 10)                1010      
    =================================================================
    Total params: 89,610
    Trainable params: 89,610
    Non-trainable params: 0
    _________________________________________________________________



```python
print(model.input)
print(model.output)
```

    KerasTensor(type_spec=TensorSpec(shape=(None, 784), dtype=tf.float32, name='dense_1_input'), name='dense_1_input', description="created by layer 'dense_1_input'")
    KerasTensor(type_spec=TensorSpec(shape=(None, 10), dtype=tf.float32, name=None), name='dense_3/Softmax:0', description="created by layer 'dense_3'")



```python
print(model.input_shape)
print(model.output_shape)
model.input
```

    (None, 784)
    (None, 10)
    <KerasTensor: shape=(None, 784) dtype=float32 (created by layer 'dense_1_input')>






```python
model.layers[0].input, model.layers[0].output, model.layers[0].trainable
```




    (<KerasTensor: shape=(None, 784) dtype=float32 (created by layer 'dense_1_input')>,
     <KerasTensor: shape=(None, 100) dtype=float32 (created by layer 'dense_1')>,
     True)

<br>


```python
model.compile(optimizer= 'rmsprop',             # set up hyperparamers
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])
```


```python
model.fit(train_images, train_labels, epochs=30, batch_size=200) 
```

    Epoch 1/30
    300/300 [==============================] - 1s 4ms/step - loss: 0.3686 - accuracy: 0.8949
    Epoch 2/30
    300/300 [==============================] - 1s 4ms/step - loss: 0.1604 - accuracy: 0.9528
    ...
    Epoch 30/30
    300/300 [==============================] - 1s 4ms/step - loss: 0.0023 - accuracy: 0.9993


```python
test_loss, test_acc = model.evaluate(test_images, test_labels) # default batch size=32
print('test_acc = ',test_acc)
```

    313/313 [==============================] - 1s 2ms/step - loss: 0.1246 - accuracy: 0.9789
    test_acc =  0.9789000153541565

<br>

<br>

### CNN 


```python
# keras.layers.Conv2D(filters, kernel_size,...)
# - filters: the dimensionality of the output space (the number of output filters).
# - parameter 수: 입력채널수 X 필터폭 X 필터높이 X 출력채널수 
# - bias 도 고려
```


```python
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape(60000, 28, 28, 1)
train_images = train_images.astype('float32')/255         # scaling

test_images = test_images.reshape(10000, 28, 28, 1)
test_images = test_images.astype('float32')/255

train_labels = to_categorical(train_labels)   # one-hot encoding
test_labels = to_categorical(test_labels)
```


```python
from keras import layers
from keras import models

model = models.Sequential()

model.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (28, 28, 1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3), activation='relu'))
# print(model.output_shape)

model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))
print(model.output_shape)

model.summary()   
```

    (None, 10)
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
    dense (Dense)                (None, 10)                5770      
    =================================================================
    Total params: 61,514
    Trainable params: 61,514
    Non-trainable params: 0
    _________________________________________________________________

```python
# number of parameters
# conv2d_1 : 3*3*1*32 + 32 = 320
# conv2d_2 : 3*3*32*64 + 64 = 18496
# conv2d_3 : 3*3*64*64 + 64 = 36928
# dense_6 : 3*3*64*10 + 10 = 5770
```

<br>


```python
model.compile(optimizer= 'rmsprop',
             loss = 'categorical_crossentropy',
             metrics = ['accuracy'])
```


```python
model.fit(train_images, train_labels, epochs=30, batch_size=200)
```

    Epoch 1/30
    300/300 [==============================] - 19s 8ms/step - loss: 0.6169 - accuracy: 0.8129
    Epoch 2/30
    300/300 [==============================] - 2s 8ms/step - loss: 0.0823 - accuracy: 0.9749
    ...
    Epoch 30/30
    300/300 [==============================] - 2s 8ms/step - loss: 7.0306e-04 - accuracy: 0.9998


```python
test_loss, test_acc = model.evaluate(test_images, test_labels)
print('test_acc = ',test_acc)
```

    313/313 [==============================] - 1s 2ms/step - loss: 0.0607 - accuracy: 0.9912
    test_acc =  0.9911999702453613

<br>

<br>

## Without Deep Learning


```python
# just for reference
from sklearn.model_selection import train_test_split
from sklearn import datasets, svm, metrics
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score
import numpy as np
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28*28*1))
X = train_images.astype('float32')/255         # scaling
y = train_labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)
```


```python
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((45000, 784), (15000, 784), (45000,), (15000,))




```python
clf = SGDClassifier(max_iter=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
```

    0.9111333333333334

