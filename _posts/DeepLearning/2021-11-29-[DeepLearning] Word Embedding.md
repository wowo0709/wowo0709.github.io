---
layout: single
title: "[Deep Learning] Word Embedding"
categories: ['AI', 'DeepLearning']
toc: true
toc_sticky: true
tag: ['Embedding']
---



## Word Embedding with Keras Embedding Layer

- 정수 인덱스를 벡터로 매핑하는 딕셔너리 구조 (인덱스 크기, 벡터 크기)
- 학습 시키는 데이터에 따라 다른 임베딩이 만들어진다.

- IMDB 영화 리뷰 데이터를 사용한 임베딩 예제
  - IMDB: (internet movie database) the world's most popular and authoritative source for movie, TV and celebrity content


```python
import tensorflow as tf
import tensorflow.keras as keras 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, Dropout
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
import os, os.path
import zipfile
from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

- 5000 개의 단어만 사용하고, 각 문장에서는 뒤에서부터 500 개의 단어만 사용하겠음.


```python
max_features = 5000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
```

```python
y_train[:1000].sum(), y_train[-1000:].sum()   # can assume equally distributed
```


    (494, 498)


```python
x_train.shape, x_test.shape, y_train.shape, y_test.shape
```


    ((25000,), (25000,), (25000,), (25000,))


```python
print(y_train[6])
print(x_train[6])
```

    1
    [1, 2, 365, 1234, 5, 1156, 354, 11, 14, 2, 2, 7, 1016, 2, 2, 356, 44, 4, 1349, 500, 746, 5, 200, 4, 4132, 11, 2, 2, 1117, 1831, 2, 5, 4831, 26, 6, 2, 4183, 17, 369, 37, 215, 1345, 143, 2, 5, 1838, 8, 1974, 15, 36, 119, 257, 85, 52, 486, 9, 6, 2, 2, 63, 271, 6, 196, 96, 949, 4121, 4, 2, 7, 4, 2212, 2436, 819, 63, 47, 77, 2, 180, 6, 227, 11, 94, 2494, 2, 13, 423, 4, 168, 7, 4, 22, 5, 89, 665, 71, 270, 56, 5, 13, 197, 12, 161, 2, 99, 76, 23, 2, 7, 419, 665, 40, 91, 85, 108, 7, 4, 2084, 5, 4773, 81, 55, 52, 1901]

<br>

```python
word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print('---review with words---')
print([id2word.get(i, ' ') for i in x_train[6]])
print('---label---')
print(y_train[6])
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/imdb_word_index.json
    1646592/1641221 [==============================] - 0s 0us/step
    ---review with words---
    ['the', 'and', 'full', 'involving', 'to', 'impressive', 'boring', 'this', 'as', 'and', 'and', 'br', 'villain', 'and', 'and', 'need', 'has', 'of', 'costumes', 'b', 'message', 'to', 'may', 'of', 'props', 'this', 'and', 'and', 'concept', 'issue', 'and', 'to', "god's", 'he', 'is', 'and', 'unfolds', 'movie', 'women', 'like', "isn't", 'surely', "i'm", 'and', 'to', 'toward', 'in', "here's", 'for', 'from', 'did', 'having', 'because', 'very', 'quality', 'it', 'is', 'and', 'and', 'really', 'book', 'is', 'both', 'too', 'worked', 'carl', 'of', 'and', 'br', 'of', 'reviewer', 'closer', 'figure', 'really', 'there', 'will', 'and', 'things', 'is', 'far', 'this', 'make', 'mistakes', 'and', 'was', "couldn't", 'of', 'few', 'br', 'of', 'you', 'to', "don't", 'female', 'than', 'place', 'she', 'to', 'was', 'between', 'that', 'nothing', 'and', 'movies', 'get', 'are', 'and', 'br', 'yes', 'female', 'just', 'its', 'because', 'many', 'br', 'of', 'overly', 'to', 'descent', 'people', 'time', 'very', 'bland']
    ---label---
    1

<br>

```python
# 각 문장이 몇개의 단어로 구성되어 있는지 확인
[len(x_train[i]) for i in range(10)]
```


    [218, 189, 141, 550, 147, 43, 123, 562, 233, 130]


```python
print(max([len(x_train[i]) for i in range(25000)]), min([len(x_train[i]) for i in range(25000)]))
print(max([len(x_test[i]) for i in range(25000)]), min([len(x_test[i]) for i in range(25000)]))
```

    2494 11
    2315 7

```python
x_train[0:2]   # words tokenized and expressed by (word) numbers
```


    array([list([1, 14, 22, 16, 43, 530, 973, 1622, 1385, 65, 458, 4468, 66, 3941, 4, 173, 36, 256, 5, 25, 100, 43, 838, 112, 50, 670, 2, 9, 35, 480, 284, 5, 150, 4, 172, 112, 167, 2, 336, 385, 39, 4, 172, 4536, 1111, 17, 546, 38, 13, 447, 4, 192, 50, 16, 6, 147, 2025, 19, 14, 22, 4, 1920, 4613, 469, 4, 22, 71, 87, 12, 16, 43, 530, 38, 76, 15, 13, 1247, 4, 22, 17, 515, 17, 12, 16, 626, 18, 2, 5, 62, 386, 12, 8, 316, 8, 106, 5, 4, 2223, 2, 16, 480, 66, 3785, 33, 4, 130, 12, 16, 38, 619, 5, 25, 124, 51, 36, 135, 48, 25, 1415, 33, 6, 22, 12, 215, 28, 77, 52, 5, 14, 407, 16, 82, 2, 8, 4, 107, 117, 2, 15, 256, 4, 2, 7, 3766, 5, 723, 36, 71, 43, 530, 476, 26, 400, 317, 46, 7, 4, 2, 1029, 13, 104, 88, 4, 381, 15, 297, 98, 32, 2071, 56, 26, 141, 6, 194, 2, 18, 4, 226, 22, 21, 134, 476, 26, 480, 5, 144, 30, 2, 18, 51, 36, 28, 224, 92, 25, 104, 4, 226, 65, 16, 38, 1334, 88, 12, 16, 283, 5, 16, 4472, 113, 103, 32, 15, 16, 2, 19, 178, 32]),
           list([1, 194, 1153, 194, 2, 78, 228, 5, 6, 1463, 4369, 2, 134, 26, 4, 715, 8, 118, 1634, 14, 394, 20, 13, 119, 954, 189, 102, 5, 207, 110, 3103, 21, 14, 69, 188, 8, 30, 23, 7, 4, 249, 126, 93, 4, 114, 9, 2300, 1523, 5, 647, 4, 116, 9, 35, 2, 4, 229, 9, 340, 1322, 4, 118, 9, 4, 130, 4901, 19, 4, 1002, 5, 89, 29, 952, 46, 37, 4, 455, 9, 45, 43, 38, 1543, 1905, 398, 4, 1649, 26, 2, 5, 163, 11, 3215, 2, 4, 1153, 9, 194, 775, 7, 2, 2, 349, 2637, 148, 605, 2, 2, 15, 123, 125, 68, 2, 2, 15, 349, 165, 4362, 98, 5, 4, 228, 9, 43, 2, 1157, 15, 299, 120, 5, 120, 174, 11, 220, 175, 136, 50, 9, 4373, 228, 2, 5, 2, 656, 245, 2350, 5, 4, 2, 131, 152, 491, 18, 2, 32, 2, 1212, 14, 9, 6, 371, 78, 22, 625, 64, 1382, 9, 8, 168, 145, 23, 4, 1690, 15, 16, 4, 1355, 5, 28, 6, 52, 154, 462, 33, 89, 78, 285, 16, 145, 95])],
          dtype=object)

<br>


```python
# 마지막 500개의 단어들만 사용한다. -> 500개보다 적으면 똑같은 길이로 만들어 준다.
maxlen = 500
x_train_p=preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test_p=preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)
print(x_train_p.shape, x_test_p.shape)
```

    (25000, 500) (25000, 500)

```python
y_train.shape, y_test.shape
```


    ((25000,), (25000,))

- Embedding()은 (number of samples, input_length)인 2D 정수 텐서를 입력받습니다. 이 때 각 sample은 정수 인코딩이 된 결과로, 정수의 시퀀스입니다. Embedding()은 워드 임베딩 작업을 수행하고 (number of samples, input_length, embedding word dimensionality)인 3D 텐서를 리턴합니다.

<br>


```python
model = Sequential()
model.add(Embedding(5000, 32, input_length=maxlen)) # input 각 단어에 대해 32-vector 로 임베딩
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, 500, 32)           160000    
    _________________________________________________________________
    flatten (Flatten)            (None, 16000)             0         
    _________________________________________________________________
    dense (Dense)                (None, 1)                 16001     
    =================================================================
    Total params: 176,001
    Trainable params: 176,001
    Non-trainable params: 0
    _________________________________________________________________

```python
model.input_shape, model.output_shape
```


    ((None, 500), (None, 1))


```python
x_train_p.shape, y_train.shape
```


    ((25000, 500), (25000,))

<br>


```python
history = model.fit(x_train_p, y_train,
                    epochs=10, batch_size=500,
                    validation_split=0.2)
```

    Epoch 1/10
    40/40 [==============================] - 2s 28ms/step - loss: 0.6844 - acc: 0.5545 - val_loss: 0.6608 - val_acc: 0.6428
    ...
    Epoch 10/10
    40/40 [==============================] - 1s 22ms/step - loss: 0.1239 - acc: 0.9635 - val_loss: 0.2851 - val_acc: 0.8860

```python
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
epochs = range(len(acc))

plt.plot(epochs, acc, '--')
plt.plot(epochs, val_acc)
plt.title('Training(--) and validation accuracy')

plt.subplot(1,2,2)
plt.plot(epochs, loss,  '--')
plt.plot(epochs, val_loss)
plt.title('Training(--) and validation loss')
```






​    
![output_19_1](https://user-images.githubusercontent.com/70505378/143830658-6be694af-8089-42fb-8220-6dd18fffa526.png)
​    

<br>

```python
# test score
scores = model.evaluate(x_test_p, y_test, verbose=0)
print('Test accuracy:', scores[1])
```

    Test accuracy: 0.8815600275993347


- 위의 결과는 500 개의 단어만 고려한 것임. 
- 각 단어를 독립적으로 다루었으며, 문장의 구성 정보를 고려하지 않음
- 문장의 구조 정보를 고려하려면 임베딩 층 위에 합성곱이나 순환신경망 층을 추가한다

<br>

<br>

## CNN (1D)


```python
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=32, input_length=maxlen))
model.add(Conv1D(128, 5, activation="relu"))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation="relu"))
model.add(MaxPooling1D(5))
model.add(Conv1D(128, 5, activation="relu"))
model.add(GlobalMaxPooling1D())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_1 (Embedding)      (None, 500, 32)           160000    
    _________________________________________________________________
    conv1d (Conv1D)              (None, 496, 128)          20608     
    _________________________________________________________________
    max_pooling1d (MaxPooling1D) (None, 99, 128)           0         
    _________________________________________________________________
    conv1d_1 (Conv1D)            (None, 95, 128)           82048     
    _________________________________________________________________
    max_pooling1d_1 (MaxPooling1 (None, 19, 128)           0         
    _________________________________________________________________
    conv1d_2 (Conv1D)            (None, 15, 128)           82048     
    _________________________________________________________________
    global_max_pooling1d (Global (None, 128)               0         
    _________________________________________________________________
    dense_1 (Dense)              (None, 128)               16512     
    _________________________________________________________________
    dropout (Dropout)            (None, 128)               0         
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 129       
    =================================================================
    Total params: 361,345
    Trainable params: 361,345
    Non-trainable params: 0
    _________________________________________________________________

<br>

```python
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['acc'])
history = model.fit(x_train_p, y_train,
                    epochs=10, batch_size=500,
                    validation_split=0.2)
```

    Epoch 1/10
    40/40 [==============================] - 5s 57ms/step - loss: 0.6925 - acc: 0.5101 - val_loss: 0.6844 - val_acc: 0.6734
    ...
    Epoch 10/10
    40/40 [==============================] - 2s 54ms/step - loss: 0.0622 - acc: 0.9817 - val_loss: 0.5348 - val_acc: 0.8644

<br>

```python
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
epochs = range(len(acc))

plt.plot(epochs, acc, '--')
plt.plot(epochs, val_acc)
plt.title('Training(--) and validation accuracy')

plt.subplot(1,2,2)
plt.plot(epochs, loss,  '--')
plt.plot(epochs, val_loss)
plt.title('Training(--) and validation loss')
```




![output_25_1](https://user-images.githubusercontent.com/70505378/143830666-285823a1-f564-493e-a6d8-ecfc7256391e.png)
    

<br>

```python
# test score
scores = model.evaluate(x_test_p, y_test, verbose=0)
print('Test accuracy:', scores[1])
```

    Test accuracy: 0.8475599884986877

```python
# prediction
model.predict(x_test_p[0:5])
```


    array([[0.01152737],
           [0.99999976],
           [0.06551089],
           [0.02209353],
           [0.9896063 ]], dtype=float32)

<br>

<br>

## GRU (RNN)


```python
x_train_p.shape
```


    (25000, 500)


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense, LSTM, GRU

model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=32, input_length=maxlen))
model.add(GRU(32))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_2 (Embedding)      (None, 500, 32)           160000    
    _________________________________________________________________
    gru (GRU)                    (None, 32)                6336      
    _________________________________________________________________
    dense_3 (Dense)              (None, 32)                1056      
    _________________________________________________________________
    dropout_1 (Dropout)          (None, 32)                0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 33        
    =================================================================
    Total params: 167,425
    Trainable params: 167,425
    Non-trainable params: 0
    _________________________________________________________________

<br>

```python
%time history = model.fit(x_train_p, y_train, epochs=10, batch_size=500, validation_split=0.2)
```

    Epoch 1/10
    40/40 [==============================] - 4s 50ms/step - loss: 0.6888 - acc: 0.5610 - val_loss: 0.6766 - val_acc: 0.6250
    ...
    Epoch 10/10
    40/40 [==============================] - 2s 43ms/step - loss: 0.1485 - acc: 0.9554 - val_loss: 0.3976 - val_acc: 0.8702
    Wall time: 19.4 s

<br>

```python
# test score
scores = model.evaluate(x_test_p, y_test, verbose=0)
print('Test accuracy:', scores[1])
```

    Test accuracy: 0.8632000088691711

```python
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
epochs = range(len(acc))

plt.plot(epochs, acc, '--')
plt.plot(epochs, val_acc)
plt.title('Training(--) and validation accuracy')

plt.subplot(1,2,2)
plt.plot(epochs, loss,  '--')
plt.plot(epochs, val_loss)
plt.title('Training(--) and validation loss')
```




![output_34_1](https://user-images.githubusercontent.com/70505378/143830667-6cc6952f-3158-4d50-9760-9238f504b7e5.png)
    

<br>

```python
# prediction
model.predict(x_test_p[0:5])
```


    array([[0.11125369],
           [0.99512935],
           [0.9718255 ],
           [0.20294943],
           [0.99939275]], dtype=float32)

- y_test[2] 는 무슨 문장일까?


```python
word2id = imdb.get_word_index()
id2word = {i: word for word, i in word2id.items()}
print([id2word.get(i, ' ') for i in x_test[2]])
print(y_test[2])
```

    ['the', 'plot', 'near', 'ears', 'recent', 'and', 'and', 'of', 'him', 'flicks', 'frank', 'br', 'by', 'excellent', 'and', 'br', 'of', 'past', 'and', 'near', 'really', 'all', 'and', 'family', 'four', 'and', 'to', 'movie', 'that', 'obvious', 'family', 'brave', 'movie', 'is', 'got', 'say', 'and', 'with', 'up', 'comment', 'this', 'and', 'been', 'of', 'entertaining', 'not', 'be', 'and', 'james', 'in', 'you', 'seen', 'and', 'and', 'portrayed', 'dirty', 'in', 'so', 'washington', 'and', 'this', 'you', 'minutes', 'no', 'all', 'station', 'all', 'after', 'and', 'promising', 'who', 'and', 'and', 'and', 'to', 'and', 'any', 'by', 'speed', 'they', 'is', 'my', 'as', 'screams', 'dirty', 'in', 'of', 'full', 'br', 'pacino', 'dignity', 'need', 'men', 'of', 'and', 'popular', 'really', 'all', 'way', 'this', 'and', 'this', 'and', 'they', 'is', 'my', 'no', 'standard', 'certainly', 'near', 'br', 'an', 'beach', 'with', 'this', 'make', 'and', 'i', 'i', 'of', 'fails', 'and', 'br', 'of', 'finished', 'wear', 'psycho', 'and', 'in', 'learn', 'in', 'twice', 'know', 'by', 'br', 'be', 'how', 'rings', 'and', 'with', 'is', 'seemed', 'fails', 'visually', 'and', 'extremely', 'movie', 'and', "it's", 'of', 'and', 'like', 'children', 'is', 'easily', 'is', 'and', 'br', 'simply', 'must', 'well', 'at', 'although', 'this', 'family', 'an', 'br', 'many', 'not', 'scene', 'that', 'it', 'time', 'seemed', 'de', 'ignored', 'up', 'they', 'boat', 'morning', 'like', 'well', 'force', 'of', 'and', 'sent', 'been', 'history', 'like', 'story', 'its', 'disappointing', 'same', 'of', 'club', 'and', 'watching', 'husband', 'reviewer', 'to', 'although', 'that', 'around', 'and', 'except', 'to', 'de', 'and', 'br', 'of', 'you', 'available', 'but', 'hours', 'animals', 'showing', 'br', 'of', 'and', 'than', 'dead', 'white', 'splatter', 'waiting', 'film', 'and', 'to', 'and', 'this', 'documentary', 'in', '3', 'and', 'of', 'accents', 'and', 'br', 'of', 'ann', 'i', 'i', 'comes', '9', 'it', 'place', 'this', 'is', 'and', 'of', 'and', 'and', 'know', 'of', 'and', 'he', 'bonus', 'film', 'were', 'central', 'to', 'one', 'oh', 'is', 'excellent', 'and', 'in', 'can', 'when', 'from', 'well', 'people', 'in', "characters'", 'chief', 'from', 'leaving', 'in', 'and', 'and', 'but', 'is', 'easily', 'of', 'and', 'he', 'and', 'speak', 'this', 'as', 'today', 'paul', 'that', 'against', 'one', 'will', 'actual', 'in', 'could', 'her', 'plot', 'and', 'and', 'few', 'grade', 'and', 'go', 'and', 'but', 'be', 'lot', 'it', 'oliver', 'movie', 'is', 'and', 'picture', 'and', 'feel', 'this', 'of', 'and', 'like', 'different', 'just', 'clichéd', 'girl', 'at', 'finds', 'is', 'and', 'no', 'and', 'glory', 'any', 'is', "children's", 'just', 'moment', 'like', 'and', 'any', 'of', 'and', 'leaving', 'for', 'as', 'it', 'even', 'cliche', 'to', 'purchased', 'is', 'money', 'easily', 'and', 'and', 'glory', 'any', 'is', 'and', 'i', 'i', 'and', 'film', 'as', 'and', 'set', 'actually', 'easily', 'like', 'and', 'sequel', 'any', 'of', 'and', 'ryan', 'made', 'film', 'is', 'and', 'br', 'and', 'constant', 'and', 'of', '90s', 'letting', 'deep', 'in', 'act', 'made', 'of', 'road', 'in', 'of', 'and', 'movie', 'and', 'rural', 'vhs', 'of', 'share', 'in', 'reaching', 'fact', 'of', 'and', 'and', 'and', 'of', '90s', 'to', 'them', 'book', 'are', 'is', 'and', 'and', 'and', 'and', 'they', 'funniest', 'is', 'white', 'courage', 'and', 'vegas', 'wooden', 'br', 'of', 'gender', 'and', 'unfortunately', 'of', '1968', 'no', 'of', 'years', 'and', 'and', 'true', 'up', 'and', 'and', 'but', '3', 'all', 'ordinary', 'be', 'and', 'to', 'and', 'were', 'deserve', 'film', 'and', 'and', 'of', 'creative', 'br', 'comes', 'their', 'kung', 'who', 'is', 'and', 'and', 'out', 'new', 'all', 'it', 'incomprehensible', 'it', 'episode', 'much', "that's", 'including', 'i', 'i', 'cartoon', 'of', 'my', 'certain', 'no', 'as', 'and', 'over', 'you', 'with', 'way', 'to', 'cartoon', 'of', 'enough', 'for', 'that', 'with', 'way', 'who', 'is', 'finished', 'and', 'they', 'of', 'and', 'br', 'for', 'and', 'and', 'stunts', 'black', 'that', 'story', 'at', 'actual', 'in', 'can', 'as', 'movie', 'is', 'and', 'has', 'though', 'songs', 'and', 'action', "it's", 'action', 'his', 'one', 'me', 'and', 'and', 'this', 'second', 'no', 'all', 'way', 'and', 'not', 'lee', 'and', 'be', 'moves', 'br', 'figure', 'of', 'you', 'boss', 'movie', 'is', 'and', '9', 'br', 'propaganda', 'and', 'and', 'after', 'at', 'of', 'smoke', 'splendid', 'snow', 'saturday', "it's", 'results', 'this', 'of', 'load', "it's", 'think', 'class', 'br', 'think', 'cop', 'for', 'games', 'make', 'southern', 'things', 'to', 'it', 'and', 'who', 'and', 'if', 'is', 'boyfriend', 'you', 'which', 'is', 'tony', 'by', 'this', 'make', 'and', 'too', 'not', 'make', 'above', 'it', 'even', 'background']
    1

<br>

<br>

## Combine CNN and RNN together


```python
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=32, input_length=maxlen))
model.add(Dropout(0.5))
model.add(Conv1D(64, 5, padding='valid', activation='relu',strides=1))
model.add(MaxPooling1D(pool_size=4))
model.add(LSTM(55))
model.add(Dense(1, activation='sigmoid'))
model.summary()
```

    Model: "sequential_3"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_3 (Embedding)      (None, 500, 32)           160000    
    _________________________________________________________________
    dropout_2 (Dropout)          (None, 500, 32)           0         
    _________________________________________________________________
    conv1d_3 (Conv1D)            (None, 496, 64)           10304     
    _________________________________________________________________
    max_pooling1d_2 (MaxPooling1 (None, 124, 64)           0         
    _________________________________________________________________
    lstm (LSTM)                  (None, 55)                26400     
    _________________________________________________________________
    dense_5 (Dense)              (None, 1)                 56        
    =================================================================
    Total params: 196,760
    Trainable params: 196,760
    Non-trainable params: 0
    _________________________________________________________________

<br>

```python
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
```


```python
%time history = model.fit(x_train_p, y_train, epochs=10, batch_size=500, validation_split=0.2)
```

    Epoch 1/10
    40/40 [==============================] - 3s 54ms/step - loss: 0.6757 - acc: 0.5888 - val_loss: 0.6171 - val_acc: 0.6452
    ...
    Epoch 10/10
    40/40 [==============================] - 2s 46ms/step - loss: 0.1449 - acc: 0.9482 - val_loss: 0.3144 - val_acc: 0.8800
    Wall time: 19.7 s

<br>

```python
# test score
scores = model.evaluate(x_test_p, y_test, verbose=0)
print('Test accuracy:', scores[1])
```

    Test accuracy: 0.8714799880981445

```python
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure(figsize=(10,6))
plt.subplot(1,2,1)
epochs = range(len(acc))

plt.plot(epochs, acc, '--')
plt.plot(epochs, val_acc)
plt.title('Training(--) and validation accuracy')

plt.subplot(1,2,2)
plt.plot(epochs, loss,  '--')
plt.plot(epochs, val_loss)
plt.title('Training(--) and validation loss')
```




![output_44_1](https://user-images.githubusercontent.com/70505378/143830668-a2f3f207-ef27-499c-83b8-7a17bf7bdaed.png)
    

<br>

<br>

## Exercise (연습)

- By default, if a GPU is available, the embedding matrix will be placed on the GPU. This achieves the best performance.
- in order to use CPU (too big to fit on GPU), you should use CPU
  - with tf.device('cpu:0'): 
  -embedding_layer = Embedding(...)
  - embedding_layer.build()


```python
import tensorflow as tf
# 문장 토큰화와 단어 토큰화
text=[['Hope', 'to', 'see', 'you', 'soon'],
      ['Nice', 'to', 'see', 'you', 'again']]

# 각 단어에 대한 정수 인코딩
text=[[0, 1, 2, 3, 4],[5, 1, 2, 3, 6]]

# 위 데이터가 아래의 임베딩 층의 입력이 된다. (훈련 없이 모양만 보기 위함)
embedding_layer = Embedding(7, 2, input_length=5)
result = embedding_layer(tf.constant([0, 1, 2, 3, 4, 5, 6]))
print(result.numpy())

# 7은 단어의 개수. 즉, 단어 집합(vocabulary)의 크기이다.
# 2는 임베딩한 후의 벡터의 크기이다.
# 5는 각 입력 시퀀스의 길이. 즉, input_length이다.

```

    [[ 0.01560372 -0.04292933]
     [ 0.0303275  -0.00451677]
     [-0.00240674 -0.01646507]
     [ 0.01426259 -0.02349484]
     [ 0.00645077 -0.04100402]
     [-0.01287322 -0.01720787]
     [ 0.02342038 -0.00832408]]

<br>

```python
# input_legnth를 지정하지 않았을 때 (가변 길이 문장, None)
model = Sequential()
model.add(Embedding(7, 2))
model.add(Flatten())
model.summary()
```

    Model: "sequential_7"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_8 (Embedding)      (None, None, 2)           14        
    _________________________________________________________________
    flatten_3 (Flatten)          (None, None)              0         
    =================================================================
    Total params: 14
    Trainable params: 14
    Non-trainable params: 0
    _________________________________________________________________

```python
# input_length를 지정했을 때 (불변 길이 문장, input_length)
model = Sequential()
model.add(Embedding(7, 2, input_length=5)) # need input_length to be connected to Flatten then Dense layers
model.add(Flatten())
model.summary()
```

    Model: "sequential_8"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding_9 (Embedding)      (None, 5, 2)              14        
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 10)                0         
    =================================================================
    Total params: 14
    Trainable params: 14
    Non-trainable params: 0
    _________________________________________________________________

