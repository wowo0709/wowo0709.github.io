---
layout: single
title: "[Tensorflow&Keras] Keras Callback"
categories: ['AI', 'TensorflowKeras']
toc: true
toc_sticky: true
tag: ['keras', 'callback']

---



## Callback - check point and early stopping

- Callback 함수: 명시적으로 호출되는 게 아니라 나중에 어떤 event 가 발생했을 때 호출되는 함수


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

import pandas as pd
import numpy as np
import os
import tensorflow as tf
```

- download wine.csv from https://codedragon.tistory.com/9480
  - class 1: red wine, 0: white wine

<br>


```python
df_all = pd.read_csv('wine.csv', header=None)
df = df_all.sample(frac=0.2)  # get only 20% of dataset
df.shape
```




    (1299, 13)




```python
df.head(1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }
    
    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>10</th>
      <th>11</th>
      <th>12</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>5510</th>
      <td>7.2</td>
      <td>0.23</td>
      <td>0.82</td>
      <td>1.3</td>
      <td>0.149</td>
      <td>70.0</td>
      <td>109.0</td>
      <td>0.99304</td>
      <td>2.93</td>
      <td>0.42</td>
      <td>9.2</td>
      <td>6</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>


```python
df[12].value_counts()
```




    0    980
    1    319
    Name: 12, dtype: int64

<br>


```python
dataset = df.values
X, y = dataset[:,0:12], dataset[:,12]

model = Sequential()
model.add(Dense(30,  input_dim=12, activation='relu'))
model.add(Dense(12, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
          optimizer='adam',
          metrics=['accuracy'])

# 모델 저장 폴더 만들기
MODEL_DIR = './model/'
if not os.path.exists(MODEL_DIR):
    os.mkdir(MODEL_DIR)

modelpath="./model/{epoch:02d}-{val_loss:.4f}.hdf5"

# 모델 업데이트 및 저장 (epoch 마다)
checkpointer = ModelCheckpoint(filepath=modelpath, 
                               monitor='val_loss', 
                               verbose=1, 
                               save_best_only=True)  # record only when imrpoved

# 테스트 오차가 줄지 않으면 학습 자동 중단 설정 (모니터할 값 저장)
early_stopping_callback = EarlyStopping(monitor='val_loss', 
                                        patience=100) # 좋아지지 않아도 몇 번까지 기다릴것인지
```


```python
y_loss, y_acc, y_vloss, y_vacc = [], [], [], []

history = model.fit(X, y, validation_split=0.2, 
                    epochs=2000, batch_size=100, verbose=0, 
                    callbacks=[early_stopping_callback,checkpointer])
y_loss = history.history['loss']
y_acc = history.history['accuracy']
y_vloss = history.history['val_loss']
y_vacc = history.history['val_accuracy']
x_len = np.arange(len(y_acc))
plt.ylim(0.,1.)
plt.title("Traing")
plt.plot(x_len, y_loss, "o", c="r", markersize=3)
plt.plot(x_len, y_acc, "o", c="b", markersize=3)
plt.show()
plt.title("Validation")
plt.ylim(0.,1.)
plt.plot(x_len, y_vloss, "o", c="r", markersize=3)
plt.plot(x_len, y_vacc, "o", c="b", markersize=3)
plt.show()
```


    Epoch 00001: val_loss improved from inf to 0.52311, saving model to ./model/01-0.5231.hdf5
    
    Epoch 00002: val_loss improved from 0.52311 to 0.43001, saving model to ./model/02-0.4300.hdf5
    
    Epoch 00003: val_loss improved from 0.43001 to 0.40039, saving model to ./model/03-0.4004.hdf5
    
    ...
    
    Epoch 00395: val_loss did not improve from 0.08230




![output_7_1](https://user-images.githubusercontent.com/70505378/142955088-c044ab71-86fb-450d-921d-3bcee8811855.png)
    




![output_7_2](https://user-images.githubusercontent.com/70505378/142955091-c15bcd1e-f69e-4f50-8a98-845764ea9999.png)
    



```python
history.history.keys()
```




    dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

<br>


```python
print("Acuracy: %.4f" %(model.evaluate(X, y)[1]))
```

    41/41 [==============================] - 0s 2ms/step - loss: 0.0510 - accuracy: 0.9823
    Acuracy: 0.9823


- 2000 epoch 전에 중간에 중단됨을 알 수 있다.
