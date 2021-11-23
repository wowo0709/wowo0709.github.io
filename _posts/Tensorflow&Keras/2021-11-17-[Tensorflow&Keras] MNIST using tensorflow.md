---
layout: single
title: "[Deep Learning] MNIST using tensorflow"
categories: ['AI', 'TensorflowKeras']
toc: true
toc_sticky: true
tag: ['MNIST','Tensorflow']
---





## MNIST with RandomForest


```python
import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras import utils
import matplotlib.pyplot as plt

(X_train, Y_class_train), (X_test, Y_class_test) = mnist.load_data()
X_train.shape, Y_class_test.shape
```


    ((60000, 28, 28), (10000,))


```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=50)
rfc.fit(X_train.reshape(X_train.shape[0], 28*28), Y_class_train)
rfc.score(X_test.reshape(X_test.shape[0], 28*28), Y_class_test)
```


    0.9685

<br>

<br>

## MNIST with Tensorflow 2.0 (version 1)

- from internet
- Multi-class logistic regression (simple MLP without hidden layer)

**step 1: importing modules**

```python
# step 1: importing modules

import tensorflow as tf
import numpy as np
```

**step 2: loading and preparing mnist dataset**


```python
# step 2: loading and preparing mnist dataset

from tensorflow.keras.datasets import mnist
num_classes = 10        # 0 to 9 digits
num_features = 28 * 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255., x_test / 255.
```


```python
x_train.shape, x_test.shape, y_train.shape, y_test.shape
```


    ((60000, 784), (10000, 784), (60000,), (10000,))

<br>

**step 3: setting up hyperparameters and dataset parameters**


```python
# step 3: setting up hyperparameters and dataset parameters

learning_rate = 0.01
training_steps = 4000
batch_size = 32
```

<br>

**step 4: shuffling and batching the dataset**


```python
# step 4: shuffling and batching the dataset

train_data=tf.data.Dataset.from_tensor_slices((x_train,y_train))
# train_data=train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)
train_data=train_data.repeat().shuffle(5000).batch(batch_size)
```


```python
train_data
```


    <BatchDataset shapes: ((None, 784), (None,)), types: (tf.float32, tf.uint8)>

<br>

**step 5: initializing weights and biases**


```python
# step 5: initializing weights and biases

W = tf.Variable(tf.random.normal([num_features, num_classes]), name="weight") # (784,10)
b = tf.Variable(tf.random.normal([num_classes]), name="bias")                # (10,)
W.shape, b.shape
```


    (TensorShape([784, 10]), TensorShape([10]))

<br>

**step 6: defining logistic regression and cost function**


```python
# step 6: defining logistic regression and cost function

def logistic_regression(x):
    # print(x.shape)    # (batch_size,784)
    return tf.nn.softmax(tf.matmul(x, W) + b)
```


```python
# for exercise
tf.nn.softmax([1.,1.,0.,0.])
```


    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([0.36552927, 0.36552927, 0.13447072, 0.13447072], dtype=float32)>


```python
def cross_entropy(y_pred, y_true):
    # print(y_pred.shape, y_true.shape)   # (batch_size, 10) (batch_size,)
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)  # to avoid log(0) error
    ce1 = tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), axis=1))
    ce0 = tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred)))       # wrong
    ce2 = tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, y_pred)) # function
    return ce1
```


```python
# for exercise
tmp_a = tf.constant([[1.,2.],[3.,4.]])
tmp_b = tf.constant([[0.,2.],[4.,6.]])
tmp_a*tmp_b, tf.reduce_sum(tmp_a*tmp_b, axis=1)
```


    (<tf.Tensor: shape=(2, 2), dtype=float32, numpy=
     array([[ 0.,  4.],
            [12., 24.]], dtype=float32)>,
     <tf.Tensor: shape=(2,), dtype=float32, numpy=array([ 4., 36.], dtype=float32)>)

<br>

**step 7: defining optimizers and accuracy metrics**


```python
# step 7: defining optimizers and accuracy metrics

def accuracy(y_pred, y_true):
    correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64)) # axis=1
    return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

optimizer = tf.optimizers.Adam(learning_rate)
```

<br>

**step 8: optimization process and updating weights and biases**


```python
# step 8: optimization process and updating weights and biases

def run_optimization(x, y):
    
    with tf.GradientTape() as tape:
        y_pred = logistic_regression(x)
        loss = cross_entropy(y_pred, y)

    gradients = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(gradients, [W, b]))  # updates
```

<br>

**step 9: the training loop**


```python
# step 9: the training loop
# dataset.take(c): Creates a Dataset with at most c elements from this dataset.
# batch_size 만큼씩  c 번 뽑는다.

lossHistory = []
accuracyHistory = []
for step, (batch_x, batch_y) in enumerate(train_data.take(training_steps), 1):

    # print(batch_x.shape, batch_y.shape)   # (batch_size,784), (batch_size,)
    run_optimization(batch_x, batch_y)

    if step % 50 == 0:
        pred = logistic_regression(batch_x)
        loss = cross_entropy(pred, batch_y)
        lossHistory.append(loss)
        acc = accuracy(pred, batch_y)
        accuracyHistory.append(acc)
        # print("step: %i, loss: %f, accuracy: %f" % (step, loss, acc))
```


```python
plt.subplot(1,2,1)
plt.plot(lossHistory)
plt.subplot(1,2,2)
plt.plot(accuracyHistory)
```




![output_19_1](https://user-images.githubusercontent.com/70505378/142134439-f5c40afe-d0b3-4941-8ae7-fdd46e3b4187.png)
    

<br>

**step 10: testing model accuracy using the test data**

```python
# step 10: testing model accuracy using the test data

pred = logistic_regression(x_test)
print("Test Accuracy: %f" % accuracy(pred, y_test))
```

    Test Accuracy: 0.903100

<br>

**step 11: visualizing the classification result**

```python
# step 11: visualizing the classification result

import matplotlib.pyplot as plt

n_images = 5
test_images = x_test[:n_images]
predictions = logistic_regression(test_images)

for i in range(n_images):
    plt.subplot(1,n_images, i+1)
    plt.imshow(np.reshape(test_images[i], [28,28]), cmap='gray')
    print('model prediction: %i' %np.argmax(predictions.numpy()[i]))
plt.show()
```

    model prediction: 7
    model prediction: 2
    model prediction: 1
    model prediction: 0
    model prediction: 4




![output_21_1](https://user-images.githubusercontent.com/70505378/142134441-b10ea099-023e-4a83-8159-f93c877628a8.png)
    

<br>

<br>

## MNIST with tensorflow 2.0 (version 2)

- all in one code routine (for easy migration from the simpler regression code)
- my version (no hidden layer)


```python
from tensorflow.keras.datasets import mnist
num_classes = 10        # 0 to 9 digits
num_features = 28 * 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255., x_test / 255.
```


```python
# hyperparameters
input_size = 784
num_classes = 10
batch_size = 32
num_epochs = 10
learning_rate = 0.01
```

<br>


```python
train_data=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data=train_data.shuffle(5000).batch(batch_size)
```


```python
W = tf.Variable(tf.random.normal([input_size, num_classes])) # (784,10)
b = tf.Variable(tf.random.normal([num_classes]))             # (10,)
```

<br>


```python
# Training
lossHistory = []
accuracyHistory = []

for e in range( num_epochs ):
    for step, (batch_x, batch_y) in enumerate(train_data, 1):
        with tf.GradientTape() as tape:
            y_pred = tf.nn.softmax(tf.matmul(batch_x, W) + b)
            y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
            batch_y_ohe = tf.one_hot(batch_y, depth = 10)
            loss = tf.reduce_mean(tf.losses.categorical_crossentropy( batch_y_ohe , y_pred ))
            # loss = tf.reduce_mean(-tf.reduce_sum(batch_y * tf.math.log(y_pred), axis=1))

        grads = tape.gradient(loss, [W, b])
        optimizer = tf.optimizers.Adam( learning_rate )
        # Adam 과 SGD 비교해 보면 좋음.
        optimizer.apply_gradients(zip(grads, [W,b]))

        corr_predict = tf.equal(tf.argmax(y_pred, 1), tf.cast(batch_y, tf.int64))
        acc = tf.reduce_mean(tf.cast(corr_predict, tf.float32))

        if step % 50 == 0:
            pred = logistic_regression(batch_x)
            loss = cross_entropy(pred, batch_y)
            lossHistory.append(loss)
            acc = accuracy(pred, batch_y)
            accuracyHistory.append(acc)
            print("epoch: %i, step: %i, loss: %f, acc: %f" % (e, step, loss, acc))
    
    # print("epoch: %i, loss: %f, accuracy: %f" % (e, loss, acc))

# Testing
test_y_pred = tf.nn.softmax(tf.matmul(x_test, W) + b)
test_pred = tf.equal(tf.argmax(test_y_pred, 1), tf.cast(y_test, tf.int64))
test_acc = tf.reduce_mean(tf.cast(test_pred, tf.float32))

print("Test accuracy: %f" % test_acc)
```

    epoch: 0, step: 50, loss: 2.907262, acc: 0.500000
    epoch: 0, step: 100, loss: 2.923770, acc: 0.656250
    ...
    epoch: 9, step: 1850, loss: 0.259415, acc: 0.906250
    Test accuracy: 0.914900

```python
plt.subplot(1,2,1)
plt.plot(lossHistory)
plt.subplot(1,2,2)
plt.plot(accuracyHistory)
```




![output_28_1](https://user-images.githubusercontent.com/70505378/142134443-4ecdf5c0-fb25-4ad8-b565-dbab095fdecb.png)
    

<br>

<br>

## MNIST tensorflow 2.0 (version 3)

- function define


```python
import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
num_classes = 10        # 0 to 9 digits
num_features = 28 * 28
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
x_train, x_test = x_train / 255., x_test / 255.

input_size = 784
num_classes = 10
batch_size = 300
num_epochs = 10
learning_rate = 0.01

train_data=tf.data.Dataset.from_tensor_slices((x_train,y_train))
train_data=train_data.shuffle(5000).batch(batch_size)

W = tf.Variable(tf.random.normal([input_size, num_classes])) # (784,10)
b = tf.Variable(tf.random.normal([num_classes]))             # (10,)

def model(X):
    return tf.nn.softmax(tf.matmul(X, W) + b)

def loss_fn(y_pred, y_true):
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.)
    y_true_ohe = tf.one_hot(batch_y, depth = 10)
    return tf.reduce_mean(tf.losses.categorical_crossentropy(y_true_ohe, y_pred))

optimizer = tf.optimizers.Adam( learning_rate )

def train_step (x, y):

    with tf.GradientTape() as tape:
        y_pred = model(x)
        loss = loss_fn (y_pred, y)
    grads = tape.gradient(loss, [W, b])
    optimizer.apply_gradients(zip(grads, [W,b]))

def accuracy(y_pred, y_true):
    corr_predict = tf.equal(tf.argmax(y_pred, 1), tf.cast(y_true, tf.int64))
    return tf.reduce_mean(tf.cast(corr_predict, tf.float32))

# Training
lossHistory = []
accuracyHistory = []

for e in range( num_epochs ):
    for step, (batch_x, batch_y) in enumerate(train_data, 1):
        with tf.GradientTape() as tape:
            train_step(batch_x, batch_y)

        if step % 50 == 0: 
            pred = model(batch_x)
            loss = loss_fn (pred, batch_y)
            acc = accuracy(pred, batch_y)
            print("epoch: %i, step: %i, loss: %f, acc: %f" % (e, step, loss, acc))
            lossHistory.append(loss)
            accuracyHistory.append(acc)

# Testing
test_y_pred = model(x_test)
test_acc = accuracy(test_y_pred, y_test)
print("Test accuracy: %f" % test_acc)

plt.subplot(1,2,1)
plt.plot(lossHistory)
plt.subplot(1,2,2)
plt.plot(accuracyHistory)
plt.show()
```

    epoch: 0, step: 50, loss: 2.445663, acc: 0.590000
    epoch: 0, step: 100, loss: 1.200333, acc: 0.790000
    ...
    epoch: 9, step: 200, loss: 0.211457, acc: 0.926667
    Test accuracy: 0.910300




![output_30_1](https://user-images.githubusercontent.com/70505378/142134445-fd638de2-cbc5-49ff-ad83-1bdef9b31bd9.png)
    

