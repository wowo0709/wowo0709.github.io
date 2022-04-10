---
layout: single
title: "[Tensorflow&Keras] Keras Introduction"
categories: ['AI', 'TensorflowKeras']
toc: true
toc_sticky: true
tag: ['keras']
---



## Keras

- from https://keras.io/guides/functional_api/


```python
import numpy as np
import tensorflow as tf
# import keras -> keras.io
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
```

## Keras Introduction

- Input() : used to instantiate a Keras tensor
  - Keras tensor: a symbolic tensor-like object, which we augment with certain attributes


```python
# 784-dimensional vector input
inputs = Input(shape=(784,))   # 784-dimensional vector. The batch size is always omitted 
                               # since only the shape of each sample is specified.
```


```python
# image input
img_inputs = Input(shape=(32, 32, 3))
```


```python
inputs.shape, img_inputs.shape, inputs.dtype, img_inputs.dtype
```


    (TensorShape([None, 784]),
     TensorShape([None, 32, 32, 3]),
     tf.float32,
     tf.float32)

<br>


```python
# create a new node in the graph of layers
dense = layers.Dense(64, activation="relu")
x = dense(inputs)
```


```python
# few more layers
x = layers.Dense(64, activation="relu")(x)
outputs = layers.Dense(10)(x)
```


```python
# create Model
model = keras.Model(inputs=inputs, outputs=outputs, name="mnist_model")
```


```python
model.summary()
```

    Model: "mnist_model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_1 (InputLayer)         [(None, 784)]             0         
    _________________________________________________________________
    dense (Dense)                (None, 64)                50240     
    _________________________________________________________________
    dense_1 (Dense)              (None, 64)                4160      
    _________________________________________________________________
    dense_2 (Dense)              (None, 10)                650       
    =================================================================
    Total params: 55,050
    Trainable params: 55,050
    Non-trainable params: 0
    _________________________________________________________________

<br>

```python
# must install pydot, graphviz
keras.utils.plot_model(model, "my_first_model.png", show_shapes=True)
```




![output_10_0](https://user-images.githubusercontent.com/70505378/142954674-91bd6e2e-5152-4815-a8c8-4c2abfb2d338.png)
    



<br>

### Using Functional API


```python
inputs = Input(shape=(784,))
x = Dense(64, activation="relu")(inputs)
x = Dense(64, activation="relu")(x)
outputs = Dense(10)(x)
```


```python
# create a Model by specifying its inputs and outputs in the graph of layers
model = Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()
```

    Model: "mnist_model"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_3 (InputLayer)         [(None, 784)]             0         
    _________________________________________________________________
    dense_48 (Dense)             (None, 64)                50240     
    _________________________________________________________________
    dense_49 (Dense)             (None, 64)                4160      
    _________________________________________________________________
    dense_50 (Dense)             (None, 10)                650       
    =================================================================
    Total params: 55,050
    Trainable params: 55,050
    Non-trainable params: 0
    _________________________________________________________________


**number of parameters to train**

* 784 * 64 + 64 = 50240
* 64 * 64 + 64 = 4160
* 64 * 10 + 10 = 650

<br>

### Using Sequential API


```python
# another type model definition
model = Sequential()
model.add(Dense(64, input_shape=(784,), activation='relu')) # 첫번째 계츧에서 input_shape 지정
model.add(Dense(64, activation='relu'))
model.add(Dense(10))
model.summary()
```

    Model: "sequential_17"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_54 (Dense)             (None, 64)                50240     
    _________________________________________________________________
    dense_55 (Dense)             (None, 64)                4160      
    _________________________________________________________________
    dense_56 (Dense)             (None, 10)                650       
    =================================================================
    Total params: 55,050
    Trainable params: 55,050
    Non-trainable params: 0
    _________________________________________________________________

<br>

### Training, evaluation, and Inference

- try to use Sequential() model, with MNIST dataset


```python
from tensorflow.keras.datasets.mnist import load_data

(x_train, y_train), (x_test, y_test) = load_data()
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    11493376/11490434 [==============================] - 0s 0us/step
    11501568/11490434 [==============================] - 0s 0us/step
    (60000, 28, 28) (10000, 28, 28) (60000,) (10000,)



```python
x_train = x_train.reshape(60000, 784).astype("float32") / 255.
x_test = x_test.reshape(10000, 784).astype("float32") / 255.
```


```python
# one-hot encoding
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)
print(y_train.shape, y_test.shape)
```

    (60000, 10) (10000, 10)



```python
model.compile(
    loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    optimizer=keras.optimizers.Adam(),
    metrics=["accuracy"],
)
# from_logits=True: inform the loss function that the output values generated by the model 
# are not normalized, a.k.a. logits. (i.e. softmax function has not been applied on them)
history = model.fit(x_train, y_train, batch_size=100, epochs=5, validation_split=0.2)

test_scores = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_scores[0])
print("Test accuracy:", test_scores[1])
```

    Epoch 1/5
    480/480 [==============================] - 3s 5ms/step - loss: 0.4050 - accuracy: 0.8880 - val_loss: 0.2059 - val_accuracy: 0.9383
    Epoch 2/5
    480/480 [==============================] - 2s 4ms/step - loss: 0.1835 - accuracy: 0.9467 - val_loss: 0.1538 - val_accuracy: 0.9558
    Epoch 3/5
    480/480 [==============================] - 2s 4ms/step - loss: 0.1368 - accuracy: 0.9596 - val_loss: 0.1383 - val_accuracy: 0.9581
    Epoch 4/5
    480/480 [==============================] - 2s 4ms/step - loss: 0.1084 - accuracy: 0.9674 - val_loss: 0.1105 - val_accuracy: 0.9683
    Epoch 5/5
    480/480 [==============================] - 2s 5ms/step - loss: 0.0875 - accuracy: 0.9737 - val_loss: 0.1087 - val_accuracy: 0.9669
    313/313 - 1s - loss: 0.1050 - accuracy: 0.9671
    Test loss: 0.10498897731304169
    Test accuracy: 0.9671000242233276


- we can see that size(train)=60000*0.8=48000 (480 steps/epoch)
- and size(val) = 60000*0.2=12000 


```python
history.history.keys()
```


    dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

<br>

### Save the model

- There are two different types of saving models
  - Tensorflow SavedModel (recommended)
    - model architecture 
    - model weight values (that were learned during training) 
    - model training config, if any (as passed to compile) 
    - optimizer and its state, if any (to restart training where you left off)
  - previous keras H5 (simplified version)
  - for more information: see https://www.tensorflow.org/guide/keras/save_and_serialize?hl=ko


```python
# method 1: savedmodel type
model.save("path_to_my_model")
del model
# Recreate the exact same model purely from the file:
model = keras.models.load_model("path_to_my_model")
```

    INFO:tensorflow:Assets written to: path_to_my_model/assets



```python
# method 2: h5 type
model.save("my_model.h5")
del model
model = keras.models.load_model("my_model.h5")
```

<br>

<br>

## To define multiple models

- a single graph of layers can be used to generate multiple models


```python
encoder_input = Input(shape=(28, 28, 1), name="img")
x = layers.Conv2D(16, 3, activation="relu")(encoder_input)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.MaxPooling2D(3)(x)
x = layers.Conv2D(32, 3, activation="relu")(x)
x = layers.Conv2D(16, 3, activation="relu")(x)
encoder_output = layers.GlobalMaxPooling2D()(x) 
            # (batch_size, rows, columns, channels)->(batch_size, channels)

encoder = Model(encoder_input, encoder_output, name="encoder")
encoder.summary()

x = layers.Reshape((4, 4, 1))(encoder_output)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
x = layers.Conv2DTranspose(32, 3, activation="relu")(x)
x = layers.UpSampling2D(3)(x)
x = layers.Conv2DTranspose(16, 3, activation="relu")(x)
decoder_output = layers.Conv2DTranspose(1, 3, activation="relu")(x)

autoencoder = Model(encoder_input, decoder_output, name="autoencoder")
autoencoder.summary()
```

    Model: "encoder"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    img (InputLayer)             [(None, 28, 28, 1)]       0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 26, 26, 16)        160       
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 24, 24, 32)        4640      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 8, 8, 32)          0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 6, 6, 32)          9248      
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 4, 4, 16)          4624      
    _________________________________________________________________
    global_max_pooling2d_1 (Glob (None, 16)                0         
    =================================================================
    Total params: 18,672
    Trainable params: 18,672
    Non-trainable params: 0
    _________________________________________________________________
    Model: "autoencoder"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    img (InputLayer)             [(None, 28, 28, 1)]       0         
    _________________________________________________________________
    conv2d_4 (Conv2D)            (None, 26, 26, 16)        160       
    _________________________________________________________________
    conv2d_5 (Conv2D)            (None, 24, 24, 32)        4640      
    _________________________________________________________________
    max_pooling2d_1 (MaxPooling2 (None, 8, 8, 32)          0         
    _________________________________________________________________
    conv2d_6 (Conv2D)            (None, 6, 6, 32)          9248      
    _________________________________________________________________
    conv2d_7 (Conv2D)            (None, 4, 4, 16)          4624      
    _________________________________________________________________
    global_max_pooling2d_1 (Glob (None, 16)                0         
    _________________________________________________________________
    reshape_1 (Reshape)          (None, 4, 4, 1)           0         
    _________________________________________________________________
    conv2d_transpose_4 (Conv2DTr (None, 6, 6, 16)          160       
    _________________________________________________________________
    conv2d_transpose_5 (Conv2DTr (None, 8, 8, 32)          4640      
    _________________________________________________________________
    up_sampling2d_1 (UpSampling2 (None, 24, 24, 32)        0         
    _________________________________________________________________
    conv2d_transpose_6 (Conv2DTr (None, 26, 26, 16)        4624      
    _________________________________________________________________
    conv2d_transpose_7 (Conv2DTr (None, 28, 28, 1)         145       
    =================================================================
    Total params: 28,241
    Trainable params: 28,241
    Non-trainable params: 0
    _________________________________________________________________

<br>

### more information about de-convolution

---

- Convolutions
  - Convolutions
  - Dilated Convolutions (a.k.s Astrous Convolutions)
  - Transposed Convolutions (a.k.a. deconvolutions or fractionally strided convolutions)
  - Separable Convolutions
  - from https://towardsdatascience.com/types-of-convolutions-in-deep-learning-717013397f4d

- Difference between UpSample2D and Conv2DTranspose

  - simple scaling up vs. trained 
  - UpSampling2D is just a simple scaling up of the image by using nearest neighbor or bilinear upsampling, so nothing smart. Advantage is it's cheap.
  - Conv2DTranspose is a convolution operation whose kernel is learnt (just like normal conv2d operation) while training your model. Using Conv2DTranspose will also upsample its input but the key difference is the model should learn what is the best upsampling for the job.
- transposed convolution
![image.png](data:image/png;base64,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)

<br>

<br>

## Simple Examples

- https://machinelearningmastery.com/how-to-make-classification-and-regression-predictions-for-deep-learning-models-in-keras/

### Classification


```python
# example of training a final classification model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
# generate 2d classification dataset
X, y = make_blobs(n_samples=100, centers=2, n_features=2, random_state=1)
scaler = MinMaxScaler()
scaler.fit(X)
X = scaler.transform(X)
# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam')
model.fit(X, y, epochs=200, verbose=0)
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense (Dense)                (None, 4)                 12        
    _________________________________________________________________
    dense_1 (Dense)              (None, 4)                 20        
    _________________________________________________________________
    dense_2 (Dense)              (None, 1)                 5         
    =================================================================
    Total params: 37
    Trainable params: 37
    Non-trainable params: 0
    _________________________________________________________________

<br>



- prediction: two types
  - class prediction
  - probability prediction


```python
# predict the result for a new data Xnew
Xnew, _ = make_blobs(n_samples=3, centers=2, n_features=2, random_state=1)
Xnew = scaler.transform(Xnew)

# make a prediction for probability
print('Probability...')
ynew = model.predict(Xnew)
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
```

    Probability...
    X=[0.89337759 0.65864154], Predicted=[0.04033506]
    X=[0.29097707 0.12978982], Predicted=[0.9291382]
    X=[0.78082614 0.75391697], Predicted=[0.04962662]

<br>

```python
# make a prediction for classes
ynew = (model.predict(Xnew) > 0.5).astype("int32")
# show the inputs and predicted outputs
print('Classes...')
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
```

    Classes...
    X=[0.89337759 0.65864154], Predicted=[0]
    X=[0.29097707 0.12978982], Predicted=[1]
    X=[0.78082614 0.75391697], Predicted=[0]

<br>

### Regression

- Question: Is it necessary to scale target values also?
- Answer: It helps to converge your GD algorithm especially when the target values are spreaded large.
- **A target variable with a large spread of values, in turn, may result in large error gradient values causing weight values to change dramatically, making the learning process unstable.**


```python
# with scaling target values
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler

X, y = make_regression(n_samples=100, n_features=2, noise=0.05, random_state=1)
scalerX, scalerY = StandardScaler(), StandardScaler()
scalerX.fit(X)
scalerY.fit(y.reshape(100,1))   # (100,) -> (100,1)
X = scalerX.transform(X)
y = scalerY.transform(y.reshape(100,1))

# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()
model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=1000, verbose=0)
```

    Model: "sequential_15"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_45 (Dense)             (None, 4)                 12        
    _________________________________________________________________
    dense_46 (Dense)             (None, 4)                 20        
    _________________________________________________________________
    dense_47 (Dense)             (None, 1)                 5         
    =================================================================
    Total params: 37
    Trainable params: 37
    Non-trainable params: 0
    _________________________________________________________________

<br>



    <keras.callbacks.History at 0x7fde0646d850>




```python
# new instances to predict
Xnew, a = make_regression(n_samples=3, n_features=2, noise=0.05, random_state=7)
Xnew = scalerX.transform(Xnew)
ynew = model.predict(Xnew)

for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
```

    X=[0.04887696 0.21052199], Predicted=[0.22459722]
    X=[ 1.80651612 -0.80617796], Predicted=[-0.05703998]
    X=[-0.82240444 -0.26142258], Predicted=[-0.58791125]



```python
scalerY.inverse_transform(ynew), a
```




    (array([[ 36.273132],
            [ 14.821406],
            [-25.613968]], dtype=float32),
     array([ 21.28207192,  22.13978868, -21.10578639]))

<br>


```python
# without target scaling

X, y = make_regression(n_samples=100, n_features=2, noise=0.05, random_state=1)
scalerX = StandardScaler()
X = scalerX.fit_transform(X)

# define and fit the final model
model = Sequential()
model.add(Dense(4, input_dim=2, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(loss='mse', optimizer='adam')
model.fit(X, y, epochs=1000, verbose=0)

Xnew, a = make_regression(n_samples=3, n_features=2, noise=0.05, random_state=7)
Xnew = scalerX.transform(Xnew)
# make a prediction
ynew = model.predict(Xnew)
# show the inputs and predicted outputs
for i in range(len(Xnew)):
	print("X=%s, Predicted=%s" % (Xnew[i], ynew[i]))
```

    X=[0.04887696 0.21052199], Predicted=[40.85008]
    X=[ 1.80651612 -0.80617796], Predicted=[11.68873]
    X=[-0.82240444 -0.26142258], Predicted=[-31.26053]

<br>


```python
# linear regression
from sklearn.linear_model import LinearRegression
X, y = make_regression(n_samples=100, n_features=2, noise=0.05, random_state=1)
scalerX = StandardScaler()
X = scalerX.fit_transform(X)

lin_model = LinearRegression()
lin_model.fit(X, y)

Xnew, a = make_regression(n_samples=3, n_features=2, noise=0.05, random_state=7)
Xnew = scalerX.transform(Xnew)

lin_model.predict(Xnew), a, Xnew
```




    (array([ 36.28583995,  14.27272315, -25.27107631]),
     array([ 21.28207192,  22.13978868, -21.10578639]),
     array([[ 0.04887696,  0.21052199],
            [ 1.80651612, -0.80617796],
            [-0.82240444, -0.26142258]]))

