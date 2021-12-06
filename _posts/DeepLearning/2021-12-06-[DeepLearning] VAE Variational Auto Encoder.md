---
layout: single
title: "[Deep Learning] VAE Variational Auto Encoder"
categories: ['AI', 'DeepLearning']
toc: true
toc_sticky: true
tag: []
---



## Variational autoencoder (VAE) - Generative model

- from https://blog.keras.io/building-autoencoders-in-keras.html
- What is a variational autoencoder? It's a type of autoencoder with added constraints on the encoded representations being learned. 
- More precisely, it is an autoencoder that learns a latent variable model for its input data. So instead of letting your neural network learn an arbitrary function, you are learning the parameters of a probability distribution modeling your data. 
- If you sample points from this distribution, you can generate new input data samples: a VAE is a "generative model".

- how does it work?
  - First, an encoder network turns the input samples x into two parameters in a latent space, which we will note z_mean and z_log_sigma. 
  - Then, we randomly sample similar points z from the latent normal distribution that is assumed to generate the data, via z = z_mean + exp(z_log_sigma) * epsilon, where epsilon is a random normal tensor.
  - Finally, a decoder network maps these latent space points back to the original input data.

- difference between VAE and standard autoencoder:
  - The main benefit of a variational autoencoder is that we're capable of learning smooth latent state representations of the input data. For standard autoencoders, we simply need to learn an encoding which allows us to reproduce the input. 
  - As you can see in the left-most figure (https://www.jeremyjordan.me/variational-autoencoders/), focusing only on reconstruction loss does allow us to separate out the classes (in this case, MNIST digits) which should allow our decoder model the ability to reproduce the original handwritten digit, but there's an uneven distribution of data within the latent space. 
  - In other words, there are areas in latent space which don't represent any of our observed data.

## A simple VAE model with MNIST

### Build Encoder


```python
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
```


```python
# original_dim = 28 * 28
# intermediate_dim = 64
# latent_dim = 2

inputs = keras.Input(shape=(28*28,))
h = layers.Dense(64, activation='relu')(inputs)
z_mean = layers.Dense(2)(h)
z_log_sigma = layers.Dense(2)(h)
```


```python
z_mean.shape, z_log_sigma.shape
```




    (TensorShape([None, 2]), TensorShape([None, 2]))

<br>

- We can use these parameters to sample new similar points from the latent space:
- layers.Lambda(func_name): Wraps arbitrary expressions as a Layer object. Inherits from Layer, Module


```python
from tensorflow.keras import backend as K

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon # latent space

z = layers.Lambda(sampling)([z_mean, z_log_sigma]) # make sampling layer (Lambda)
```


```python
z.shape
```




    TensorShape([None, 2])

<br>

- Finally, we can map these sampled latent points back to reconstructed inputs:


```python
# Create encoder
encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
encoder.summary()
```

    Model: "encoder"
    __________________________________________________________________________________________________
    Layer (type)                    Output Shape         Param #     Connected to                     
    ==================================================================================================
    input_5 (InputLayer)            [(None, 784)]        0                                            
    __________________________________________________________________________________________________
    dense_20 (Dense)                (None, 64)           50240       input_5[0][0]                    
    __________________________________________________________________________________________________
    dense_21 (Dense)                (None, 2)            130         dense_20[0][0]                   
    __________________________________________________________________________________________________
    dense_22 (Dense)                (None, 2)            130         dense_20[0][0]                   
    __________________________________________________________________________________________________
    lambda_3 (Lambda)               (None, 2)            0           dense_21[0][0]                   
                                                                     dense_22[0][0]                   
    ==================================================================================================
    Total params: 50,500
    Trainable params: 50,500
    Non-trainable params: 0
    __________________________________________________________________________________________________



```python
encoder.output_shape, encoder.input_shape
```




    ([(None, 2), (None, 2), (None, 2)], (None, 784))




```python
keras.utils.plot_model(encoder, "encoder_info.png", show_shapes=True)
```




![output_14_0](https://user-images.githubusercontent.com/70505378/144794929-659b3fd2-45ac-4a19-91c5-4bd41bcdf0b4.png)
    

<br>

### Build Decoder


```python
# Create decoder
latent_inputs = keras.Input(shape=(2,), name='z_sampling')
x = layers.Dense(64, activation='relu')(latent_inputs)
outputs = layers.Dense(28*28, activation='sigmoid')(x)

decoder = keras.Model(latent_inputs, outputs, name='decoder')
decoder.summary()
keras.utils.plot_model(decoder, "decoder_info.png", show_shapes=True)
```

    Model: "decoder"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    z_sampling (InputLayer)      [(None, 2)]               0         
    _________________________________________________________________
    dense_23 (Dense)             (None, 64)                192       
    _________________________________________________________________
    dense_24 (Dense)             (None, 784)               50960     
    =================================================================
    Total params: 51,152
    Trainable params: 51,152
    Non-trainable params: 0
    _________________________________________________________________






![output_16_1](https://user-images.githubusercontent.com/70505378/144794932-dcc2ed37-0217-4b14-a5a1-24dfaa064904.png)
    




```python
decoder.input_shape, decoder.output_shape
```




    ((None, 2), (None, 784))

<br>

### Build VAE (Encoder+Decoder)


```python
# instantiate VAE model
outputs = decoder(encoder(inputs)[2])    # take only z-value
vae = keras.Model(inputs, outputs, name='vae_mlp')
vae.summary()
keras.utils.plot_model(vae, "vae_info.png", show_shapes=True)
```

    Model: "vae_mlp"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    input_5 (InputLayer)         [(None, 784)]             0         
    _________________________________________________________________
    encoder (Functional)         [(None, 2), (None, 2), (N 50500     
    _________________________________________________________________
    decoder (Functional)         (None, 784)               51152     
    =================================================================
    Total params: 101,652
    Trainable params: 101,652
    Non-trainable params: 0
    _________________________________________________________________






![output_19_1](https://user-images.githubusercontent.com/70505378/144794934-559a0257-d473-4da2-bf5c-58fb6ba4878b.png)
    




```python
z_mean, z_log_sigma, z
```




    (<KerasTensor: shape=(None, 2) dtype=float32 (created by layer 'dense_21')>,
     <KerasTensor: shape=(None, 2) dtype=float32 (created by layer 'dense_22')>,
     <KerasTensor: shape=(None, 2) dtype=float32 (created by layer 'lambda_3')>)




```python
inputs, outputs
```




    (<KerasTensor: shape=(None, 784) dtype=float32 (created by layer 'input_5')>,
     <KerasTensor: shape=(None, 784) dtype=float32 (created by layer 'decoder')>)



- What we've done so far allows us to instantiate 3 models:
  - an end-to-end autoencoder mapping inputs to reconstructions
  - an encoder mapping inputs to the latent space
  - a generator that can take points on the latent space and will output the corresponding reconstructed samples.
- We train the model using the end-to-end model, with a custom loss function: the sum of a reconstruction term, and the KL divergence regularization term.

<br>

### Assign Loss and Compile


```python
# for exercise
# By default, loss functions return one scalar loss value per input sample, e.g.
tf.keras.losses.mean_squared_error(tf.ones((2, 2,)), tf.zeros((2, 2)))
```




    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1., 1.], dtype=float32)>

<br>

- As it turns out, by placing a larger emphasis on the KL divergence term we're also implicitly enforcing that the learned latent dimensions are uncorrelated (through our simplifying assumption of a diagonal covariance matrix).


```python
# reconstruction_loss = keras.losses.mse(inputs, outputs)
reconstruction_loss = keras.losses.binary_crossentropy(inputs, outputs)
reconstruction_loss *= 28*28    # loss for the entire input image shape (28*28)
kl_loss = -0.5 * K.sum(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma), axis=1)
beta = 2.0
vae_loss = K.mean(reconstruction_loss + beta * kl_loss)  # more weight on KL_loss ?
```

- All loss functions in Keras always take two parameters y_true and y_pred.
- If you have more (here, you have three), you should use model.add_loss(). (no restriction)


```python
vae.add_loss(vae_loss) # assign loss
vae.compile(optimizer='adam')
```


```python
reconstruction_loss.shape, kl_loss.shape, vae_loss.shape
```




    (TensorShape([None]), TensorShape([None]), TensorShape([]))

<br>

### Training

- train with MNIST dataset


```python
from keras.datasets import mnist
import numpy as np

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))

vae.fit(x_train, x_train,
        epochs=20,
        batch_size=32,
        validation_data=(x_test, x_test))
```

    Epoch 1/20
    1875/1875 [==============================] - 8s 4ms/step - loss: 189.4539 - val_loss: 168.8576
    Epoch 2/20
    1875/1875 [==============================] - 9s 5ms/step - loss: 166.1879 - val_loss: 164.4573
    Epoch 3/20
    1875/1875 [==============================] - 11s 6ms/step - loss: 163.2706 - val_loss: 162.6243
    Epoch 4/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 161.5531 - val_loss: 161.0191
    Epoch 5/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 160.0789 - val_loss: 159.5590
    Epoch 6/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 158.7723 - val_loss: 158.5069
    Epoch 7/20
    1875/1875 [==============================] - 8s 4ms/step - loss: 157.5742 - val_loss: 157.0777
    Epoch 8/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 156.4659 - val_loss: 156.4031
    Epoch 9/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 155.5817 - val_loss: 155.4868
    Epoch 10/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 154.8719 - val_loss: 154.8092
    Epoch 11/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 154.2903 - val_loss: 154.4484
    Epoch 12/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 153.8224 - val_loss: 153.8707
    Epoch 13/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 153.3987 - val_loss: 153.4907
    Epoch 14/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 153.0817 - val_loss: 153.3089
    Epoch 15/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 152.7346 - val_loss: 153.1525
    Epoch 16/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 152.4254 - val_loss: 152.8617
    Epoch 17/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 152.2219 - val_loss: 152.5894
    Epoch 18/20
    1875/1875 [==============================] - 8s 4ms/step - loss: 151.9763 - val_loss: 152.3004
    Epoch 19/20
    1875/1875 [==============================] - 7s 4ms/step - loss: 151.7122 - val_loss: 152.5168
    Epoch 20/20
    1875/1875 [==============================] - 9s 5ms/step - loss: 151.5326 - val_loss: 152.1393



- Because our latent space is two-dimensional, there are a few cool visualizations that can be done at this point. One is to look at the neighborhoods of different classes on the latent 2D plane:

<br>

### Result (latent space)


```python
import matplotlib.pyplot as plt

x_test_encoded = np.array(encoder.predict(x_test, batch_size=16))
print("Shapes of x_test_encoded and y_test: ", x_test_encoded.shape, y_test.shape)

plt.figure(figsize=(12, 12))
x_test_encoded.shape          # 3 (mean, log_sigma, z) * 10000 * 2
x_test_encoded[0,:,1].shape, y_test.shape

plt.scatter(x_test_encoded[2,:,0], x_test_encoded[2,:,1], c=y_test, s=5)
plt.colorbar()
plt.show()
```

    Shapes of x_test_encoded and y_test:  (3, 10000, 2) (10000,)




![output_35_1](https://user-images.githubusercontent.com/70505378/144794935-785c885a-49f7-4b53-bc00-d52e7a42bfc6.png)
    


- Each of these colored clusters is a type of digit. Close clusters are digits that are structurally similar (i.e. digits that share information in the latent space).

- Because the VAE is a generative model, we can also use it to generate new digits! Here we will scan the latent plane, sampling latent points at regular intervals, and generating the corresponding digit for each of these points. This gives us a visualization of the latent manifold that "generates" the MNIST digits.


```python
# Display a 2D manifold of the digits

from scipy.stats import norm

n = 30         # figure with 15x15 digits
figure = np.zeros((28 * n, 28 * n))
# We will sample n points within [-15, 15] standard deviations
grid_x = norm.ppf(np.linspace(0.05, 0.95, n)) 
grid_y = norm.ppf(np.linspace(0.05, 0.95, n))

for i, yi in enumerate(grid_x):
    for j, xi in enumerate(grid_y):
        z_sample = np.array([[xi, yi]])
        x_decoded = decoder.predict(z_sample)
        digit = x_decoded[0].reshape(28, 28)
        figure[i * 28: (i + 1) * 28,
               j * 28: (j + 1) * 28] = digit

plt.figure(figsize=(10, 10))
plt.imshow(figure)
plt.show()
```


![output_38_0](https://user-images.githubusercontent.com/70505378/144794936-39db2d1a-5539-49c8-92a3-df914cbe6b18.png)
    


- You will get different distributions and different effect by giving various values of beta in the loss.
- For more information, see
  - https://towardsdatascience.com/understanding-variational-autoencoders-vaes-f70510919f73 or
  - https://www.jeremyjordan.me/variational-autoencoders/

<br>

<br>
