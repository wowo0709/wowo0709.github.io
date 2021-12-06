---
layout: single
title: "[Deep Learning] Unconditional GAN"
categories: ['AI', 'DeepLearning']
toc: true
toc_sticky: true
tag: []
---



## Unconditional GAN with fashion-MNIST

- https://machinelearningmastery.com/how-to-develop-a-conditional-generative-adversarial-network-from-scratch/



### Setup


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```


```python
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Reshape, Flatten
from tensorflow.keras.layers import Conv2D, Conv2DTranspose
from tensorflow.keras.layers import LeakyReLU, Dropout
import tensorflow as tf
from tensorflow import keras

(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

print('Train', trainX.shape, trainy.shape)
print('Test', testX.shape, testy.shape)
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    32768/29515 [=================================] - 0s 0us/step
    40960/29515 [=========================================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    26427392/26421880 [==============================] - 0s 0us/step
    26435584/26421880 [==============================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    16384/5148 [===============================================================================================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    4423680/4422102 [==============================] - 0s 0us/step
    4431872/4422102 [==============================] - 0s 0us/step
    Train (60000, 28, 28) (60000,)
    Test (10000, 28, 28) (10000,)



```python
for i in range(100):
	plt.subplot(10, 10, 1 + i)
	plt.axis('off')
	plt.imshow(trainX[i], cmap='gray')  # cmap = gray or gray_r
plt.show()
```


![output_4_0](https://user-images.githubusercontent.com/70505378/144795765-26c5466d-6b0a-47d6-b9a8-5af858540bd0.png)
    

<br>

### Build Discriminator


```python
def define_discriminator(in_shape=(28,28,1)):
	model = Sequential()
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same', input_shape=in_shape))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(128, (3,3), strides=(2,2), padding='same'))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Flatten())
	model.add(Dropout(0.4))
	model.add(Dense(1, activation='sigmoid'))
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
	return model
```


```python
define_discriminator().summary()
keras.utils.plot_model(define_discriminator(), "gan_encoder.png", show_shapes=True)
```

    Model: "sequential_4"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    conv2d_8 (Conv2D)            (None, 14, 14, 128)       1280      
    _________________________________________________________________
    leaky_re_lu_8 (LeakyReLU)    (None, 14, 14, 128)       0         
    _________________________________________________________________
    conv2d_9 (Conv2D)            (None, 7, 7, 128)         147584    
    _________________________________________________________________
    leaky_re_lu_9 (LeakyReLU)    (None, 7, 7, 128)         0         
    _________________________________________________________________
    flatten_4 (Flatten)          (None, 6272)              0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 6272)              0         
    _________________________________________________________________
    dense_4 (Dense)              (None, 1)                 6273      
    =================================================================
    Total params: 155,137
    Trainable params: 155,137
    Non-trainable params: 0
    _________________________________________________________________


    /usr/local/lib/python3.7/dist-packages/keras/optimizer_v2/optimizer_v2.py:356: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      "The `lr` argument is deprecated, use `learning_rate` instead.")






![output_7_2](https://user-images.githubusercontent.com/70505378/144795768-1a5bc28d-7070-41ff-98ef-c6178af913fa.png)
    



- not compiled yet (intentionally), and returns the model

<br>

### Build Generator


```python
# take as input a point in the latent space smfd outputs a single 28x28 grayscale image
def define_generator(latent_dim):
	model = Sequential()
	# foundation for 7x7 image
	n_nodes = 128 * 7 * 7
	model.add(Dense(n_nodes, input_dim=latent_dim))
	model.add(LeakyReLU(alpha=0.2))
	model.add(Reshape((7, 7, 128)))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) # upsample to 14x14
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')) # upsample to 28x28
	model.add(LeakyReLU(alpha=0.2))
	model.add(Conv2D(1, (7,7), activation='tanh', padding='same'))        # generate
	return model
```


```python
define_generator(100).summary()
keras.utils.plot_model(define_generator(100), "gan_encoder.png", show_shapes=True)
```

    Model: "sequential_6"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    dense_6 (Dense)              (None, 6272)              633472    
    _________________________________________________________________
    leaky_re_lu_12 (LeakyReLU)   (None, 6272)              0         
    _________________________________________________________________
    reshape (Reshape)            (None, 7, 7, 128)         0         
    _________________________________________________________________
    conv2d_transpose (Conv2DTran (None, 14, 14, 128)       262272    
    _________________________________________________________________
    leaky_re_lu_13 (LeakyReLU)   (None, 14, 14, 128)       0         
    _________________________________________________________________
    conv2d_transpose_1 (Conv2DTr (None, 28, 28, 128)       262272    
    _________________________________________________________________
    leaky_re_lu_14 (LeakyReLU)   (None, 28, 28, 128)       0         
    _________________________________________________________________
    conv2d_12 (Conv2D)           (None, 28, 28, 1)         6273      
    =================================================================
    Total params: 1,164,289
    Trainable params: 1,164,289
    Non-trainable params: 0
    _________________________________________________________________






![output_11_1](https://user-images.githubusercontent.com/70505378/144795769-bed77420-06c1-481e-95de-9f8c2e7cee1e.png)
    



<br>

### Build GAN (Generator+Discriminator)


```python
# define the combined generator and discriminator model, for updating the generator
def define_gan(generator, discriminator):
	# make weights in the discriminator not trainable
	discriminator.trainable = False
	# connect them
	model = Sequential()
	model.add(generator)
	model.add(discriminator)
	opt = Adam(lr=0.0002, beta_1=0.5)
	model.compile(loss='binary_crossentropy', optimizer=opt)
	return model
```


```python
all = define_gan(define_generator(100), define_discriminator())
all.summary()
keras.utils.plot_model(all, "all.png", show_shapes=True)
```

    Model: "sequential_10"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    sequential_8 (Sequential)    (None, 28, 28, 1)         1164289   
    _________________________________________________________________
    sequential_9 (Sequential)    (None, 1)                 155137    
    =================================================================
    Total params: 1,319,426
    Trainable params: 1,164,289
    Non-trainable params: 155,137
    _________________________________________________________________


    /usr/local/lib/python3.7/dist-packages/keras/optimizer_v2/optimizer_v2.py:356: UserWarning: The `lr` argument is deprecated, use `learning_rate` instead.
      "The `lr` argument is deprecated, use `learning_rate` instead.")






![output_14_2](https://user-images.githubusercontent.com/70505378/144795770-4726014e-4c95-4ed1-9d93-843f5df4b4d5.png)
    



<br>

### Util functions


```python
# load fashion mnist images
def load_real_samples():
	(trainX, _), (_, _) = fashion_mnist.load_data()
	X = np.expand_dims(trainX, axis=-1)   # expand to 3d, e.g. add channels (60000,28,28,1)
	X = X.astype('float32')               # convert from ints to floats
	X = (X - 127.5) / 127.5               # scale from [0,255] to [-1,1]
	return X
```

- We will require one batch (or a half) batch of real images from the dataset each update to the GAN model. A simple way to achieve this is to select a random sample of images from the dataset each time.


```python
# select real samples
def generate_real_samples(dataset, n_samples):
	# choose random instances
	ix = np.random.randint(0, dataset.shape[0], n_samples)
	# select images
	X = dataset[ix]
	# generate class labels
	y = np.ones((n_samples, 1))   # y_real = 1
	return X, y
```

- Next, we need inputs for the generator model. These are random points from the latent space, specifically Gaussian distributed random variables.


```python
# generate points in latent space as input for the generator
def generate_latent_points(latent_dim, n_samples):
	# generate points in the latent space
	x_input = np.random.randn(latent_dim * n_samples)
	# reshape into a batch of inputs for the network
	x_input = x_input.reshape(n_samples, latent_dim)
	return x_input
```

- Next, we need to use the points in the latent space as input to the generator in order to generate new images.


```python
# use the generator to generate n fake examples, with class labels
def generate_fake_samples(generator, latent_dim, n_samples):
	# generate points in latent space
	x_input = generate_latent_points(latent_dim, n_samples)
	# predict outputs
	X = generator.predict(x_input)
	# create class labels
	y = np.zeros((n_samples, 1))  # y_fake = 0
	return X, y
```

<br>

### Training

We are now ready to fit the GAN models.


```python
# train the generator and discriminator
def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=10, n_batch=128):   # n_epochs=100
	bat_per_epo = int(dataset.shape[0] / n_batch)
	half_batch = int(n_batch / 2)

	for i in range(n_epochs):
		for j in range(bat_per_epo):
			X_real, y_real = generate_real_samples(dataset, half_batch)
			d_loss1, _ = d_model.train_on_batch(X_real, y_real)
			X_fake, y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
			d_loss2, _ = d_model.train_on_batch(X_fake, y_fake)
			X_gan = generate_latent_points(latent_dim, n_batch)
			y_gan = np.ones((n_batch, 1))
			g_loss = gan_model.train_on_batch(X_gan, y_gan)

            # loss for the discriminator on real and fake, and loss for the generator
			print('>epoch:%d, batch:%d/%d, d1=%.3f, d2=%.3f g=%.3f' %
				(i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))

	g_model.save('generator.h5')
```


```python
latent_dim = 100

discriminator = define_discriminator()
generator = define_generator(latent_dim)
gan_model = define_gan(generator, discriminator)

dataset = load_real_samples()

train(generator, discriminator, gan_model, dataset, latent_dim)
```


    >epoch:1, batch:1/468, d1=0.712, d2=0.695 g=0.693
    >epoch:1, batch:2/468, d1=0.649, d2=0.697 g=0.690
    ...
    >epoch:10, batch:467/468, d1=0.687, d2=0.692 g=0.708
    >epoch:10, batch:468/468, d1=0.680, d2=0.684 g=0.728
    WARNING:tensorflow:Compiled the loaded model, but the compiled metrics have yet to be built. `model.compile_metrics` will be empty until you train or evaluate the model.

<br>

### Result

- now, we generate 100 random items of clothing


```python
# create and save a plot of generated images (reversed grayscale)
def show_plot(examples, n):
    plt.figure(figsize=(12,12))
    for i in range(n * n):
        plt.subplot(n, n, 1 + i)    # define subplot
        plt.axis('off')             # turn off axis
        plt.imshow(examples[i, :, :, 0], cmap='gray')
    plt.show()
```


```python
# load model
model = keras.models.load_model('generator.h5')
# generate images
latent_points = generate_latent_points(100, 100)
# generate images
X = model.predict(latent_points)
# plot the result
show_plot(X, 10)
```




![output_30_1](https://user-images.githubusercontent.com/70505378/144795771-250a17d4-fb39-468f-ba9b-3219f0a31c56.png)
    

<br>
