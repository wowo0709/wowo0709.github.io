---
layout: single
title: "[Machine Learning] Final Summary"
categories: ['AI', 'DeepLearning']
toc: true
toc_sticky: true
tag: []
---

<br>

## week 9

### Clustering

Scaling required!!

#### Agglomerative Clustering

* `linkage`: dataframe, metric, method
* `dendrogram`: link_dist, labels

#### KMeans

* `KMeans`: n_clusters
  * labels\_, cluster\_centers\_

#### DBSCAN

* core point, border point, noise point

* `DBSCAN`: eps, min_samples, metric
  * 0~n: class samples
  * -1: noise points

<br>

### Dimension Reduction

* `SelectPercentile`: score_func, percentile
* `PCA`: n_components
* `TSNE`: n_components, perplexity(number of nearest neighbors that is used in other manifold learning algorithms)

<br>

<br>

## week 10

### Tensorflow 1.0

```python
# version 1
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# placeholder()
X = tf.placeholder("float") # X를 담을 공간
Y = tf.placeholder("float") # Y를 담을 공간
W = tf.Variable(np.random.randn(), name = "W") 
b = tf.Variable(np.random.randn(), name = "b") 

# session
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])
result = tf.multiply(x1, x2)

with tf.Session() as sess:
  output = sess.run(result)
  print(output)
    
# tf.global_variables_initializer
sess = tf.Session()
sess.run(tf.global_variables_initializer())
lossHistory = []

# feed_dict
for i in range(300): 
    sess.run([W_update, b_update], feed_dict={X: x, Y: y})
    cost_val, W_val, b_val = sess.run([cost, W, b], feed_dict={X: x, Y: y})
    
    lossHistory.append(cost_val)

sess.close()
```





<br>

### Tensorflow 2.0

```python
# tf.GradientTape() (tape.gradient())
for i in range(300):
    with tf.GradientTape() as tape:
        y_pred = W * x + b
        cost = tf.reduce_mean(tf.square(y_pred - y))
    
    W_grad, b_grad = tape.gradient(cost, [W,b])  # dCost/dw, dCost/db
    
    W.assign_sub(learning_rate * W_grad)
    b.assign_sub(learning_rate * b_grad)
    # optimizer = tf.optimizers.Adam( learning_rate )
    # optimizer.apply_gradients(zip(grads, [W,b]))
    lossHistory.append(cost)
    if i % 10 == 0:
        print("{:5}|{:10.4f}|{:10.4}|{:10.6f}".format(i, W.numpy(), b.numpy(), cost))
       
```



<br>

### Keras

```python
model = Sequential()

model.add(Flatten(input_shape=(1,)))
model.add(Dense(2, activation='sigmoid'))
# or
# model.add(Dense(2, activation='sigmoid'), input_shape=(1,)) # more common
model.summary()
'''
Model: "sequential_4"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
flatten (Flatten)            (None, 1)                 0         
_________________________________________________________________
dense_4 (Dense)              (None, 2)                 4         
=================================================================
Total params: 4
Trainable params: 4
Non-trainable params: 0
_________________________________________________________________
'''
# regression
model.compile(optimizer=SGD(learning_rate=0.1), 
              loss='mse',
              metrics=['accuracy'])    
# classification
model.compile(optimizer=Adam(learning_rate=1e-3), 
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10, 
          batxh_size=100,
          verbose=0,
          validation_split=0.2)

model.evaluate(x_test, y_test, epochs=10, batch_size=10)
model.predict(x_input_data, batch_size=100)

model.save("model_name.h5")
# and later
model = load_model("model_name.h5")
```

<br>

<Br>

## week 11

### Keras Introduction

```python
# Functional API
inputs = Input(shape=(784,))
x = Dense(64, activation="relu")(inputs)
x = Dense(64, activation="relu")(x)
outputs = Dense(10)(x)
# create a Model by specifying its inputs and outputs in the graph of layers
model = Model(inputs=inputs, outputs=outputs, name="mnist_model")
model.summary()
'''
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
- 784 * 64 + 64 = 50240
- 64 * 64 + 64 = 4160
- 64 * 10 + 10 = 650
'''

# Sequential API
model = Sequential()
model.add(Dense(64, input_shape=(784,), activation='relu')) # 첫번째 계츧에서 input_shape 지정
model.add(Dense(64, activation='relu'))
model.add(Dense(10))
model.summary()
'''
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
'''
```

<br>

### Optimizers

#### Gradient Descent

![image-20211212222617604](https://user-images.githubusercontent.com/70505378/145714907-9b117993-6a6a-46b4-9ddf-7428e801653c.png)

#### Stochastic gradient descent

![image-20211212222638911](https://user-images.githubusercontent.com/70505378/145714909-4a650eeb-ea39-4f72-b7a2-22ac4e69ddef.png)

#### Mini-batch gradient descent

![image-20211212222700795](https://user-images.githubusercontent.com/70505378/145714910-1dd09b0b-efaf-4397-936a-d7ee9ae3ba5b.png)

#### Momentum

![image-20211212222715586](https://user-images.githubusercontent.com/70505378/145714911-3a836db8-5af5-4aca-b222-cb717b175f97.png)

#### NAG(Nesterov accelerated gradient)

![image-20211212222738712](https://user-images.githubusercontent.com/70505378/145714912-dd60321d-6619-4392-ab1e-a00d44ac286e.png)

#### Adagrad

*  가중치 별로 다른 갱신
* 과거에 많이 변경되지 않은 매개변수에 더 큰 learning rate  적용

![image-20211212222853627](https://user-images.githubusercontent.com/70505378/145714913-f793a438-121f-4fbc-b63f-e1eb007b9ddb.png)

#### RMSProp

* Adagrad 알고리즘은 너무 급격히 감소하여 global optimum에 도달하지 못 하는 경우 발생
* 처음부터 모든 gradient Gt를 합산하는 대신 **지수 평균**을 사용하여 최근 것 사용

![image-20211212222913644](https://user-images.githubusercontent.com/70505378/145714914-9dfe8f81-a774-4980-a5c9-da8b8b5ce05a.png)

#### Adam

* RMSProp + Momentum

![image-20211212223031983](https://user-images.githubusercontent.com/70505378/145714915-3806f1b7-aaa6-4518-b7a1-2aac5dbf22e8.png)

#### Summary

![image-20211212223121646](https://user-images.githubusercontent.com/70505378/145714917-ace1b3ad-14d6-4512-a9b2-842a8fbb0423.png)

<br>

<br>

## week 12

### MLP

* `parameters = (input_shape) * (# of neurons) + (# of neurons)`

```python
# Transfer learning
conv_base = VGG16(weights = 'imagenet',   # loading 할 weights
                 include_top=False,
                 input_shape=(150, 150, 3))
conv_base.trainable = False 

model = models.Sequential()
model.add(conv_base) # 특징 추출기
model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

'''
Model: "sequential_1"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
vgg16 (Functional)           (None, 4, 4, 512)         14714688  
_________________________________________________________________
flatten (Flatten)            (None, 8192)              0         
_________________________________________________________________
dense_2 (Dense)              (None, 256)               2097408   
_________________________________________________________________
dense_3 (Dense)              (None, 1)                 257       
=================================================================
Total params: 16,812,353
Trainable params: 16,812,353
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-5),
              metrics=['acc'])


history = model.fit_generator(
    generator=train_generator, 
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

 
```

<br>

### CNN

* `parameters = (kernel_size) * (input_depth) * (# of neurons) + (# of neurons)`

```python
model = models.Sequential()

model.add(layers.Conv2D(32,(3,3), activation = 'relu', 
                        input_shape=(img_width, img_height, 3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3), activation = 'relu'))
model.add(layers.MaxPooling2D((2,2)))

model.add(layers.Flatten())
model.add(layers.Dense(512, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d (Conv2D)              (None, 148, 148, 32)      896       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 74, 74, 32)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 72, 72, 64)        18496     
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 36, 36, 64)        0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 34, 34, 128)       73856     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 17, 17, 128)       0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 15, 15, 128)       147584    
_________________________________________________________________
max_pooling2d_3 (MaxPooling2 (None, 7, 7, 128)         0         
_________________________________________________________________
flatten (Flatten)            (None, 6272)              0         
_________________________________________________________________
dense (Dense)                (None, 512)               3211776   
_________________________________________________________________
dense_1 (Dense)              (None, 1)                 513       
=================================================================
Total params: 3,453,121
Trainable params: 3,453,121
Non-trainable params: 0
_________________________________________________________________
'''

model.compile(loss='binary_crossentropy',
              optimizer=optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
# Use ImageDataGenerator
datagen = ImageDataGenerator(rescale = 1./255)
train_generator = datagen.flow_from_directory(directory=train_dir,
											   target_size=(img_width,img_height),
											   classes=['dogs','cats'],
											   class_mode='binary',
											   batch_size=20)

validation_generator = datagen.flow_from_directory(directory=validation_dir,
											   target_size=(img_width,img_height),
											   classes=['dogs','cats'],
											   class_mode='binary',
											   batch_size=20)

history = model.fit_generator(
    generator=train_generator, 
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50)
```

<br>

<br>

## week 13

### Embedding/RNN

* Embedding
  * `parameters = (input_shape(단어 개수)) * (output_shape(단어 벡터 차원))`
  * `output = (None, None, output_shape)`
  * **input_length 지정 시**
    * `output = (None, input_length, output_shape)`

* SimpleRNN:
  * `parameters = (Embedding output_shape * 2) * (# of neurons) + (# of neurons)`
  * `output = (None, # of neurons)`
  * **return_sequences = True 지정 시**
    * `output = (None, input_length, # of neurons)`

  * SimpleRNN이 한 가지 다른 점은 넘파이 예제처럼 하나의 시퀀스가 아니라 다른 케라스 층과 마찬가지로 시퀀스 배치를 처리한다는 것입니다. 즉, (timesteps, input_features) 크기가 아니라 (batch_size, timesteps, input_features) 크기의 입력을 받습니다.
  * 케라스에 있는 모든 순환 층과 동일하게 SimpleRNN은 두 가지 모드로 실행할 수 있습니다. 각 타임스텝의 출력을 모은 전체 시퀀스를 반환하거나(크기가 (batch_size, timesteps, output_features)인 3D 텐서), 입력 시퀀스에 대한 마지막 출력만 반환할 수 있습니다(크기가 (batch_size, output_features)인 2D 텐서). 이 모드는 객체를 생성할 때 return_sequences 매개변수로 선택할 수 있습니다.


```python
model = Sequential()
model.add(Embedding(10000, 32)) # 문장 길이(단어 개수) 10000, 단어벡터 차원 32
model.add(SimpleRNN(32))
model.summary()
model.input_shape, model.output_shape

'''
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
'''

# 뒤에 flatten 이나 Dense layer 에 연결하기위해서는 length 가 고정되어야 함
model = Sequential()
model.add(Embedding(10000, 32, input_length=20)) # 인풋 20개
model.add(SimpleRNN(32))
model.summary()
model.input_shape, model.output_shape

'''
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
'''

# RNN layer의 중간 결과들을 저장
model = Sequential()
model.add(Embedding(10000, 32, input_length=20))
model.add(SimpleRNN(32, return_sequences=True)) # 중간 결과 반환
model.summary()

'''
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
'''
```

<br>

* 네트워크의 표현력을 증가시키기 위해 여러 개의 순환 층을 차례대로 쌓는 것이 유용할 때가 있다. 이런 설정에서는 중간 층들이 전체 출력 시퀀스를 반환하도록 설정해야 한다:

```python
model = Sequential()
model.add(Embedding(10000, 32, input_length=20))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32, return_sequences=True))
model.add(SimpleRNN(32))  # 맨 위 층만 마지막 출력을 반환합니다.
model.summary()

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
```

<br>

### Vectorizing

* `CountVectorizer`
* `TfidfVectorizer`: tokenizer, max_features

![TF-IDF를 활용한 클래스 유사도 분석과 추천 서버 구축 1편 | 클래스101 기술 블로그](https://class101.dev/images/thumbnails/tf-idf.png)

```python
cv = TfidfVectorizer(tokenizer=twitter_tokenizer, max_features=3000)
X_train = cv.fit_transform(text_train)
X_test = cv.transform(text_test) # cv.fit_transform(text_test) (X)

X_train.shape, y_train.shape, X_test.shape, y_test.shape

# ((2000, 3000), (2000,), (1000, 3000), (1000,))
```

```python
# CountVectorizer
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
]
vectorizer1 = CountVectorizer()
X = vectorizer1.fit_transform(corpus)
print(vectorizer1.get_feature_names())
print(X.toarray())

'''
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
[[0 1 1 1 0 0 1 0 1]
 [0 2 0 1 0 1 1 0 1]
 [1 0 0 1 1 0 1 1 1]
 [0 1 1 1 0 0 1 0 1]]
'''

# TfidfVectorier
vectorizer2 = TfidfVectorizer()
X = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names())
print(X.toarray().round(2))

'''
['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
[[0.   0.47 0.58 0.38 0.   0.   0.38 0.   0.38]
 [0.   0.69 0.   0.28 0.   0.54 0.28 0.   0.28]
 [0.51 0.   0.   0.27 0.51 0.   0.27 0.51 0.27]
 [0.   0.47 0.58 0.38 0.   0.   0.38 0.   0.38]]
'''
```

<br>

* `Word2Vec`: sentence_list, sg(skip-gram), size, window, min_count

```python
model = Word2Vec(data,         # 리스트 형태의 데이터
                 sg=1,         # 0: CBOW, 1: Skip-gram
                 size=100,     # 벡터 크기
                 window=3,     # 고려할 앞뒤 폭(앞뒤 3단어)
                 min_count=3,  # 사용할 단어의 최소 빈도(3회 이하 단어 무시)
                 workers=4)    # 동시에 처리할 작업 수(코어 수와 비슷하게 설정)

model.wv['대한민국']
'''
array([-2.4107585e-02, -7.4946046e-02,  1.5689157e-03,  1.7300507e-02,
        7.7659652e-02, -4.3071166e-02,  8.3631985e-02,  1.6745523e-01,
       -8.2903586e-02, -1.7553378e-02,  3.9016213e-02, -1.0054115e-01,
        4.1688729e-02,  1.7242630e-01, -1.8903978e-02,  1.2952442e-01,
        4.8356697e-02,  4.0910381e-01, -7.0913650e-02, -5.0823655e-02,
        1.4685905e-01, -1.2997684e-01,  2.2543812e-02, -3.7712879e-02,
        9.6920088e-02,  1.3099691e-01, -1.3746825e-01, -1.0660959e-01,
        1.1127534e-01,  1.2975276e-01, -2.8525587e-02, -1.2853998e-01,
       -8.3741836e-02, -9.9310517e-02, -2.4495709e-01, -4.1113162e-01,
        1.0418992e-02,  7.9034410e-02,  1.3711397e-01, -5.1028132e-02,
       -1.4102933e-01, -4.6473064e-02, -7.5484976e-02, -6.2391542e-02,
       -4.0519308e-02, -1.5226401e-01, -1.3334070e-01, -1.7248647e-01,
       -9.5049895e-02,  9.9440172e-02, -2.9708706e-02,  8.7483376e-02,
        8.1404611e-02,  1.3708833e-01, -1.1457676e-01, -9.5910830e-03,
       -6.4596653e-02, -2.4731688e-01,  3.0563422e-02,  1.2345860e-01,
       -3.4807574e-02,  1.6530770e-01,  1.2371200e-01, -1.2324062e-02,
        1.4210464e-01, -1.4213949e-01,  1.7249145e-01, -7.8410409e-02,
       -6.2629886e-02, -9.0875283e-02,  2.9489502e-02,  2.1956262e-01,
        3.4037119e-01,  1.0848373e-01,  3.6547065e-02, -1.5146755e-01,
        5.6681294e-02,  6.6085658e-03,  1.9274153e-02,  1.9991216e-01,
       -1.5090431e-01,  9.0067700e-02,  5.1970325e-02,  2.0268182e-01,
        4.6885550e-02, -5.2929554e-02,  6.6083498e-02, -5.8406308e-02,
       -1.1952946e-01,  5.5076398e-02,  1.2351151e-04, -3.8982730e-02,
       -1.3962780e-01,  1.2789361e-01, -1.5078008e-01, -1.4386822e-01,
       -1.3026667e-01, -1.1459819e-01, -7.1221814e-02,  1.1928054e-01],
      dtype=float32)
'''

print(model.wv.most_similar("대한민국"))
'''
[('대한', 0.9968054294586182), ('민국', 0.9958725571632385), ('터닝포인트', 0.9953158497810364), ('근', 0.9948737621307373), ('터닝', 0.994050920009613), ('마감', 0.993889570236206), ('국내증시', 0.9935024976730347), ('정치인', 0.992567777633667), ('글로벌', 0.9920015335083008), ('외국인', 0.9918369650840759)]
'''

# a:b = c: ? 
model.wv.most_similar(positive=['한국', '미국'], negative=['서울'])
'''
[('핵', 0.6568202376365662),
 ('미', 0.6307210922241211),
 ('북핵', 0.6297447681427002),
 ('북', 0.6209843754768372),
 ('북ㆍ미', 0.6095261573791504),
 ('ㆍ', 0.6072773337364197),
 ('성명', 0.601407527923584),
 ('정상회담', 0.6000897884368896),
 ('변', 0.5984941720962524),
 ('월말', 0.5965142250061035)]
'''

print(model.wv.similarity("한국","미국"))
print(model.wv.similarity("한국","일본"))
print(model.wv.similarity("미국","일본"))
'''
0.19900209
0.45370853
0.7131777
'''
```

<br>

<br>

## week 14

### AutoEncoder

```python
# MLP
input_img = keras.Input(shape=(784,))
encoded = layers.Dense(128, activation='relu')(input_img)
encoded = layers.Dense(64, activation='relu')(encoded)
encoded = layers.Dense(32, activation='relu')(encoded)

decoded = layers.Dense(64, activation='relu')(encoded)
decoded = layers.Dense(128, activation='relu')(decoded)
decoded = layers.Dense(784, activation='sigmoid')(decoded)

encoder = keras.Model(input_img, encoded)
autoencoder = keras.Model(input_img, decoded)

autoencoder.compile(optimizer='adam', loss='mse')
# input = output = X
autoencoder.fit(x_train, x_train, epochs=10, batch_size=128, shuffle=True,
                validation_data=(x_test, x_test))

# CNN
# Encoder
input_img = keras.Input(shape=(28, 28, 1))

x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(input_img)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
x = layers.MaxPooling2D(2, 2)(x)
x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D(2, 2)(x)
encoder = keras.Model(input_img, encoded)
# Decoder
# at this point the representation of 'encoded' is (3, 3, 64) 
# Decconvolution
x = layers.Conv2DTranspose(32, kernel_size=3, strides=2, activation='relu', 
                           padding='valid')(encoded)
x = layers.Conv2DTranspose(16, kernel_size=3, strides=2, activation='relu', 
                           padding='same')(x)
x = layers.Conv2DTranspose(1, kernel_size=3, strides=2, activation='sigmoid', 
                           padding='same')(x)
decoded = layers.Reshape([28,28])(x)

autoencoder = keras.Model(input_img, decoded)
autoencoder.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
input_1 (InputLayer)         [(None, 28, 28, 1)]       0         
_________________________________________________________________
conv2d (Conv2D)              (None, 28, 28, 16)        160       
_________________________________________________________________
max_pooling2d (MaxPooling2D) (None, 14, 14, 16)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 14, 14, 32)        4640      
_________________________________________________________________
max_pooling2d_1 (MaxPooling2 (None, 7, 7, 32)          0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 7, 7, 64)          18496     
_________________________________________________________________
max_pooling2d_2 (MaxPooling2 (None, 3, 3, 64)          0         
_________________________________________________________________
conv2d_transpose (Conv2DTran (None, 7, 7, 32)          18464     
_________________________________________________________________
conv2d_transpose_1 (Conv2DTr (None, 14, 14, 16)        4624      
_________________________________________________________________
conv2d_transpose_2 (Conv2DTr (None, 28, 28, 1)         145       
_________________________________________________________________
reshape (Reshape)            (None, 28, 28)            0         
=================================================================
Total params: 46,529
Trainable params: 46,529
Non-trainable params: 0
_________________________________________________________________
'''
```

<br>

### VAE

#### Encoder

```python
# original_dim = 28 * 28
# intermediate_dim = 64
# latent_dim = 2

inputs = keras.Input(shape=(28*28,))
h = layers.Dense(64, activation='relu')(inputs)
z_mean = layers.Dense(2)(h)
z_log_sigma = layers.Dense(2)(h)

z_mean.shape, z_log_sigma.shape
# (TensorShape([None, 2]), TensorShape([None, 2]))

from tensorflow.keras import backend as K

def sampling(args):
    z_mean, z_log_sigma = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], 2),
                              mean=0., stddev=0.1)
    return z_mean + K.exp(z_log_sigma) * epsilon # latent space

z = layers.Lambda(sampling)([z_mean, z_log_sigma]) # make sampling layer (Lambda)

# Create encoder
encoder = keras.Model(inputs, [z_mean, z_log_sigma, z], name='encoder')
encoder.summary()
'''
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
'''
```

![output_14_0](https://user-images.githubusercontent.com/70505378/144794929-659b3fd2-45ac-4a19-91c5-4bd41bcdf0b4.png)

#### Decoder

```python
# Create decoder
latent_inputs = keras.Input(shape=(2,), name='z_sampling')
x = layers.Dense(64, activation='relu')(latent_inputs)
outputs = layers.Dense(28*28, activation='sigmoid')(x)

decoder = keras.Model(latent_inputs, outputs, name='decoder')

decoder.summary()
keras.utils.plot_model(decoder, "decoder_info.png", show_shapes=True)

'''
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
'''
```

![output_16_1](https://user-images.githubusercontent.com/70505378/144794932-dcc2ed37-0217-4b14-a5a1-24dfaa064904.png)

#### VAE (Encoder+Decoder)

```python
outputs = decoder(encoder(inputs)[2])    # take only z-value
vae = keras.Model(inputs, outputs, name='vae_mlp')
vae.summary()
keras.utils.plot_model(vae, "vae_info.png", show_shapes=True)

'''
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
'''
```

![output_19_1](https://user-images.githubusercontent.com/70505378/144794934-559a0257-d473-4da2-bf5c-58fb6ba4878b.png)

<br>

<br>

## Lab3

**Caption testing on pre-trained model**

1. 모델 다운로드
2. 라이브러리 임포트
3. 학습된 모델, 이미지, 토크나이저 로드

**Training using pre-trained model and smaller dataset**

1. Extract features on photo images. 
   * load model to extract features on out images
2. Prepare text data (photograph - description)
   * load the file containing all of the descriptions
   * map the descriptions to corresponding photo
   * clean the text of descriptions
   * summarize the size of the vocabulary. (faster training)
   * save the dictionary of image identifiers - descriptions

**Retrain model using the prepared data**

1. Load the training data
2. Load the prepared descriptions
3. Load the prepared photos
4. Load the tokenizer
5. Transform data to input-output pairs for training the model.
6. Define the model
7. Fit the model

**Evaluate the trained model with test dataset**

1. Load the dataset, its features and descriptions, and the tokenizer
2. Define a function that can generate a description for a photo using the trained model
3. Evaluate a trained model against a given test dataset (BLEU score)





























