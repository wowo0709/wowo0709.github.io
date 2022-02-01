---
layout: single
title: "[Computer Vision] 4(2). 고급 CNN 아키텍처의 이해 (2편. GoogLeNet, Inception 모듈)"
categories: ['AI', 'ComputerVision']
---



<br>

# 고급 CNN 아키텍처의 이해 (2편. GoogLeNet, Inception 모듈)

GoogLeNet은 2014년 ILSVRC에서 VGGNet을 제치고 1위를 차지한 네트워크로, 구조적으로 **인셉션 블록(네트워크)**이라는 개념을 도입한다는 점에서 선형 네트워크와 매우 다르다. 

<br>

### GoogLeNet 아키텍쳐 개요

---

**동기**

첫번째로 GoogLeNet은 CNN 계산 용량을 최적화하는(매개변수의 수를 줄이는) 것에 목표를 두었다. 결과적으로 GoogLeNet의 매개변수는 약 500만 개로, 이는 AlexNet보다 12배 가볍고 VGG-16보다 21배 가벼운 수준이다. 

두번째로 이들은 '인셉션 모듈'을 사용하여 병렬 계층 블록을 사용해 깊을 뿐만 아니라 규모도 큰 네트워크를 구성하여 정확도까지 높일 수 있었다. 

<br>

**아키텍처**

![KakaoTalk_20210811_160509070](https://user-images.githubusercontent.com/70505378/128999103-bdf14db8-bd71-4a89-bf9c-607947c8a836.png)

<br>

<img src="https://user-images.githubusercontent.com/70505378/128999097-f545db2c-9932-4a2d-be6b-b95b4353d315.png" alt="KakaoTalk_20210811_160456318" style="zoom:67%;" />

인셉션 모듈은 수직 수평으로 겹쳐놓은 계층 블록이다. 각 모듈에서 입력 특징 맵은 한두 개의 서로 다른 계층으로 구성된 4개의 병렬 하위 블록에 전달된다. 

위 그림에서 모든 합성곱과 최대풀링 계층은 패딩 옵션으로 'SAME'을 사용한다. 합성곱 게층은 별도로 지정하지 않는 한 보폭으로 s=1을 사용하고 활성화 함수로 ReLU를 사용한다. 

<img src="https://user-images.githubusercontent.com/70505378/128999105-7d8f8be8-633a-4c91-9af3-14d09b8db933.png" alt="KakaoTalk_20210811_161354464" style="zoom:67%;" />

_첫번째 인셉션 모듈_
- 입력으로 (28, 28, 192) 크기의 특징
- 첫번째 병렬 하위 블록은 (64, 1)크기의 커널로 (28, 28, 64)의 텐서를 생성
- 두번째 병렬 하위 블록은 두 개의 합성곱으로 구성(각각 (96,1), (128, 3))되어 (28,28,128) 의 텐서를 생성
- 세번째 병렬 하위 블록은 두 개의 합성곱으로 구성(각각 (16,1), (32,5))되어 (28, 28, 32) 의 텐서를 생성
- 네번째 병렬 하위 블록은 최대풀링을 거쳐 합성곱 계층((32,1))에서 (28,28,32)의 텐서를 생성
- 최종 출력인 (28, 28, 256) 크기의 텐서가 다음 인셉션 모듈의 입력이 된다. 
- 인셉션 모듈에서의 필터 개수 N은 모듈 깊이가 깊어질수록 증가한다. 

_마지막 계층_
- 마지막 인셉션 모듈에서 특징은 평균풀링을 통해 (7,7,1024)에서 (1,1,1024)로 변환되고 마지막으로 밀집계층에서 1000개의 출력으로 변환된다. 

네트워크 구조에서 보듯이 이 네트워크는 두 개의 보조 블록을 갖도록 구성될 수 있으며 여기서도 예측을 도출한다. 이 보조 블록의 목적인 다음 절에서 자세히 다룬다. 

전체적으로 GoogLeNet은 22 게층의 깊이를 갖는 아키텍처(훈련 가능한 계층만)로 60개 이상의 합성곱과 밀집계층으로 구성되지만 그럼에도 AlexNet보다 12배나 적은 매개변수를 갖는다. 

<br>

**기여 - 규모가 큰 블록과 병목을 보편화**


인셉션 모듈에서 각 병렬 하위 계층의 결과를 하나로 연결해 최종 결과를 만드는 이 병렬 처리의 이점은 여러가지가 있다. 
1. 척도가 다양한 데이터 처리를 가능하게 해준다. 
    - 다양한 커널을 이용하여 최적의 커널 크기를 선택할 필요가 없다. 
    - 광범위한 정보를 잡아낼 수 있다. 
    - 서로 다른 계층에서 매핑된 특징을 연결하는 것으로 CNN에 비선형성이 추가된다. 

2. 병목 계층으로 1x1 합성곱 계층을 사용
    - 병목 계층(차원과 매개변수 개수를 줄이는 중간 계층)으로 1x1 합성곱 계층을 사용하여 입력의 공간 구조에 영향을 주지 않으면서 채널을 줄이고, 따라서 매개변수의 수를 줄임
    

✋ 일반적으로 GoogLeNet은 **Inception V1**으로 불린다. 이후에 개선된 **Inception V2, V3**에서는 5x5, 7x7 합성곱 계층을 그보다 작은 합성곱 계층으로 대체하고 정보 손실을 줄이기 위해 병목 계층의 초매개변수를 개선하거나 'BatchNorm' 계층을 추가하는 등 몇 가지 개선사항이 포함되어 있다. 

<br>

3. 완전 연결 계층 대신 풀링 계층 사용
    - 마지막 인셉션 모듈 이후 완전 연결 계층 대신 평균풀링 계층(윈도우 크기 7x7, 보폭 1)을 사용하여 매개변수 하나 없이 특징 볼륨을 7x7x1024에서 1x1x1024로 줄인다. 
    - 물론 표현력을 약간 잃게 되기는 하지만 계산상의 막대한 이익을 얻는다. 
    
4. 중간 손실로 경사 소실 문제 해결
    - 아키텍처를 소개할 때 언급했듯이 GoogLeNet에는 훈련에 사용되어(그 후에는 제거됨) 예측을 생성하는 두 개의 보조 분기가 있다. 이 보조 분기는 훈련 동안 다양한 네트워크 깊이에서 추가적인 분류 손실을 도입하여 첫번재 계층과 예측 사이의 거리를 줄인다. 이는 기울기 소실에 대한 해결책이다. 
    - 또한 여러 손실에 의해 영향을 받는 계층의 견고함도 다소 개선한다. 이는 주요 네트워크 뿐만 아니라 그보다 짧은 보조 분기에서도 유용한 차별적 특징을 추출하도록 학습해야 하기 때문이다. 

<br>

<br>

### 텐서플로와 케라스로 구현하기

---

**케라스 함수형 API로 Inception 모듈 구현하기**

순차형 API는 경로가 여럿인 아키텍처에는 잘 적용하지 못한다. 케라스 함수형 API가 텐서플로 패러다임에 더 가까우며 계층을 구성하는 파이썬 변수가 다음 계층에 매개변수로 전달돼 그래프를 구성한다. 

다음 코드는 두 API로 구성된 상당히 단순화시킨 모델을 보여준다. 


```python
import tensorflow as tf
from tensorflow import keras
```


```python
from keras.models import Sequential, Model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Input

# 순차형 API
model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), input_shape=input_shape))
model.add(MaxPooling2D(pool_size(2,2)))
model.add(Flatten())
model.add(Dense(10, activation='softmax'))

# 함수형 API
inputs = Input(shape=input_shape)
conv1 = Conv2D(32, kernel_size=(5,5))(inputs)
maxpool1 = MaxPooling2D(pool_size=(2,2))(conv1)
predictions = Dense(10, activation='softmax')(Flatten()(maxpool1))
model = Model(inputs=inputs, ouputs=predictions)
```


```python
# 원시버전의 인셉션 블록

from keras.layers import Conv2D, MaxPooling2D, concatenate

def naive_inception_block(previous_layer, filters=[64, 128, 32]):
    conv1x1 = Conv2D(filters[0], kernel_size=(1,1), padding='same',
                     activation='relu')(previous_layer)
    conv3x3 = Conv2D(filters[1], kernel_size=(3,3), padding='same',
                     activation='relu')(previous_layer)
    conv5x5 = Conv2D(filters[2], kernel_size=(5,5), padding='same',
                     activation='relu')(previous_layer)
    max_pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(previous_layer)
    
    return concatenate([conv1x1, conv3x3, conv5x5, max_pool], axis=-1)
```


```python
# GoogLeNet의 인셉션 블록

from keras.layers import Conv2D, MaxPooling2D, concatenate

def inception_v1_block(previous_layer):
    # 첫번째 병렬 계층
    conv1x1_2 = Conv2D(96, (1,1), padding='same', activation='relu')(previous_layer)
    conv1x1_3 = Conv2D(16, (1,1), padding='same', activation='relu')(previous_layer)
    max_pool_4 = MaxPooling2D((3,3), strides=(1,1), padding='same')(previous_layer)
    # 두번째 병렬 계층
    conv1x1_1 = Conv2D(64, (1,1), padding='same', activation='relu')(previous_layer)
    conv3x3_2 = Conv2D(128, (3,3), padding='same', activation='relu')(conv1x1_2)
    conv5x5_3 = Conv2D(32, (3,3), padding='same', activation='relu')(conv1x1_3)
    conv1x1_4 = Conv2D(32, (1,1), padding='same', activation='relu')(max_pool_4)
    # 최종 계층
    return concatenate([conv1x1_1, conv3x3_2, conv5x5_3, conv1x1_4], axis=-1)
```

<br>

**텐서플로 모델과 텐서플로 허브**

구글은 인셉션 네트워크를 직접 사용하는 방법 또는 새로운 애플리케이션을 위해 이 네트워크를 다시 훈련하는 방법을 설명하는 몇 가지 스크립트와 튜토리얼([tensorflow/models 깃 저장소](https://github.com/tensorflow/models/tree/master/research))을 제공한다. 

게다가 사전에 훈련된 버전의 인셉션 V3는 **텐서플로 허브**에서 사용할 수 있다. 


```python
import tensorflow_hub as hub
from keras.models import Sequential
from keras.layers import Dense

num_classes = 1000

# feture_vector: 최종 출력 직전의 밀집 계층을 제외한 모델 반환(출력은 최종 합성곱 계층의 특징맵)
# (include_top = false)
url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2"
# 탠서플로 허브의 모델을 가져올 때는 hub.KerasLayer 사용
hub_feature_extractor = hub.KerasLayer(     # Layer로서의 TF-Hub 모델
url,                      # TF-Hub 모델 URL
trainable=False,         # 모델 계층들을 훈련 가능하게 할지 여부를 설정하는 플래그
input_shape=(299,299,3),  # 예산 입력 형상 (tfhub.dev에서 확인)
output_shape=(2048,),     # 출력 형상(입력 형상과 동일, 모델 페이지에서 확인)
dtype=tf.float32)         # 예상 dtype

inception_model = Sequential([hub_feature_extractor, 
                              Dense(num_classes, activation='softmax')], 
                              name="inception_tf_hub")
```

<br>

**케라스 모델**


```python
# VGG와 동일한 파라미터 시그니처를 가짐
inceptionV3 = tf.keras.applications.InceptionV3(include_top=True, weights='imagenet', 
                                                input_tensor=None, input_shape=None, 
                                                pooling=None, classes=1000)
```
