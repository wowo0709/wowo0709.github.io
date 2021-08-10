---
layout: single
title: "[Computer Vision] 4(1). 고급 아키텍처의 이해 (1편. VGG)"
---



<br>

# 고급 CNN 아키텍처의 이해 (1편. VGG)

ImageNet 분류 대회(ImageNet Large Scale Visual Recognition Challenge, ILSVRC)는 ImageNet에 있는 수백만 개의 이미지를 1000개의 세밀한 클래스로 분류하는 대회이다. 2012년 AlexNet 알고리즘이 의미 있고 상징적인 승리를 거둔 뒤에도 여전히 연구원들에게는 대표적인 도전 과제이다. 

여기서는 ILSVRC 문제를 해결하는 AlexNet 알고리즘을 따르는 전통적인 딥러닝 기법의 일부를 설명하고 이 기법이 나오게 된 이유와 그것이 기여한 부분에 대해 설명한다. 

<br>

## VGG, 표준 CNN 아키텍처

### VGG 아키텍처 개요

---

**동기**

- 활성화 함수로 ReLU를 사용해 경사 소실 문제를 피해서 훈련을 개선한다.
- CNN에 드롭아웃을 적용한다. 
- 합성곱과 풀링 계층으로 구성된 블록과 최종 예측을 위한 밀집 계층을 결합한 전형적인 CNN 아키텍처
- 인위적으로 데이터셋을 늘리기(원본 샘플을 무작위로 편집해 다양한 훈련 이미지 개수를 확대) 위해 무작위 변환(이미지 변환, 좌우 반전 등)을 적용한다. 

수많은 연구원들의 주 목적인 여러 어려운 점이 있더라도 '그래도 더 깊이 들어가는 것'이다. VGG는 2014년에 ILSVRC 대회에서 7.3 %의 top-5 오차율을 기록해 AlexNet이 기록한 16.4%보다 절반 이하의 낮은 수치를 기록했다. 

✋ **top-5 오차율**은 어떤 기법의 상위 5개 예측 안에 정확한 클래스가 포함되면 제대로 예측한 것으로 간주하는 메트릭이다. 실제로 많은 애플리케이션에서 다수의 클래스 후보를 그보다 적은 수로 줄일 수 있는 기법을 갖는 것 만으로 충분하다. 최종 선택은 사용자에게 맡기는 것이다. 

<br>

**아키텍처**

시몬얀과 지서맨은 논문 [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556) 에서 네트워크를 이전 네트워크보다 더 깊게 개발하는 방법을 설명한다. 이 논문에서 제시한 11개 ~ 25개의 계층으로 구성된 6개의 다양한 CNN 계층은 모두 **몇 개의 합성곱 계층이 연이어 나오고 그 뒤를 최대풀링 계층이 따르는 블록 5개(훈련 시 드롭아웃 포함), 마지막으로 3개의 밀집 계층**으로 구성되어 있다. 

그 중 가장 성능이 우수해 지금도 보편적으로 사용되는 아키텍처는 **VGG-16**과 **VGG-19** 이다. 16과 19는 각각 계층의 깊이이며, 훈련할 수 없는 계층인 최대풀링 계층과 드롭아웃 계층은 포함하지 않는다. 즉 각각의 VGG 네트워크는 13개와 16개의 합성곱 계층을 포함한다. 

두 네트워크는 각각 1억 3800만 개와 1억 4400만 개의 매개변수를 가지고 있지만, 이들은 아키텍처 깊이에도 불구하고 이 값들을 확인하기 위한 새로운 접근 방식을 고안했다. 

<br>

**기여 - CNN 아키텍처 표준화**

<br>

**1. 규모가 큰 합성곱을 여러 작은 합성곱으로 대체**

작은 커널을 갖는 여러 개의 합성곱 계층은 큰 커널을 갖는 하나의 합성곱 계층과 **같은 수용영역(ERF)**을 갖는다. 
> R<sub>i</sub> = R<sub>i-1</sub> + (k<sub>i</sub> - 1) * PI(j=1 ~ i-1) s<sub>j</sub>
- R은 ERF의 크기, k는 커널 크기, PI는 총곱, s는 스트라이드

AlexNet의 필터는 11x11로 규모가 크지만, VGG 네트워크는 이보다 작은 합성곱 계층을 더 많이 포함해 더 큰 ERF를 얻을 수 있다. 

- 매개변수 개수를 줄인다. 

실제로 11x11 합성곱 계층에 N개의 필터를 적용한다는 것은 커널만을 위해 훈련시켜야 할 값이 11x11xDxN = 121DN 개라는 것을 뜻한다. 반면 5개의 3x3 합성곱에는 커널을 위한 가중치가 총 1x(3x3xDxN) + 4x(3x3xNxN) = 9DN + 36N<sup>2</sup> 개가 있다. 즉, 이는 N < 3.6D 이기만 하면 매개변수의 개수가 더 작다는 뜻이다. 

- 비선형성을 증가시킨다. 

합성곱 계층 개수가 커지면, 그리고 각 합성곱 계층 다음에 ReLU 같은 '비선형' 활성화 함수가 오면 네트워크가 복잡한 특징을 학습할 수 있는 능력이 증대된다. 

<br>

**2. 특징 맵 깊이를 증가**

각 합성곱 블록에 대한 특징 맵의 깊이를 두 배로 늘렸다(첫번째 합성곱 다음에 64에서 512로). 각 집합 다음에는 윈도우 크기가 2x2이고 보폭이 2인 최대풀링 계층이 나오므로 깊이가 두 배로 늘고 공간 차원은 반으로 줄게 된다. 

이를 통해 공간 정보를 분류에 사용할 더 복잡하고 차별적인 특징으로 인코딩할 수 있다. 

<br>

**3. 척도 변경을 통한 데이터 보강**

이 논문에서는 **척도 변경**이라고 하는 **데이터 보강** 기법도 소개한다. 이 기법은 훈련이 반복될 때마다 이미지 배치를 적절한 입력 크기로 자르기 전에 그 척도를 무작위로 조정한다. 

이렇게 무작위로 변환함으로써 네트워크는 다양한 척도의 샘플을 경험하게 되고 이러한 척도 변경에도 불구하고 이미지를 적절히 분류하는 방법을 학습하게 된다. 다양한 범위의 현실적인 변환을 포괄하는 이미지에서 훈련했기 때문에 네트워크가 더 견고해진다. 

또한 테스트 시점에 무작위로 척도를 변경하고 이미지를 자를 것을 제안한다. 이 방식으로 쿼리 이미지를 여러 버전으로 생성하고 이를 모두 네트워크에 제공하면 해당 네트워크가 특별히 익숙한 척도의 콘텐츠를 제공할 가능성이 증가할 것이라는 직관에 의거한다. 최종 예측은 각 버전에 대한 겨로가의 평균을 구해 얻는다. 

<br>

**4. 완전 연결 계층을 합성곱 계층으로 대체**

마지막에 여러 개의 완전 연결 계층을 사용하는 전통적인 VGG 아키텍처와 달리 대신 크기가 좀 더 큰 커널(7x7 과 3x3)을 적용한 합성곱 계층을 사용할 것을 제안한다. 

첫번째 합성곱 세트는 특징 맵의 공간 크기를 1x1로 줄이고 특징 맵의 깊이를 4096으로 늘린다. 마지막으로 1x1 합성곱 계층이 예측해야 할 클래스의 개수만큼의 필터와 함께 사용된다. 그 결과 얻게된 1x1xN 벡터는 softmax 함수로 정규화된 다음 평명화되어 최종 클래스 예측으로 출력된다. 

✋ **1x1 합성곱**은 일반적으로 공간 구조에 영향을 주지 않고 입력 볼륨의 깊이를 바꿀 때 사용한다. 

밀집 계층을 두지 않는 이러한 네트워크를 **완전 합성곱 계층(fully convolutional network, FCN)**이라고 한다. 

<br>

**텐서플로와 케라스로 구현하기**

<br>

- **텐서플로 모델**

텐서플로에서 직접 VGG 아키텍처 구현을 공식적으로 제공하지는 않지만, [tensorflow/models 깃 저장소](https://github.com/tensorflow/models)에서 깔끔하게 구현된 VGG-16과 VGG-19 네트워크를 사용할 수 있다. 

두 네트워크 외에도 특정 네트워크를 찾아볼 때 이 저장소를 검색하는 것을 추천한다. 

<br>

- **케라스 모델**

케라스 API에서는 이 아키텍처의 구현물을 공식적으로 제공하며 [tf.keras.applications 패키지](https://www.tensorflow.org/api_docs/python/tf/keras/applications)를 통해 접근할 수 있다.

이 패키지에는 그 외에도 잘 알려진 모델이 포함되어 있으며 각 모델에 대해 '사전에 훈련된' 매개변수(특정 데이터셋에서 사전에 훈련시켜 저장해둔 매개변수)도 제공한다. 

- 모델 가져오기


```python
import tensorflow as tf

vgg_net = tf.keras.applications.VGG16(include_top=True, weights='imagenet', 
                                      input_tensor=None, input_shape=None, 
                                      pooling=None, classes=1000)
```

✋ **'include_top=False'**로 지정하면 최종 분류를 위한 밀집 계층을 제외한 네트워크를 얻을 수 있다. 이 경우 네트워크를 확장하기에 용이한다. **'pooling'** 매개변수는 'include_top=False'로 지정한 경우에 네트워크의 출력 결과인 특징 맵에 적용할 풀링 계층을 지정한다. 

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/vgg16/vgg16_weights_tf_dim_ordering_tf_kernels.h5
    553467904/553467096 [==============================] - 16s 0us/step

- 테스트 이미지 가져오고 np.array 형태로 변환

```python
# 테스트 이미지 가져오기
from PIL import Image
import os
import numpy as np

# 이미지 경로 지정
data_dir = "./images/"
files = os.listdir(data_dir)

# 이미지를 가져오고, np.array 형태로 바꿔주어야 함
images = []
for file in files:
    path = os.path.join(data_dir, file)
    images.append(np.array(Image.open(path))) # 한번에 이미지를 np.array 형태로 가져옴
```

- 데이터 전처리


```python
from keras.applications.vgg16 import preprocess_input

# VGG16의 기본 입력 텐서 형태는 (224, 224, 3)
# 0으로 초기화된 array 생성
resized_images = np.array(np.zeros((len(images), 224, 224, 3)))
for i in range(len(images)):
    # 이미지 크기를 변환하여 array에 담는다. 
    resized_images[i] = tf.image.resize(images[i], (224, 224))

# mobilenet_v2 모듈에 포함된 preprocess_input() 메서드를 사용하여 입력값을 전처리
preprocessed_images = preprocess_input(resized_images)
```

- 예측


```python
from keras.applications.vgg16 import decode_predictions

# 예측 결과
y_pred = vgg_net.predict(preprocessed_images)
# 결과값 해석
topK = 5
y_pred_top = decode_predictions(y_pred, top=topK)
```

- 추론 결과 확인


```python
# 추론 결과 확인
from matplotlib import pyplot as plt

for i in range(len(images)):
    plt.imshow(images[i])
    plt.show()
    
    for k in range(topK):
        print("{0}: {1} %".format(y_pred_top[i][k][1], round(y_pred_top[i][k][2] * 100, 1)))
```


![output_16_0](https://user-images.githubusercontent.com/70505378/128867650-c741b525-b1d4-4462-9bdd-dc185dc9bb80.png)
    


    great_grey_owl: 99.7 %
    hare: 0.0 %
    redshank: 0.0 %
    prairie_chicken: 0.0 %
    bustard: 0.0 %




![output_16_2](https://user-images.githubusercontent.com/70505378/128867653-897b80a9-3ae8-4dca-9121-91cf783b0c2b.png)
    


    strainer: 55.6 %
    matchstick: 23.6 %
    coil: 1.7 %
    abacus: 1.1 %
    thimble: 1.0 %




![output_16_4](https://user-images.githubusercontent.com/70505378/128867657-edefa118-0185-4caf-b2c8-bf5a058f986a.png)
    


    black-and-tan_coonhound: 98.8 %
    redbone: 0.7 %
    Doberman: 0.2 %
    bluetick: 0.2 %
    Gordon_setter: 0.0 %




![output_16_6](https://user-images.githubusercontent.com/70505378/128867659-d636c87e-c1c8-400d-ad4a-b2007b237993.png)
    


    Indian_elephant: 50.6 %
    African_elephant: 15.7 %
    tusker: 6.8 %
    Mexican_hairless: 4.0 %
    Komodo_dragon: 2.3 %




![output_16_8](https://user-images.githubusercontent.com/70505378/128867662-5786a7fb-f822-4ff0-b1ec-5527760468dd.png)
    


    cabbage_butterfly: 7.3 %
    ladybug: 6.9 %
    long-horned_beetle: 6.7 %
    acorn: 6.4 %
    ant: 6.1 %




![output_16_10](https://user-images.githubusercontent.com/70505378/128867664-a37e3200-6285-47e1-a58a-7d255fe51004.png)
    


    acorn_squash: 27.3 %
    hot_pot: 18.3 %
    butternut_squash: 13.7 %
    spaghetti_squash: 7.6 %
    plate: 5.6 %




![output_16_12](https://user-images.githubusercontent.com/70505378/128867668-91ba2eda-0007-48d3-a924-98128592bc3f.png)
    


    lion: 100.0 %
    cheetah: 0.0 %
    lynx: 0.0 %
    leopard: 0.0 %
    cougar: 0.0 %




![output_16_14](https://user-images.githubusercontent.com/70505378/128867800-48562f0a-e8e6-4148-a0e3-59d074893ac4.png)
    


    orange: 69.4 %
    lemon: 16.3 %
    butternut_squash: 2.8 %
    acorn_squash: 1.4 %
    cucumber: 1.0 %




![output_16_16](https://user-images.githubusercontent.com/70505378/128867778-9bc0af08-b41a-4cfc-918d-09426369cf60.png)
    


    plate: 14.7 %
    pot: 13.1 %
    broccoli: 11.7 %
    guacamole: 8.2 %
    hot_pot: 6.4 %




![output_16_18](https://user-images.githubusercontent.com/70505378/128867780-bf133321-200e-4409-8493-a9e1d6208369.png)
    


    macaw: 98.6 %
    knot: 0.5 %
    lorikeet: 0.2 %
    panpipe: 0.1 %
    hummingbird: 0.1 %




![output_16_20](https://user-images.githubusercontent.com/70505378/128867785-955b9785-9cf5-467d-8f80-0929fd5eda8a.png)
    


    carbonara: 70.0 %
    plate: 14.5 %
    head_cabbage: 2.6 %
    burrito: 2.3 %
    guacamole: 2.3 %




![output_16_22](https://user-images.githubusercontent.com/70505378/128867786-ddb9aafc-2451-4bf4-907e-48dadf505a81.png)
    


    soup_bowl: 40.4 %
    consomme: 31.0 %
    plate: 9.5 %
    cucumber: 2.5 %
    tray: 2.2 %




![output_16_24](https://user-images.githubusercontent.com/70505378/128867791-89b9d4fc-e74a-4a44-b36d-201c414d8291.png)
    


    Arctic_fox: 39.8 %
    grey_fox: 25.2 %
    kit_fox: 8.3 %
    timber_wolf: 6.7 %
    badger: 6.6 %




![output_16_26](https://user-images.githubusercontent.com/70505378/128867792-199ab519-818c-42f0-975c-bda324b225e1.png)
    


    strawberry: 98.1 %
    bucket: 0.5 %
    pot: 0.4 %
    strainer: 0.3 %
    hip: 0.1 %




![output_16_28](https://user-images.githubusercontent.com/70505378/128867795-b4f78430-5a33-4534-bc52-15ce907dd08a.png)
    


    hip: 22.0 %
    necklace: 18.2 %
    bell_pepper: 14.0 %
    strawberry: 6.0 %
    chain: 6.0 %




![output_16_30](https://user-images.githubusercontent.com/70505378/128867797-27dbdff0-6fac-4d7b-9cbf-6da536dd9e60.png)
    


    cucumber: 12.3 %
    zucchini: 9.1 %
    hot_pot: 7.9 %
    plate: 5.3 %
    banana: 4.8 %

