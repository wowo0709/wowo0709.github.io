---
layout: single
title: "[Computer Vision] 3(1). 합성곱 신경망 CNN"
---



# 합성곱 신경망 CNN

텐서플로의 강력한 API를 사용해 CNN(합성곱 신경망)이 무엇인지 알아보고 구현해보자. 

<br>

### 다차원 데이터를 위한 신경망

---

CNN은 초기 신경망의 단점을 해결하기 위해 도입됐다. 여기서는 이 이슈를 해결하고 CNN이 이 문제를 어떻게 다루는지 설명한다. 

<br>

##### 완전 연결 네트워크의 문제점

완전 연결 네트워크는 이미지를 처리할 때 다음의 두 가지 주요 문제점을 갖는다. 
- **매개변수의 폭발적 증가**: 이미지는 (H, W, D)의 형상으로 이루어진다. 이는 HxWxD 개의 매개변수가 하나의 뉴런 당 필요하다. 
- **공간 추론의 부족**: 완전 연결 신경망에서는 뉴런이 모든 값을 받는다. 이는 모든 픽셀 값이 계층별로 **원래 위치와 상관없이** 결합된다는 것을 뜻하고, 이 때문에 완전 연결 계층에서는 픽셀 사이의 근접성/공간 개념이 사라진다. 

<br>

<br>

### CNN 도입

---

이미지의 경우, CNN은 '3차원 데이터(H, W, D)'를 입력으로 취하고 뉴런을 그와 비슷한 볼륨으로 정렬한다. <br>
완전 연결 계층과 달리, CNN의 각 뉴런은 이전 계층에서 이웃한 영역에 속한 일부 요소에만 접근한다. 이 영역을 뉴런의 **수용 영역**(또는 필터 크기)라고 한다. <br>
뉴런을 이전 계층의 이웃한 뉴런과만 연결함으로써 CNN은 훈련시킬 **'매개변수 개수를 급격히 줄일'** 뿐 아니라 **'이미지 특징의 위치 정보를 보존'**한다. 

<br>

<br>

### CNN 작업

---

이러한 CNN 아키텍쳐 패러다임으로 몇 가지 새로운 유형의 계층을 추가로 도입해 **'다차원성'**과 **'지역적 연결성'**을 효율적으로 활용한다. 

<br>

##### <span style="color:rgb(243, 178, 39)">합성곱 계층</span>

![KakaoTalk_20210807_165351662](https://user-images.githubusercontent.com/70505378/128715877-469a753b-8c3d-4799-a471-79de5c76cd49.png)

<br>

**개념**

- 합성곱 계층에서는 동일한 출력 채널에 연결된 모든 뉴런이 똑같은 가중치와 편향값을 공유함으로써 매개변수의 개수를 더 줄일 수 있다. 

- CNN의 뉴런은 하나의 필터로 작용한다. 이 필터는 입력 행렬에서 슬라이딩하며 가중치와 해당 영역의 입력값들을 선형 결합(각 원소에 대해 스칼라 곱을 하고 모두 더함)한 뒤에 활성화 함수를 적용한다. (물론 편향값도 더한다.)
  ![KakaoTalk_20210807_152021663](https://user-images.githubusercontent.com/70505378/128716665-2510a137-4a5b-42cc-925b-5f73c3c867db.png)

- 이 연산을 통해 크기 (H, W, D)의 입력값을 크기 (H0, W0, D)의 출력 값으로 바꿀 수 있다. (H0, W0는 필터가 슬라이딩할 수 있는 횟수)

  ![KakaoTalk_20210807_152218127](https://user-images.githubusercontent.com/70505378/128715865-a0778d13-eb9b-4fd1-9db6-bb19f485f92b.png)

<br>

**속성**

- N개의 다양한 뉴런의 집합을 갖는 합성곱 계층은 형상이 (D, k, k)(필터가 정사각형인 경우)인 N개ㅡ이 가중치 행렬(**필터** 또는 **커널**)과 N개의 편향값으로 정의된다. 따라서 이 계층에서 훈련시킬 값은 N x (Dk^2 + 1)개 뿐이다. 이 수식에서 알 수 있듯이, 완전연결계층과 달리 합성곱 계층은 **데이터 차원이 매개변수 개수에 영향을 주지 않는다.**
✋ 데이터의 차원이 매개변수의 개수에 영향을 주지는 않지만, 일반적으로 이미지 샘플들을 배치 형태로 만들기 위해서는 데이터 처리와 네트워크 작업의 활성화를 위해 **이미지를 전처리해서 모두 동일한 크기를 갖도록 한다.**
- 훈련을 통해 CNN 계층의 필터는 특정 '지역 특징'에 반응하는 데 정말 탁월해진다. 이는 앞 단계의 CNN 계층에서는 저차원 특징(선의 방향이나 색의 변화)에 반응하고, 뒷 단계의 CNN 계층에서는 좀 더 추상적이고 발전된 특징(얼굴 형태, 특정 객체의 윤곽 등)에 반응하게 된다는 것이다. 그리고 이는 그 특징의 위치와 상관없이 반응할 수 있다. 
- 입력 이미지에 필터를 적용하여 생성된 필터의 응답 맵을 **특징 맵**이라 하고 N개의 필터(뉴런)가 있는 계층에서 반환하는 N개의 특징 맵 스택을 **특징 볼륨**(형상은 (H0, W0, N))이라 한다. 

<br>

**초매개변수**

- 합성곱 계층은 필터 개수 N, 입력 깊이 D(입력 채널의 개수), 필터 크기(k<sub>H</sub>, k<sub>W</sub>)와 더불어 몇 가지의 초매개변수로 결정된다. 
- 초매개변수의 종류는 다음과 같다. 
    - stride(보폭, s): 필터가 움직일 때의 보폭. 보폭이 커지면 결과 특징맵은 희소해진다. 
    - padding(패딩, p): 필터를 적용하기 전에 이미지 주위를 0으로(또는 다른 수로) 패딩하여 크기를 키워서 필터가 이미지를 차지할 수 있는 위치의 수를 증가시킬 수 있다. 이는 특징맵의 크기가 너무 작아지는 것을 방지할 수 있다. 
- N, k, s, p의 매개변수들은 합성곱 계층의 출력 형상을 결정한다. 
![KakaoTalk_20210807_154656167](https://user-images.githubusercontent.com/70505378/128715869-b939d19e-330f-4439-b959-ecdb4773e8ea.png)

<br>

**텐서플로/케라스 메서드**

이미지 합성곱의 경우 기본적으로 저차원 API의 **tf.nn.conv2d()**를 사용한다. 주요 매개변수는 다음과 같다. 
- **input**: 형상이 (B, H, W, D)인 입력 이미지의 배치. 
- **filter**: N개의 필터가 쌓여 형상이 (k<sub>H</sub>, k<sub>W</sub>, D, N)인 텐서가 됨
- **strides**: 배치로 나눈 입력의 각 차원에 대한 보폭을 나타내는 4개의 정수 리스트. 일반적으로 [1, s<sub>H</sub>, s<sub>W</sub>, 1]을 사용
- **padding**: 배치로 나눈 입력의 각 차원 전후에 붙이는 패딩을 나타내는 4x2 개의 정수 리스트 또는 사전 정의된 패딩 상수("VALID", "SAME" 등)
    - VALID: 이미지에 패딩을 더하지 않는다. 
    - SAME: 합성곱 출력이 보폭이 1인 입력과 '동일한' 높이와 너비를 갖도록 하는 p를 적용한다. 
- **name**: 이 연산을 식별하는 이름(계산 그래프 적용)

<br>

**합성곱 계층 구현하기**

실제 합성곱 계층의 경우 필터를 훈련 가능하게 만들어야 한다. 

먼저 텐서플로 API로 CNN 계층을 직접 구현해보고, CNN 계층의 역할을 하는 케라스의 간단한 API를 살펴본다. 


```python
import tensorflow as tf
```

<br>

* 직접 구현하기


```python
k, D, N = 3, 10, 10 # 커널 크기 정의
```


```python
# 훈련 가능한 변수를 초기화
kernels_shape = [k, k, D, N]
# glorot 객체는 Glorot 분포를 따르는 값을 생성하기 위해 정의됨
# (다른 유명한 매개변수는 임의의 초기화 기법이 존재하거나 텐서플로에서 다룬다)
glorot_uni_initializer = tf.initializers.GlorotUniform()

# 커널 생성. 글로럿 분포를 따르는 훈련 가능한 이름이 filters인 커널. 
kernels = tf.Variable(glorot_uni_initializer(kernels_shape), 
                         trainable=True, name="filters")
bias = tf.Variable(tf.zeros(shape=[N]), trainable=True, name="bias")

# 합성곱 계층을 컴파일된 함수로 정의
@tf.function
def conv_layer(x, kernels, bias, s):
    # 특징맵 추출
    z = tf.nn.conv2d(x, kernels, strides=[1, s, s, 1], padding='VALID')
    # 마지막으로 편향값과 활성화 함수 적용
    return tf.nn.relu(z+bias)
```


```python
# 케라스의 Layer 클래스 상속
class SimpleConvolutionLayer(tf.keras.layers.Layer):
    def __init__(self, num_kernels=32, kernel_size=(3,3), stride=1):
        """계층 초기화
        
        param num_kernels: 합성곱 계층의 커널 수
        param kernel_size: 커널 크기(H x W)
        param stride: 수직/수평 보폭
        
        """
        super.__init__()
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.stride = stride
        
    def build(self, input_shape):
        """계층 구성, 계층 매개변수와 변수를 초기화
        
        이 함수는 계층이 최초로 사용될 때 내부적으로 호출됨
        param input_shape: 계층의 입력 형상((B x H x W x C))
        
        """
        num_input_ch = input_shape[-1] # 입력 채널 수
        # 이제 필요한 커널 텐서의 형상을 확인(H x W x C x N)
        kernels_shape = (*self.kernel_size, num_input_ch, self.num_kernels)
        # 필터값을 Glorot 분포를 따르는 값으로 초기화
        glorot_init = tf.initializers.GlorotUniform()
        # 변수를 계층에 추가하기 위한 메서드
        self.kernels = self.add_weight(
                            name='kernels', 
                            shape=kernel_shape, 
                            initializer=glorot_init,
                            trainable = True)
        # 편향값에도 동일하게 적용
        self.bias = self.add_weight(
                            name='kernels', 
                            shape=num_kernels, 
                            initializer=glorot_init,
                            trainable = True)
        
    def call(self, inputs):
        """계층을 호출, 해당 계층의 연산을 입력 텐서에 적용"""
        return conv_layer(inputs, self.kernels, self.bias, self.stride)
```

<br>

* 케라스 API

케라스의 API로 구현 시 표현이 명확하다는 장점도 있지만, 케라스 API가 일반적인 계층의 초기화를 캡슐화해서 제공하기 때문에 개발 속도를 높여준다.

tf.keras.layers 모듈을 사용하면 다음과 같이 단일 호출에서 비슷한 합성곱 계층을 인스턴스화 할 수 있다. 


```python
conv = tf.keras.layers.Conv2D(filters=N, kernel_size=(k,k),
                              padding='valid', activation='relu')
```

tf.keras.layers.Conv2D에는 가중치 정규화 같은 여러 개념을 캡슐화하는 추가 매개변수가 많다. 따라서 고급 CNN을 구성할 때 이러한 개념을 다시 개발하느라 시간 낭비하는 대신 이 클래스를 사용하는 것이 좋다. 

<br>

##### <span style="color:rgb(243, 178, 39)">풀링 계층</span>

합성곱 계층과 일반적으로 함께 사용되는 계층 유형으로 '풀링(pooling)' 계층이 있다. 

<br>

**개념 및 초매개변수**

- 풀링 계층에는 '훈련 가능한 매개변수'가 없다. 단지 각 뉴런은 자기 '윈도우'(수용 영역)의 값을 취하고 사전에 정의된 함수로 계산한 하나의 출력을 반환한다. 
- 풀링에는 대표적으로 다음 두 가지가 있다. 
    - **최대 풀링**: 풀링된 영역의 깊이마다 최댓값만 반환
    - **평균 풀링**: 풀링된 영역의 깊이마다 평균값을 반환
    <img src="https://user-images.githubusercontent.com/70505378/128716771-4f3e1cb4-4d04-4ce5-8855-b19df28d5af5.png" alt="KakaoTalk_20210807_163855983" style="zoom:50%;" />
    
- 보통 풀링 계층은 윈도우를 서로 겹치지 않게 하기 위해 '윈도우 크기'와 동일한 크기의 '보폭' 값을 사용한다. 풀링 계층은 '**데이터의 공간 차원을 줄여서**' 네트워크에서 필요한 매개변수의 전체 개수를 줄이고 계산 시간을 단축시키는 것을 목적으로 한다. 
- 따라서 훈련 가능한 커널이 없다는 점만 제외하면 합성곱 계층과 비슷한 초매개변수(패딩, 보폭 등)를 가지고 있는 풀링 계층은 데이터 차원을 제어하는 사용하기 쉽고 가벼운 솔루션이다. 

<br>

**텐서플로/케라스 메서드**

tf.nn 패키지에서는 tf.nn.max_pool()과 tf.nn.avg_pool을 지원한다. 두 메서드의 매개변수는 다음과 같다. 
- **value**: 형상이 (B, H, W, D)인 입력 이미지의 배치. 
- **ksize**: 차원별 윈도우 크기를 나타내는 4개의 정수 리스트. 일반적으로 [1, k, k, 1]을 사용. 
- **strides**: 배치로 나눈 입력의 각 차원에 대한 보폭을 나타내는 4개의 정수 리스트. 일반적으로 [1, s<sub>H</sub>, s<sub>W</sub>, 1]을 사용
- **padding**: 배치로 나눈 입력의 각 차원 전후에 붙이는 패딩을 나타내는 4x2 개의 정수 리스트 또는 사전 정의된 패딩 상수("VALID", "SAME" 등)
    - VALID: 이미지에 패딩을 더하지 않는다. 
    - SAME: 합성곱 출력이 보폭이 1인 입력과 '동일한' 높이와 너비를 갖도록 하는 p를 적용한다. 
- **name**: 이 연산을 식별하는 이름(계산 그래프 적용)

<br>

여기서도 마찬가지로 더 높은 수준의 케라스 API를 사용해 인스턴스화를 좀 더 간결하게 할 수 있다. 


```python
avg_pool = tf.keras.layers.AvgPool2D(pool_size=k, strides=[s,s], padding='valid')
max_pool = tf.keras.layers.MaxPool2D(pool_size=k, strides=[s,s], padding='valid')
```

풀링 계층에는 훈련 가능한 가중치가 없으므로 텐서플로에서는 실제로 풀링 연산과 그에 대응하는 계층 사이에 차이가 없다. 덕분에 이 연산은 가벼울 뿐만 아니라 인스턴스화 하기도 쉽다. 

<br>

##### <span style="color:rgb(243, 178, 39)">완전 연결 계층</span>

CNN 에서도 일반 네트워크와 같은 방식으로 FC 계층이 사용된다. 

<br>

**CNN에서의 사용법**

때에 따라 예를 들어 공간적으로 거리가 먼 특징을 결합하기 위해 뉴런이 전체 입력 맵에 접근하는 것이 유리할 수 있지만, 완전 연결 계층은 이 장 앞에서 언급했듯이 공간 정보의 손실이나 매개변수가 매우 많아진다는 등의 단점이 있다. 

게다가 다른 CNN 계층과는 달리 밀집 계층은 입력과 출력 크기에 의해 정의되기 때문에 설정과 다른 입력 크기를 갖는 입력에는 동작하지 않을 수 있다. 이는 곧 다양한 크기의 이미지에 적용될 가능성을 잃게 됨을 의미한다. 

다만 이러한 단점에도 불구하고, 밀집 연결 계층은 여전히 CNN에서 보편적으로 사용된다. 
이 계층은 일반적으로 마지막 계층에서 예츨 들어 다차원 특징을 1차원 분류 벡터로 변환하기 위해 사용된다. 

<br>

**텐서플로/케라스 메서드**

다차원 텐서를 밀집 계층에 전달하기 전에는 반드시 **평면화**를 하여 배치로 나뉜 칼럼 벡터로 형상이 조정해야 한다. (높이, 너비, 깊이 차원을 단차원으로 평면화)


```python
flatten = tf.keras.layers.Flatten()
fc = tf.keras.layers.Dense(units=output_size, activation='relu')
```

<br>

<br>

### 유효 수용 영역

**유효 수용 영역**은 입력 이미지에서 거리가 먼 요소를 상호 참조하고 결합하는 네트워크 능력에 영향을 줄 수 있는 딥러닝에서 아주 중요한 개념이다. 

참고 논문: [Understanding the Effective Receptive Field in Deep Covolutional Neural Networks published in Advances in Neural Information Processing Systems](https://proceedings.neurips.cc/paper/2016/hash/c8067ad1937f728f51288b3eb986afaa-Abstract.html)

간단하게 정리하면, 흔히 윈도우/커널의 크기로 생각되는 뉴런의 (유효) 수용 영역이 그 픽셀의 공간적 위치에 따라 달라질 수 있다는 것이다. 윈도우 내의 모든 입력 값이 뉴런에게 같은 영향을 주는 것이 아니며(패딩 등의 영향으로), 이에 따라 인간의 눈의 중심이 되는 '중심와'와 비교하여 중심부의 픽셀들이 영향을 강하게 준다고 주장한다. 이에 따라 유효 수용 영역은 단순히 그 크기와 동일한 것이 아니라, '중간 계층 개수, 필터 크기, 보폭' 등에 의해 직접적으로 영향을 받는다는 것이다. 

CNN의 지역 연결성으로 인해 계층과 그 계층의 초매개변수가 네트워크 아키텍처를 정의할 때 네트워크를 통한 시각 정보 흐름에 어떤 영향을 미치는지 염두에 둬야 한다. 

<br>

<br>

### 텐서플로로 CNN 구현하기

대부분의 최신 컴퓨터 비전 알고리즘은 **합성곱, 풀링, 완전연결 계층**의 조합으로 구성된 CNN을 기반으로 몇 가지 변형과 기법을 사용합니다. 

##### 첫 CNN 구현

첫 합성곱 신경망으로 **LeNet-5**를 구현해봅니다. 
![KakaoTalk_20210808_144107416](https://user-images.githubusercontent.com/70505378/128717252-26b04bf6-e4b4-44f9-94cf-2adb62f33034.png)

<br>

<img src="https://user-images.githubusercontent.com/70505378/128717298-f3b33b8b-4049-4224-b2e5-86d3ff7ef169.png" alt="KakaoTalk_20210808_144013917" style="zoom: 67%;" />

- 위의 초기 LeNet-5 아키텍쳐에서 최근에는 평균 풀링 대신 최대 풀링을 사용하고, 활성화 함수로 tanh 대신 ReLU 함수를 사용합니다. 

<br>

**텐서플로와 케라스 구현**


```python
img_height, img_width, img_channels = 28, 28, 1
num_classes = 10
```


```python
# 순차형 API
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()

# 첫번째 블록
# 맨 처음 블록에는 input_shape를 지정해주어야 함
model.add(Conv2D(6, kernel_size=(5,5), padding='same', activation='relu', 
                 input_shape=(img_height, img_width, img_channels)))
model.add(MaxPooling2D(pool_size=(2,2)))

# 두번째 블록
model.add(Conv2D(16, kernel_size=(5,5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

# 밀집 계층
model.add(Flatten())
model.add(Dense(120, activation='relu'))
model.add(Dense(84, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))
```


```python
# 함수형 API
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

class LeNet5(Model):
    def __init__(self, num_classes): # 모델과 계층 생성
        super(LeNet5, self).__init__()
        self.conv1 = Conv2D(6, kernel_size=(5,5), padding='same', 
                            activation='relu')
        self.conv2 = Conv2D(16, kernel_size=(5,5), activation='relu')
        self.max_pool = MaxPooling2D(pool_size=(2,2))
        self.flatten = Flatten()
        self.dense1 = Dense(120, activation='relu')
        self.dense2 = Dense(84, activation='relu')
        self.dense3 = Dense(num_classes, activation='softmax')
        
    def call(self, x):
        x = self.max_pool(self.conv1(x)) # 첫번째 블록
        x = self.max_pool(self.conv2(x)) # 두번째 블록
        x = self.flatten(x)
        x = self.dense3(self.dense2(self.dense1(x))) # 밀집 계층
        return x
```

함수형 API를 사용하면 네트워크 내부에서 특정 계층을 여러 회 재사용하는 경우나 계층에 여러 입력 또는 여러 출력이 있을 경우처럼 더 복합적인 신경망을 구성할 수 있다. 

<br>

**MNIST에 적용**

모델을 생성했으니 이제 컴파일하고 훈련을 한다. 


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
x_train = x_train.reshape((-1, 28, 28, 1))
x_test = x_test.reshape((-1, 28, 28, 1))
```


```python
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
              metrics = ['accuracy'])

# 몇가지 콜백 지정
callbacks = [
    # 3세대가 지나도 'val_loss'가 개선되지 않으면 훈련을 중단함. 
    tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss'), 
    # 그래프, 메트릭을 텐서보드에 기록(파일을 './logs' 에 저장)
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)]

# 훈련 시작
model.fit(x_train, y_train, batch_size=32, epochs=80, 
          validation_data = (x_test, y_test), callbacks=callbacks)
```

    Epoch 1/80
    1875/1875 [==============================] - 9s 4ms/step - loss: 0.5818 - accuracy: 0.8179 - val_loss: 0.1620 - val_accuracy: 0.9470
    Epoch 2/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.1322 - accuracy: 0.9596 - val_loss: 0.0875 - val_accuracy: 0.9712
    Epoch 3/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0941 - accuracy: 0.9708 - val_loss: 0.0693 - val_accuracy: 0.9767
    Epoch 4/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0763 - accuracy: 0.9764 - val_loss: 0.0598 - val_accuracy: 0.9812
    Epoch 5/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0641 - accuracy: 0.9798 - val_loss: 0.0566 - val_accuracy: 0.9808
    Epoch 6/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0554 - accuracy: 0.9828 - val_loss: 0.0517 - val_accuracy: 0.9827
    Epoch 7/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0499 - accuracy: 0.9849 - val_loss: 0.0497 - val_accuracy: 0.9846
    Epoch 8/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0440 - accuracy: 0.9865 - val_loss: 0.0504 - val_accuracy: 0.9840
    Epoch 9/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0402 - accuracy: 0.9871 - val_loss: 0.0444 - val_accuracy: 0.9860
    Epoch 10/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0370 - accuracy: 0.9879 - val_loss: 0.0433 - val_accuracy: 0.9855
    Epoch 11/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0332 - accuracy: 0.9897 - val_loss: 0.0413 - val_accuracy: 0.9867
    Epoch 12/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0309 - accuracy: 0.9902 - val_loss: 0.0430 - val_accuracy: 0.9866
    Epoch 13/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0288 - accuracy: 0.9910 - val_loss: 0.0404 - val_accuracy: 0.9872
    Epoch 14/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0263 - accuracy: 0.9917 - val_loss: 0.0458 - val_accuracy: 0.9846
    Epoch 15/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0244 - accuracy: 0.9923 - val_loss: 0.0350 - val_accuracy: 0.9886
    Epoch 16/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0221 - accuracy: 0.9933 - val_loss: 0.0411 - val_accuracy: 0.9867
    Epoch 17/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0211 - accuracy: 0.9931 - val_loss: 0.0353 - val_accuracy: 0.9886
    Epoch 18/80
    1875/1875 [==============================] - 7s 4ms/step - loss: 0.0191 - accuracy: 0.9940 - val_loss: 0.0402 - val_accuracy: 0.9875

    <tensorflow.python.keras.callbacks.History at 0x207316ccd90>

