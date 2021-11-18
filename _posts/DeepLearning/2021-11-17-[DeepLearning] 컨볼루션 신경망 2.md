---
layout: single
title: "[Deep Learning] 컨볼루션 신경망 2"
categories: ['AI', 'DeepLearning']
toc: true
toc_sticky: true
tag: ['최적화', '대표적인분류기', '전이학습']
---

<br>

## 컨볼루션 신경망의 학습

### 목적 함수

* **분류 문제**: 교차 엔트로피(Cross Entropy)

  ![image-20211117223115159](https://user-images.githubusercontent.com/70505378/142219604-ef75779e-2a3c-4c26-ba20-c459bd7593b2.png)

* **회귀 문제**: 평균 제곱 오차(Mean Squared Error)

  ![image-20211117223149509](https://user-images.githubusercontent.com/70505378/142219610-498e0f4a-43a3-40c9-8189-0234b6720018.png)

### 최적화기

* **경사 하강법**: 그래디언트의 절댓값이 최소인 방향으로 이동

  ![image-20211117223349923](https://user-images.githubusercontent.com/70505378/142219616-1ad8d491-5f63-4ffa-9570-5f552be5d36d.png)

* **모멘텀**: 경사 하강법 이동에 가속도의 개념을 활용

  ![image-20211117223509124](https://user-images.githubusercontent.com/70505378/142219618-12ccf4e7-b072-4273-b2ea-47dc7065c45f.png)

* **NAG(Nesterov accelerated gradient)**: 모멘텀의 발전된 형태. 선험적인 결과를 사용. 

  ![image-20211117223517334](https://user-images.githubusercontent.com/70505378/142219620-47793199-d99d-4861-b0f5-cbd56356cfcc.png)

* **AdaGrad**: 가중치별로 다른 학습율 사용. 변화가 많았던 가중치에는 작은 학습율 적용. 

  ![image-20211117223530228](https://user-images.githubusercontent.com/70505378/142219624-fe4ed681-7279-4fc6-9099-c6fa5a91fc38.png)

* **AdaDelta**: Adagrad의 변형. 과거 그레디언트의 영향을 점점 축소. 

  ![image-20211117223557355](https://user-images.githubusercontent.com/70505378/142219625-7e61305b-0fa3-4759-87d9-c868f5fdbcb9.png)

* **Adam**: 가중치 별로 다른 학습율 사용. 그레디언트의 1차 및 2차 모멘텀 사용. 

  ![image-20211117223607445](https://user-images.githubusercontent.com/70505378/142219627-81a87d32-002d-4477-984a-5a33d069bb2c.png)

* **RMSprop**: 가중치 별로 별도의 학습율 사용. 학습율을 가중치 별 누적합의 제곱근으로 나누어서 조정. 

  ![image-20211117223620584](https://user-images.githubusercontent.com/70505378/142219629-9ff5edb2-057e-48f1-979a-c2c28daabfb0.png)

<br>

<br>

## 대표적인 컨볼루션 신경망 모델

### ILSVRC 대회

`ILSVRC(ImageNet Large Scale Visual Recognition Challenge)` 대회는 이미지 분류 데이터인 **ImageNet 데이터셋**을 사용하여 모델들의 Top-5 error rate를 평가하는 대회입니다. 

![image-20211117224458760](https://user-images.githubusercontent.com/70505378/142219630-f3e6d5a4-df39-4dd9-8333-cdfcf1afd7f3.png)

2010~2017 년 동안 개최된 이 대회에서 우승한 모델은 아래와 같으며, 아래 모델들 중 대표적인 몇 가지 모델에 대해 살펴보겠습니다. 

![image-20211117224553491](https://user-images.githubusercontent.com/70505378/142219635-816e3d3b-a89f-4c50-a8aa-f6da1654b26b.png)

<br>

### LeNet

![image-20211117224626778](https://user-images.githubusercontent.com/70505378/142219637-be9f9e70-cc3c-49be-94bc-22b5d1bef538.png)

1998년 Yann LeCun등이 제안한 `LeNet` 모델은 5계층 구조를 이루고 있습니다. 

> [Conv - Pool] - [Conv - Pool] - [Conv] - [FC] - [FC(SM)]

* 풀링: 가중치 * (2x2 블록의 합) + 편차항
* 시그모이드 활성화 함수 사용

LeNet은 이후 GoogLeNet의 근간이 됩니다. 

<br>

### AlexNet

![image-20211117224851414](https://user-images.githubusercontent.com/70505378/142219640-76314aa5-fd89-4b3c-b1ce-484f439107e3.png)

2012년도 ILSVRC 대회에서 우승을 차지한 `AlexNet`은 16.43%의 Top-5 오차율을 보였습니다. 

AlexNet은 이 대회에서 최초로 딥러닝을 이용해 우승을 차지한 모델이며, 직전 년도 대비 9.4%의 엄청난 성능 개선을 보였습니다. 

AlexNet이 우승을 차지한 이후로 많은 딥러닝 모델들이 출현하였습니다. 

<br>

AlexNet은 8계층의 구조를 사용하였습니다. 

> [Conv - Pool - Norm] - [Conv - Pool - Norm] - [Conv] - [Conv] - [Conv - Pool] - [FC] - [FC] - [FC(SM)]

* ReLU 함수를 사용한 첫 모델
* FC 층에 드롭아웃 기법 사용
* 최대값 풀링 사용

<br>

**Norm 계층**은 국소 반응 정규화 연산 층으로, **인접한 여러 층의 출력값들을 이용하여 출력 값을 조정**합니다. 

![image-20211117225701917](https://user-images.githubusercontent.com/70505378/142219644-a428bf89-243a-41ca-a6c8-c91489d99e6d.png)

<br>

### VGGNet

VGG 모델은 사이머니언과 지서만이 제안한 모델이고, 모델이 사용하는 계층 수에 따라 `VGG-16`, `VGG-19` 등 여러 이름으로 불립니다. 

2014년도 ILSVRC 대회에서 7.32%의 Top-5 오차율로 2등을 차지하였으며, 단순한 구조를 보입니다. 

![image-20211117230013529](https://user-images.githubusercontent.com/70505378/142219646-fb22aed4-3ab5-4c0f-b12b-d0cd0cd6cfaa.png)

* 모든 층에서 **3x3 필터** 사용
* 큰 필터가 필요한 경우 작은 필터를 여러 번 사용해 같은 효과를 내면서 가중치의 수를 줄임
  * **3x3** 필터 2회 적용 -> **5x5** 필터 적용 효과
  * **3x3** 필터 3회 적용 -> **7x7** 필터 적용 효과

![image-20211117230154503](https://user-images.githubusercontent.com/70505378/142219648-657c4bda-d750-454e-99d0-403129b2f02d.png)

<br>

### GoogLeNet

구글의 체게디 등이 개발한 `GoogLeNet`은 2014년 ILSVRC 대회에서 6.67%의 Top-5 오차율을 보이며 우승을 차지하였고, 22개의 계층 구조를 가집니다. 

![image-20211117230322896](https://user-images.githubusercontent.com/70505378/142219649-747cd253-dd6d-44dc-8979-d3dd799f9206.png)

GoogLeNet은 최초로 **인셉션 모듈**을 제안하였습니다. 

* 직전 층의 처리결과에 1x1, 3x3, 5x5 컨볼루션을 병렬적으로 적용
* 이들 크기의 수용장에 있는 특징들을 동시에 추출

**1x1 컨볼루션**은 다음과 같은 효과를 줍니다. 

* 동일한 위치의 피처 맵의 값을 필터의 가중치와 선형 결합
* 1x1 컨볼루션 필터의 개수를 조정하여 **출력되는 피처 맵의 개수를 조정**
  * (224, 224, 500) * (1, 1, 500) x 120 -> (224, 224, 120)

또한 중간에서 **보조 분류기**를 통해 **그레디언트 정보를 제공**함으로써 **기울기 소멸 문제를 완화**하였습니다. 

<br>

결과적으로, GoogLeNet은 22개 층 모델이지만, AlexNet 모델에 비해 가중치의 개수는 10% 밖에 늘지 않았습니다. 

<br>

### ResNet

카이밍 허 팀에서 개발한 `ResNet`은 2015년 ILSVRC 대회에서 3.75%의 Top-5 오차율로 우승을 차지한 모델입니다. 

ResNet은 152개 층의 아주 깊은 신경망을 사용합니다. 

![image-20211117230944468](https://user-images.githubusercontent.com/70505378/142219652-ae5e8026-afdc-4ac9-adbd-83c668692305.png)

> [Conv - Mpool] - [Conv-ReLU-Conv-ReLU-Conv-ReLU]x3 - [Conv-ReLU-Conv-ReLU-Conv-ReLU]x8 - [Conv-ReLU-Conv-ReLU-Conv-ReLU]x36 - [Conv-ReLU-Conv-ReLU-Conv-ReLU]x3 - [APool] - [FC] - [SM]

<br>

ResNet은 다수의 층을 사용하면서 생길 수 있는 기울기 소멸 문제를 **잔차 모듈**을 사용함으로써 해결하였습니다. 

![image-20211117231213135](https://user-images.githubusercontent.com/70505378/142219653-51b217ea-6676-49cd-967a-11d38c5ab95b.png)

잔차 모듈의 특징은 다음과 같습니다. 

* 기대하는 출력과 유사한 입력이 들어오면 영벡터에 가까운 값을 학습
  * 입력의 작은 변화에 민감 -> 잔차 학습
* 다양한 경로를 통해 복합적인 특징 추출
  * 필요한 출력이 얻어지면 컨볼루션 층을 건너뛸 수 있음
  * 다양한 조합의 특징 추출 가능

<br>

### DenseNet

가오 후앙 등이 개발한 `DenseNet`은 2016년 ILSVRC에 출전한 모델이며, 각 층은 모든 앞 단계에서 올 수 있는 지름길이 연결되어 있습니다. 

![image-20211117231627204](https://user-images.githubusercontent.com/70505378/142219657-789266b6-b2e5-4209-81bf-0c2e201ea331.png)

DenseNet의 **H<sub>i</sub>** 노드 연산은 다음과 같습니다. 

* **배치 정규화(BN) - ReLU - 3x3 컨볼루션**
* 각 층은 입력 특징지도와 같은 차원의 특징 지도 생성
* **병목층**
  * 1x1 컨볼루션
  * 출력되는 특징지도의 채널 수 축소
* **병목층이 있는 층**
  *  BN-ReLU-(1x1 컨볼루션)-BN-ReLU-(3x3 컨볼루션)

<br>

결과적으로 DenseNet은 밀집 블록(Dense block)과 전이 층(Transition layer)으로 구성되어 있는 구조입니다. 

밀집 블록은 **H<sub>i</sub>** 연산이 연쇄적으로 일어나는 블록이고, 전이층은 1x1 컨볼루션과 평균값 풀링으로 이루어져 있습니다. 

![image-20211117232443203](https://user-images.githubusercontent.com/70505378/142219658-038844ae-906f-4043-9dd7-b05f3f691cef.png)

<br>

### DPN(Dual Path Network) 모델

`DPN 모델`은 ResNet과 DenseNet을 결합한 모델입니다. 

* **ResNet**
  * 이전 단계의 동일한 특징 정보가 각 단계에 전달되어 이들 **특징을 재사용**하도록 하는 경향
  * 상대적으로 이전 단계의 특징들로부터 **새로운 특징을 만드는 것에는 소극적**
* **DenseNet**
  * **새로운 특징이 추출**될 가능성이 높음
  * **이전에 추출된 특징이 다시 추출**될 가능성도 높음

DPN 모델은 마이클로 블록에서 두 모델의 특징을 결합하였습니다. 

![image-20211117232709561](https://user-images.githubusercontent.com/70505378/142219661-11ba8113-a845-4144-90bf-8fed8d408140.png)

<br>

<br>

## 딥러닝 신경망의 전이 학습

* 큰 규모의 딥러닝 신경망을 학습시킬 때는, 많은 학습 데이터와 상당한 학습 시간이 필요
* 학습된 모델을 자신의 문제에 적용해볼 수도 있고, 일부를 재활용할 수도 있음
* 이미지의 특징을 추출하는 부분은 그대로 사용하고, 분류를 수행하는 부분만 교체하여 학습된 모델을 나의 목적에 맞게 사용할 수 있음



<br>

<br>

