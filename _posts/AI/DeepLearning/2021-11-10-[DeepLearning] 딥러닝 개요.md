---
layout: single
title: "[Deep Learning] 딥러닝 개요"
categories: ['AI', 'DeepLearning']
toc: true
toc_sticky: true
tag: ['기울기소멸', '가중치초기화', '과적합문제']
---

<br>

## 딥러닝

**일반 신경망**

* 소수의 은닉층 포함

![image-20211110182258289](https://user-images.githubusercontent.com/70505378/141092129-2721daf6-2830-4ca6-80d3-728cac264233.png)

* 원시 데이터에서 직접 특징을 추출해서 만든 **특징 벡터**를 입력으로 사용
* 특징 벡터들의 품질에 영향

![image-20211110182551433](https://user-images.githubusercontent.com/70505378/141092135-64527181-fd5c-43a8-91fa-37d2fdd0c085.png)

**딥러닝 신경망(심층 신경망)**

* 다수의 은닉층 포함

![image-20211110182438669](https://user-images.githubusercontent.com/70505378/141092130-f37dd1bd-fc6a-4286-a82d-39b214de9bfc.png)

* **특징 추출**과 **학습**을 함께 수행
* 데이터로부터 **효과적인 특징**을 학습을 통해 추출 -> 우수한 성능

![image-20211110182633927](https://user-images.githubusercontent.com/70505378/141092136-01678825-20b2-469f-b365-aa1302892a72.png)

<br>

<br>

## 기울기 소멸 문제 (Vanishing gradient problem)

`기울기 소멸 문제`란 은닉층이 많은 다층 퍼셉트론에서, 출력층에서 아래 층으로 갈수록 전달되는 오차가 크게 줄어들어, 학습이 되지 않는 현상을 말합니다. 

이는 특히 은닉층의 뉴런의 활성화 함수로 **시그모이드 함수** 또는 **하이퍼볼릭 탄젠트 함수**를 사용하는 경우에 두드러지게 나타납니다. 

![image-20211110182917051](https://user-images.githubusercontent.com/70505378/141092137-122bbcc2-9dd1-41d7-a3ca-137008e814bc.png)

이는 학습 과정에서 **오차 역전파** 시 오차의 미분치가 역으로 전달되어 학습이 수행되는데, 활성화 함수의 도함수 값이 0~1 사이의 값을 가지기 때문에 전달되는 양이 점점 작아짐으로써 나타나는 현상입니다. 

<br>

### 기울기 소멸 문제 완화

따라서 이를 해결하기 위해 은닉층의 활성화 함수로 `ReLU(Rectified Linear Unit)` 함수를 사용할 수 있습니다. 

![image-20211110183205598](https://user-images.githubusercontent.com/70505378/141092142-915dc896-ee2d-48c3-b5d6-e090a41b7bdb.png)

**ReLU 함수**를 사용할 경우 아래 그림과 같이 오차의 미분치가 보존되어 전달됩니다. 

![image-20211110183239879](https://user-images.githubusercontent.com/70505378/141092147-2af9db8b-469c-4b74-953e-50a016ee6ef5.png)

<br>

#### ReLU 함수 사용과 함수 근사

* 함수를 부분적인 평면 타일들로 근사하는 형태
* **출력**이 0 이상인 것들에 의해 계산되는 결과
  * **입력의 선형결합**(입력과 가중치 행렬의 곱들의 합)의 결과

![image-20211110183412860](https://user-images.githubusercontent.com/70505378/141092148-caf9aac7-af04-4d18-906f-716a296a8516.png)

<br>

#### ReLU와 변형된 형태

ReLU 함수는 여러 형태의 변환된 함수들을 가집니다. 

* **ReLU**

![image-20211110183534629](https://user-images.githubusercontent.com/70505378/141092149-c16b68cb-e572-4d22-973d-5eb4e0f020ca.png)

* **Reaky ReLU**

![image-20211110183752271](https://user-images.githubusercontent.com/70505378/141092151-5400aa7f-359a-4e08-b4c3-2032009013e8.png)

* **ELU (Exponential Linear Unit)**

![image-20211110183838197](https://user-images.githubusercontent.com/70505378/141092152-3839eb54-546b-41c5-975d-f78490c28a6a.png)

* **Maxout**

![image-20211110183851721](https://user-images.githubusercontent.com/70505378/141092157-7aedc0b9-0696-4905-ad51-6659d370b4c7.png)

* **PReLU (Parametric ReLU)**

![image-20211110183941550](https://user-images.githubusercontent.com/70505378/141092158-5e1ef982-252e-4cd3-a1fe-f0e69c90a494.png)

<br>

<br>

## 가중치 초기화

**가중치 초기화**

* 신경망의 성능에 큰 영향을 주는 요소
* 보통 가중치의 초기값으로 0에 가까운 무작위 값 사용

**개선된 가중치 초기화 방법**

각 노드의 입력 노드 개수 n<sub>i</sub>와 출력 노드의 개수 n<sub>i+1</sub>를 사용하는 방법

* **균등 분포**

![image-20211110184143413](https://user-images.githubusercontent.com/70505378/141092160-071b8ed4-924c-41f3-b892-080dde4ca97a.png)

* **제이비어(Xavier) 초기화**

![image-20211110184206961](https://user-images.githubusercontent.com/70505378/141092111-8b1bc522-acf6-4ca9-8b5a-4df6ee2b6fb4.png)

* **허(He) 초기화**

![image-20211110184225694](https://user-images.githubusercontent.com/70505378/141092119-81f0b321-4ad7-4949-b6d8-36a870a87e9e.png)

<br>

<br>

## 과적합 문제

**과적합**

* 모델이 **학습 데이터에 지나치게 맞추어진** 상태
* 데이터는 잡음이나 오류를 포함하므로 학습 데이터셋에 지나치게 맞추어진 모델은 테스트 데이터셋에서 좋은 성능을 보일 수 없다. 

![image-20211110184338991](https://user-images.githubusercontent.com/70505378/141092120-4ffb6d6f-6fee-40f8-922a-a3c3af3fb0a2.png)

**과적합 완화기법**

* 조기 종료
* 규제화
* 드롭아웃
* 미니배치
* 배치 정규화
* ...

<br>

### 조기 종료

모델 훈련 중 검증 데이터셋에 대한 성능을 관찰하며 성능이 좋아지지 않으면 학습을 종료한다. 

<Br>

### 규제화 기법

오차 함수를 **오차 항과 모델 복잡도 항**으로 정의한다. 

모델이 복잡해지면 과적합이 될 수 있으므로, **모델 복잡도를 패널티 항으로 추가**한다. 

> **오차 함수 = (오차 항) + α (모델 복잡도 항)**

**모델 복잡도 항**에는 크게 **모든 가중치들에 대해 골고루 규제를 가하는 `L1 규제(Lasso)`**와 **큰 가중치들에 대해 더욱 강한 규제를 가하는 `L2 규제(Ridge)`**가 있다. 

* **L1 규제(Lasso)**

![image-20211110185012388](https://user-images.githubusercontent.com/70505378/141092123-215541b0-fced-4da9-8fd5-359d05b86da9.png)

* **L2 규제(Ridge)**

![image-20211110184955561](https://user-images.githubusercontent.com/70505378/141092122-4e959657-8392-443e-a66e-f4f275963c54.png)

<br>

### 드롭 아웃

* 일정 확률로 노드들을 무작위로 선택하여, 선택된 노드의 앞뒤로 연결된 가중치 연결선을 없는 것으로 간주하고 학습
* 미니배치나 학습주기마다 드롭아웃할 노드들을 새롭게 선택하여 학습
* 추론 시에는 드롭아웃을 하지 않고 전체 학습된 신경망을 사용하여 출력 계산

<br>

### 미니 배치

* 전체 데이터: batch / 전체 데이터의 일부: mini-batch
* 학습 데이터가 큰 경우에는 미니배치 단위로 학습
* 경사 하강법 적용 시 **미니배치의 그레디언트**를 사용
  * 미니 배치에 속하는 각 데이터의 그레디언트의 평균 값을 사용
* 미니 배치를 사용하여 **데이터에 포함된 오류에 대해 둔감한 학습**이 가능
  * **과적합 문제 완화**에 도움

<br>

### 배치 정규화

* **내부 공변량 이동**

  * 오차 역전파 알고리즘을 통한 학습 시 발생
  * **이전 층들의 학습**에 의해 해당 층들의 가중치가 바뀌게 되면, **현재 층에 전달되는 데이터의 분포와 현재 층이 학습했던 시점의 분포 사이에 차이가 발생**. 이로 인해 **학습 속도 저하**

* **배치 정규화**

  * 신경망의 각 층에서 미니배치 B의 각 데이터에 가중치 연산을 적용한 결과인 **x<sub>i</sub>**의 분포를 정규화

    1. 𝒙𝑖의 평균 𝝁<sub>𝐵</sub>가 **0**이 되고 표준편차 𝝈<sub>𝐵</sub>는 **I**가 되도록 변환

    2.  크기조정(scaling) 파라미터 𝛾와 이동(shifting) 파라미터 𝛽 적용
    3. 변환된 데이터 𝒚𝑖 생성

  * 가중치 연산 결과의 미니 배치: B = {x1, x2, ..., xm}
  * 배치 정규화 적용 결과: {y1, y2, ..., ym}

![image-20211110185833783](https://user-images.githubusercontent.com/70505378/141092125-9e998a13-b47f-4d47-aa51-5bfe7c888c24.png)

![image-20211110185842202](https://user-images.githubusercontent.com/70505378/141092127-d183813a-60dd-44a7-b866-33217b680d62.png)

<br>

<br>

