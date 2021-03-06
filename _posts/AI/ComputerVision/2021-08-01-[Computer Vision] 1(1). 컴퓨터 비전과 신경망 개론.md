---
layout: single
title: "[Computer Vision] 1(1). 컴퓨터 비전과 신경망 개론"
categories: ['AI', 'ComputerVision']
---

<br>

# 컴퓨터 비전과 신경망 개론

컴퓨터 비전은 **디지털 이미지에서 정보를 자동으로 추출하는 것**이다. 추출된 특징은 기존의 특징들과 비교되어 기존의 클래스에 속하거나 새로운 클래스를 만든다. 

<br>

### 주요 작업 및 애플리케이션

---

##### <span style = "color:rgb(243, 178, 39)">1. 콘텐츠 인식</span>

<br>

**객체 분류(classification)**

사전 정의된 집합의 이미지에 적절한 레이블(혹은 클래스)을 할당하는 작업

**객체 식별(recognition)**

클래스의 특정 인스턴스를 인식하는 작업. 데이터셋을 클러스터링하는 절차로 볼 수 있음. 

**객체 탐지와 위치 측정(detection & localization)**

이미지 내의 특정 요소를 탐지, 객체의 경계 상자 제공. 

**객체 및 인스턴스 분할**

발전된 형태의 탐지 기법. 특정 클래스 혹은 특정 클래스의 인스턴스에 속한 **모든 픽셀에 레이블을 단 마스크**를 반환. 

분할 알고리즘은 객체 분할 알고리즘과 인스턴스 분할 알고리즘으로 나눌 수 있으며, 객체 분할 알고리즘은 **같은 클래스에 속한 픽셀 전체에 동일한 마스크를 반환**하지만 인스턴스 분할 알고리즘은 **식별된 인스턴스 별로 다른 마스크를 반환**한다. 

**자세 추정**

자체 추정은 객체의 고정 유무에 따라 다른 목적성을 갖는다. 

고정된 객체의 경우, 일반적으로 **3차원 공간에서 카메라를 기준으로 객체의 위치와 방향을 추정**한다. 

고정되지 않은 객체의 경우, **하부 요소들의 상대적인 위치를 추정**한다. 

<br>

##### <span style = "color:rgb(243, 178, 39)">2. 동영상 분석</span>

<br>

**인스턴스 추적**

동영상 스트림에서 특정 요소의 위치를 추정. 추적은 프레임마다 탐지와 식별 기법을 적용함으로써 이루어지며, 위치를 부분적으로 예측하기 위해 이전 결과를 사용해 인스턴스의 움직임을 모델링하는 것이 효율적이다. 이것을 **움직임 연속성**이라 한다. (단, 빠르게 움직이는 객체와 같은 경우 움직임 연속성이 중요하지 않을 수 있음)

**행동 인식**

이미지 시퀀스를 놓고 봐야하는 작업. **사전 정의된 집합 중 특정 행동을 인식하는 것**

**움직임 추정**

동영상에 포착된 **실제 속도/궤도**를 추정하는 작업

<br>

##### <span style = "color:rgb(243, 178, 39)">3. 콘텐츠 - 인식 이미지본</span>

이미지 자체의 개선을 위한 작업. 잡음 제거, 흐릿한 부분 제거, 고해상도 변환 등. 

이는 사진/그림 애플리케이션 등에 많이 사용. 

<br>

##### <span style = "color:rgb(243, 178, 39)">4. 장면 복원</span>

하나 이상의 이미지가 주어졌을 때 **장면의 3차원 기하학적 구조를 복원**하는 작업

<br>

<br>

### 컴퓨터 비전의 약력

---

* 60년대 **인공지능** 연구 학회의 한 영역으로 시작. 
* 90년대 **주성분 분석(PCA)**와 **SIFT(Scale Invariant Feature Transform)**가 등장. 핵심은 하나의 이미지에서 특징을 추출하고 다른 특징들과 비교한다는 것. 
* 머신러닝의 등장. 이미지 특징을 기반으로 이미지를 분류하는 통계적인 방법으로 접근.  **서포트 벡터머신(SVM)**은 추출된 특징을 기반으로 한 클래스를 다른 클래스에서 구분하기 위한 함수를 학습하고, 이 함수를 적용하여 그 이미지를 클래스 중 하나로 매핑한다. 

* 이후 **랜덤 포레스트, 단어 주머니(BoW), 베이즈 모델, 신경망 등** 다른 머신러닝 알고리즘이 조정, 발전되어 왔음. 

<br>

<br>

### 딥러닝의 출현

---

##### <span style = "color:rgb(243, 178, 39)">1. 초기 시도와 실패</span>

* 최초로 고안된 신경망의 기반을 이루는 블록인 뉴런에서 고안된 **퍼셉트론**은 비선형 문제(XOR 문제)를 해결하지 못하였다. 이는 퍼셉트론이 선형 함수로 모델링되었기 때문에 당연한 결과이다. 
* 확장성을 갖기에는 계산적으로 너무 무거워 큰 문제로 확장될 수 없었다. 

<br>

##### <span style = "color:rgb(243, 178, 39)">2. 복귀 이유</span>

* **인터넷의 폭발적인 발전**으로 인한 대량 데이터셋의 공유, 새로운 콘텐츠의 범람. 
* GPU에서 비롯한 **강력한 컴퓨팅 파워**. CUDA는 프로그래밍에 있어 GPU의 힘을 활용한다. 

<br>

##### <span style = "color:rgb(243, 178, 39)">3. 왜 딥러닝인가</span>

* 딥러닝은 말 그대로 더 깊은 신경망, 즉 여러 개의 '은닉 계층'이 있는 신경망을 다시 그룹으로 묶는다. 각 계층은 입력을 처리하고 그 결과를 다음 계층으로 전달하여 점점 더 추상적인 정보만 추출하도록 훈련된다. 
* 클라우드 컴퓨팅 분야의 발전이 훈련 절차를 병렬로 처리할 수 있게 해주어 딥러닝 모델의 학습 속도와 정확도를 큰 폭으로 향상시켰다. 
* 더 깊이 있는 모델, 더 고도화된 훈련 기법, 그리고 휴대용 기기에 적용할 만한 더 가벼운 솔루션의 개발 등이 활발하게 이루어지고 있다. 

