---
layout: single
title: "[Machine Learning] 신경망"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: ['NeuralNetwork', Perceptron', 'RBF']
---

<br>

## 신경망

* **인간 두뇌에 대한 계산적 모델**을 통해 인공지능을 구현하려는 분야
* **신경세포**

![image-20211108181011146](https://user-images.githubusercontent.com/70505378/140719007-553adb4c-c7ad-46cf-887c-8e00ece21cb3.png)

## 퍼셉트론

* _로젠블랏_이 제안한 학습 가능한 신경망 모델

![image-20211108181150842](https://user-images.githubusercontent.com/70505378/140719011-1a26638a-afe3-4b51-8fdb-160be6b974c9.png)

* **OR 연산**을 수행하는 퍼셉트론

![image-20211108181228528](https://user-images.githubusercontent.com/70505378/140719013-53cb686b-15ac-499c-82cb-98c7883cb3d4.png)

<br>

* **선형 분리 가능 문제**

![image-20211108181311316](https://user-images.githubusercontent.com/70505378/140719015-a8e0c60e-fc48-496b-a4d0-b947a0fe2e0c.png)

* **선형 분리불가 문제** (XOR 문제)

![image-20211108181337496](https://user-images.githubusercontent.com/70505378/140719017-973b3710-bf85-43c4-9a5c-184b35c274b4.png)

<br>

<br>

## 다층 퍼셉트론 (Multi Layer Perceptron, MLP)

* 여러 개의 퍼셉트론을 층 구조로 구성한 신경망 모델

![image-20211108181427721](https://user-images.githubusercontent.com/70505378/140719020-db5a465c-9611-49a0-8bb4-7233a7ec57d0.png)

<br>

### 다층 퍼셉트론의 동작

![image-20211108181803553](https://user-images.githubusercontent.com/70505378/140719022-06bbcd2d-7559-4480-8be9-97a98306032c.png)

<br>

### 다층 퍼셉트론의 학습

입력-출력 (x, y)의 학습 데이터에 대해 출력값과 f(x)의 차이, **오차(error)가 최소가 되도록 가중치 w를 결정**하는 것

* 학습 가능한 다층 퍼셉트론
  * 학습 알고리즘: 오차(오류) 역전파 알고리즘
  * 계단모양 전달 함수를 미분가능한 **시그모이드 함수**로 대체

![image-20211108181644127](https://user-images.githubusercontent.com/70505378/140719021-dc8e88d3-ffc4-44bc-b2e1-cc694e6e96c5.png)

* **학습 목표**

  * 기대 출력과 MLP 출력이 최대한 비슷해지도록 가중치를 변경하는 것

    ![image-20211108181937335](https://user-images.githubusercontent.com/70505378/140719023-0db98e47-8e22-4edf-a210-f338850994d8.png)

  * 경사 하강법

    ![image-20211108182420045](https://user-images.githubusercontent.com/70505378/140719026-17045d12-a8d7-48b2-aecd-012c1b6ac076.png)

  * 오차 역전파 알고리즘

![image-20211108182508780](https://user-images.githubusercontent.com/70505378/140719027-a87743a2-6e56-42ff-8a29-03d1ff6be024.png)

<br>

### 다중 분류

#### 소프트맥스 층

* 최종 출력을 **분류 확률**로 변환하는 층

![image-20211108182621530](https://user-images.githubusercontent.com/70505378/140719029-407b6806-a2ce-4492-905c-e936eb3fd2ce.png)

#### 최대 가능도 추정

* 데이터의 가능도를 최대로 하는 파라미터를 추정하는 것

![image-20211108182928352](https://user-images.githubusercontent.com/70505378/140719030-1fbc30f5-2e5e-459b-9a5c-68a53ccce0e9.png)

#### 교차 엔트로피

* 오차함수 E(w): 음의 로그 가능도로 정의

![image-20211108183003766](https://user-images.githubusercontent.com/70505378/140719031-c7a74652-1fe7-47f1-8aba-f851a1d89a09.png)

<br>

<br>

## RBF 망

### RBF 함수

* 기준 벡터와 입력 벡터 사이의 유사도를 측정하는 함수

![image-20211108183112107](https://user-images.githubusercontent.com/70505378/140719033-bbe70bad-ed75-4fc1-9042-000800878a27.png)

![image-20211108183126125](https://user-images.githubusercontent.com/70505378/140719034-b429c033-d7ee-4cd1-a9bd-a5a136294b18.png)

<br>

### RBF 망

* 어떤 함수 f<sub>k</sub>(x)를 다음과 같이 **RBF 함수들의 선형 결합 형태로 근사**시키는 모델

![image-20211108183232944](https://user-images.githubusercontent.com/70505378/140719037-6b4a5303-b7b0-42a4-860f-d58ad6034bfb.png)

![image-20211108183249376](https://user-images.githubusercontent.com/70505378/140719039-345e820d-f8a0-44ee-a4db-c61dd4ef3962.png)

![image-20211108183354546](https://user-images.githubusercontent.com/70505378/140719042-909fc766-ad0b-4bd4-afd5-1fa994537304.png)

<br>

### RBF 망의 학습

* 오차 함수 E

  ![image-20211108183424351](https://user-images.githubusercontent.com/70505378/140719044-3f8b72f1-6a1c-428c-973a-b2eb2d4b009a.png)

* **경사 하강법** 사용

  * 전체 데이터로부터 대표값을 추출해 기준 벡터 Mu와 파라미터 Beta 계산
  * 가중치 w를 경사 하강법을 이용해 갱신

* 부류 별 군집화 결과를 사용한 기준 벡터 Mu와 파라미터 Beta 초기화

  * 군집 중심: 기준(평균) 벡터 Mu
  * 분산의 역수: Beta

  ![image-20211108183636260](https://user-images.githubusercontent.com/70505378/140719045-45ac6e69-9947-4060-b11f-f7a9cb1cc0a3.png)

<br>

### RBF 망을 이용한 분류의 예

![image-20211108183703769](https://user-images.githubusercontent.com/70505378/140719047-985862c0-2dab-4a54-8564-c4933d57b908.png)



<br>























