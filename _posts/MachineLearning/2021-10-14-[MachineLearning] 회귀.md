---
layout: single
title: "[Machine Learning] 회귀"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: ['Regression']
---

<br>

## 회귀

`회귀 (Regression)` 문제는 **학습 데이터에 부합**되는 **출력 값**이 **실수**인 함수를 찾는 문제입니다. 

![image-20211014211942555](https://user-images.githubusercontent.com/70505378/137318180-81f969da-bb60-4d17-aa87-dd8c2f2e43a9.png)

<br>

### 회귀에서의 성능

회귀의 성능 지표로는 여러가지가 있는데, 그 중 가장 대표적으로 `MSE(Mean Squared Error)`를 사용합니다. 

![image-20211014212044866](https://user-images.githubusercontent.com/70505378/137318183-6ecfee27-2824-4d09-a216-1f295d51577a.png)

이 성능은 모델의 종류(함수의 종류)에 영향을 받습니다. 

<br>

### 회귀의 과적합과 부적합

* `과적합`(과대적합, 과잉적합): 지나치게 복잡한 모델(함수) 사용
* `부적합`(과소적합): 지나치게 단순한 모델 사용

![image-20211014212217022](https://user-images.githubusercontent.com/70505378/137318185-0bbad167-e89f-4040-8119-8062e28dc9fe.png)

#### 회귀의 과적합 대응 방법

회귀 문제에서 과적합에 대응하는 방법으로는 `가중치 규제` 방법이 있습니다. 

이는 모델의 복잡도를 성능 평가에 반영하는 것으로, 모델이 지나치게 복잡해지지 않도록 규제하는 것입니다. 

> <span style="color:blue">**목적 함수 = 손실 함수 + (가중치)\*(모델 복잡도)**</span>

**가중치 규제** 방법에는 크게 `L1 규제(Lasso)`, `L2 규제(Ridge)`, `엘라스틱넷 규제`가 있습니다.  

* L1 규제

![image-20211014212812407](https://user-images.githubusercontent.com/70505378/137318186-11173188-9d6f-47d2-87e1-b395eb0f8093.png)

* L2 규제

![image-20211014212829930](https://user-images.githubusercontent.com/70505378/137318187-98651528-205e-47a9-a7e6-85192d98e08c.png)

* 엘라스틱 넷 규제

![image-20211014212854706](https://user-images.githubusercontent.com/70505378/137318190-80e27844-e7b6-43e6-8593-b2da9f97da37.png)

<br>

<br>

## 로지스틱 회귀

로지스틱 회귀는 회귀의 출력 값에 `로지스틱 함수(시그모이드 함수)` 적용함으로써 점수((-INF, INF))를 확률([0, 1])로 변환합니다. 이 때 확률은 해당 데이터가 해당 클래스에 속할 확률입니다. 

즉, 로지스틱 회귀는 분류기로 사용될 수 있습니다. 

![image-20211014213155140](https://user-images.githubusercontent.com/70505378/137318192-49263fad-dbb6-4236-88cb-9deb5e1004d3.png)

<br>

로지스틱 회귀에서는 분류에 사용되는 손실 함수인 **크로스 엔트로피 손실 함수(Cross-Entropy Loss)**가 사용되며, 그 중 이진 분류에서는 **Binary Cross-Entropy**가 사용됩니다. 

![image-20211014213345192](https://user-images.githubusercontent.com/70505378/137318195-2ec96e5c-57a6-4ca1-810d-eb2ba0be3c76.png)

<br>





















