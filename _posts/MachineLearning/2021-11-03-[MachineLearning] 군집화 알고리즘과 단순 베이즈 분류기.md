---
layout: single
title: "[Machine Learning] 군집화 알고리즘과 단순 베이즈 분류기"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: ['Agglomerative', 'Divisive', 'KMeans', 'NaiveBayesianClassifier']
---

<br>

## 군집화 알고리즘

`군집화 알고리즘`이란 데이터를 유사한 것들끼리 모으는 것입니다. 

**군집 간**의 **유사도는 작게(거리는 크게)**, **군집 내**의 **유사도는 크게(거리는 작게)**하는 것입니다. 

### 계층적 군집화 (Hierarchical clustering)

* 군집화의 결과가 군집들이 계층적인 구조를 갖도록 하는 것
* **병합형(agglomerative) 계층적 군집화**
  * 각 데이터가 하나의 군집을 구성하는 상태에서 시작하여, 가까이에 있는 군집들을 결합하는 과정을 반복하여 계층적인 군집 형성
* **분리형(divisive) 계층적 군집화**
  * 모든 데이터를 포함한 군집에서 시작하여 유사성을 바탕으로 군집을 분리하여 점차 계층적인 구조를 갖도록 구성

![image-20211103215106346](https://user-images.githubusercontent.com/70505378/140064902-59b52742-7a0b-4281-ab1f-12cb51f2b428.png)

<br>

### 분할 군집화

* 계층적 구조를 만들지 않고 전체 데이터를 유사한 것들끼리 나누어서 묶는 것
* **K-Means** 알고리즘

#### K-Means 알고리즘

* 전체 분산값 **_V_**를 최소화하는 S<sub>i</sub>를 찾는 것이 알고리즘의 목표

  ![image-20211103215730452](https://user-images.githubusercontent.com/70505378/140064912-4b354ae0-39af-494c-a1e6-a1ba25bc16b8.png)

* **과정**

1. **군집**의 **초기 중심위치**를 무작위로 선정
2. 군집 중심을 기준으로 **군집 재구성**
3. 군집별 중심을 **군집별 평균 위치로 재조정**
4. 2~3 과정을 군집 중심이 변하지 않을 때까지 반복 

* **특성**
  * 군집의 개수 k는 미리 지정
  * 초기 군집 위치에 민감 

![image-20211103215127720](https://user-images.githubusercontent.com/70505378/140064908-30fd2ce5-dfe2-48b1-8221-a0642d529f37.png)



<br>

<br>

## 단순 베이즈 분류기

* 부류(class)의 결정 지식을 조건부 확률(conditional probability)로 결정

  * P(c|x1, x2, ..., xn): 속성값에 대한 부류의 조건부 확률
    * c: 부류
    * xi: 속성값

* 베이즈 정리

  ![image-20211103215913601](https://user-images.githubusercontent.com/70505378/140064913-895eb33c-7ee7-4311-9f07-b0213b5f1116.png)

* 가능도(likelihood)의 조건부 독립(conditional independence) 가정

  ![image-20211103215940521](https://user-images.githubusercontent.com/70505378/140064915-903e0a80-cc83-4b3c-a8ce-51adaf0fe06d.png)

* **예시**

![image-20211103220159246](https://user-images.githubusercontent.com/70505378/140064918-02f82a95-c105-4f48-8d23-ac3f83233c1c.png)



<br>

<br>

