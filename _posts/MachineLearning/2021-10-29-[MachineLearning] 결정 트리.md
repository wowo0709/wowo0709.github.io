---
layout: single
title: "[Machine Learning] 결정트리"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: ['DecisionTree']
---

<br>

## 결정 트리의 형태

### 결정 트리

* **트리 형태**로 의사결정 **지식**을 표현한 것
  * 내부 노드(internal node): 비교 속성
  * 간선(edge): 속성 값 
  * 단말 노드(terminal node): 부류(class), 대표값

![image-20211029191221150](https://user-images.githubusercontent.com/70505378/139424514-15c2d53c-f969-4669-8f75-800bc7ebbd27.png)

<br>

<br>

## 결정 트리 학습 알고리즘

### 결정 트리 알고리즘

* **모든 데이터를 포함한 하나의 노드**로 구성된 트리에서 시작
* **반복적인 노드 분할** 과정
  1. **분할 속성** 선택
  2. 속성값에 따라 **서브 트리** 생성
  3. 데이터를 속성값에 따라 **분배**

![image-20211029191404485](https://user-images.githubusercontent.com/70505378/139424518-a921c0b4-ba43-4208-8eb8-8dbeb3512865.png)

<br>

### 분할 속성의 결정

* **어떤 속성**을 선택하는 것이 효율적인가?
  * 분할한 결과가 가능하면 <span style="color:red">**동질적인 것(불순도가 낮은 것)**</span>으로 만드는 속성을 선택

#### 엔트로피

* **엔트로피(Entropy)**

  * **불순도**를 나타내는 척도로, [-1, INF) 범위의 값을 가지며 클수록 불순도가 높음
  * 단위로는 **bits**를 사용(정보량)
  * **p(c)**는 부류 c에 속하는 것의 비율

  ![image-20211029191713552](https://user-images.githubusercontent.com/70505378/139424519-c533c00c-5c8b-4476-b49b-70af697f5b92.png)
  * 2개 부류가 있는 경우의 엔트로피

    ![image-20211029191823557](https://user-images.githubusercontent.com/70505378/139424521-2c699d20-e45c-48e5-8861-eb9665048270.png)

<br>

* **정보 이득**
  * <span style="color:red">**IG = I - I<sub>res</sub>**</span>
    * I<sub>res</sub>: 특정 속성으로 분할한 후의 각 부분집합의 정보량의 가중평균
  * 정보이득이 클수록 우수한 분할 속성

![image-20211029192032488](https://user-images.githubusercontent.com/70505378/139424522-f0628ec4-4fb8-4bb9-a830-1bcf93ca56a1.png)

<br>

👍 **예시**

학습 데이터

![image-20211029192519803](https://user-images.githubusercontent.com/70505378/139424523-241e4982-fd0d-4891-8eab-a3182b4f536d.png)

_(pattern 기준 분할 시)_

![image-20211029194029725](https://user-images.githubusercontent.com/70505378/139424524-4d828f5a-b6fe-4335-b32a-40f25ee3238d.png)

_(outline 기준 분할 시)_

![image-20211029194201509](https://user-images.githubusercontent.com/70505378/139424526-9c16855d-9c98-42fa-9be7-1cd0e3a1b92b.png)

_(dot 기준 분할 시)_

![image-20211029194259092](https://user-images.githubusercontent.com/70505378/139424528-87ba67e8-d999-4791-b1ba-1b11c8ba637e.png)

<br>

최종 분할 선택

* IG(Pattern) = 0.246
* IG(Outline) = 0.151
* IG(Dot) = 0.048

![image-20211029194505828](https://user-images.githubusercontent.com/70505378/139424529-39a5134f-db80-4c20-83e5-550bfb9c8f49.png)

분할 2회 수행 이후 최종 결정 트리

![image-20211029194556877](https://user-images.githubusercontent.com/70505378/139424530-140284ae-16d4-45d6-ada8-a37eea5b17e8.png)

![image-20211029194615010](https://user-images.githubusercontent.com/70505378/139424531-8f59b9f0-43b8-44bf-9d79-23ffc9e3734a.png)

<br>

* **정보이득 척도의 단점**
  * **속성 값이 많은 것** 선호
    * 예) 학번, 이름 등
  * **속성 값이 많으면** 데이터집합을 **많은 부분집합으로 분할**
  * 테스트 데이터에 대해 좋은 성능을 보이기 어려움
* **개선 척도**
  * 정보 이득비
  * 지니 지수

<br>

#### 정보 이득비

* **정보 이득비(Information gain ratio)**

  * 속성값이 많은 속성에 대해 불이익

  ![image-20211029194933411](https://user-images.githubusercontent.com/70505378/139424533-181ec140-92f2-438e-a37c-c50d513080bb.png)

* **I(A)**

  * 속성 A의 속성값을 부류(class)로 간주하여 계산한 엔트로피
  * 속성값이 많을수록 커지는 경향

  ![image-20211029195021545](https://user-images.githubusercontent.com/70505378/139424535-839f1f69-9999-4585-8ee6-402b23d88131.png)

  👍 **예시**

  ![image-20211029195551657](https://user-images.githubusercontent.com/70505378/139424537-613984d3-b8a6-4bd4-b22a-49c2d8e0226b.png)

<br>

#### 지니 지수

* **지니 지수(Gini Index)**

  * 데이터 집합에 대한 Gini 값

    * i, j가 부류를 나타낼 때

    ![image-20211029195658100](https://user-images.githubusercontent.com/70505378/139424540-10b64e10-232a-41fb-bb3e-cfc626323a0f.png)

    ![image-20211029195715422](https://user-images.githubusercontent.com/70505378/139424542-efffb46d-96cd-4b07-bcd5-1e16bd3ccb10.png)

  * 속성 A에 대한 지니 지수값 가중평균

    ![image-20211029195742274](https://user-images.githubusercontent.com/70505378/139424543-a158feab-4f62-4217-8aba-dbb19bf7dc08.png)

  * 지니 지수 이득 (Gini index gain)

    ![image-20211029195804703](https://user-images.githubusercontent.com/70505378/139424544-ad0f2195-032e-4bb5-93c6-3773b5b66a66.png)

👍 **예시**

![image-20211029195953996](https://user-images.githubusercontent.com/70505378/139424546-274f19c8-f022-4a91-8639-ff4e76f469dd.png)

#### 분할속성 평가 척도 비교

![image-20211029200024141](https://user-images.githubusercontent.com/70505378/139424549-86c0969c-ccae-449a-a308-cc1a0e1afc66.png)

<br>

<br>

## 결정트리를 이용한 회귀

회귀를 위한 결정트리에서는 출력값이 수치값인 데이터를 사용한다. 

![image-20211029200129055](https://user-images.githubusercontent.com/70505378/139424552-86a1bffd-a896-4fa4-839c-c38b38e5feb8.png)

<br>

### 분류를 위한 결정트리와 차이점

* **단말 노드**가 부류(class)가 아닌 **수치값**
* 해당 조건을 만족하는 것들이 가지는 **대표값**

### 분할 속성 선택

* **표준편차 축소 SDR**를 최대로하는 속성 선택

  ![image-20211029200326316](https://user-images.githubusercontent.com/70505378/139424555-a575e29e-c479-435a-b347-ee4c66ebcd94.png)

  * 표준편차 SD

    ![image-20211029200350400](https://user-images.githubusercontent.com/70505378/139424559-5151355a-7091-4893-9c01-5ff9d46a527d.png)

  * SD(A)

    * 속성 A를 기준으로 **분할 후**의 **부분 집합별 표준편차**의 **가중 평균**

<br>

👍 **예시**

![image-20211029200519408](https://user-images.githubusercontent.com/70505378/139424562-25cd0d85-ca47-48fc-8d7a-fd7f96de72f2.png)

![image-20211029200618042](https://user-images.githubusercontent.com/70505378/139424565-f4f1110d-9bc8-4243-a9c4-74fac2cfb9aa.png)

<br>

<br>
