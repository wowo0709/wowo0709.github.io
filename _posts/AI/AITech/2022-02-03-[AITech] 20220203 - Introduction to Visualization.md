---
layout: single
title: "[AITech] 20220203 - Introduction to Visualization"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 학습 내용 정리

### 데이터 시각화란?

**데이터 시각화**는 데이터를 그래픽 요소로 매핑하여 시각적으로 표현하는 것으로, 다양한 요소가 포함되는 Task입니다. 

* 목적, 독자, 데이터, 스토리, 방법, 디자인

시각화는 사용하는 목적에 따라 다르고, 분야나 독자 등 여러 요소에 따라 바뀔 수 있기 때문에 **정답이란 것은 없습니다.**

다만, 지금까지 연구되고 사용된 시각화 모법 사례를 통해 **좋은 시각화**를 만들 수는 있습니다. 따라서 데이터 시각화 강의에서는 다음의 2가지를 그 목표로 합니다. 

1. 목적에 따라 시각화를 선택하고 사용할 수 있다. 
2. 시각화 결과를 효과적으로 수용할 수 있다. 

<br>

### 시각화의 요소

#### 데이터 이해하기

시각화를 진행하려면 당연히 **데이터**가 필요합니다. 이 데이터에는 어떤 종류의 데이터들이 있고, 그 데이터에서 어떤 정보들을 추출할 수 있을까요?

**정형 데이터**

![image-20220203125919149](https://user-images.githubusercontent.com/70505378/152280146-5b49fae3-c247-4d2e-9426-649c3f229e6d.png)

정형 데이터는 일반적으로 테이블 형태로 제공되는 데이터로 csv, tsv 파일 등이 있습니다. 이 때 각 row는 1개 item(data), columns은 1개 atribute(feature)라고 합니다. 

정형 데이터는 가장 쉽게 시각화 할 수 있는 데이터 형태이며 통계적 특성, feature 간의 관계, data 간의 관계 등을 주로 나타냅니다. 

**시계열 데이터**

![image-20220203130116482](https://user-images.githubusercontent.com/70505378/152280147-aa4e6320-8362-4cee-868f-1b6b4a62b428.png)

시계열 데이터는 시간 흐름에 따른 데이터를 나타냅니다. 기온, 주가 등의 정형데이터와 음성, 비디오 같은 비정형 데이터가 존재합니다. 

시계열 데이터에서는 시간 흐름에 따른 추세, 계절성, 주기성 등을 관찰할 수 있습니다. 

**지리 데이터**

![image-20220203130353027](https://user-images.githubusercontent.com/70505378/152280151-bb85d1ab-be33-4110-a733-38e7228ead33.png)

지리/지도 데이터에서는 **지도 정보와 보고자 하는 정보 사이의 조화**가 중요시되며, 이 때문에 지도 정보를 단순화시킬 수도 있습니다. 

거리, 경로, 분포, 위경도 등 다양한 특성들을 관찰할 수 있습니다. 

**관계형(네트워크) 데이터**

![image-20220203130404340](https://user-images.githubusercontent.com/70505378/152280153-47c0fc92-86e2-4ae5-ad35-03cc5ee48611.png)

관계형 데이터란 **객체와 객체 간의 관계**를 나타내는 데이터이며, 따라서 여러 객체 간의 관계를 시각화하기 위해 **그래프** 형태로 주로 표현합니다. 

객체는 Node, 관계는 Link로 표현하고 크기, 색, 수 등으로 객체와 관계의 가중치를 표현합니다. 

**계층적 데이터**

![image-20220203130501494](https://user-images.githubusercontent.com/70505378/152280154-582b266b-e59c-47d6-81b0-d84a6231cb7f.png)

관계형 데이터 중에서도 객체 간의 포함관계가 뚜렷한 데이터를 계층적 데이터라고 하며 회사 조직도, 가계도 등이 있습니다. 

Tree, Treemap, Sunburst 등으로 표현할 수 있습니다. 

**다양한 비정형 데이터**

이외에도 여러 다양한 비정형 데이터들을 존재합니다. 

<br>

위처럼 데이터는 여러 종류로 구분이 가능한데요, 대표적으로 가장 많이 사용하는 분류 방법으로 다음과 같은 분류 방법이 있습니다. 

* **수치형(Numerical)**
  * 연속형(Continuous): 실수 전체 범위에서 값을 가질 수 있는 데이터(길이, 무게, 온도 등)
  * 이산형(Discrete): 나올 수 있는 값이 정해져 있는 데이터(주사위 눈, 사람 수 등)
* **범주형(Categorical)**
  * 명목형(Norminal): 순서가 없는 카테고리 데이터(혈액형, 종교 등)
  * 순서형(Ordinal): 순서가 있는 카테고리 데이터(학년, 별점, 등급 등)

<br>

#### 시각화 이해하기

* **마크(mark)**: 점, 선, 면 등의 이미지에서 가장 기본이 되는 시각적 요소. 

  ![image-20220203131132137](https://user-images.githubusercontent.com/70505378/152280141-b50fa160-2643-4d4e-8bbe-da293f83c4d9.png)

* **채널(Channel)**: 기하학적 원형(geometric primitive)의 차원(dimensionality)과 독립적으로 각 마크를 변경할 수 있는 요소들

  ![image-20220203131140300](https://user-images.githubusercontent.com/70505378/152280143-eef23a8d-9137-48b0-9751-9f969e41ad5e.png)

* **전주의적 특성(Pre-attentive Attribute)**
  * 특별한 주의 없이도 자연스럽게 인지하게 되는 요소
  * 시각적으로 다양한 전주의적 속성이 존재
  * 하지만 동시에 사용하면 인지하기 어려움
  * 전주의적 특성을 적절하게 사용할 때 **시각적 분리(visual popout)**가 일어남

![image-20220203131332156](https://user-images.githubusercontent.com/70505378/152280145-bb61ca43-3cf7-4320-81ad-392f7fc921d6.png)

<br>

## 참고 자료

* 도서
  * Visualization Analysis&Design
  * Fundamentals of Data Visualization
* 사이트
  * https://kaggle.com
  * https://observablehq.com/
  * https://dataviztoday.com/
  * https://medium.com/nightingale
  * http://ieeevis.org/year/2021/welcome  
