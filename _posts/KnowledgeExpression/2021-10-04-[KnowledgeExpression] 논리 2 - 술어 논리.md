---
layout: single
title: "[Knowledge Expression] 논리 2 - 술어 논리"
categories: ['AI', 'KnowledgeExpression']
toc: true
toc_sticky: true
tag: []
---

<br>

## 논리

**말로 표현된 문장**들에 대한 타당한 추론을 위해, **기호**를 사용하여 문장들을 **표현**하고 **기호의 조작**을 통해 문장들의 **참 또는 거짓을 판정**하는 분야

<br>

## 술어 논리란?

### 술어 논리

명제의 내용을 다루기 위해 **변수, 함수** 등을 도입하고 이들의 값에 따라 참, 거짓이 결정되도록 명제 논리를 확장한 논리

<br>

### 술어

* 문장의 `주어 + 서술어` 형태에서 **서술어**에 해당

* 대상의 **속성**이나 **대상간의 관계**를 기술하는 기호

* **참** 또는 **거짓**을 부여하는 명제의 기본 형식

* 예

  ![image-20211006101636687](https://user-images.githubusercontent.com/70505378/136129765-113f3d03-e1eb-44e4-afd6-aa7c5852148a.png)

<br>

<br>

## 술어 논리의 구문

### 존재 한정사와 전칭 한정사

* 변수의 범위를 고려한 지식을 표현

* 예

  ![image-20211006101939688](https://user-images.githubusercontent.com/70505378/136129801-771d0e8a-5e07-4281-b96f-27a4aa006f21.png)

<br>

### 함수

* 주어진 인자에 대해서 **참, 거짓 값이 아닌 일반적인 값을 반환**
* 술어나 다른 함수의 인자로 사용

### 항

* 함수의 인자가 될 수 있는 것
* 항이 될 수 있는 것: 개체상수, 변수, 함수
  1. 개체 상수, 변수는 항이다. 
  2. t1, t2, ..., tn이 모두 항이고, f가 n개의 인자를 갖는 함수 기호일 때, f(t1, t2, ..., tn)은 항이다. 
  3. 1과 2에 의해 만들어질 수 있는 것만 항이다. 

<br>

### 술어 논리식에 대한 정형식

![image-20211006102310049](https://user-images.githubusercontent.com/70505378/136129832-48c5c847-bedf-4a99-833e-8669eeeff24a.png)

<br>

<br>

## 술어 논리의 종류

### 일차 술어논리 (first-order predicate logic, FOL)

**술어 기호의 인자로 사용될 수 있는 객체나 대상 만을 변수화할 수 있고**, 이들 변수에만 전칭 한정사와 존재 한정사를 쓸 수 있도록 한 술어논리

![image-20211006103107447](https://user-images.githubusercontent.com/70505378/136129854-aaef36d9-fe70-4f89-a5e2-d554fd8ab661.png)

<br>

### 고차 술어논리 (high-order predicate logic)

**함수나 술어기호도 변수화할 수 있고**, 이들 변수에 대해서도 전칭 한정사와 존재 한정사를 쓸 수 있도록 한 술어논리

![image-20211006103057834](https://user-images.githubusercontent.com/70505378/136129876-13c26a25-0395-4bf5-a3d4-42710de8e3ec.png)

<br>

<br>

## 술어 논리의 지식 표현

![image-20211006104642987](https://user-images.githubusercontent.com/70505378/136129889-886397e4-7e94-4a29-a0c1-805a01b575d1.png)

<br>

<br>

## 술어 논리의 추론

### 술어논리식의 CNF(논리곱 정규형)로의 변환 과정

**1. 전칭 한정사와 존재 한정사를 논리식의 맨 앞으로 끌어내는 변환**

**2. 전칭 한정사에 결합된 변수**

* 임의의 값 허용

**3. 존재 한정사에 결합된 변수**

* 대응되는 술어 기호를 참으로 만드는 값을 변수에 대응시킴

* <span style="color:blue">**스콜렘 함수** (Skolem function)</span>

  * 존재 한정사에 결합된 변수를 해당 술어의 **전칭 한정사**에 결합된 다른 **변수**들의 새로운 **함수로 대체**: <span style="color:red">**s(x)**</span>

  * 예

    ![image-20211006105340992](https://user-images.githubusercontent.com/70505378/136129924-8367fd89-7445-4444-800a-902022b8a35e.png)

정리하면 **술어논리식의 CNF로의 변환과정**은, 

* **전칭 한정사**만 존재하는 변수의 경우 논리식에서 전칭 한정사 삭제. 이는 변수에 임의의 값을 허용함을 뜻함. 
* **존재 한정사**만 존재하는 변수의 경우 대응되는 술어기호를 참으로 만드는 값을 변수에 대응. 
* **존재 한정사와 전치 한정사**가 함께 사용되는 경우 스콜렘 함수 사용

<br>

<br>

### 단일화 과정

논리융합을 적용할 때, 대응되는 리터럴이 같아지도록 변수의 값을 맞춰주는 과정

![image-20211006110108084](https://user-images.githubusercontent.com/70505378/136129938-e7f5b106-7121-459c-b07c-4d310e8acd4a.png)

<br>

### 술어 논리로 지식의 증명

<br>

![image-20211006110237664](https://user-images.githubusercontent.com/70505378/136129942-2b53d6fb-5579-477f-933c-3e0b2d3058f4.png)

<br>

![image-20211006110325029](https://user-images.githubusercontent.com/70505378/136129945-7d4df133-b81e-421d-9a31-5d12111469d1.png)

<br>

<br>

## 논리 프로그래밍 언어

* Horn 절

  * 논리식을 논리합의 형태로 표현할 때, `ㄱA(x) or ㄱB(x) or ㄱC(x)`와 같이 긍정인 리터럴을 최대 하나만 허용

* Prolog

  * Horn 절만 허용하는 논리 프로그래밍 언어

  ![image-20211006110557748](https://user-images.githubusercontent.com/70505378/136129947-adc935a8-f692-4864-ac5a-eb3c8485d283.png)

  * 백트래킹을 이용하여 실행

<br>

<br>







































