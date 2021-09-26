---
layout: single
title: "[Knowledge Expression] 논리 1 - 명제 논리"
categories: ['AI', 'KnowledgeExpression']
toc: true
toc_sticky: true
tag: []
---

<br>

## 논리

**말로 표현된 문장**들에 대한 타당한 추론을 위해, **기호**를 사용하여 문장들을 **표현**하고 **기호의 조작**을 통해 문장들의 **참 또는 거짓을 판정**하는 분야

<br>

## 명제 논리란?

### 명제

참, 거짓을 분명하게 판정할 수 있는 문장

**명제 O**: 아리스토텔레스는 플라톤의 제자이다. 1 + 1 = 3. 

**명제 X**: 일어나서 아침 먹자. 

* 명제 기호의 **진리값**을 사용하여 명제들에 의해 표현되는 문장들의 진리값 결정

* 문장 자체의 내용에 대해서는 무관심, **문장의 진리값**에만 관심

<br>

#### 기본 명제

하나의 진술(statement)로 이루어진 최소 단위의 명제

예 ) 

* 알렉산더는 아시아를 넘본다. ➡ P
* 징기스칸은 유럽을 넘본다. ➡ Q

#### 복합 명제

기본 명제들이 결합되어 만들어진 명제

예 )

* 알렉산더는 아시아를 넘보고, 징기스칸을 유럽을 넘본다. ➡ P ∧ Q

<br>

## 명제 논리의 구문

### 논리식

명제를 기호로 표현한 형식

* 논리 기호

![image-20210926125004541](https://user-images.githubusercontent.com/70505378/134793788-6f0d2e13-3689-4a56-94bb-bfa9be7d1139.png)

### 리터럴

명제 기호 P 또는 명제 기호의 부정 ㄱP

### 절

리터럴들이 **논리합**으로만 연결되거나 (**논리곱**으로 연결된 논리식)

![image-20210926125214769](https://user-images.githubusercontent.com/70505378/134793789-6c7fa25b-48cd-49a7-9cdd-4a2770925d2d.png)

### 논리곱 정규협(cojunctive normal form, CNF)

논리합 절들이 논리곱으로 연결되어 있는 논리식

![image-20210926125339432](https://user-images.githubusercontent.com/70505378/134793790-3f18ed2b-f2a7-445a-b189-08d5dc5404df.png)

### 논리합 정규형(disjunctive normal form, DNF)

논리곱 절들이 논리합으로 연결되어 있는 논리식

![image-20210926125354684](https://user-images.githubusercontent.com/70505378/134793792-310721cd-d0cf-4fd0-874b-5782a9be00ef.png)

### 정형식

논리에서 **문법에 맞는** 논리식

📌 **명제 논리에 대한 정형식**

1. 진리값 T, F와 명제 기호들 P, Q, R... 은 정형식이다. 
2. p와 q가 정형식이면, 논리 기호를 사용하여 구성되는 논리식 ㄱp, p or p ➡ p, p and q 도 정형식이다. 
3. 1과 2에 의해 정의되는 논리식만 정형식이다. 

<br>

<br>

## 명제 논리의 의미

### 진리표

논리 기호에 따라 참, 거짓 값을 결합하는 방법을 나타낸 표

![image-20210926125752439](https://user-images.githubusercontent.com/70505378/134793802-614f7a2f-8041-4361-954e-3b6d11973592.png)

### 논리식의 해석

`논리식의 해석`이란 논리식의 **진리값을 결정**하는 것입니다. 

이를 위해서는 먼저 논리식의 **명제 기호**에 **참 또는 거짓**을 할당해야 합니다. 

![image-20210926130249566](https://user-images.githubusercontent.com/70505378/134793804-240817fd-a349-4182-b5c1-e4ebf0d354be.png)

해석이 주어지면, **진리표**를 사용하여 논리식의 진리값을 결정할 수 있습니다. 

![image-20210926130337858](https://user-images.githubusercontent.com/70505378/134793805-34b2af94-9199-4527-9c17-03ba504a1b1a.png)

따라서, <span style="color:red">**n개의 명제 기호가 논리식**에 사용된다면, 각각 T 또는 F 값을 가질 수 있기 때문에, **총 2<sup>n</sup>개의 해석**이 존재합니다.</span>

<br>

#### 타당한 논리식 (항진식, Valid logical expression)

모든 가능한 해석에 대해서 **항상 참**인 논리식

![image-20210926131005510](https://user-images.githubusercontent.com/70505378/134793806-c289db45-b562-4279-8304-b9ad0fdafb6e.png)

#### 항위식 (Contradiction)

모든 가능한 해석에 대해서 **항상 거짓**이 되는 논리식

![image-20210926131038916](https://user-images.githubusercontent.com/70505378/134793820-b67da456-27da-40dd-8809-c02cddf0b8c6.png)

#### 충족 가능한 논리식

참으로 만들 수 있는 해석이 **하나라도 있는**, 즉 **모델이 존재**하는 논리식

![image-20210926131250739](https://user-images.githubusercontent.com/70505378/134793829-34815ef6-43a8-41f4-a5c7-91c7aa3d0b84.png)

#### 충족 불가능한 논리식

참으로 만들 수 있는 해석이 **전혀 없는**, 즉 **모델이 존재하지 않는** 논리식 (**항위식**인 논리식) 

<br>

### 동치 관계의 논리식

어떠한 해석에 대해서도 **같은 진리값**을 갖는 두 논리식

![image-20210926131511888](https://user-images.githubusercontent.com/70505378/134793830-5a4e2910-04aa-434a-84b0-f96542a52e57.png)

#### 동치관계를 이용한 논리식의 변환

논리식의 동치관계를 이용하면 **임의의 논리식**을 **논리곱 정규형(CNF)**과 같은 **정형식**으로 변환할 수 있습니다. 

![image-20210926131718118](https://user-images.githubusercontent.com/70505378/134793831-b2e68832-8f5d-4140-aa3a-c440398b0624.png)

<br>

### 논리적 귀결

* **Δ** : 정형식(wff)의 집합. 
* **ω**: 정형식. 

`Δ` 에 있는 모든 정형식을 참(T)으로 만드는 모델(해석)이, `ω`를 참(T) 으로 만든다.

➡ `Δ` 는  `ω`를 **논리적으로 귀결**한다. (logically entail)

➡ `ω` 는 `Δ` 를 **논리적으로 따른다.** (logically follow)

➡ `ω` 는 `Δ` 의 **논리적 결론이다.** (logical consequence)

<br>

**표기법**: **Δ |= ω**

➡ Δ 가 참이면,  ω 도 참이다. 

![image-20210926132852167](https://user-images.githubusercontent.com/70505378/134793832-b998e282-ccf4-48f5-b08e-9f539c52ff86.png)

![image-20210926132901981](https://user-images.githubusercontent.com/70505378/134793833-9fa3eda6-daa2-435b-bb75-aa1e7f246129.png)

<br>

<br>

## 명제 논리의 추론





## 명제 논리의 지식 표현

























