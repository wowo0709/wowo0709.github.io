---
layout: single
title: "[Knowledge Expression] 지식표현과 추론"
categories: ['AI', 'KnowledgeExpression']
toc: true
toc_sticky: true
tag: []
---

<br>

## 지식 표현과 추론

`Knowledge Expression` 카테고리의 포스팅들에서는 인공지능에서의 지식 표현 방법과 추론 방법에 대해 알아봅니다. 

비단 인공지능에서 사용되는 내용 뿐만이 아닌 전체적인 내용을 다룰 예정입니다. 

목차는 다음과 같습니다. 

* 규칙
* 프레임
* 논리
* 의미망
* 스크립트
* 온톨로지
* 함수에 의한 지식 표현
* 불확실한 지식 표현
* 규칙 기반 시스템
* 심볼 그라운딩 문제와 프레임 문제
* CYC 프로젝트

<br>

## 지식 표현

이번 포스팅에서는 `지식 표현`에 대한 개괄적인 내용을 다루겠습니다. 

<br>

### 데이터 피라미드

**데이터 (data)**

* 특정 분야에서 <span style="color:blue">**관측된 아직 가공되지는 않은 것**</span>
* 사실인 것처럼 관측되지만 <span style="color:blue">**오류**</span>나 <span style="color:blue">**잡음**</span>을 포함 가능

**정보 (information)**

* 데이터를 <span style="color:blue">**가공**</span>하여 어떤 <span style="color:blue">**목적이나 의미**</span>를 갖도록 한 것

**지식 (knowledge)**

* 정보를 <span style="color:blue">**취합하고 분석**</span>하여 얻은 대상에 대해 <span style="color:blue">**이해**</span>한 것

**지혜 (wisdom)**

* 경험과 학습을 통해서 얻은 <span style="color:blue">**지식보다 높은 수준의 통찰**</span>

![image-20210923120927208](https://user-images.githubusercontent.com/70505378/134452119-0b4d9105-f7a4-4c3b-8326-c3ff0e8d017d.png)

<br>

<br>

### 지식

`지식`이란 경험이나 교육을 통해 얻어진 전문적인 이해와 체계화된 문제 해결 능력이며, 어떤 주제나 분야에 대한 이론적 또는 실제적인 이해 또는 현재 알려진 사실과 정보의 모음을 뜻합니다. 

지식의 표현 여부에 따라 **암묵지**와 **형식지**로 나눌 수 있습니다. 

* 암묵지: 형식을 갖추어 **표현하기 어려운**, 학습과 경험을 통해 쌓은 지식
* 형식지: 비교적 쉽게 형식을 갖추어 **표현될 수 있는 지식**

또한 지식의 기술 방법에 따라 **절차적 지식**과 **선언적 지식**으로 나눌 수 있습니다. 

* 절차적 지식: 문제해결의 절차 기술
* 선언적 지식: 어떤 대상의 성질, 특성이나 관계 서술

<br>

이러한 지식을 **프로그램이 쉽게 처리**할 수 있도록 **정형화된 형태**로 표현하는 것이 지식 표현의 목적입니다. 

