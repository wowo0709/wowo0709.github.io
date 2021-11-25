---
layout: single
title: "[Tensorflow&Keras] 딥러닝 프레임워크 텐서플로우"
categories: ['AI', 'TensorflowKeras']
toc: true
toc_sticky: true
tag: ['Tensorflow']
---



## 텐서플로우 소개

* **Google**에서 2015년 공개한 **오픈소스 라이브러리**
  * 파이썬/C++ API를 이용해 쉽게 사용 가능
  * 초창기에는 대량의 수치 계산을 효과적으로 지원하기 위해 개발됨
  * 현재는 기계학습과 딥러닝 응용 시스템을 개발하는데 널리 활용
  * 실제 내부 코드는 C/C++로 개발되어 컴파일 속도 빠름
  * CPU, GPU, 안드로이드 모바일 운영체제에서도 사용 가능
* **데이터 플로우 그래프**를 이용하여 계산과정과 모델을 표현
* 기계학습 알고리즘을 구현하고 실행하기 위한 프로그래밍 인터페이스
* 신경망과 딥러닝 관련 라이브러리 제공
  * 그레디언트 기반의 학습 알고리즘 사용
  * 다양한 최적화 모듈 제공

<br>

## 데이터 플로우 그래프

텐서플로는 일련의 계산 과정을 그래프를 이용하여 표현합니다. 

* **노드**: 수학적 연산
* **간선**: 노드 사이에 전달되는 데이터 배열(텐서)

👍 **예시**

![image-20211124204940729](https://user-images.githubusercontent.com/70505378/143238298-fac54384-ab6b-4149-a186-186081007702.png)

그리고 이는 **텐서보드(Tensorboard)**에서 확인할 수 있습니다. 

* **Operation**
  * Op라고도 부른다. 
  * 하나 이상의 텐서를 받아 계산을 수행하고 결과를 하나 이상의 텐서로 반환
* **Tensor**
  * 모든 데이터는 텐서를 통해 표현
  * 다차원 배열
* **Session**
  * 그래프를 실행하기 위해 세션 객체 필요
* **Variables**
  * 그래프 실행 시 파라미터를 저장하고 갱신하기 위해 사용

<br>

<br>

## 변수와 플레이스홀더

**Variable**

* 모델의 **학습 가능한 변수**를 정의할 때 Variable을 사용함

* 가중치, 편향 등의 파라미터를 저장하는 데 사용

* 정의할 때 반드시 초기화되어야 함

* 예

  ![image-20211124205259318](https://user-images.githubusercontent.com/70505378/143238300-b575e210-6b99-49a0-a474-59c019c3d78b.png)

**Placeholder**

* 데이터플로우 **그래프 실행 시 데이터(텐서)를 전달**하기 위해 사용

* 그래프 실행 시 값이 제공되어야 함

* 예

  ![image-20211124205340969](https://user-images.githubusercontent.com/70505378/143238303-56aaa2cd-cc7e-49ba-8ef8-5bb221cea80f.png)

<br>

<br>

## 텐서 보드

* 텐서플로우에서 제공하는 **시각적 도구**
* **데이터플로우 그래프**, **학습 과정의 성능 변화** 등을 보여줄 수 있음
* 프로그램 실행 중 해당 정보를 로그 파일에 저장
  * **tf.summary.FileWriter(폴더이름, 저장내용)**

![image-20211124205503210](https://user-images.githubusercontent.com/70505378/143238306-02880854-20d8-4888-972b-1d8d5c5b8a25.png)

텐서보드를 사용한 모습

![image-20211124205731903](https://user-images.githubusercontent.com/70505378/143238307-75feb9ed-cc51-4b66-a4db-175b4c004d82.png)

<br>

<br>

## 텐서

`텐서`란 3차원 이상의 다차원 배열로, 텐서플로에서 사용하는 기본 자료형입니다. 

**rank, shape, type**의 세가지 속성을 가짐

### rank

`rank`: 텐서의 차원수 (dimension, order, degree 라고도 함)

* rank 0 텐서: 스칼라 값
* rank 1 텐서: 1차원 배열 (rank 0 텐서의 배열)
* rank 2 텐서: 2차원 배열 (rank 1 텐서의 배열)
* rank n 텐서: n-텐서 (rank n-1 텐서의 배열)

![image-20211124210026662](https://user-images.githubusercontent.com/70505378/143238308-0cd83161-4269-4fae-bc98-cf24dc781bb5.png)

### shape

`shape`: 텐서의 구조

* shape [m] : rank 1인 텐서로 원소를 m개 가지고 있음
* shape [i1, i2,…, ik] : shape [i2, …, ik]인 원소를 i1 개 가지고 있음
* 예
  * **b = [6,3,5,2,7] -> b의 shape : [5]**
  * **c = [[1,2,3],[3,6,1],[5,8,2],[6,1,2],[8,2,3]] -> c의 shape : [5,3]**
  * **d = [[[1,2],[3,4],[4,5]], [[3,2],[5,7],[2,8]], [[5,1],[5,4],[8,2]]] -> d의 shape: [3,3,2]**

![image-20211124210307157](https://user-images.githubusercontent.com/70505378/143238310-4d3b6850-1299-4caa-8d32-9a4e8f3fc5d0.png)

### type

`type`: 텐서 구성 원소의 자료형

| 텐서플로 자료형 | 파이썬 자료형 | 의미        |
| --------------- | ------------- | ----------- |
| CT_FLOAT        | float32       | 32비트 실수 |
| DT_INT16        | int16         | 16비트 정수 |
| DT_INT32        | int32         | 32비트 정수 |
| DT_INT64        | int64         | 64비트 정수 |
| DT_STRING       | string        | 문자열      |
| DT_BOOL         | bool          | 불리언      |

<br>

<br>

## 텐서 변환 연산

| 함수        | 용도                                             |
| ----------- | ------------------------------------------------ |
| shape       | 텐서의 형상 정보 확인                            |
| size        | 텐서의 크기 확인                                 |
| rank        | 텐서의 rank 확인                                 |
| reshape     | 텐서의 원소는 유지하면서 구조를 변경             |
| squeeze     | 텐서의 크기가 1인 차원을 삭제                    |
| expand_dims | 텐서에 차원을 추가                               |
| slice       | 텐서의 일부분 선택                               |
| split       | 텐서를 한 차원을 기준으로 여러 개의 텐서로 분리  |
| tile        | 한 텐서를 여러 번 중복해서 늘려 새로운 텐서 생성 |
| concat      | 한 차원을 기준으로 텐서를 이어 붙임              |
| reverse     | 텐서의 지정된 차원의 구성요소를 역전시킴         |
| transpose   | 텐서를 전치 시킴(축을 교환)                      |
| gather      | 주어진 인덱스에 따라 텐서의 원소 수집            |

<br>

* **shape, size, rank**

  * 예

    ![image-20211124210814912](https://user-images.githubusercontent.com/70505378/143238312-4d8c298e-6d49-49ce-948c-7a5c84ab2f31.png)

* **reshape**: 텐서의 기존의 원소를 새로운 shape에 따라 재배치

  * <span style="color:blue">**reshape(텐서, shape)**</span>

  * 예

    ![image-20211124211435383](https://user-images.githubusercontent.com/70505378/143238326-7b15cc92-0d91-45ca-b565-2c9db1ee9b93.png)

* **squeeze**: 텐서에서 크기가 1인 축(axis)을 제거

  * <span style="color:blue">**squeeze(텐서)**</span>

  * 예

    ![image-20211124211058946](https://user-images.githubusercontent.com/70505378/143238317-4d9af748-1789-42a9-818c-03815d1cbc9a.png)

* **expand_dims**: 지정한 축의 위치에 차원 추가

  * <span style="color:blue">**expand_dims(텐서, 확장축)**</span>

  * 예

    ![image-20211124211149973](https://user-images.githubusercontent.com/70505378/143238321-1e67d7e1-bc60-415b-9f3a-816f57171314.png)

* **slice**: 텐서에서 일부분 선택

  * <span style="color:blue">**slice(텐서, [축0의 추출시작위치, 축1의 추출시작위치, …, 축d-1의 추출시작위치],  [축0의 길이, 축1의 길이, …, 축d-1의길이])**</span>

  * 예

    ![image-20211124211357289](https://user-images.githubusercontent.com/70505378/143238322-2a68cc84-050f-437a-b4ab-efa85b91498e.png)

* **split**: 지정된 축을 따라 하나의 텐서를 주어진 개수의 부분 텐서로 분리

  * <span style="color:blue">**split(텐서, 텐서분할개수, 축)**</span>

  * 예

    ![image-20211124211605804](https://user-images.githubusercontent.com/70505378/143238327-f3e369f7-6661-4eae-99f3-3630b1c065bd.png)

* **concat**: 한 축을 기준으로 두 텐서를 이어붙임

  * <span style="color:blue">**concat([텐서1, 텐서2], 차원)**</span>

  * 예

    ![image-20211124211736396](https://user-images.githubusercontent.com/70505378/143238332-e5fbe8c6-09ab-4d88-af8a-f22d66d51614.png)

* **reverse**: 텐서의 축을 기준으로 원소를 역순으로 배열

  * <span style="color:blue">**reverse(텐서, [축]**</span>

  * 예

    ![image-20211124211834587](https://user-images.githubusercontent.com/70505378/143238335-1b816a41-b706-4fd4-b5d7-be41319291fc.png)

* **transpose**: 텐서의 지정된 축을 지정된 순서로 바꿈

  * <span style="color:blue">**transpose(텐서, {perm=[축의 순열]})**</span>

    * {perm = [축의 순열]}이 없으면 default로 [1,0]으로 간주

  * 예

    ![image-20211124212032543](https://user-images.githubusercontent.com/70505378/143238336-7ee5454f-8dc0-4f2c-abbb-029e6a924f05.png)

* **gather**: 지정된 인덱스의 원소들로 이루어진 텐서를 생성

  * <span style="color:blue">**gather(텐서, [인덱스(들)]**</span>

  * 예

    ![image-20211124212238328](https://user-images.githubusercontent.com/70505378/143238337-c65694f9-3a4e-4485-beff-1f8311ce8fa1.png)

* **one-hot**: 정수값을 one-hot 벡터로 변환

  * <span style="color:blue">**one_hot(텐서, depth=전체원소의 가짓수)**</span>

  * 예

    ![image-20211124212320818](https://user-images.githubusercontent.com/70505378/143238340-461a77c7-bc56-44af-9ce9-4ba83988823f.png)

<br>

<br>

## 텐서 산술 연산

| 함수        | 용도                      |
| ----------- | ------------------------- |
| add         | 덧셈                      |
| subtract    | 뺄셈                      |
| multiply    | 곱셈                      |
| truediv     | 실수값 몫을 구하는 나눗셈 |
| trancatediv | 정수 몫을 구하는 나눗셈   |
| trancatemod | 나머지 연산               |
| abs         | 절댓값                    |
| negative    | 음수로 변환한 값          |
| sign        | 수의 부호                 |
| reciprocal  | 역수                      |
| square      | 제곱                      |
| round       | 반올림 값                 |
| pow         | 거듭제곱                  |
| log         | 로그함수 값               |
| exp         | 지수함수 값               |
| sin         | 사인값                    |
| cos         | 코사인 값                 |

<br>

<br>

## 텐서 축약 연산

텐서의 크기를 줄이는 함수: <span style="color:blue">**reduce_X(텐서, 축)**</span>

| 함수             | 용도                                                         |
| ---------------- | ------------------------------------------------------------ |
| reduce_max       | 지정된 축의 원소들 중 최댓값                                 |
| reduce_min       | 지정된 축의 원소들 중 최솟값                                 |
| reduce_mean      | 지정된 축의 원소들 중 평균값                                 |
| reduce_sum       | 지정된 축의 원소들의 합                                      |
| reduce_prod      | 지정된 축의 원소들의 곱                                      |
| arg_max          | 지정된 축의 원소들 중 최댓값인 원소의 위치                   |
| arg_min          | 지정된 축의 원소들 중 최솟값인 원소의 위치                   |
| reduce_logsumexp | 지정된 축의 원소들에 exp를 적용하여 sum을 한 다음 log를 한 결과 |

<br>

![image-20211126042443962](https://user-images.githubusercontent.com/70505378/143493137-305c0f7b-350a-4273-a35a-509d0b3c3e67.png)





<br>

<br>

## 텐서 행렬 연산

| 함수               | 용도                                       |
| ------------------ | ------------------------------------------ |
| diag               | 주어진 원소를 대각에 배치한 대각 행렬 변환 |
| transpose          | 전치 행렬                                  |
| matmul             | 텐서의 행렬 곱셈                           |
| matrix_determinant | 정방 행렬의 행렬식                         |
| matrix_inverse     | 정방 행렬의 역행렬                         |

<br>

<br>

## 기타 텐서 연산 함수

**기타 유용한 함수**

| 함수       | 용도                                                         |
| ---------- | ------------------------------------------------------------ |
| one_hot    | 주어진 정수값을 지정한 차원의 one-hot 벡터로 표현            |
| cast       | 텐서의 원소의 자료형을 지정한 자료형으로 변환                |
| stack      | 텐서를 지정된 축에 따라 쌓아 새로운 텐서를 생성              |
| ones_like  | 텐서의 shape과 같은 크기의 텐서를 만들어 각 원소를 1로 초기화 |
| zeros_like | 텐서의 shape과 같은 크기의 텐서를 만들어 각 원소를 0으로 초기화 |
| zeros      | 주어진 shape과 같은 크기의 텐서를 만들어 각 원소를 0으로 초기화 |
| ones       | 주어진 shape과 같은 크기의 텐서를 만들어 각 원소를 1로 초기화 |
| where      | Boolean 텐서에서 true인 원소의 위치에 대한 첨자 변환         |

![image-20211126042929653](https://user-images.githubusercontent.com/70505378/143493139-dc52a928-4cd9-4732-aead-d6d386370832.png)

<br>

**난수 생성 함수**

| 함수            | 용도                                                       |
| --------------- | ---------------------------------------------------------- |
| random_normal   | 정규분포 형태의 난수 생성                                  |
| truncted_normal | (2*표준편차)를 벗어나는 것을 제외한 정규분포에서 난수 생성 |
| random_uniform  | 균등 분포에서 난수 생성                                    |
| random_shuffle  | 첫번째 축을 중심으로 텐서의 원소를 섞음                    |
| set_random_seed | 난수의 초기값(seed) 생성                                   |



