---
layout: single
title: "[AITech] 20220119 - Vector&Matrix Basics"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Vector', 'Matrix']
---



<br>

## 강의 복습 내용

### 벡터

* **벡터**는 숫자를 원소로 가지는 리스트 또는 배열이다. 
  
  * **벡터의 차원**은 벡터가 가진 원소의 개수이다. 
  
* **벡터**는 n차원 공간에서 한 점을 나타낸다. 
  * 벡터에 숫자를 곱해주면 길이만 변한다. (스칼라곱)
  * 벡터는 같은 모양을 가지면 덧셈, 뺄셈, 성분곱(Hadamard product)을 계산할 수 있다. 
    * 벡터 덧셈은 다른 벡터로부터 상대적 위치 이동을 표현합니다. 
    * 벡터 뺄셈은 벡터의 방향을 뒤집은 덧셈입니다. 
  
  ```python
  import numpy as np
  
  x = np.array([1,7,2])
  y = np.array([5,2,1])
  
  # 벡터 덧셈
  print(x+y)
  # 벡터 뺄셈
  print(x-y)
  # 벡터 내적
  print(x*y)
  '''
  [6 9 3]
  [-4  5  1]
  [ 5 14  2]
  '''
  ```

* **벡터의 노름**은 **원점에서부터의 거리**를 말합니다. 

  ![노름(norm)](https://t1.daumcdn.net/cfile/tistory/99D721445BD00C6E1A)

  ```python
  def l1_norm(x):
      x_norm = np.abs(x)
      x_norm = np.sum(x_norm)
      return x_norm
  
  def l2_norm(x):
      x_norm = x*x
      x_norm = np.sum(x_norm)
      x_norm = np.sqrt(x_norm)
      return x_norm
  
  x = np.array([1,2,3])
  print(f"l1 norm: {l1_norm(x)}")
  print(f"l2 norm: {l2_norm(x)}")
  '''
  l1 norm: 6
  l2 norm: 3.7416573867739413
  '''
  ```

* 서로 다른 노름이 중요한 이유

  * 노름의 종류에 따라 **기하학적 성질**이 달라진다. 
  * 머신러닝에선 각 성질들이 필요할 대가 있으므로 둘 다 사용한다. 

  ![image-20220119112430894](https://user-images.githubusercontent.com/70505378/150055194-34fd93f3-c1e7-445e-bae5-10ab8320cb10.png)

  * L1, L2 노름을 이용해 **두 벡터 사이의 거리**를 계산할 수 있다. 
  * L2 노름을 이용해 **두 벡터 사이 각도**를 계산할 수 있다. 

  ```python
  def angle(x,y):
      v = np.inner(x,y) / (l2_norm(x)*l2_norm(y))
      theta = np.arccos(v)
      return theta # Pi 기준 표시
  
  x, y = np.array([1,10,3]), np.array([-1,-10,-3])
  print(angle(x,y))
  # 3.141592653589793
  ```

* **내적**은 **정사영된 벡터의 길이**와 관련 있다. 

  * 내적은 두 벡터의 **유사도**를 측정하는 데 사용 가능하다. 

  ![image-20220119113026170](https://user-images.githubusercontent.com/70505378/150055199-3e4817d2-36ed-48a9-8ece-35b3a388a8a4.png)

<br>

### 행렬

* **행렬**은 벡터를 원소로 가지는 **2차원 배열**이다. 

  * 행렬은 **행**과 **열**이라는 인덱스를 가진다. 

  ![image-20220119113221006](https://user-images.githubusercontent.com/70505378/150055203-94431aca-0f80-443a-9bfe-bd3284897dac.png)

  * **전치 행렬**은 행과 열의 인덱스가 바뀐 행렬이다. 

  ![image-20220119113259537](https://user-images.githubusercontent.com/70505378/150055206-fde116e1-e729-4adc-badd-c85201a60274.png)

  * 행렬끼리 같은 모양을 가지면 덧셈, 뺄셈, 성분곱, 스칼라곱을 계산할 수 있다. 
  * **행렬 곱셈**은 **i번째 행벡터와 j번재 열벡터 사이의 내적**을 계산하고, **행렬 내적**은 **i번째 행벡터와 j번째 행벡터 사이의 내적(XY<sup>T</sup>**을 계산한다. 
    * 수학에서 말하는 내적과는 다르므로 주의!

  ```python
  X = np.array([[1,-2,3],
                [7,5,0],
                [-2,-1,2]])
  Y = np.array([[0,1,10],
                [1,-1,7],
                [-2,1,0]])
  
  # 행렬곱
  print(X @ Y)
  # 행렬 내적
  print(np.inner(X,Y)) # = X @ Y.T
  '''
  [[ -8   6  -4]
   [  5   2 105]
   [ -5   1 -27]]
  [[28 24 -4]
   [ 5  2 -9]
   [19 13  3]]
  '''
  ```

  

* **행렬을 이해하는 방법 1**

  * 벡터가 공간에서 한 점을 나타낸다면, 행렬을 **여러 점들**을 나타낸다. 
  * 행렬의 행벡터 xi는 i번째 데이터를 의미한다. 
  * 행렬의 xij는 i번째 데이터의 j번째 변수의 값을 말한다. 

   

* **행렬을 이해하는 방법 2**

  * 행렬은 **벡터 공간에서 사용되는 연산자**로 이해한다. 
  * 행렬 곱을 통해 벡터를 **다른 차원의 공간**으로 보낼 수 있다. 
  * 행렬 곱을 통해 **패턴을 추출**할 수도 있고, **데이터를 압축**할 수도 있다. 
    * 모든 선형변환은 행렬곱으로 계산할 수 있다!

  ![image-20220119114208634](https://user-images.githubusercontent.com/70505378/150055207-b575bca8-e772-4339-bf0f-5ac106ff97be.png)

  * 역행렬 이해하기

    * 어떤 행렬 A의 연산을 거꾸로 되돌리는 행렬을 **역행렬**이라 부르고 A<sup>-1</sup>라 표기한다. (AA<sup>-1</sup> = A<sup>-1</sup>A = I)

      ![image-20220119114654455](https://user-images.githubusercontent.com/70505378/150055214-792166a7-7145-4145-b5c2-f70a3590f1de.png)

      ![image-20220119114636254](https://user-images.githubusercontent.com/70505378/150055212-79f4a05c-5c8d-4b73-b20b-97c6dc9bc695.png)

      

    * 역행렬은 **행과 열 숫자가 같고 행렬식이 0이 아닌 경우**에만 계산할 수 있다. 

    * 만일 역행렬을 계산할 수 없다면 **유사 역행렬** 또는 **무어-펜로즈 역행렬** A<sup>+</sup>을 이용한다. 

      ![image-20220119114552134](https://user-images.githubusercontent.com/70505378/150055211-3e39b636-a6f3-4af6-ab75-ff475b0b37a2.png)

      ![image-20220119114535231](https://user-images.githubusercontent.com/70505378/150055210-7aa5dd1b-d297-4648-8564-466deb0d19c1.png)

    ```python
    X = np.array([[1,-2,3],
                  [7,5,0],
                  [-2,-1,2]])
    
    # 역행렬
    print(np.linalg.inv(X))
    print(X @ np.linalg.inv(X))
    # 유사 역행렬
    print(np.linalg.pinv(X))
    print(X @ np.linalg.pinv(X))
    '''
    [[ 0.21276596  0.0212766  -0.31914894]
     [-0.29787234  0.17021277  0.44680851]
     [ 0.06382979  0.10638298  0.40425532]]
    [[ 1.00000000e+00 -1.38777878e-17  0.00000000e+00]
     [-2.22044605e-16  1.00000000e+00 -5.55111512e-17]
     [-2.77555756e-17  0.00000000e+00  1.00000000e+00]]
    [[ 0.21276596  0.0212766  -0.31914894]
     [-0.29787234  0.17021277  0.44680851]
     [ 0.06382979  0.10638298  0.40425532]]
    [[ 1.00000000e+00 -2.08166817e-16  5.55111512e-16]
     [-1.66533454e-16  1.00000000e+00  3.33066907e-16]
     [ 1.66533454e-16  8.32667268e-17  1.00000000e+00]]
    '''
    ```

**응용**

* 연립 방정식 풀기

  * `np.linalg.pinv`를 이용하면 연립방정식의 해를 구할 수 있다. 

  ![image-20220119115006710](https://user-images.githubusercontent.com/70505378/150055217-5a13cccc-ce7e-49b1-9d4b-d69e60d4f494.png)

  * `np.linalg.pinv`를 이용하면 데이터를 선형 모델로 해석하는 **선형 회귀식**을 찾을 수 있다. 

  ![image-20220119115141877](https://user-images.githubusercontent.com/70505378/150055221-bff6b7cf-f2bb-4b15-8d61-6c43c88dc21c.png)

  ```python
  # Scikit Learn을 활용한 회귀분석
  from sklearn.linear_model import LinearRegression
  model = LinearRegression()
  model.fit(X, y)
  y_test = model.predict(x_test)
  
  # Moore-Penrose 역행렬
  X_ = np.array([np.append(x,[1]) for x in X])
  beta = np.linalg.pinv(X_) @ y
  t_test = np.append(x,[1]) @ beta
  ```

  

<br>
