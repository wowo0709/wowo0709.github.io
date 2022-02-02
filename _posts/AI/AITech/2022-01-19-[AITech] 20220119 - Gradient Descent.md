---
layout: single
title: "[AITech] 20220119 - Gradient Descent"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 강의 복습 내용

### 경사하강법 (순한맛)

**미분**

* **미분**은 **변수의 움직임에 따른 함수값의 변화를 측정하기 위한 도구**로 최적화에서 제일 많이 사용하는 기법이다. 

  ```python
  import sympy as sym
  from sympy.abc import x
  
  sym.diff(sym.poly(x**2 + 2*x + 3),x)
  # Poly(2𝑥+2,𝑥,𝑑𝑜𝑚𝑎𝑖𝑛=ℤ)
  ```

* 미분은 함수 f의 주어진 점 (x, f(x))에서의 **접선의 기울기**를 구한다. 

  * 한 점에서 접선의 기울기를 알면 어느 방향으로 점을 움직여야 함수값이 증가/감소하는지 알 수 있다. 
  * **미분값을 더하면 '경사상승법'**이라 하며, 함수의 **극댓값**의 위치를 구할 때 사용한다. (목적함수 최대화)

  ![image-20220119134819687](https://user-images.githubusercontent.com/70505378/150077661-2655dee2-debc-4cdc-bcff-dc71318ad39d.png)

  * **미분값을 빼면 '경사하강법'**이라 하며 함수의 **극솟값**의 위치를 구할 때 사용한다. (목적함수 최소화)

  ![image-20220119134943930](https://user-images.githubusercontent.com/70505378/150077664-9418387e-8c0c-4ee8-930a-b8373ca1198e.png)

  * **극값에서는 미분값이 0**이므로 더 이상 업데이트가 일어나지 않는다. 

* 경사하강법: 알고리즘

```python
# pseudo code
# Input: gradient, init, lr, eps, Output: var
var = init
grad = gradient(var)
while(bas(grad) > eps): # 종료 조건
    var = var - lr * grad # x 값 갱신
    grad = gradient(var)
```

```python
# python code
def func(val):
    fun = sym.poly(x**2 + 2*x + 3)
    return fun.subs(x, val), fun

def func_gradient(fun, val):
    _, function = fun(val)
    diff = sym.diff(function, x)
    return diff.subs(x, val), diff

def gradient_descent(fun, init_point, lr_rate = 1e-2, epsilon = 1e-5):
    cnt = 0
    val = init_point
    diff, _ = func_gradient(fun, init_point)
    while np.abs(diff) > epsilon:
        val = val - lr_rate*diff
        diff, _ = func_gradient(fun, val)
        cnt += 1
        
    print(f"함수: {fun(val)[1]}, 연산횟수: {cnt}, 최소점: ({val},{fun(val)[0]})")
    
    
gradient_descent(fun=func, init_point=np.random.uniform(-2,2))
# 함수: Poly(x**2 + 2*x + 3, x, domain='ZZ'), 연산횟수: 632, 최소점: (-0.999995083760464,2.00000000002417)
```

**변수가 벡터라면?**

* 벡터가 입력인 다변수 함수의 경우 **편미분**을 사용한다. 

  ```python
  import sympy as sym
  from sympy.abc import x, y
  
  sym.diff(sym.poly(x**2 + 2*x*y + 3) + sym.cos(x + 2*y), x)
  # 2𝑥+2𝑦−sin(𝑥+2𝑦)
  ```

* 각 변수별로 편미분을 계산한 **그레디언트 벡터**를 이용하여 경사하강/경사상승법에 사용할 수 있다. 

  ![image-20220119135959609](https://user-images.githubusercontent.com/70505378/150077665-bb495cc7-cd9a-4f9a-8ed5-fa356d8916c5.png)

* 경사하강법: 알고리즘

```python
# Pseudo code
# Input: gradient, init, lr, eps, Output: var

var = init
grad = gradient(var)
while(norm(grad) > eps): # 벡터의 경우 절댓값 대신 노름(norm)을 계산해서 종료조건 설정
    var = var - lr * grad
    grad = gradient(var)
```

```python
# python code
def eval_(fun, val):
    val_x, val_y = val
    fun_eval = fun.subs(y, val_y)
    return fun_eval

def func_multi(val):
    x_, y_ = val
    func = sym.poly(x**2 + 2*y**2)
    return eval_(func, [x_, y_]), func

def func_gradient(fun, val):
    x_, y_ = val
    _, function = fun(val)
    diff_x = sym.diff(function, x)
    diff_y = sym.diff(function, y)
    grad_vec = np.array([eval_(diff_x, [x_, y_]), eval_(diff_y, [x_, y_])], dtype=float)
    return grad_vec, [diff_x, diff_y]

def gradient_descent(fun, init_point, lr_rate=1e-2, epsilon=1e-5):
    cnt = 0
    val = init_point
    diff, _ = func_gradient(fun, val)
    while np.linalg.norm(diff) > epsilon:
        val = val - lr_rate*diff
        diff, _ = func_gradient(fun, val)
        cnt += 1
        
    print(f"함수: {fun(val)[1]}, 연산횟수: {cnt}, 최소점: ({val},{fun(val)[0]})")
    
    
pt = [np.random.uniform(-2,2), np.random.uniform(-2,2)]
gradient_descent(fun=func_multi, init_point=pt)
```



<br>

### 경사하강법 (매운맛)

**경사하강법을 이용한 선형 회귀**

앞서 **역행렬을 이용하면 선형회귀모델을 구할 수 있다**고 했다. 

**경사 하강법**을 이용하면 역행렬을 이용하지 않고 적절한 선형 모델을 찾을 수 있다. 

* 선형회귀의 목적식은 \|\|y − Xβ\|\|<sub>2</sub>이고 **이를 최소화하는 beta**를 찾아야 하므로 이에 대한 그레디언트 벡터를 구한다. 

  * 계산의 편의를 위해 목적식을 제곱한 항을 목적식으로 사용한다. 

  ![image-20220119141342983](https://user-images.githubusercontent.com/70505378/150077669-4d96609d-c9d5-49e7-b522-0e897bf3c9b5.png)

  * 이제 목적식을 최소화하는 beta를 구하는 경사하강법 알고리즘은 다음과 같다. 

  ![image-20220119141422702](https://user-images.githubusercontent.com/70505378/150077673-26e78fbc-0c86-4219-8216-586e450f312d.png)

* 경사하강법 기반 선형회귀 알고리즘

  * 종료 조건을 일정 횟수로 설정할 경우 **학습 횟수**와 **학습률**이 중요한 parameter가 된다. 두 값에 따라 적절한 모델을 찾을 수도, 찾지 못 할수도 있다. 

```python
# Pseudo code
# Input: X, y, lr, T, Output: beta

for t in range(T): # 종료조건을 일정 학습 횟수로 설정
    error = y - X * beta 
    grad = -transpose(X) @ error # 미분식
    beta = beta - lr * grad # beta 업데이트
```

```python
# Python code
X = np.array([[1,1], [1,2], [2,2], [2,3]])
y = np.dot(X, np.array([1,2])) + 3

beta_gd = [10.1, 15.1, -6.5] # [1,2,3]이 정답
X_ = np.array([np.append(x,[1]) for x in X]) # intercept 항 추가

for t in range(5000):
    error = y - X_ @ beta_gd
    # error = error / np.linalg.norm(error)
    grad = -np.transpose(X_) @ error
    beta_gd = beta_gd - 0.01*grad
    
print(beta_gd)
# [1.00000367 1.99999949 2.99999516]
```

**확률적 경사 하강법(Stochastic Gradient Descent, SGD)**

* 이론적으로 경사 하강법은 **미분가능하고 볼록한 함수**에 대해서는 **적절한 학습률과 학습 횟수 하에 수렴이 보장**된다. 

  * 특히 선형 회귀의 경우 목적식이 beta에 대해 볼록하기 때문에 수렴이 보장된다. 

* 하지만 **비선형회귀** 문제의 경우 목적식이 볼록하지 않을 수 있으므로 **수렴이 항상 보장되지는 않는다.**

  * 특히 딥러닝을 사용하는 경우 목적식은 대부분 볼록 함수가 아니다. 

* **확률적 경사 하강법**

  * 확률적 경사 하강법은 모든 데이터를 사용해서 업데이트 하는 대신, **데이터 한 개 또는 일부(미니 배치)를 활용하여 업데이트한다.**

  * SGD는 데이터의 일부를 가지고 파라미터를 업데이트 하기 때문에 연산자원을 좀 더 효율적으로 활용하는 데 도움이 된다. 

  * 미니 배치는 확률적으로 선택하므로 **목적식 모양이 바뀌게 된다.**

    * 따라서 최적화 과정에서 local minimum에 빠져서 grad = 0이 되더라도 탈출 할 수 있다. 

    ![image-20220119142539708](https://user-images.githubusercontent.com/70505378/150077675-d4f05fd0-8386-45ef-ba2a-0d1e2d6f8224.png)

  * SGD는 볼록이 아닌 목적식에서도 사용 가능하므로 경사하강법보다 **머신러닝 학습에 더 효율적**이다. 

  ![image-20220119142854277](https://user-images.githubusercontent.com/70505378/150077677-bb528f36-87ce-453d-8eb5-088bb450824b.png)







<br>
