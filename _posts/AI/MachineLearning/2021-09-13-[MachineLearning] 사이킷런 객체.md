---
layout: single
title: "[Machine Learning] 사이킷런 객체"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
---

## 사이킷런 API

이번 포스팅에서는 사이킷런의 객체들을 어떻게 생성하고 다루는 지에 대해 간단히 살펴봅니다. 

사이킷런의 객체들은 다음 메서드를 갖습니다. 
- fit(): estimators
- transform(): transformers
- fit_transform(): fit() + transform()
- predict(): prediction

<br>

## 평균과 표준편차

```python
import numpy as np
```


```python
# 평균과 표준편차를 구하는 함수
def _mean_and_std(X, axis=0): # axis=0: 열 단위
   
    X = np.asarray(X)
    mean_ = X.mean(axis)
    std_ = X.std(axis)

    return mean_, std_
```


```python
data = np.array([[1,2,3],[4,5,6]])

data
```




    array([[1, 2, 3],
           [4, 5, 6]])




```python
print(_mean_and_std(data, 0)) # 열방향
print(_mean_and_std(data, 1)) # 행방향
```

    (array([2.5, 3.5, 4.5]), array([1.5, 1.5, 1.5]))
    (array([2., 5.]), array([0.81649658, 0.81649658]))

<br>

<br>

## fit과 transform

```python
# 스탠다드 스케일러 클래스에서 불필요한 부분들을 제외한 코드
class my_StandardScaler():

    def __init__(self):
        self.mean_, self.std_ = 0., 0.
    
    '''전달된 객체의 평균과 표준편차를 계산'''
    def fit(self, X, y=None):
        X = X.astype(np.float32)
        self.mean_, self.std_ = _mean_and_std(X, axis=0)
        return self
    '''self.mean_, self.std_를 이용하여 스케일링 수행'''
    def transform(self, X, y=None):
        X = X.astype(np.float32)
        X -= self.mean_
        X /= self.std_
        return X
    '''fit+transform'''
    def fit_transform(self, X, y=None):
        X = X.astype(np.float32)
        self.mean_, self.std_ = _mean_and_std(X, axis=0)
        X -= self.mean_
        X /= self.std_
        return X
    '''역변환'''
    def inverse_transform(self, X):
        X = X.astype(np.float32)
        X *= self.std_
        X += self.mean_
        return X
```


```python
X = np.array([1,2,3,4,5,6,7,8,9,10])
sc = my_StandardScaler()
```


```python
dir(sc)
```




    ['__class__',
     '__delattr__',
     '__dict__',
     '__dir__',
     '__doc__',
     '__eq__',
     '__format__',
     '__ge__',
     '__getattribute__',
     '__gt__',
     '__hash__',
     '__init__',
     '__init_subclass__',
     '__le__',
     '__lt__',
     '__module__',
     '__ne__',
     '__new__',
     '__reduce__',
     '__reduce_ex__',
     '__repr__',
     '__setattr__',
     '__sizeof__',
     '__str__',
     '__subclasshook__',
     '__weakref__',
     'fit',
     'fit_transform',
     'inverse_transform',
     'mean_',
     'std_',
     'transform']




```python
# 전달한 인스턴스의 평균과 표준편차 계산
sc.fit(X)
```




    <__main__.my_StandardScaler at 0x2b63775c160>




```python
# 계산된 평균과 표준편차
sc.mean_, sc.std_
```




    (5.5, 2.8722813)




```python
# 스케일링 수행
X_sc = sc.transform(X)

X_sc
```




    array([-1.5666989 , -1.2185436 , -0.87038827, -0.52223295, -0.17407766,
            0.17407766,  0.52223295,  0.87038827,  1.2185436 ,  1.5666989 ],
          dtype=float32)




```python
# 역변환 수행
sc.inverse_transform(X_sc)
```




    array([ 1.       ,  1.9999998,  3.       ,  4.       ,  5.       ,
            6.       ,  7.       ,  8.       ,  9.       , 10.       ],
          dtype=float32)




```python
sc.fit_transform(X) == X_sc
```




    array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            True])

