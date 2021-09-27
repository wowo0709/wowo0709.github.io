---
layout: single
title: "[Machine Learning] Linear Classification 2 (Practice)"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: ['Classification']
---



# Linear Classification 2

## Example 1: Binary Classification

Iris dataset을 사용한 binary classification 실습

* Setosa VS non-Setosa

### Import dataset


```python
from sklearn.datasets import load_iris
iris = load_iris()
print(type(iris)) 
```

    <class 'sklearn.utils.Bunch'>



```python
print(iris.feature_names) # 4 features
print(iris.target_names)  # 3 labels
```

    ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
    ['setosa' 'versicolor' 'virginica']



```python
X, y = iris.data, iris.target
print(X.shape, type(X))
print(y.shape, type(y))
```

    (150, 4) <class 'numpy.ndarray'>
    (150,) <class 'numpy.ndarray'>

<br>

### Make to binary classification dataset


```python
X2 = X[:, :2]   # first two features
```


```python
%matplotlib inline
import matplotlib.pyplot as plt

markers = ['o', '+', '^']
for i in range(3):
    xs = X2[:, 0][y == i]
    ys = X2[:, 1][y == i]
    plt.scatter(xs, ys, marker=markers[i])
plt.legend(iris.target_names)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
```




    Text(0, 0.5, 'Sepal width')




![output_8_1](https://user-images.githubusercontent.com/70505378/134864974-68e38547-f9b5-4fb7-9b83-adc714bcbacf.png)
    


위의 label 분포를 보면 setosa와 versicolor&virginica 로 크게 이진 분류를 할 수 있다. 

따라서 label의 virginica를 모두 versicolor로 바꿔서 이진 분류 데이터로 바꾼다. 


```python
y2 = y.copy()      # y의 복사본을 만든다
y2[(y2==2)] = 1    # y중에 2의 값을 모두 1로 바꾼다 -> 이진 분류
y2

markers = ['o', '+', '^']
for i in range(3):
    xs = X2[:, 0][y2 == i]
    ys = X2[:, 1][y2 == i]
    plt.scatter(xs, ys, marker=markers[i])
binary_names = ['setosa', 'non-setosa']
plt.legend(binary_names)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
```




    Text(0, 0.5, 'Sepal width')




![output_10_1](https://user-images.githubusercontent.com/70505378/134864980-4ebd610d-5a3f-4bf6-a3cf-e858d2c21f5b.png)
    

<br>

### Split train/test dataset


```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=17)
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```




    ((105, 2), (45, 2), (105,), (45,))




```python
markers = ['o', '+', '^']
for i in range(3):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, marker=markers[i])
binary_names = ['setosa', 'non-setosa']
plt.legend(binary_names)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")
```




    Text(0, 0.5, 'Sepal width')




![output_13_1](https://user-images.githubusercontent.com/70505378/134864981-130538e8-8909-4ace-91f1-f7586ee372f6.png)
    

<br>

### Select Model and Train


```python
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(max_iter=2000, random_state=42)
clf.fit(X_train, y_train)
```




    SGDClassifier(max_iter=2000, random_state=42)




```python
clf.coef_, clf.intercept_
```




    (array([[  98.14963797, -139.44757308]]), array([-90.4188475]))




```python
a = clf.coef_[0,0]
b = clf.coef_[0,1]
c = clf.intercept_
print(a, b, c)
```

    98.1496379726464 -139.44757307589103 [-90.4188475]

<br>

### Inference and Plotting


```python
import numpy as np
%matplotlib inline
import matplotlib.pyplot as plt

markers = ['o', '+', '^']
for i in range(2):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, marker=markers[i])

binary_names = ['setosa', 'non-setosa']
plt.legend(binary_names)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

XX = np.linspace(4, 8, 40)
# 결정 경계선
plt.plot(XX, (-a/b)*XX - c/b, "k-", linewidth=2)
```




    [<matplotlib.lines.Line2D at 0x17c7dd13c70>]




![output_19_1](https://user-images.githubusercontent.com/70505378/134864985-fc996d0f-30bd-4ed1-a42c-e73166db69b4.png)
    



```python
print(clf.predict([[4.5, 3.5]]))  # 0
```

    [0]



```python
print(clf.score(X2, y2))
```

    0.9933333333333333



```python
print(clf.score(X_test, y_test))
```

    1.0

<br>

#### Use KFold and cross_val_score


```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import SGDClassifier

cv = KFold(n_splits=5, shuffle=True)

'''
X2.shape[0]
y

for train_index, test_index in cv.split(X2):
   print("TRAIN:", train_index, "TEST:", test_index)
   X_train, X_test = X2[train_index], X2[test_index]
   y_train, y_test = y2[train_index], y2[test_index]
'''

# 위의 과정을 한 번에 해 주는 함수가 있음.
score = cross_val_score(SGDClassifier(), X2, y2, cv=cv)
print(score, score.mean())
```

    [0.93333333 1.         1.         0.96666667 0.8       ] 0.9400000000000001

<br>

<br>

## Example 2: More difficult classification

Still binary classification. 

* Virginica VS non-Viginica

### Make dataset


```python
iris = load_iris()
X, y = iris.data, iris.target
X3 = X[:,[0,2]] # feature selection
y3 = y.copy()
y3[y3 == 1] = 0 # 0, 1 -> 0
y3[y3 == 2] = 1 # 2 -> 1
X_train, X_test, y_train, y_test = train_test_split(X3, y3, test_size=0.3, random_state=17)

markers = ['o', '+', '^']
for i in range(3):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, marker=markers[i])
binary_names = ['non-virginica', 'virginica']
plt.legend(binary_names)
plt.xlabel("Sepal length")
plt.ylabel("Petal length")
```




    Text(0, 0.5, 'Petal length')




![output_26_1](https://user-images.githubusercontent.com/70505378/134864988-1225233d-4125-4206-a1b9-6eef7fd2b7f0.png)
    

<br>

### Model selection and training

첫번째 모델은 alpha 값을 작게 주어 에러를 줄이는 데 더 집중하도록 한다. 


```python
clf1 = SGDClassifier(penalty='l2', alpha=0.0001, random_state=42)
clf1.fit(X_train, y_train)
clf1.score(X_test, y_test)
```




    0.9111111111111111

<br>

두번째 모델은 alpha 값을 크게 주어 가중치를 줄이는 데 더 집중하도록 한다. 


```python
clf2 = SGDClassifier(penalty='l2', alpha=1, random_state=42)
clf2.fit(X_train, y_train)
clf2.score(X_test, y_test)
```




    0.6888888888888889

너무 강하게 가중치를 규제하면 성능이 떨어진다. 

<br>

### Plotting


```python
x0, x1 = np.meshgrid(
        np.linspace(3, 9, 200).reshape(-1, 1),
        np.linspace(1, 8, 200).reshape(-1, 1),
    )
X_new = np.c_[x0.ravel(), x1.ravel()]

y_predict1 = clf1.predict(X_new)
y_predict2 = clf2.predict(X_new)

zz1 = y_predict1.reshape(x0.shape)
zz2 = y_predict2.reshape(x0.shape)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1)

# clf1 -> train
plt.plot(X_train[y_train==1, 0], X_train[y_train==1, 1], "rx", label="virginica")
plt.plot(X_train[y_train==0, 0], X_train[y_train==0, 1], "yo", label="non-virginica")
plt.contourf(x0, x1, zz1)
plt.xlabel("setal length")
plt.ylabel("petal length")
plt.legend(loc="lower right")
# clf1 -> test
plt.subplot(2,2,2)
plt.plot(X_test[y_test==1, 0], X_test[y_test==1, 1], "rx", label="virginica")
plt.plot(X_test[y_test==0, 0], X_test[y_test==0, 1], "yo", label="non-virginica")
plt.contourf(x0, x1, zz1)
plt.xlabel("setal length")
plt.ylabel("petal length")
plt.legend(loc="lower right")
# clf2 -> train
plt.subplot(2,2,3)
plt.plot(X_train[y_train==1, 0], X_train[y_train==1, 1], "rx", label="virginica")
plt.plot(X_train[y_train==0, 0], X_train[y_train==0, 1], "yo", label="non-virginica")
plt.contourf(x0, x1, zz2)
plt.xlabel("setal length")
plt.ylabel("petal length")
plt.legend(loc="lower right")
# clf2 -> test
plt.subplot(2,2,4)
plt.plot(X_test[y_test==1, 0], X_test[y_test==1, 1], "rx", label="virginica")
plt.plot(X_test[y_test==0, 0], X_test[y_test==0, 1], "yo", label="non-virginica")
plt.contourf(x0, x1, zz2)
plt.xlabel("setal length")
plt.ylabel("petal length")
plt.legend(loc="lower right")
plt.show()
```


![output_33_0](https://user-images.githubusercontent.com/70505378/134864989-7fa9bbfa-e482-4bc8-be7a-a7c15fc5dffb.png)
    

<br>

```python
clf1.coef_, clf2.coef_
```


    (array([[-151.63660655,  225.78490314]]), array([[-0.07619048,  0.42329004]]))

