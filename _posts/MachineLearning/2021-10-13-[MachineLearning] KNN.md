---
layout: single
title: "[Machine Learning] KNN"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: []

---

<br>



## KNN Classifier

### Setup


```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
```

### Import dataset


```python
iris = load_iris()
X, y = iris.data, iris.target
```


```python
import matplotlib.pyplot as plt
plt.scatter(X[:,0], X[:,1], c=y)
```




![output_5_1](https://user-images.githubusercontent.com/70505378/137089437-e9d876a9-5a93-4d27-8db6-006cd03d9a11.png)
    

<br>

### Split train/test dataset


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

clf = SGDClassifier()
knn = KNeighborsClassifier(n_neighbors=3)

knn.fit(X_train, y_train)
print("kNN score: {:.2f}".format(knn.score(X_test, y_test)))
clf.fit(X_train, y_train)
print("Linear Reg score: {:.2f}".format(clf.score(X_test, y_test)))
```

    kNN score: 0.98
    Linear Reg score: 0.93

<br>

### Evaluation


```python
for i in range(1,30,3):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    print("K가", i, "일때 정확도: {:.2f}".format(knn.score(X_test, y_test)))
```

    K가 1 일때 정확도: 0.93
    K가 4 일때 정확도: 0.93
    K가 7 일때 정확도: 0.93
    K가 10 일때 정확도: 0.96
    K가 13 일때 정확도: 0.96
    K가 16 일때 정확도: 0.93
    K가 19 일때 정확도: 0.91
    K가 22 일때 정확도: 0.93
    K가 25 일때 정확도: 0.96
    K가 28 일때 정확도: 0.98

<br>

### Cross validation


```python
from sklearn.model_selection import cross_val_score, KFold
cross_val_score(knn, X, y, cv=5).mean().round(4)
```




    0.94




```python
cross_val_score(clf, X, y, cv=5).mean().round(4)
```




    0.8067

<br>

### Observation

- 선형 알고리즘이 성능이 더 좋지 않은 것으로 나타난다. (하지만 데이터 사이즈가 작아 불확실)
- scaling 한 후에 다시 한 번 확인해 보자 
- 선형 모델이나 SVM, 신경망에서는 반드시 scaling 을 해야 한다.


```python
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_sc = sc.fit_transform(X)
cross_val_score(clf, X_sc, y, cv=5).mean().round(4)
```




    0.9267




```python
cross_val_score(knn, X_sc, y, cv=5).mean().round(4)  # 별 차이가 없음
```




    0.9267
