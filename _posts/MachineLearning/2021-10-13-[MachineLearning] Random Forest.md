---
layout: single
title: "[Machine Learning] Random Forest"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: []
---

<br>



## RandomForest Classifier


```python
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn import model_selection, svm, metrics
from sklearn.model_selection import GridSearchCV

import numpy as np
import pandas as pd
```

<br>

### DecisionTree


```python
cancer = load_breast_cancer()
np.random.seed(9)
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, stratify=cancer.target)
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
```


    DecisionTreeClassifier()


```python
print(clf.score(X_test, y_test))
```

    0.916083916083916

```python
X_train.shape, X_test.shape, cancer.data.shape
```


    ((426, 30), (143, 30), (569, 30))

<br>

### feature_importances_:
- In trees, it tells how much each feature contributes to decreasing the 
  weighted impurity. 
- in Random Forest, it averages the decrease in impurity over trees.


```python
# 결정 트리를 사용한 경우의 중요 변수

list(zip(cancer.feature_names, clf.feature_importances_.round(4)))[:10]
```


    [('mean radius', 0.0),
     ('mean texture', 0.0417),
     ('mean perimeter', 0.0),
     ('mean area', 0.0),
     ('mean smoothness', 0.0),
     ('mean compactness', 0.0),
     ('mean concavity', 0.0),
     ('mean concave points', 0.0426),
     ('mean symmetry', 0.0114),
     ('mean fractal dimension', 0.0)]

<br>


```python
df = pd.DataFrame({'feature':cancer.feature_names,'importance':clf.feature_importances_ })
df=df.sort_values('importance', ascending=False)
print(df.head(10))
```

                     feature  importance
    22       worst perimeter    0.694689
    27  worst concave points    0.121068
    7    mean concave points    0.042647
    1           mean texture    0.041720
    21         worst texture    0.039639
    13            area error    0.017216
    20          worst radius    0.017188
    15     compactness error    0.012042
    8          mean symmetry    0.011405
    14      smoothness error    0.002385

<br>

```python
x = df.feature
y = df.importance
ypos = np.arange(len(x))

plt.figure(figsize=(10,7))
plt.barh(x, y)
plt.yticks(ypos, x)
plt.xlabel('Importance')
plt.ylabel('Variable')
plt.xlim(0, 1)
plt.ylim(-1, len(x))
plt.show()
```


![output_9_0](https://user-images.githubusercontent.com/70505378/137091543-9f7a6874-d5a5-429a-ae9f-93f1bbba626b.png)
    

<br>

### Random Forest (Bagging)

```python
# 랜덤 포레스트를 사용한 경우의 중요 특성

rfc = RandomForestClassifier(n_estimators=500)
rfc.fit(X_train, y_train)
print(rfc.score(X_test, y_test))

df = pd.DataFrame({'feature':cancer.feature_names,'importance':rfc.feature_importances_ })
df=df.sort_values('importance', ascending=False)
x = df.feature
y = df.importance
ypos = np.arange(len(x))

plt.figure(figsize=(10,7))
plt.barh(x, y)
plt.yticks(ypos, x)
plt.xlabel('Importance')
plt.ylabel('Variable')
plt.xlim(0, 1)
plt.ylim(-1, len(x))
plt.show()
```

    0.951048951048951




![output_10_1](https://user-images.githubusercontent.com/70505378/137091548-974e0922-4d39-483e-9661-e2923f01eeb8.png)
    


- 골고루 사용됨을 알 수 있다.

<br>

### Gradient Boosting (Boosting)

**Boosting 방법을 써보자**


```python
from sklearn.ensemble import GradientBoostingClassifier


gbc = GradientBoostingClassifier(n_estimators=500)
gbc.fit(X_train, y_train)
print(gbc.score(X_test, y_test))
```

    0.9790209790209791



```python
df = pd.DataFrame({'feature':cancer.feature_names,'importance':gbc.feature_importances_ })
df = df.sort_values('importance', ascending=False)
x = df.feature
y = df.importance
ypos = np.arange(len(x))

plt.figure(figsize=(10,7))
plt.barh(x, y)
plt.yticks(ypos, x)
plt.xlabel('Importance')
plt.ylabel('Variable')
plt.xlim(0, 1)
plt.ylim(-1, len(x))
plt.show()
```


![output_14_0](https://user-images.githubusercontent.com/70505378/137091551-c04e56f6-2534-4667-986a-81f2e8585ca7.png)
    

<br>

<br>

## Grid Search

**SVM과 비교**

### Grid Search:

- The grid search will try all combinations of parameter values and select the set of parameters which provides the most accurate model.
- 그리드 서치의 매개변수를 설정한다(C, gamma)


```python
params = [{"C": [1,10,100,1000], "kernel":["linear"]},
          {"C": [1,10,100,1000], "kernel":["rbf"], "gamma":[0.001, 0.0001]}
         ]

clf = GridSearchCV(svm.SVC(), params, n_jobs=-1 )
clf.fit(X_train, y_train)
print('최적값 :', clf.best_estimator_)
print('최적 score :', clf.best_score_)

#테스트 데이터로 최종 평가
score = clf.score(X_test, y_test)
print('최종 평가 =',score)
```

    최적값 : SVC(C=100, kernel='linear')
    최적 score : 0.9601094391244871
    최종 평가 = 0.965034965034965

<br>

<br>

## Tuning the hyper-parameters of an estimator

### A search consists of:

- an estimator (regressor or classifier such as sklearn.svm.SVC());
- a parameter space;
- a method for searching or sampling candidates;
- a cross-validation scheme; and
- a score function.


```python
# to find the names and current values for all parameters for a given 
# estimator,
clf.get_params()
```




    {'cv': None,
     'error_score': nan,
     'estimator__C': 1.0,
     'estimator__break_ties': False,
     'estimator__cache_size': 200,
     'estimator__class_weight': None,
     'estimator__coef0': 0.0,
     'estimator__decision_function_shape': 'ovr',
     'estimator__degree': 3,
     'estimator__gamma': 'scale',
     'estimator__kernel': 'rbf',
     'estimator__max_iter': -1,
     'estimator__probability': False,
     'estimator__random_state': None,
     'estimator__shrinking': True,
     'estimator__tol': 0.001,
     'estimator__verbose': False,
     'estimator': SVC(),
     'n_jobs': -1,
     'param_grid': [{'C': [1, 10, 100, 1000], 'kernel': ['linear']},
      {'C': [1, 10, 100, 1000], 'kernel': ['rbf'], 'gamma': [0.001, 0.0001]}],
     'pre_dispatch': '2*n_jobs',
     'refit': True,
     'return_train_score': False,
     'scoring': None,
     'verbose': 0}

<br>





