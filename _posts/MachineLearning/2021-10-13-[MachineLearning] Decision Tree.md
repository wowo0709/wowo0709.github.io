---
layout: single
title: "[Machine Learning] Decision Tree"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: []

---





## Decision Tree Classifier

- Iris dataset


```python
import numpy as np
import matplotlib.pyplot as plt
```


```python
from sklearn.datasets import load_iris
iris = load_iris()

X = iris.data[:, :2] 
y = iris.target
```


```python
X.shape, y.shape
```




    ((150, 2), (150,))




```python
plt.scatter(X[:,0], X[:,1], c=y)
```




![output_4_1](https://user-images.githubusercontent.com/70505378/137089819-18f8e549-2f4d-419e-a333-8a9f43908528.png)
    

<br>

```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=2)
clf.fit(X, y)
clf.score(X, y)
```




    0.7733333333333333

<br>

<br>

## Plotting Decision Trees
- using tree.plot_tree
- using graphviz library


```python
from sklearn import tree
tree.plot_tree(clf, filled=True) # filled=True -> paint to indicate majority class
```




![output_7_1](https://user-images.githubusercontent.com/70505378/137089824-687afce7-b2a5-432e-a454-31cb3bcc6320.png)
    <br>


## Graphviz 설치

- graphviz.org 사이트에서 다운로드 후 설치
- 윈도우 버전:
- https://graphviz.gitlab.io/_pages/Download/Download_windows.html


```python
# !pip install graphviz   or
# conda install python-graphviz (in cmd 창)
# note that the above two commands install graphviz library in different places.
```

- export_graphviz(): Export a decision tree in DOT format. This function generates a GraphViz representation of the decision tree, which is then written into out_file
- dot file: DOT is a graph description language. DOT graphs are typically files with the filename extension gv or dot.


```python
# graphvis 실행시 path 문제가 있는 경우
# - 내 PC (오른쪽 마우스 클릭) -> 속성 -> 고급시스템 -> 환경변수 (시스템변수) -> path 에 graphviz path 추가
```


```python
# export_graphviz(): Export a decision tree in DOT format.
```


```python
from sklearn.tree import export_graphviz
import graphviz
export_graphviz(
    clf,
    out_file = "./iris.dot",
    feature_names = iris.feature_names[:2],
    class_names = iris.target_names,
    filled = True
    )
```


```python
with open("./iris.dot") as f:
    dot_graph = str(open("./iris.dot", "rb").read(), "utf8")
graphviz.Source(dot_graph)
```

![image-20211013165001045](C:\Users\wjsdu\AppData\Roaming\Typora\typora-user-images\image-20211013165001045.png)

<br>

<br>

## Plot borderline


```python
%matplotlib inline
import matplotlib.pyplot as plt
import numpy as np

plt.xlim(4, 8.5)
plt.ylim(1.5, 4.5)

markers = ['o', '+', '^']
for i in range(3):
    xs = X[:, 0][y == i]
    ys = X[:, 1][y == i]
    plt.scatter(xs, ys, marker=markers[i])

plt.legend(iris.target_names)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")


# 결정 트리 경계선: 실선은 루트 노드 점선은 자식 노드
xx = np.linspace(5.45, 5.45, 3)
yy = np.linspace(1.5, 4.5, 3)
plt.plot(xx, yy, '-k') # 검정색 실선

xx = np.linspace(4, 5.45, 3)
yy = np.linspace(2.8, 2.8, 3)
plt.plot(xx, yy, '--b') # 파란색 점선

xx = np.linspace(6.15, 6.15, 3)
yy = np.linspace(1.5, 4.5, 3)
plt.plot(xx, yy, '--r') # 붉은색 점선

```




    [<matplotlib.lines.Line2D at 0x1de1bc071c0>]




![output_16_1](https://user-images.githubusercontent.com/70505378/137090232-a4c9bf92-2a55-48e6-bfdc-dd613dfe4111.png)
    



```python
print(clf.predict([[5.5, 4]]))   # prediction
```

    [1]



```python
print(clf.predict_proba([[5.5, 4]]))   # prediction probability
```

    [[0.11627907 0.65116279 0.23255814]]

<br>

<br>

## Tree Hyper parameters

### Breast Cancer classification


```python
from sklearn.datasets import load_breast_cancer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split 
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```


```python
cancer = load_breast_cancer() 
```


```python
# dir(cancer)
cancer.feature_names
```


    array(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
           'mean smoothness', 'mean compactness', 'mean concavity',
           'mean concave points', 'mean symmetry', 'mean fractal dimension',
           'radius error', 'texture error', 'perimeter error', 'area error',
           'smoothness error', 'compactness error', 'concavity error',
           'concave points error', 'symmetry error',
           'fractal dimension error', 'worst radius', 'worst texture',
           'worst perimeter', 'worst area', 'worst smoothness',
           'worst compactness', 'worst concavity', 'worst concave points',
           'worst symmetry', 'worst fractal dimension'], dtype='<U23')

<br>


```python
np.random.seed(9)
# X_train, X_test, y_train, y_test = train_test_split(
#    cancer.data, cancer.target, stratify=cancer.target) 
# stratify: If not None, data is split in a stratified fashion, using this as the class labels.
```


```python
X_train, X_test, y_train, y_test = train_test_split(cancer.data, cancer.target) 
```


```python
X_train.shape, y_train.shape
```


    ((426, 30), (426,))

<br>


```python
clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)
print(clf.score(X_test, y_test))
```

    0.9370629370629371


- feature importance: The importance of a feature is computed as the (normalized) total reduction of the criterion brought by that feature. It is also known as the Gini importance.
- The higher the value the more important the feature bold text.
- 결정트리를 만드는데 기여한 정도


```python
list(zip(cancer.feature_names, clf.feature_importances_.round(4)))
```


    [('mean radius', 0.0074),
     ('mean texture', 0.043),
     ('mean perimeter', 0.0),
     ('mean area', 0.0),
     ('mean smoothness', 0.0),
     ('mean compactness', 0.0),
     ('mean concavity', 0.0),
     ('mean concave points', 0.0),
     ('mean symmetry', 0.0),
     ('mean fractal dimension', 0.0),
     ('radius error', 0.0),
     ('texture error', 0.0),
     ('perimeter error', 0.0),
     ('area error', 0.0033),
     ('smoothness error', 0.0),
     ('compactness error', 0.0188),
     ('concavity error', 0.0),
     ('concave points error', 0.0),
     ('symmetry error', 0.0093),
     ('fractal dimension error', 0.0),
     ('worst radius', 0.7116),
     ('worst texture', 0.0591),
     ('worst perimeter', 0.0),
     ('worst area', 0.0),
     ('worst smoothness', 0.0),
     ('worst compactness', 0.0211),
     ('worst concavity', 0.0106),
     ('worst concave points', 0.1157),
     ('worst symmetry', 0.0),
     ('worst fractal dimension', 0.0)]

<br>


```python
df = pd.DataFrame({'feature':cancer.feature_names,'importance':clf.feature_importances_ })
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>feature</th>
      <th>importance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mean radius</td>
      <td>0.007389</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mean texture</td>
      <td>0.043044</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mean perimeter</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>mean area</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>mean smoothness</td>
      <td>0.000000</td>
    </tr>
  </tbody>
</table>
</div>

<br>


```python
df = df.sort_values(by='importance', ascending=False) 
print(df.head(20))
```

                       feature  importance
    20            worst radius    0.711625
    27    worst concave points    0.115708
    21           worst texture    0.059071
    1             mean texture    0.043044
    25       worst compactness    0.021073
    15       compactness error    0.018815
    26         worst concavity    0.010635
    18          symmetry error    0.009318
    0              mean radius    0.007389
    13              area error    0.003323
    9   mean fractal dimension    0.000000
    6           mean concavity    0.000000
    28          worst symmetry    0.000000
    2           mean perimeter    0.000000
    3                mean area    0.000000
    4          mean smoothness    0.000000
    24        worst smoothness    0.000000
    23              worst area    0.000000
    22         worst perimeter    0.000000
    5         mean compactness    0.000000



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

![output_32_0](https://user-images.githubusercontent.com/70505378/137090235-d696679e-d479-4067-a5ab-445a8b01badf.png)
    <br>

<br>


## DecisionTree Regressor

```python
# exercise for tree regressor() 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
X = iris.data[:,:2]
y = iris.data[:,2]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tr_reg1 = DecisionTreeRegressor(max_depth=2)
tr_reg2 = DecisionTreeRegressor(max_depth=5)
tr_reg1.fit(X_train,y_train)
tr_reg2.fit(X_train,y_train)
tr_reg1.score(X_test, y_test), tr_reg2.score(X_test, y_test)
```




    (0.8273026760637123, 0.8239910886453354)




```python
tree.plot_tree(tr_reg1, filled=True)
```




![output_35_1](https://user-images.githubusercontent.com/70505378/137090237-5c972c5e-cb48-4824-8cf2-e6c4616fa73b.png)
    

<br>

```python
# predicting petal width (y) from petal length (x)
X = iris.data[:,2]
y = iris.data[:,3]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

tr_reg3 = DecisionTreeRegressor(max_depth=2)
tr_reg3.fit(X_train.reshape(-1,1),y_train)
tr_reg3.score(X_test.reshape(-1,1), y_test)

from sklearn.metrics import mean_squared_error, r2_score 
y_pred = tr_reg3.predict(X_train.reshape(-1,1)) 
plt.scatter(X_train, y_train, c='b', s = 5) 
plt.scatter(X_train, y_pred, c ='r', s = 3) 

mse = mean_squared_error(y_train, y_pred) 
rmse = np.sqrt(mean_squared_error(y_train, y_pred)) 
r2 = r2_score(y_train, y_pred)           # same as score(x,y)
print('MSE: ', mse, 'R2 score: ', r2)

```

    MSE:  0.042516993988801044 R2 score:  0.931000996993349




![output_36_1](https://user-images.githubusercontent.com/70505378/137090241-71aefbf7-9d8a-4712-89f6-ecf5fca550d8.png)
    

<br>

```python
tree.plot_tree(tr_reg3, filled=True)
```




![output_37_1](https://user-images.githubusercontent.com/70505378/137090242-8f04c1f0-af70-4e3c-802c-6368e1173e7a.png)
    
