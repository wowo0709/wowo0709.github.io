---
layout: single
title: "[Machine Learning] SVM"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: []
---



<br>

## Linear Classification

### Binary classification


```python
import numpy as np
from sklearn.datasets import load_iris
iris = load_iris()
```


```python
X, y = iris.data, iris.target
X2 = X[:, :2]
y2 = y.copy()              # y의 복사본을 만든다
y2[(y2==2)] = 1            # y중에 2의 값을 모두 1로 바꾼다
y2
```


    array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
           1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

<br>


```python
from sklearn.model_selection import train_test_split
np.random.seed(13)
X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3)
```


```python
X_train.shape, X_test.shape
```


    ((105, 2), (45, 2))

<br>


```python
import matplotlib.pyplot as plt
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




![output_6_1](https://user-images.githubusercontent.com/70505378/136890971-e2cbedc7-3e41-4755-9e0c-285570f0b2f9.png)
    

<br>

```python
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier(max_iter=1000)
clf.fit(X_train, y_train)
```


    SGDClassifier()


```python
a = clf.coef_[0,0]
b = clf.coef_[0,1]
c = clf.intercept_
```


```python
clf.score(X_train, y_train), clf.score(X_test, y_test)
```


    (0.9809523809523809, 0.9777777777777777)

<br>

### Multi-class: use all 3 classes

하나의 클래스와 나머지 클래스를 구분하는 선을 예측


```python
# use all classes
np.random.seed(17)
X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3)

X_train.shape, y_train.shape

markers = ['o', '+', '^']
for i in range(3):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i]
    plt.scatter(xs, ys, marker=markers[i])

plt.legend(iris.target_names)
plt.xlabel("Sepal length")
plt.ylabel("Sepal width")

clf = SGDClassifier(max_iter=1000)
clf.fit(X_train, y_train)
print("Coefficients: ", clf.coef_, clf.intercept_)
print("multi-class score: ", clf.score(X_test, y_test))

for i in range(3):
    a = clf.coef_[i,0]
    b = clf.coef_[i,1]
    c = clf.intercept_[i]
    xx = np.linspace(4,9,100)
    yy = -a/b * xx - c/b
    plt.plot(xx, yy, c='r')
```

    Coefficients:  [[-114.57251644  165.80131533]
     [   2.3557126   -69.49352179]
     [  66.21199204  -76.72634271]] [  78.79909348  104.38733447 -178.46683519]
    multi-class score:  0.5777777777777777



![output_11_1](https://user-images.githubusercontent.com/70505378/136890974-ee26ed8c-b289-4c1a-b7bb-120d6deba1f3.png)

<br>

**test set에 대해 경계선을 그려보자.**


```python
# contour
h = .02  # step size in the mesh
x_min, x_max = X2[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X2[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])  # column 으로 붙이기
print(Z.shape)
Z = Z.reshape(xx.shape)
print(Z.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired) 
# plt.contourf(xx, yy, Z)      # Z: height values over which the contour is drawn
# plt.axis('tight')
#-----------------------

markers = ['o', '+', '^']
colors = "rbg"

for i in range(3):
    xs = X_test[:, 0][y_test == i]
    ys = X_test[:, 1][y_test == i]
    plt.scatter(xs, ys, marker=markers[i], c=colors[i])
   
for i in range(3):
    a = clf.coef_[i,0]
    b = clf.coef_[i,1]
    c = clf.intercept_[i]
    xx = np.linspace(4,8,100)
    yy = -a/b * xx - c/b
    plt.plot(xx, yy, c='k')

plt.plot()
```

    (61600,)
    (220, 280)




![output_13_2](https://user-images.githubusercontent.com/70505378/136890975-b625036c-0dd4-44e4-8472-360eb1297ed0.png)
    

<br>

```python
from sklearn.metrics import confusion_matrix
y_pred = clf.predict(X_test)
confusion_matrix(y_test, y_pred)  # one vs. rest
```


    array([[12,  0,  0],
           [ 1,  0, 18],
           [ 0,  0, 14]], dtype=int64)

<br>

- **one-vs-all (one-vs-rest)**

![image-20211012131523357](https://user-images.githubusercontent.com/70505378/136891372-95ab559d-74c2-40d2-a37a-89eb6d295636.png)

<br>

<br>

## Linear SVM Classifier
- `C` 가 증가하면 곡선이 디테일해지고 (margin이 hard해진다)
- `gamma` 가 증가하면 섬들이 많이 생긴다 (이웃의 수가 적어진다)

### SVM

![image-20211012140306119](https://user-images.githubusercontent.com/70505378/136895701-afa16d5e-a1b7-4131-b534-020c0ce6cfd3.png)

#### SVM Optimization

![image-20211012140425282](https://user-images.githubusercontent.com/70505378/136895703-761d4c6d-cac5-450e-b7fb-cebe611ec47d.png)



#### Loss Function

SVM uses **hinge loss.**

![image-20211012140447032](https://user-images.githubusercontent.com/70505378/136895704-1395cc44-cd0a-43b4-8e07-04eedb8522ed.png)

![image-20211012140605301](https://user-images.githubusercontent.com/70505378/138541000-3f681056-5297-497f-8c39-b66ef65772aa.png)

<br>

### Iris data


```python
X, y = iris.data, iris.target
X2 = X[:, :2]

X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3)
# X_train, X_test, y_train, y_test = train_test_split(X2, y, test_size=0.3)
```


```python
from sklearn.svm import SVC

lin_clf = SGDClassifier(max_iter=1000)
lin_clf.fit(X_train, y_train)

svm_clf = SVC(kernel="linear", C=10)
svm_clf.fit(X_train, y_train)
```


    SVC(C=10, kernel='linear')

<br>


```python
print(svm_clf.score(X_test, y_test), lin_clf.score(X_test, y_test))
y_pred = svm_clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
y_pred = lin_clf.predict(X_test)
print(confusion_matrix(y_test, y_pred))
```

    0.9777777777777777 0.9777777777777777
    [[11  1]
     [ 0 33]]
    [[11  1]
     [ 0 33]]

<br>

SVM과 선형 분류기의 차이는?


```python
svm_clf.coef_, svm_clf.intercept_, lin_clf.coef_, lin_clf.intercept_
```


    (array([[ 3.33199106, -4.07243689]]),
     array([-5.14593311]),
     array([[ 38.06228374, -62.28373702]]),
     array([-9.57793523]))

<br>


```python
plt.figure(figsize=(8,6)) 
plt.xlim(3.9,7.1) 
plt.ylim(1.9,4.5)
w = svm_clf.coef_[0]
v = svm_clf.intercept_[0]
XX = np.linspace(4, 8, 30)

decision_boundary = -w[0]/w[1] * XX - v/w[1]
margin = 1/(np.sqrt(w[0]**2 + w[1]**2))
gutter_up = decision_boundary + margin 
gutter_down = decision_boundary - margin
svs = svm_clf.support_vectors_
plt.scatter(svs[:, 0], svs[:, 1], s=180)  # support vectors
# plt.scatter(svs[:, 0], svs[:, 1], s=180, facecolors='#AAFFAA') 
# print(svs)

plt.plot(XX, decision_boundary, "k-")
plt.plot(XX, gutter_up, "k--")
plt.plot(XX, gutter_down, "k--")
markers = ['o', '+', '^'] 
for i in range(3):
    xs = X_train[:, 0][y_train == i]
    ys = X_train[:, 1][y_train == i] 
    plt.scatter(xs, ys, marker=markers[i])
binary_names = ['setosa', 'non-setosa'] 
plt.legend(binary_names)
plt.xlabel("Sepal length") 
plt.ylabel("Sepal width")

# 선형분류 결정 경계선
a = lin_clf.coef_[0,0]
b = lin_clf.coef_[0,1]
c = lin_clf.intercept_
plt.plot(XX, (-a/b * XX - c/b), "r-")
```




![output_13_2](https://user-images.githubusercontent.com/70505378/136890975-b625036c-0dd4-44e4-8472-360eb1297ed0.png)
    

<br>

```python
svs.shape   # support vectors
```


    (4, 2)

<br>

<br>

## Non-linear SVM: Kernel Trick (커널 기법)

![image-20211012140635999](https://user-images.githubusercontent.com/70505378/136895708-d580f0e5-0622-4d7e-be95-0522a100d50d.png)

![image-20211012140710207](https://user-images.githubusercontent.com/70505378/136895709-952efa5b-4c77-4100-b987-a652f95d692f.png)

<br>


```python
from sklearn.datasets import load_breast_cancer
cancer = load_breast_cancer() 
```


```python
X, y = cancer.data, cancer.target
```


```python
X.shape, y.shape
```


    ((569, 30), (569,))

<br>


```python
from sklearn.multiclass import OneVsRestClassifier

# SVC uses one-vs-one
classifier = OneVsRestClassifier(SVC(kernel='rbf', C=1000, gamma=0.1, probability=True))
                                    # enable prob estimates
classifier = classifier.fit(X_train, y_train)
classifier.score(X_train, y_train), classifier.score(X_test, y_test)
```


    (1.0, 0.9777777777777777)


```python
clf = SGDClassifier(max_iter=1000)
clf.fit(X_train, y_train)
clf.score(X_test, y_test)
```


    0.9777777777777777

<br>


```python
svm_clf = SVC(kernel="linear")
svm_clf.fit(X_train, y_train)
print("SVM score:", svm_clf.score(X_test, y_test))
```

    SVM score: 0.9777777777777777

<br>

<br>

## Nonlinear by Polynomial features

![image-20211012140809531](https://user-images.githubusercontent.com/70505378/136895710-1d8706bc-80c5-4149-acbb-890586b24b46.png)


```python
from sklearn.datasets import make_moons
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
```


```python
from sklearn.datasets import make_moons
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)

def plot_dataset(X, y, axes):
    plt.plot(X[:, 0][y==0], X[:, 1][y==0], "bs")
    plt.plot(X[:, 0][y==1], X[:, 1][y==1], "g^")
    plt.axis(axes)
    plt.grid(True, which='both')
    plt.xlabel(r"$x_1$", fontsize=20)
    plt.ylabel(r"$x_2$", fontsize=20, rotation=0)

plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()
```


![output_68_0](https://user-images.githubusercontent.com/70505378/136890994-2981344b-d77c-4bc6-b6f0-ff9c7cf2124c.png)
    

<br>

```python
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
```


```python
clf = Pipeline([
        ("poly_features", PolynomialFeatures(degree=3)),
        ("scaler", StandardScaler()),
        ("svm_clf", LinearSVC(C=10, loss="hinge", random_state=42))
    ])

clf.fit(X, y)
```

    Pipeline(steps=[('poly_features', PolynomialFeatures(degree=3)),
                    ('scaler', StandardScaler()),
                    ('svm_clf', LinearSVC(C=10, loss='hinge', random_state=42))])

<br>


```python
clf.steps
```


    [('poly_features', PolynomialFeatures(degree=3)),
     ('scaler', StandardScaler()),
     ('svm_clf', LinearSVC(C=10, loss='hinge', random_state=42))]

- Pipeline of transforms with a final estimator. 
  - Sequentially apply a list of transforms and a final estimator. 
  - Intermediate steps of the pipeline must be ‘transforms’, that is, they must implement fit and transform methods. 
  - The final estimator only needs to implement fit. 
  - The transformers in the pipeline can be cached using memory argument.
- The Pipeline is built using a list of (key, value) pairs, where the key is a string containing the name you want to give this step and value is an estimator object:


```python
def plot_predictions(clf, axes):
    x0s = np.linspace(axes[0], axes[1], 100)
    x1s = np.linspace(axes[2], axes[3], 100)
    x0, x1 = np.meshgrid(x0s, x1s)
    X = np.c_[x0.ravel(), x1.ravel()]
    y_pred = clf.predict(X).reshape(x0.shape)
    y_decision = clf.decision_function(X).reshape(x0.shape)
    plt.contourf(x0, x1, y_pred, cmap=plt.cm.brg, alpha=0.2)
    plt.contourf(x0, x1, y_decision, cmap=plt.cm.brg, alpha=0.1)

plot_predictions(clf, [-1.5, 2.5, -1, 1.5])
plot_dataset(X, y, [-1.5, 2.5, -1, 1.5])
plt.show()
```


![output_73_0](https://user-images.githubusercontent.com/70505378/136890995-48444261-de6c-471e-b72c-cbf4444ee917.png)
    

<br>

<br>

## SVM Classifier example

gamma 값과 C 값을 조정함에 따라 train과 test set에 대한 모델의 score를 보자. 

### Ex 1. Iris dataset


```python
# for train and test data
iris = load_iris()
X = iris.data[:, [0, 1]]
y = iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
```

<br>

#### For train set


```python
# Training classifiers
clf1 = SVC(gamma=.1, C=1, kernel='rbf', probability=True)
clf2 = SVC(gamma=.1, C=100, kernel='rbf', probability=True)
clf3 = SVC(gamma=100, C=1, kernel='rbf', probability=True)
clf4 = SVC(gamma=100, C=100, kernel='rbf', probability=True)
clf5 = SVC(gamma=1000, C=1000, kernel='rbf', probability=True)

clf1.fit(X_train, y_train)
clf2.fit(X_train, y_train)
clf3.fit(X_train, y_train)
clf4.fit(X_train, y_train)
clf5.fit(X_train, y_train)


# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1, 2], [0, 1]),
                        [clf1, clf2, clf3, clf4, clf5],
                        ['gamma=0.1, C=1', 'gamma=0.1, C=100',
                         'gamma=100, C=1', 'gamma=100, C=100',
                        'gamma=1000, C=1000']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X_train[:, 0], X_train[:, 1], c=y_train,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()
# 확인
print (clf1.score(X_train, y_train), 
       clf2.score(X_train, y_train),
       clf3.score(X_train, y_train),
       clf4.score(X_train, y_train),
       clf5.score(X_train, y_train))
```


![output_42_0](https://user-images.githubusercontent.com/70505378/136890978-60542185-e04c-4d2d-b9ce-188d66772fcb.png)
    


    0.7666666666666667 0.8 0.925 0.9333333333333333 0.9333333333333333

<br>

#### For test set


```python
# for test data
# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(3, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1, 2], [0, 1]),
                        [clf1, clf2, clf3, clf4, clf5],
                        ['gamma=0.1, C=1', 'gamma=0.1, C=100',
                         'gamma=100, C=1', 'gamma=100, C=100',
                        'gamma=1000, C=1000']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()
# 확인
print (clf1.score(X_test, y_test), 
       clf2.score(X_test, y_test),
       clf3.score(X_test, y_test),
       clf4.score(X_test, y_test),
       clf5.score(X_test, y_test))
```


![output_44_0](https://user-images.githubusercontent.com/70505378/136890981-9ef25665-ce02-4214-9a28-4d2263585d7c.png)
    


    0.8333333333333334 0.8666666666666667 0.7 0.7333333333333333 0.4

<span style="color:red">**위에서 보면 알 수 있듯이, gamma와 C 값을 너무 키우게 되면 overfitting이 발생한다.**</span>

<br>

### Ex 2. XOR problem

- binary classification
- target to predict is a XOR of the inputs
- illustrate decision function learned by SVC


```python
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn import svm

xx, yy = np.meshgrid(np.linspace(-3, 3, 500),
                     np.linspace(-3, 3, 500))
```


```python
np.random.seed(0)
X = np.random.randn(300, 2)
Y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
```


```python
X.shape, Y.shape
```


    ((300, 2), (300,))


```python
plt.scatter(X[:,0], X[:,1], c=Y)
```




![output_50_1](https://user-images.githubusercontent.com/70505378/136890982-7b01ebfb-33c7-4680-97c8-c67e6e36fc06.png)
    

<br>

```python
# fit the model
clf = svm.SVC(gamma='auto')   # gamma = 'auto': uses 1/n_features
clf.fit(X, Y)

# plot the decision function for each datapoint on the grid
# ravel(): Return a contiguous flattened array.
Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])  # evaluate the decision function
```


```python
Z.shape, xx.shape
```


    ((250000,), (500, 500))


```python
Z = Z.reshape(xx.shape)    # 500 x 500
```


```python
np.c_[xx.ravel(), yy.ravel()].shape
```


    (250000, 2)

<br>


```python
plt.imshow(Z, interpolation='nearest',
           extent=(xx.min(), xx.max(), yy.min(), yy.max()), aspect='auto',
           origin='lower', cmap=plt.cm.PuOr_r)
contours = plt.contour(xx, yy, Z, levels=[0], linewidths=2,
                       linestyles='dashed')
plt.scatter(X[:, 0], X[:, 1], s=30, c=Y, cmap=plt.cm.Paired,
            edgecolors='k')
plt.xticks(())
plt.yticks(())
plt.axis([-3, 3, -3, 3])
plt.show()
```


![output_55_0](https://user-images.githubusercontent.com/70505378/136890984-f24b22d2-c185-48b2-b79f-226fc3605cc6.png)
    

<br>

<br>

## SVM Regression

`epsilon` 값을 키운다는 것은 street의 너비를 키운다는 것이다. 따라서 오차에 관대해진다. 

### SVM Regression

![image-20211012141035504](https://user-images.githubusercontent.com/70505378/136895711-efa76f7d-f238-496e-bf22-61a6c36ca52e.png)

### Loss function

SVM Regressor uses **epsilon-insensitive loss.**

![image-20211012141049098](https://user-images.githubusercontent.com/70505378/136895713-a5b86405-6f85-48d9-8216-31f3db97ed0c.png)

![image-20211012141203277](https://user-images.githubusercontent.com/70505378/136895714-290db344-ba7d-4eec-9050-6db22f042936.png)

<br>


```python
import numpy as np
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

np.random.seed(21)

N = 1000    
def makeData(x):    
    r = [a/10 for a in x]
    y = np.sin(x) + np.random.normal(0, 0.2, len(x))
    return np.array(y + r)

x = [i/100 for i in range(N)]
y = makeData(x)
x = np.array(x).reshape(-1,1)

plt.scatter(x, y, s=5, color="blue")
plt.show()
```


![output_57_0](https://user-images.githubusercontent.com/70505378/136890987-a6741515-32dc-4260-a0c5-c785a6a1a386.png)
    

<br>

### Change kernel and epsilon

RBF kernel은 차원을 높여 학습을 하기 때문에 non-linear dataset에 대한 예측이 가능하다. 


```python
svr1 = SVR(kernel='linear', epsilon=0.1).fit(x, y)
svr2 = SVR(epsilon=0.01).fit(x, y)
svr3 = SVR(epsilon=1.).fit(x, y)

f, axarr = plt.subplots(1, 3, sharex='col', sharey='row', figsize=(14, 6))

for idx, svr_n, tt in zip(range(3),
                          [svr1, svr2, svr3],
                          ['linear, epsilon=0.1', 'rbf, epsilon=0.01','rbf, epsilon=1.0']):

    axarr[idx].scatter(x, y, s=5, color="blue", label="original")
    yfit = svr_n.predict(x)
    axarr[idx].plot(x, yfit, lw=2, color="red")
    axarr[idx].set_title(tt)

plt.show()

print (svr1.score(x, y), svr2.score(x, y), svr3.score(x, y))
```


![output_59_0](https://user-images.githubusercontent.com/70505378/136890991-ece5088e-c61a-4974-8bcc-6d43140591c6.png)
    


    0.08016199956060532 0.9272981404051117 0.5802043013218763

<br>

<br>

## Comparison of many classifiers

- Decision Tree
- Knn
- SVC
- VotingClassifier (soft voting): The idea behind the VotingClassifier is to combine conceptually different machine learning classifiers and use a majority vote or the average predicted probabilities (soft vote) to predict the class labels. Such a classifier can be useful for a set of equally well performing model in order to balance out their individual weaknesses.


```python
xx, yy = np.meshgrid(np.linspace(0,2,3), np.linspace(0,2,3))
print(xx, '\n', yy)
xx.shape
```

    [[0. 1. 2.]
     [0. 1. 2.]
     [0. 1. 2.]] 
     [[0. 0. 0.]
     [1. 1. 1.]
     [2. 2. 2.]]
    
    (3, 3)


```python
xx.ravel()
```


    array([0., 1., 2., 0., 1., 2., 0., 1., 2.])


```python
np.c_[xx.ravel(), yy.ravel()]
```


    array([[0., 0.],
           [1., 0.],
           [2., 0.],
           [0., 1.],
           [1., 1.],
           [2., 1.],
           [0., 2.],
           [1., 2.],
           [2., 2.]])

<br>


```python
from itertools import product

import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier

# Loading some example data
iris = datasets.load_iris()
X = iris.data[:, [0, 1]]
y = iris.target

# Training classifiers
clf1 = DecisionTreeClassifier(max_depth=6)
clf2 = KNeighborsClassifier(n_neighbors=7)
clf3 = SVC(gamma=1e2, C=100, kernel='rbf', probability=True)
eclf = VotingClassifier(estimators=[('dt', clf1), ('knn', clf2), ('svc', clf3)],
                        voting='soft', weights=[2, 1, 2])

clf1.fit(X, y)
clf2.fit(X, y)
clf3.fit(X, y)
eclf.fit(X, y)

# Plotting decision regions
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

f, axarr = plt.subplots(2, 2, sharex='col', sharey='row', figsize=(10, 8))

for idx, clf, tt in zip(product([0, 1], [0, 1]),
                        [clf1, clf2, clf3, eclf],
                        ['Decision Tree (depth=4)', 'KNN (k=7)',
                         'Kernel SVM', 'Soft Voting']):

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    axarr[idx[0], idx[1]].contourf(xx, yy, Z, alpha=0.4)
    axarr[idx[0], idx[1]].scatter(X[:, 0], X[:, 1], c=y,
                                  s=20, edgecolor='k')
    axarr[idx[0], idx[1]].set_title(tt)

plt.show()
print(clf1.score(X, y),clf2.score(X, y),clf3.score(X, y),eclf.score(X, y))
```


![output_65_0](https://user-images.githubusercontent.com/70505378/136890992-e9026d34-69e2-4007-9c01-8c3ae00dab22.png)
    


    0.8533333333333334 0.8266666666666667 0.9266666666666666 0.86

<br>

<br>
