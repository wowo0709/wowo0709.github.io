---
layout: single
title: "[Machine Learning] Linear Regression"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
---



## Example of one feature (x1)

### Make dataset

 ex1: linear regression with two variables (y = wx + b)


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

# 임시 데이터 생성
n = 100
x = np.random.randn(n)                # batch size
y = x*20 + 10                         # w=20, b=10
y = y + np.random.randn(n) * 10       # add noise

plt.scatter(x,y)
```


![output_2_1](https://user-images.githubusercontent.com/70505378/133963094-8f09217d-66d4-49e6-8a81-3597bf6d7d11.png)
    

<br>

### Train

```python
# --------
# Start with random parameter
w=np.random.randn()   
b=np.random.randn()

lr = 0.1          # learning rate
n_epoch = 200     # number of epoch
lossHistory = []  

for epoch in range(n_epoch):
    y_pred = w*x + b
    loss = ((y_pred - y)**2).mean()     # mean square error
    lossHistory.append(loss)
    
    # update parameters by differentiation of MSE
    w = w - lr* ((y_pred - y)*x).mean()
    b = b - lr* (y_pred - y).mean()
    if epoch %10 == 0:
        print('epoch=', epoch, 'loss=', loss, 'w=', w, 'b=', b)
        
print('---------------------------')
print('epoch=', epoch, 'loss=', loss, 'w=', w, 'b=', b)

```

    epoch= 0 loss= 608.135326701648 w= 0.689784764278889 b= 0.9972609240805469
    epoch= 10 loss= 207.53080423004252 w= 11.670716818468396 b= 7.227075715189502
    epoch= 20 loss= 152.28938253673073 w= 15.930927964152566 b= 9.20608673708662
    epoch= 30 loss= 144.4693105859433 w= 17.59153654618134 b= 9.821009291157926
    ...
    ---------------------------
    epoch= 199 loss= 143.13519198204307 w= 18.663985686793474 b= 10.075984418251208

<br>

### Plotting

```python
plt.figure(figsize=(4,4))
plt.scatter(x,y)

xx = np.linspace(-5,5,100) 
yy = w * xx + b
plt.plot(xx,yy,c='r') 
plt.show()

fig = plt.figure()
plt.plot(np.arange(0, n_epoch), lossHistory)
fig.suptitle("Training Loss")
plt.xlabel("Epoch #")
plt.ylabel("Loss")
plt.show()
```


![output_4_0](https://user-images.githubusercontent.com/70505378/133963183-8321d157-603c-4ea5-b7e8-e85174e775b9.png)
    



![output_4_1](https://user-images.githubusercontent.com/70505378/133963185-9df65612-28e6-4f30-bb96-3bd602c832bc.png)
    

<br>

<br>

## Training two features (x1, x2)

ex 2 : training two parameters w1, w2 and b (y = w1*x1 + w2*x2 + b)


```python
import numpy as np
import pandas as pd

n=100
x1 = np.random.randn(n)             # randn=normal distribution in (-1,1), rand=(0,1)
x2 = np.random.randn(n)

y = x1*30 + x2*40 + 50
y = y + np.random.randn(n)*20      # add noise

w1 = np.random.rand()               # initial guess
w2 = np.random.rand()
b = np.random.rand()

lr = 0.1                            # learning rate
n_epoch = 200                      # no of epoch
lossHistory = []

for epoch in range(n_epoch):
    y_pred = w1*x1 + w2*x2 + b
    error = ((y_pred - y)**2).mean()
    lossHistory.append(error)

    w1 = w1 - lr* ((y_pred - y)*x1).mean()
    w2 = w2 - lr* ((y_pred - y)*x2).mean()
    b = b - lr* (y_pred - y).mean()
    if epoch %10 == 0:
        print('epoch=', epoch, 'loss=', loss, 'w=', w, 'b=', b)
        
print('---------------------------')
print('epoch=', epoch, 'error=', error, 'w1=', w1.round(2), 'w2=', w2.round(2), 'b=', b.round(2))
```

    epoch= 0 loss= 143.13519198204307 w= 18.663985686793474 b= 5.527577031987033
    epoch= 10 loss= 143.13519198204307 w= 18.663985686793474 b= 34.161569950771764
    epoch= 20 loss= 143.13519198204307 w= 18.663985686793474 b= 42.63247728631368
    ...
    ---------------------------
    epoch= 199 error= 409.6591467434651 w1= 27.94 w2= 40.08 b= 46.28

<br>

```python
plt.figure(figsize = (8,4))
ax1 = plt.subplot(121, projection='3d')
ax1.scatter3D(x1, x2, y);

xx = np.linspace(-3,3,100) 
yy = np.linspace(-2,2,100)
zz = w1*x1 + w2*x2 + b
ax1.plot(xx, yy, w1*xx + w2*yy + b, c='r') 

ax2 = plt.subplot(122)
ax2.plot(np.arange(0, n_epoch), lossHistory)
ax2.set_title("Training Loss")
ax2.set_xlabel("Epoch #")
ax2.set_ylabel("Loss")

plt.subplots_adjust(wspace=0.5)
plt.show()
```


![output_7_0](https://user-images.githubusercontent.com/70505378/133963296-26c045f6-1f54-4894-9eaa-38f1a5e1a1bf.png)

<br>

<br>    

## Using regression model (LinearRegression)

### Using model

선형 회귀 모델인 `LinearRegression`을 사용합니다. 


```python
# ex3: using regression function (LinearRegression)

from sklearn.linear_model import LinearRegression

# Make it to matrix(two features)
X = np.concatenate([x1.reshape(n,1), x2.reshape(n,1)], axis=1)

model = LinearRegression()        # create model
model.fit(X,y)                    # train model
print("score: ",model.score(X,y))
print('w1=', model.coef_[0], 'w2=', model.coef_[1], 'b=', model.intercept_)

# prediction
new_X=[1,3]
print('Real Value: ', 1*30 + 3*40 + 50)        # y 
print('Predicted Value', *model.predict([new_X]))  # model predict(inference)
```

    score:  0.8465475643687691
    w1= 27.93626338823067 w2= 40.08110377245416 b= 46.27791539580053
    Real Value:  200
    Predicted Value 194.4574901013937

<br>

### Plotting


```python
w1, w2, b = model.coef_[0], model.coef_[1], model.intercept_

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(x1, x2, y);

xx = np.linspace(-3,3,100) 
yy = np.linspace(-2,2,100)
zz = w1*x1 + w2*x2 + b
ax.plot(xx, yy, w1*xx + w2*yy + b, c='r') 
```




![output_10_1](https://user-images.githubusercontent.com/70505378/133963406-160c8aec-6ff9-4f4d-8efa-b0cb287cf75e.png)

<br>

<br>    

## Use make_regression

`make_regression` 함수를 이용하면 회귀에 적합한 데이터셋을 생성할 수 있습니다. 


```python
from sklearn.datasets import make_regression     # 회귀에 적합한 데이터셋 생성
from sklearn.preprocessing import StandardScaler 
from sklearn.linear_model import LinearRegression
X, y = make_regression(n_samples=2000, n_features=2, noise=1.5, random_state=1)
X = StandardScaler().fit_transform(X)
print(X[:5], y[:5])
```

    [[ 0.33762316 -0.38981751]
     [-1.02672037  2.22938631]
     [ 0.09896413  0.63153974]
     [ 3.97755099 -1.64591196]
     [ 1.14153039 -0.70330793]] [-26.77111241 176.82634938  55.25266631 -79.36653137 -41.50945283]

<br>

```python
model = LinearRegression()        # create model
model.fit(X,y)                    # train model
model.score(X,y)                  # evaluate model
```


    0.9996931455705321





    


# 
