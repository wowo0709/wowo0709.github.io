---
layout: single
title: "[Machine Learning] Linear Regression 3 (Regularization)"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: ['Regression', 'Regularization']
---

<br>

이번 포스팅에서는 회귀 문제와 함께 가중치 규제인 `Ridge(L2)`와 `Lasso(L1)` 규제에 대해 알아보도록 하겠습니다. 

<br>

## Setup


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

from sklearn.linear_model import LinearRegression, Ridge, Lasso
```

<br>

<br>

## Linear Regression model (without Regularization)

* n_features - number of features to be considered
* noise - deviation from straight line
* n_samples - number of samples

<br>

### Make dataset


```python
from sklearn.datasets import make_regression
X,y = make_regression(n_features=1, noise=10, n_samples=1000, random_state=42)   
```


```python
X[:5], y[:5]
```


    (array([[-1.75873949],
            [ 1.03184454],
            [-0.48760622],
            [ 0.18645431],
            [ 0.72576662]]),
     array([-32.77038605,   3.50459106, -17.93030767,  -3.99020124,
             13.10526434]))


```python
plt.xlabel('Feature - X')
plt.ylabel('Target - y')
plt.scatter(X, y, s=5)
```




<img width="391" alt="output_7_1" src="https://user-images.githubusercontent.com/70505378/134345917-52aa9b97-6c7e-4bf9-bf9a-230f3df58c19.png">
    

<br>

### Regression


```python
lr = LinearRegression()
lr.fit(X, y)
lr.coef_, lr.intercept_
```




    (array([16.63354605]), 0.04526205905820929)




```python
# Predicting using trained model
y_pred = lr.predict(X)
```

<br>

### Plotting

* Blue dots represent maps to actual target data
* Orange dots represent predicted data 


```python
plt.scatter(X, y, s=5, label='training')
plt.scatter(X, y_pred, s=5, label='prediction')
plt.xlabel('Feature - X')
plt.ylabel('Target - y')
plt.legend()
plt.show()
```


<img width="391" alt="output_12_0" src="https://user-images.githubusercontent.com/70505378/134345978-ae5c9dd5-5989-4a80-821b-f4b1d4b1fce9.png">
    

<br>

<br>

## Regularized Regression Methods 

여기서 살펴볼 `Lasso(L1)`과 `Ridge(L2)` 규제는 **가중치 규제**에 해당합니다. 

가중치 규제가 필요한 이유는 무엇일까요? 

이를 이해하기 위해서는 먼저 `Overfitting(과대적합)`과 `Underfitting(과소적합)`에 대해 알아야 합니다.

<br>

### Overfitting/Underfitting

![image-20210922211723077](https://user-images.githubusercontent.com/70505378/134346029-2d881d58-0ec2-4c07-8305-bfbda74005d2.png)

 쉽게 말하면, 과대적합과 과소적합은 다음과 같습니다. 

* `과대적합`: 모델이 훈련 데이터에만 적응하여 높은 성능을 보이고, 테스트 데이터에 있어서는 낮은 성능을 보이는 경우 (<span style="color:red">분산이 크다 (high variance)</span>)
* `과소적합`: 모델이 훈련 데이터에 대한 적응도 부족하여 훈련 데이터와 테스트 데이터 모두에 있어서 낮은 성능을 보이는 경우 (<span style="color:red">편향이 크다 (high bias)</span>)

![image-20210922212335941](https://user-images.githubusercontent.com/70505378/134346037-598d0dd0-8cbe-4c27-ae79-5ffe69caa69c.png)

이 중 더 중요한 이슈는 무엇일까요? 바로 **과대 적합**입니다. 

<br>

과소 적합은 더 많은 데이터셋을 모으거나 모델의 복잡도를 높이는 등 명확한 해결책이 제시되어 있지만, 과대 적합은 그것을 해결하는 데 명확한 정답이 존재하지 않습니다. 

따라서 모델을 학습시키는 데 과대 적합 문제와 더 자주 마주하게 되고, 그것을 해결하기 위한 방법들도 많이 생겨나게 되었죠. 

* 더 다양한 데이터셋 수집 (More various data)
* 모델을 단순화 (Simplify the model)
* 일부 특성만을 이용 (Feature selection)
* <span style="color:blue">**데이터 보강 (Data augmentation)**</span>
* <span style="color:blue">**가중치 규제 (Regularization)**</span>
* <span style="color:blue">**조기 종료 (Early stopping)**</span>
* <span style="color:blue">**드롭아웃 (Dropouts)**</span>

이 중 여기서 살펴볼 방법은 `가중치 규제`입니다. 

<br>

### Regularization

![image-20210922212431808](https://user-images.githubusercontent.com/70505378/134346039-501aa532-1a6a-43cb-9d01-f96f6ec3d4eb.png)

첫번째 규제는 `Ridge(L2) 규제`입니다. 이는 가중치의 L2 term을 최소화하는 것으로, 영향이 너무 큰 가중치를 줄이는 역할을 합니다. 

두번째 규제는 `Lasso(L1) 규제`입니다. 이는 가중치의 L1 term을 최소화하는 것으로, 모든 가중치의 영향을 동등하게 줄입니다. 이 때 값이 작은 가중치들이 먼저 사라지기 때문에, 영향이 작은 가중치를 무시하는 역할을 합니다. 

세번째 규제는 `Elastic net 규제`로, L1과 L2 규제를 모두 사용하는 규제입니다.  

<br>

L1과 L2 규제의 경우 규제의 정도를 조절하기 위한 하이퍼파라미터 `alpha`가 존재하고, 엘라스틱 넷의 경우 `gamma`가 추가로 존재합니다. 

<br>

그럼 이제 이 가중치 규제를 사용하면 예측이 어떤 식으로 변화하는 지 살펴보겠습니다. 

<br>

### Make outliers


```python
outliers = y[950:] - 600; outliers
```




    array([-620.72518918, -607.24456936, -602.35967987, -589.77927836,
           -606.97474711, -602.5249083 , -617.53354476, -581.41160958,
           -568.58829982, -588.48103465, -576.37267804, -608.20115427,
           -627.62758019, -629.16862648, -600.74874687, -603.71586107,
           -597.22691815, -589.89284288, -587.08855694, -612.90456844,
           -607.72930237, -606.38449017, -592.78147515, -564.34789926,
           -579.47960861, -596.20989757, -608.437806  , -595.54249235,
           -605.14184967, -585.14253937, -602.46852941, -591.20272709,
           -576.61995697, -624.29969481, -633.11859313, -584.0344489 ,
           -580.06411958, -602.36414388, -600.03658325, -598.97777085,
           -600.7449772 , -588.33620239, -610.44463741, -620.9629963 ,
           -613.84011222, -622.5064205 , -586.40905438, -591.93411712,
           -562.64219745, -611.96087644])




```python
import numpy as np
y_out = np.append(y[:950], outliers)
```


```python
plt.scatter(X, y_out, s=5)
```




<img width="384" alt="output_17_1" src="https://user-images.githubusercontent.com/70505378/134346107-4fa8b007-1ff2-4a59-95fc-9cba47bc9cf8.png">
    

<br>

### Regression without Regularization


```python
lr = LinearRegression()
lr.fit(X, y_out)
y_out_pred = lr.predict(X)
```


```python
plt.scatter(X, y_out, s=5, label='actual')
plt.scatter(X, y_out_pred, s=5, label='prediction with outliers')
plt.scatter(X, y_pred,s=5, c='k', label='prediction without outlier')
plt.legend() 
plt.title('Linear Regression')
```




    Text(0.5, 1.0, 'Linear Regression')




<img width="384" alt="output_20_1" src="https://user-images.githubusercontent.com/70505378/134346142-3383b75e-e811-4d4c-a42b-08d63dce27b7.png">
    



```python
lr.coef_, lr.intercept_
```




    (array([14.75586098]), -29.918438428236247)

<br>

### With Ridge(L2) Regularization

* 영향이 너무 큰 가중치들을 줄인다. 


```python
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1000)
ridge.fit(X, y_out)
y_ridge_pred = ridge.predict(X)
```


```python
plt.scatter(X, y_out,s=5,label='actual')
plt.scatter(X, y_out_pred,s=5, c='r' , label='LinearRegression with outliers')
plt.scatter(X, y_ridge_pred,s=5,c='k', label='RidgeRegression with outliers')
plt.legend()
plt.title('Linear Regression')
```




    Text(0.5, 1.0, 'Linear Regression')




<img width="384" alt="output_24_1" src="https://user-images.githubusercontent.com/70505378/134346146-791192e9-a49a-4a27-8b2a-029b48a0add0.png">
    



```python
ridge.coef_, ridge.intercept_    # 기울기 coefficient(w) 가 값이 훨씬 작아짐.
```




    (array([7.21930478]), -29.77274130320161)

<br>

### With Lasso(L1) Regularization

* 영향이 적은 가중치들을 무시한다. 


```python
from sklearn.linear_model import Lasso

lasso = Lasso(alpha=1000)
lasso.fit(X, y_out)
y_lasso_pred = lasso.predict(X)
```


```python
plt.scatter(X, y_out,s=5,label='actual')
plt.scatter(X, y_out_pred,s=5, c='r' , label='LinearRegression with outliers')
plt.scatter(X, y_ridge_pred,s=5,c='k', label='RidgeRegression with outliers')
plt.scatter(X, y_lasso_pred,s=5,c='y', label='LassoRegression with outliers')
plt.legend()
plt.title('Linear Regression')
```




    Text(0.5, 1.0, 'Linear Regression')




<img width="384" alt="output_28_1" src="https://user-images.githubusercontent.com/70505378/134346148-57dcebb2-55f9-4af2-b777-392daf485248.png">
    



```python
lasso.coef_, lasso.intercept_
```




    (array([0.]), -29.63317730014139)

<br>

<br>

## Effects of alpha using Ridge on Coefficients 

여기서는 가중치 규제의 하이퍼파라미터 `alpha`의 변화에 따라 예측이 어떻게 달라지는 지 보겠습니다. 

### Make dataset


```python
X, y, w = make_regression(n_samples=1000, n_features=10, coef=True,
                          random_state=42, bias=3.5)
# w: The coefficient of the underlying linear model. It is returned only if coef is True.
```


```python
w
```


    array([32.12551734, 76.33080772, 33.6926875 ,  9.42759779,  5.16621758,
           58.28693612, 29.43481665,  7.18075454, 10.30191944, 75.31997019])

<br>

**0.001 ~ 10^5 사이의 200개 표본 alpha를 생성합니다.**


```python
alphas = np.logspace(-3, 5, 200)
alphas[:20], alphas[-20:]
```


    (array([0.001     , 0.00109699, 0.00120338, 0.00132009, 0.00144812,
            0.00158857, 0.00174263, 0.00191164, 0.00209705, 0.00230043,
            0.00252354, 0.00276829, 0.00303677, 0.00333129, 0.00365438,
            0.00400881, 0.0043976 , 0.00482411, 0.00529198, 0.00580523]),
     array([ 17225.85965399,  18896.52339691,  20729.21779595,  22739.65752358,
             24945.0813523 ,  27364.39997075,  30018.35813576,  32929.71255097,
             36123.42699709,  39626.88638701,  43470.13158125,  47686.11697714,
             52310.99308056,  57384.41648302,  62949.88990222,  69055.13520162,
             75752.50258772,  83099.41949353,  91158.88299751, 100000.        ]))

<br>

**릿지 규제를 적용합니다.**


```python
coefs = []
for a in alphas:
    ridge = Ridge(alpha=a) # 값이 큰 w 값을 작게!
    ridge.fit(X, y)
    coefs.append(ridge.coef_)
```


```python
w
```


    array([32.12551734, 76.33080772, 33.6926875 ,  9.42759779,  5.16621758,
           58.28693612, 29.43481665,  7.18075454, 10.30191944, 75.31997019])

* **alpha가 작을 때**


```python
coefs[:5]
```


    [array([32.12547819, 76.33071941, 33.69264671,  9.42759217,  5.16620902,
            58.28688033, 29.43478769,  7.18074889, 10.30191419, 75.31989693]),
     array([32.12547439, 76.33071085, 33.69264275,  9.42759163,  5.16620819,
            58.28687492, 29.43478488,  7.18074834, 10.30191368, 75.31988982]),
     array([32.12547022, 76.33070145, 33.69263841,  9.42759103,  5.16620728,
            58.28686898, 29.4347818 ,  7.18074774, 10.30191312, 75.31988203]),
     array([32.12546566, 76.33069115, 33.69263365,  9.42759038,  5.16620628,
            58.28686247, 29.43477841,  7.18074708, 10.30191251, 75.31987348]),
     array([32.12546064, 76.33067984, 33.69262843,  9.42758966,  5.16620518,
            58.28685533, 29.43477471,  7.18074636, 10.30191184, 75.3198641 ])]

* **alpha**가 클 때

`alpha` 값이 커지면 w 값이 굉장히 작아지는 것을 볼 수 있습니다. 


```python
coefs[-5:]
```


    [array([0.36905501, 0.94856311, 0.38307061, 0.18174143, 0.03080575,
            0.87029109, 0.43753906, 0.11359349, 0.22092661, 1.11708021]),
     array([0.3367566 , 0.86565699, 0.34953102, 0.1659322 , 0.02807343,
            0.79439856, 0.3993917 , 0.10368337, 0.2017385 , 1.01966879]),
     array([0.30725841, 0.78992058, 0.31890168, 0.15147756, 0.02558406,
            0.72504039, 0.36452758, 0.09462722, 0.18418953, 0.93064419]),
     array([0.28032211, 0.72074665, 0.29093448, 0.1382649 , 0.02331595,
            0.66166797, 0.33267124, 0.08635322, 0.16814427, 0.84930233]),
     array([0.2557289 , 0.65757725, 0.26540173, 0.12619038, 0.02124935,
            0.60377642, 0.30356917, 0.07879532, 0.15347768, 0.77499522])]

<br>

**alpha 값에 따른 가중치들의 값을 그래프로 그려보겠습니다.**


```python
ax = plt.gca()
# Get the current Axes instance on the current figure matching the given keyword 
# args, or create one.

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Ridge coefficients as a function of the regularization')
plt.show()
```


<img width="382" alt="output_42_0" src="https://user-images.githubusercontent.com/70505378/134346212-3ea93b56-0911-4efd-a552-6982bc3f6bc0.png">
    <br>


- Conclusion
    * 큰 가중치들을 작게 만드는 것을 볼 수 있다. 
    * As alpha tends toward zero the coefficients found by Ridge regression stabilize towards the randomly sampled vector w (similar to LinearRegression).
    * For big alpha (strong regularisation) the coefficients are smaller (eventually converging at 0) leading to a simpler and biased solution.

<br>

<br>

## Effects of alpha using Lasso on Coefficients 

라쏘 규제의 가중치 그래프도 그려보겠습니다. 


```python
# lasso 
X, y, w = make_regression(n_samples=1000, n_features=10, coef=True,
                          random_state=42, bias=3.5)

alphas = np.logspace(-3, 5, 200)
coefs = []
for a in alphas:
    lasso = Lasso(max_iter=10000, alpha=a) # 작은 값을 무시!
    lasso.fit(X, y)
    coefs.append(lasso.coef_)

ax = plt.gca()
# Get the current Axes instance on the current figure matching the given keyword 
# args, or create one.

ax.plot(alphas, coefs)
ax.set_xscale('log')
plt.xlabel('alpha')
plt.ylabel('weights')
plt.title('Lasso coefficients as a function of the regularization')
plt.show()

```


<img width="382" alt="output_45_0" src="https://user-images.githubusercontent.com/70505378/134346218-5e86b02d-4ee6-4aae-a471-f7dceab2d46e.png">
    <br>


* Conclusion
    * 작은 가중치들을 무시하는 것을 볼 수 있다. 

<br>

<br>

## Example

- https://www.analyticsvidhya.com/blog/2016/01/ridge-lasso-regression-python-complete-tutorial/
- RSS (residual sum of square): sum of square of error

<br>

### Setup


```python
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
%matplotlib inline
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 12, 10
```

<br>

### Make dataset


```python
#Define input array with angles from 60deg to 300deg converted to radians
x = np.array([i*np.pi/180 for i in range(60,300,4)])
np.random.seed(10)  #Setting seed for reproducibility
y = np.sin(x) + np.random.normal(0,0.15,len(x))
data = pd.DataFrame(np.column_stack([x,y]),columns=['x','y'])
plt.plot(data['x'],data['y'],'.')
```




<img width="715" alt="output_52_1" src="https://user-images.githubusercontent.com/70505378/134346313-db94bc7f-ba3d-48ac-a0b4-80feab22fced.png">
    

<br>

### Polynomial(Non-Linear) regression 

with powers of x from 1 to 15

가중치 규제의 영향을 보기 위해 비선형 회귀를 진행합니다. 


```python
for i in range(2,16):  # power of 1 is already there
    colname = 'x_%d'%i      # new var will be x_power
    data[colname] = data['x'] ** i
data.head()
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
      <th>x</th>
      <th>y</th>
      <th>x_2</th>
      <th>x_3</th>
      <th>x_4</th>
      <th>x_5</th>
      <th>x_6</th>
      <th>x_7</th>
      <th>x_8</th>
      <th>x_9</th>
      <th>x_10</th>
      <th>x_11</th>
      <th>x_12</th>
      <th>x_13</th>
      <th>x_14</th>
      <th>x_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>1.1</td>
      <td>1.1</td>
      <td>1.1</td>
      <td>1.2</td>
      <td>1.3</td>
      <td>1.3</td>
      <td>1.4</td>
      <td>1.4</td>
      <td>1.5</td>
      <td>1.6</td>
      <td>1.7</td>
      <td>1.7</td>
      <td>1.8</td>
      <td>1.9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.1</td>
      <td>1</td>
      <td>1.2</td>
      <td>1.4</td>
      <td>1.6</td>
      <td>1.7</td>
      <td>1.9</td>
      <td>2.2</td>
      <td>2.4</td>
      <td>2.7</td>
      <td>3</td>
      <td>3.4</td>
      <td>3.8</td>
      <td>4.2</td>
      <td>4.7</td>
      <td>5.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1.2</td>
      <td>0.7</td>
      <td>1.4</td>
      <td>1.7</td>
      <td>2</td>
      <td>2.4</td>
      <td>2.8</td>
      <td>3.3</td>
      <td>3.9</td>
      <td>4.7</td>
      <td>5.5</td>
      <td>6.6</td>
      <td>7.8</td>
      <td>9.3</td>
      <td>11</td>
      <td>13</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1.3</td>
      <td>0.95</td>
      <td>1.6</td>
      <td>2</td>
      <td>2.5</td>
      <td>3.1</td>
      <td>3.9</td>
      <td>4.9</td>
      <td>6.2</td>
      <td>7.8</td>
      <td>9.8</td>
      <td>12</td>
      <td>16</td>
      <td>19</td>
      <td>24</td>
      <td>31</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.3</td>
      <td>1.1</td>
      <td>1.8</td>
      <td>2.3</td>
      <td>3.1</td>
      <td>4.1</td>
      <td>5.4</td>
      <td>7.2</td>
      <td>9.6</td>
      <td>13</td>
      <td>17</td>
      <td>22</td>
      <td>30</td>
      <td>39</td>
      <td>52</td>
      <td>69</td>
    </tr>
  </tbody>
</table>

</div>

<br>

```python
from sklearn.linear_model import LinearRegression

def linear_regression(data, power, models_to_plot):
    # initialize predictors:
    predictors=['x']
    if power>=2:
        predictors.extend(['x_%d'%i for i in range(2,power+1)])
    
    # Fit the model
    linreg = LinearRegression(normalize=True)
    linreg.fit(data[predictors],data['y'])
    y_pred = linreg.predict(data[predictors])
    # Check if a plot is to be made for the entered power
    if power in models_to_plot:
        plt.subplot(models_to_plot[power])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for power: %d'%power)
    
    # Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([linreg.intercept_])
    ret.extend(linreg.coef_)
    return ret
```


```python
#Initialize a dataframe to store the results:
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['model_pow_%d'%i for i in range(1,16)]
coef_matrix_simple = pd.DataFrame(index=ind, columns=col)

```


```python
# Define the powers to plot
models_to_plot = {1:231,3:232,6:233,9:234,12:235,15:236}

#Iterate through all powers and assimilate results
for i in range(1,16):
    coef_matrix_simple.iloc[i-1,0:i+2] = linear_regression(data, power=i, models_to_plot=models_to_plot)
```


<img width="849" alt="output_57_0" src="https://user-images.githubusercontent.com/70505378/134346315-f5c7e10e-1de9-4694-89b2-e745fd010e5a.png">
    


- As the **model complexity increases**, the models tends to fit even smaller deviations in the training data set, possibly leading to **overfitting**.
- **The size of coefficients** increase exponentially with increase in **model complexity**.
- What does a large coefficient signify? It means that we’re putting a lot of emphasis on that feature, i.e. the particular feature is a good predictor for the outcome. When it becomes too large, the algorithm starts modelling intricate relations to estimate the output and ends up overfitting to the particular training data.

<br>

### Ridge


```python
from sklearn.linear_model import Ridge
def ridge_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha

    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([ridgereg.intercept_])
    ret.extend(ridgereg.coef_)
    return ret
```


```python
#Initialize predictors to be set of 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Set the different values of alpha to be tested
alpha_ridge = [1e-15, 1e-10, 1e-8, 1e-4, 1e-3,1e-2, 1, 5, 10, 20]

#Initialize the dataframe for storing coefficients.
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_ridge[i] for i in range(0,10)]
coef_matrix_ridge = pd.DataFrame(index=ind, columns=col)

models_to_plot = {1e-15:231, 1e-10:232, 1e-4:233, 1e-3:234, 1e-2:235, 1:236}
for i in range(10):
    coef_matrix_ridge.iloc[i,] = ridge_regression(data, predictors, alpha_ridge[i], models_to_plot)
```



<img width="849" alt="output_61_1" src="https://user-images.githubusercontent.com/70505378/134346321-f012055c-f55c-4d60-b7db-0e10e588a7b9.png">
    



```python
# Set the display format to be scientific for ease of analysis
pd.options.display.float_format = '{:,.2g}'.format
coef_matrix_ridge
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
      <th>rss</th>
      <th>intercept</th>
      <th>coef_x_1</th>
      <th>coef_x_2</th>
      <th>coef_x_3</th>
      <th>coef_x_4</th>
      <th>coef_x_5</th>
      <th>coef_x_6</th>
      <th>coef_x_7</th>
      <th>coef_x_8</th>
      <th>coef_x_9</th>
      <th>coef_x_10</th>
      <th>coef_x_11</th>
      <th>coef_x_12</th>
      <th>coef_x_13</th>
      <th>coef_x_14</th>
      <th>coef_x_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alpha_1e-15</th>
      <td>0.87</td>
      <td>94</td>
      <td>-3e+02</td>
      <td>3.8e+02</td>
      <td>-2.3e+02</td>
      <td>65</td>
      <td>0.56</td>
      <td>-4.3</td>
      <td>0.39</td>
      <td>0.2</td>
      <td>-0.028</td>
      <td>-0.0069</td>
      <td>0.0012</td>
      <td>0.00019</td>
      <td>-5.6e-05</td>
      <td>4.1e-06</td>
      <td>-7.8e-08</td>
    </tr>
    <tr>
      <th>alpha_1e-10</th>
      <td>0.92</td>
      <td>11</td>
      <td>-29</td>
      <td>31</td>
      <td>-15</td>
      <td>2.9</td>
      <td>0.17</td>
      <td>-0.091</td>
      <td>-0.011</td>
      <td>0.002</td>
      <td>0.00064</td>
      <td>2.4e-05</td>
      <td>-2e-05</td>
      <td>-4.2e-06</td>
      <td>2.2e-07</td>
      <td>2.3e-07</td>
      <td>-2.3e-08</td>
    </tr>
    <tr>
      <th>alpha_1e-08</th>
      <td>0.95</td>
      <td>1.3</td>
      <td>-1.5</td>
      <td>1.7</td>
      <td>-0.68</td>
      <td>0.039</td>
      <td>0.016</td>
      <td>0.00016</td>
      <td>-0.00036</td>
      <td>-5.4e-05</td>
      <td>-2.9e-07</td>
      <td>1.1e-06</td>
      <td>1.9e-07</td>
      <td>2e-08</td>
      <td>3.9e-09</td>
      <td>8.2e-10</td>
      <td>-4.6e-10</td>
    </tr>
    <tr>
      <th>alpha_0.0001</th>
      <td>0.96</td>
      <td>0.56</td>
      <td>0.55</td>
      <td>-0.13</td>
      <td>-0.026</td>
      <td>-0.0028</td>
      <td>-0.00011</td>
      <td>4.1e-05</td>
      <td>1.5e-05</td>
      <td>3.7e-06</td>
      <td>7.4e-07</td>
      <td>1.3e-07</td>
      <td>1.9e-08</td>
      <td>1.9e-09</td>
      <td>-1.3e-10</td>
      <td>-1.5e-10</td>
      <td>-6.2e-11</td>
    </tr>
    <tr>
      <th>alpha_0.001</th>
      <td>1</td>
      <td>0.82</td>
      <td>0.31</td>
      <td>-0.087</td>
      <td>-0.02</td>
      <td>-0.0028</td>
      <td>-0.00022</td>
      <td>1.8e-05</td>
      <td>1.2e-05</td>
      <td>3.4e-06</td>
      <td>7.3e-07</td>
      <td>1.3e-07</td>
      <td>1.9e-08</td>
      <td>1.7e-09</td>
      <td>-1.5e-10</td>
      <td>-1.4e-10</td>
      <td>-5.2e-11</td>
    </tr>
    <tr>
      <th>alpha_0.01</th>
      <td>1.4</td>
      <td>1.3</td>
      <td>-0.088</td>
      <td>-0.052</td>
      <td>-0.01</td>
      <td>-0.0014</td>
      <td>-0.00013</td>
      <td>7.2e-07</td>
      <td>4.1e-06</td>
      <td>1.3e-06</td>
      <td>3e-07</td>
      <td>5.6e-08</td>
      <td>9e-09</td>
      <td>1.1e-09</td>
      <td>4.3e-11</td>
      <td>-3.1e-11</td>
      <td>-1.5e-11</td>
    </tr>
    <tr>
      <th>alpha_1</th>
      <td>5.6</td>
      <td>0.97</td>
      <td>-0.14</td>
      <td>-0.019</td>
      <td>-0.003</td>
      <td>-0.00047</td>
      <td>-7e-05</td>
      <td>-9.9e-06</td>
      <td>-1.3e-06</td>
      <td>-1.4e-07</td>
      <td>-9.3e-09</td>
      <td>1.3e-09</td>
      <td>7.8e-10</td>
      <td>2.4e-10</td>
      <td>6.2e-11</td>
      <td>1.4e-11</td>
      <td>3.2e-12</td>
    </tr>
    <tr>
      <th>alpha_5</th>
      <td>14</td>
      <td>0.55</td>
      <td>-0.059</td>
      <td>-0.0085</td>
      <td>-0.0014</td>
      <td>-0.00024</td>
      <td>-4.1e-05</td>
      <td>-6.9e-06</td>
      <td>-1.1e-06</td>
      <td>-1.9e-07</td>
      <td>-3.1e-08</td>
      <td>-5.1e-09</td>
      <td>-8.2e-10</td>
      <td>-1.3e-10</td>
      <td>-2e-11</td>
      <td>-3e-12</td>
      <td>-4.2e-13</td>
    </tr>
    <tr>
      <th>alpha_10</th>
      <td>18</td>
      <td>0.4</td>
      <td>-0.037</td>
      <td>-0.0055</td>
      <td>-0.00095</td>
      <td>-0.00017</td>
      <td>-3e-05</td>
      <td>-5.2e-06</td>
      <td>-9.2e-07</td>
      <td>-1.6e-07</td>
      <td>-2.9e-08</td>
      <td>-5.1e-09</td>
      <td>-9.1e-10</td>
      <td>-1.6e-10</td>
      <td>-2.9e-11</td>
      <td>-5.1e-12</td>
      <td>-9.1e-13</td>
    </tr>
    <tr>
      <th>alpha_20</th>
      <td>23</td>
      <td>0.28</td>
      <td>-0.022</td>
      <td>-0.0034</td>
      <td>-0.0006</td>
      <td>-0.00011</td>
      <td>-2e-05</td>
      <td>-3.6e-06</td>
      <td>-6.6e-07</td>
      <td>-1.2e-07</td>
      <td>-2.2e-08</td>
      <td>-4e-09</td>
      <td>-7.5e-10</td>
      <td>-1.4e-10</td>
      <td>-2.5e-11</td>
      <td>-4.7e-12</td>
      <td>-8.7e-13</td>
    </tr>
  </tbody>
</table>
</div>

<br>

### Lasso


```python
from sklearn.linear_model import Lasso
def lasso_regression(data, predictors, alpha, models_to_plot={}):
    #Fit the model
    lassoreg = Lasso(alpha=alpha,normalize=True, max_iter=1e5)
    lassoreg.fit(data[predictors],data['y'])
    y_pred = lassoreg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    if alpha in models_to_plot:
        plt.subplot(models_to_plot[alpha])
        plt.tight_layout()
        plt.plot(data['x'],y_pred)
        plt.plot(data['x'],data['y'],'.')
        plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    ret = [rss]
    ret.extend([lassoreg.intercept_])
    ret.extend(lassoreg.coef_)
    return ret
```


```python
#Initialize predictors to all 15 powers of x
predictors=['x']
predictors.extend(['x_%d'%i for i in range(2,16)])

#Define the alpha values to test
alpha_lasso = [1e-15, 1e-10, 1e-8, 1e-5,1e-4, 1e-3,1e-2, 1, 5, 10]

#Initialize the dataframe to store coefficients
col = ['rss','intercept'] + ['coef_x_%d'%i for i in range(1,16)]
ind = ['alpha_%.2g'%alpha_lasso[i] for i in range(0,10)]
coef_matrix_lasso = pd.DataFrame(index=ind, columns=col)

#Define the models to plot
models_to_plot = {1e-10:231, 1e-5:232,1e-4:233, 1e-3:234, 1e-2:235, 1:236}

#Iterate over the 10 alpha values:
for i in range(10):
    coef_matrix_lasso.iloc[i,] = lasso_regression(data, predictors, alpha_lasso[i], models_to_plot)
```



<img width="849" alt="output_65_1" src="https://user-images.githubusercontent.com/70505378/134346322-80cedbd6-8b7c-4991-aa37-e31457dfa8dd.png">
    




```python
coef_matrix_lasso
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
      <th>rss</th>
      <th>intercept</th>
      <th>coef_x_1</th>
      <th>coef_x_2</th>
      <th>coef_x_3</th>
      <th>coef_x_4</th>
      <th>coef_x_5</th>
      <th>coef_x_6</th>
      <th>coef_x_7</th>
      <th>coef_x_8</th>
      <th>coef_x_9</th>
      <th>coef_x_10</th>
      <th>coef_x_11</th>
      <th>coef_x_12</th>
      <th>coef_x_13</th>
      <th>coef_x_14</th>
      <th>coef_x_15</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>alpha_1e-15</th>
      <td>0.96</td>
      <td>0.22</td>
      <td>1.1</td>
      <td>-0.37</td>
      <td>0.00089</td>
      <td>0.0016</td>
      <td>-0.00012</td>
      <td>-6.4e-05</td>
      <td>-6.3e-06</td>
      <td>1.4e-06</td>
      <td>7.8e-07</td>
      <td>2.1e-07</td>
      <td>4e-08</td>
      <td>5.4e-09</td>
      <td>1.8e-10</td>
      <td>-2e-10</td>
      <td>-9.2e-11</td>
    </tr>
    <tr>
      <th>alpha_1e-10</th>
      <td>0.96</td>
      <td>0.22</td>
      <td>1.1</td>
      <td>-0.37</td>
      <td>0.00088</td>
      <td>0.0016</td>
      <td>-0.00012</td>
      <td>-6.4e-05</td>
      <td>-6.3e-06</td>
      <td>1.4e-06</td>
      <td>7.8e-07</td>
      <td>2.1e-07</td>
      <td>4e-08</td>
      <td>5.4e-09</td>
      <td>1.8e-10</td>
      <td>-2e-10</td>
      <td>-9.2e-11</td>
    </tr>
    <tr>
      <th>alpha_1e-08</th>
      <td>0.96</td>
      <td>0.22</td>
      <td>1.1</td>
      <td>-0.37</td>
      <td>0.00077</td>
      <td>0.0016</td>
      <td>-0.00011</td>
      <td>-6.4e-05</td>
      <td>-6.3e-06</td>
      <td>1.4e-06</td>
      <td>7.8e-07</td>
      <td>2.1e-07</td>
      <td>4e-08</td>
      <td>5.3e-09</td>
      <td>2e-10</td>
      <td>-1.9e-10</td>
      <td>-9.3e-11</td>
    </tr>
    <tr>
      <th>alpha_1e-05</th>
      <td>0.96</td>
      <td>0.5</td>
      <td>0.6</td>
      <td>-0.13</td>
      <td>-0.038</td>
      <td>-0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>7.7e-06</td>
      <td>1e-06</td>
      <td>7.7e-08</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0</td>
      <td>-7e-11</td>
    </tr>
    <tr>
      <th>alpha_0.0001</th>
      <td>1</td>
      <td>0.9</td>
      <td>0.17</td>
      <td>-0</td>
      <td>-0.048</td>
      <td>-0</td>
      <td>-0</td>
      <td>0</td>
      <td>0</td>
      <td>9.5e-06</td>
      <td>5.1e-07</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-4.4e-11</td>
    </tr>
    <tr>
      <th>alpha_0.001</th>
      <td>1.7</td>
      <td>1.3</td>
      <td>-0</td>
      <td>-0.13</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1.5e-08</td>
      <td>7.5e-10</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>alpha_0.01</th>
      <td>3.6</td>
      <td>1.8</td>
      <td>-0.55</td>
      <td>-0.00056</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>alpha_1</th>
      <td>37</td>
      <td>0.038</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
    </tr>
    <tr>
      <th>alpha_5</th>
      <td>37</td>
      <td>0.038</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
    </tr>
    <tr>
      <th>alpha_10</th>
      <td>37</td>
      <td>0.038</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
      <td>-0</td>
    </tr>
  </tbody>
</table>
</div>

<br>

<br>

## 정리

- Conclusion
  - Ridge: It includes all (or none) of the features in the model. Thus, the major advantage of ridge regression is **coefficient shrinkage** and **reducing model complexity**.
  - Lasso: Along with **shrinking coefficients**, lasso performs **feature selection** as well. (Remember the ‘selection‘ in the lasso full-form) As we observed, some of the coefficients become exactly zero, which is equivalent to the particular feature being excluded from the model.
  - 계수 `alpha` 는 Ridge보다 Lasso에서 더 큰 영향을 보인다. Ridge에서는 10e-2 ~ 10e-4 값을 일반적으로 사용하는 데 반해, Lasso에서는 10e-2 이상의 값에서는 표현력을 잃는다. 

- Typical use cases:

  - Ridge: It is majorly used to **prevent overfitting**. Since it includes all the features, it is not very useful in case of exorbitantly high #features, say in millions, as it will pose computational challenges.
  - Lasso: Since it provides sparse solutions, it is generally **the model of choice** (or some variant of this concept) for modelling cases where the **# of features are in millions or more**. In such a case, getting a sparse solution is of great computational advantage as the features with zero coefficients can simply be ignored.
