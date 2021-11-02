---
layout: single
title: "[Machine Learning] Dimension Reduction"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: ['SelectPercentile', PCA', 't-SNE']
---

<br>

## Dimensionality Reduction

- Feature Elimination: You reduce the feature space by eliminating features. This has a disadvantage though, as you gain no information from those features that you have dropped.
- Feature Selection: You apply some statistical tests in order to rank them according to their importance and then select a subset of features for your work. This again suffers from information loss and is less stable as different test gives different importance score to features. 
- Feature Extraction: You create new independent features, where each new independent feature is a combination of each of the old independent features. These techniques can further be divided into linear and non-linear dimensionality reduction techniques.
- tSNE and PCA are feature extraction
- https://www.datacamp.com/community/tutorials/introduction-t-sne?utm_source=adwords_ppc&utm_campaignid=1455363063&utm_adgroupid=65083631748&utm_device=c&utm_keyword=&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=278443377086&utm_targetid=aud-299261629574:dsa-429603003980&utm_loc_interest_ms=&utm_loc_physical_ms=1009871&gclid=CjwKCAjwtNf6BRAwEiwAkt6UQn9Fh31RQWu68b19VdBqQhZWcl_EiKf-R1fW_5heDab7jEZLOYWqOxoCvHoQAvD_BwE

<br>

## PCA (Principal Component Analysis)

분산(variance)이 가장 큰 방향으로 데이터를 사상(projection)

![image-20211101145900790](https://user-images.githubusercontent.com/70505378/139628598-65e56d7f-e3a2-4ade-913e-8b2e5afb3c1c.png)

![image-20211101145955194](https://user-images.githubusercontent.com/70505378/139628605-2b79b3f4-d811-4234-b498-d99a58780ffd.png)

- breastcancer example
- use some important data only (about 30%), SelectPercentile
- tSNE 

### Setup


```python
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier 

from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
import seaborn as sns
%matplotlib inline
```


```python
cancer = load_breast_cancer()
X_all = cancer.data
y = cancer.target 
X_all = StandardScaler().fit_transform(X_all)
```


```python
X_all.shape
```


    (569, 30)


```python
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

**30개의 특성을 모두 사용한 경우**


```python
rfc = RandomForestClassifier(n_estimators=200)
```


```python
cross_val_score(rfc, X_all, y, cv=5).mean().round(4)
```


    0.9631

<br>

### Feature Selection

**Feature Selection: importance of feature selection**

- It enables the machine learning algorithm to train faster.
- It reduces the complexity of a model and makes it easier to interpret.
- It improves the accuracy of a model if the right subset is chosen.
- It reduces Overfitting.

**SelectPercentile()**

- `SelectPercentile(score_func, percentile)`: Select features according to a percentile 
  of the highest scores.
    - score_func : callable Function taking two arrays X and y, and returning a pair of arrays (scores, pvalues) or a single array with scores. Default is f_classif. The default function only works with classification tasks.
    - percentile : int, optional, default=10, Percent of features to keep.

**Chi-squared statistics**

- 상관관계를 계산하여 우연히 어떤 관계가 발생한 것인지 아니면 충분히 연관성이 있는지 알려주는 방법.
- Chi2 test:
  - 두 범주형 변수간의 상관관계를 측정하는 통계적 기법
  - observed value (input feature) 가 expected value (expected output)와 얼마나 관련이 있는지 결정
  - problem of feature selection.
  - X^2 = sum[(Oi - Ei)^2 / Ei], where
    - Oi: observed frq in each category (input)
    - Ei: expected freq (label)
    - k: number of categories
    - sum[(관측값 - 기댓값)^2 / 기댓값]
  - When two features are independent, the observed count is close to the expected count, thus we will have smaller Chi-Square value.
  - So high Chi-Square value indicates that the hypothesis of independence is incorrect.
  - **In simple words, higher the Chi-Square value the feature is more dependent on the response and it can be selected for model training.**

#### 6개의 특성만 사용하는 경우


```python
from sklearn.feature_selection import SelectPercentile, chi2
fs = SelectPercentile(chi2, percentile = 20) # 20%만 사용
sc = StandardScaler()
X_P = fs.fit_transform(cancer.data, y)
X_P = sc.fit_transform(X_P)
```


```python
fs.get_support()   # 20% - 6개의 특성 선택
```


    array([False, False,  True,  True, False, False, False, False, False,
           False, False, False, False,  True, False, False, False, False,
           False, False,  True, False,  True,  True, False, False, False,
           False, False, False])


```python
cancer.feature_names[fs.get_support()]
```


    array(['mean perimeter', 'mean area', 'area error', 'worst radius',
           'worst perimeter', 'worst area'], dtype='<U23')


```python
cross_val_score(rfc, X_P, y).mean().round(4)
```


    0.9315

<br>

#### 2개의 특성만 사용하는 경우


```python
# 상위 6%의 유효한 특성만 선택 )
fs = SelectPercentile(chi2, percentile = 6)
X_P = fs.fit_transform(cancer.data, y)
X_P = sc.fit_transform(X_P)
cancer.feature_names[fs.get_support()]
```


    array(['mean area', 'worst area'], dtype='<U23')


```python
cross_val_score(rfc, X_P, y).mean().round(4)
```


    0.9174

<br>


```python
m = ['v', 'o']
c = ['r','b']
plt.figure(figsize=(8,6))
for i in range(len(y)):
    plt.scatter(cancer.data[:,0][i],cancer.data[:,1][i], marker=m[y[i]], c=c[y[i]], s=5)
plt.show()
```


![output_26_0](https://user-images.githubusercontent.com/70505378/139628663-84d9fb60-c4d9-4e0e-991b-5e732d36946d.png)
    

<br>

```python
m = ['v', 'o']
c = ['r','b']
plt.figure(figsize=(8,6))
for i in range(len(y)):
    plt.scatter(X_P[:,0][i],X_P[:,1][i], marker=m[y[i]], c=c[y[i]], s=5)
plt.show()
```


![output_27_0](https://user-images.githubusercontent.com/70505378/139628666-ed9568d7-f405-4128-aac5-e7f00b608fda.png)

<br>    


### Feature Extraction

- `PCA(n_components)`: Dimension reduction using Principal components analysis
- linear method

#### PCA로 2개의 차원만 사용하는 경우


```python
pca = PCA(n_components=2)
pca_result = pca.fit_transform(X_all)
```


```python
pca_result #  after dimensionality reduction, there usually isn’t a particular 
           # meaning assigned to each principal component. The new components are 
           # just the two main dimensions of variation.
```


    array([[ 9.19283683,  1.94858307],
           [ 2.3878018 , -3.76817174],
           [ 5.73389628, -1.0751738 ],
           ...,
           [ 1.25617928, -1.90229671],
           [10.37479406,  1.67201011],
           [-5.4752433 , -0.67063679]])


```python
m = ['v', 'o']
c = ['r','b']
plt.figure(figsize=(8,6))
for i in range(len(y)):
    plt.scatter(pca_result[:,0][i],pca_result[:,1][i], marker=m[y[i]], c=c[y[i]], s=5)
plt.show()
```


![output_32_0](https://user-images.githubusercontent.com/70505378/139628668-07d7d7b4-6360-4e07-b832-d068b1d9b7d4.png)
    

<br>

```python
pca.components_.round(3) # 기존의 30개의 특성에 각각 어떤 가중치를 곱했는지 파악
```


    array([[ 0.219,  0.104,  0.228,  0.221,  0.143,  0.239,  0.258,  0.261,
             0.138,  0.064,  0.206,  0.017,  0.211,  0.203,  0.015,  0.17 ,
             0.154,  0.183,  0.042,  0.103,  0.228,  0.104,  0.237,  0.225,
             0.128,  0.21 ,  0.229,  0.251,  0.123,  0.132],
           [-0.234, -0.06 , -0.215, -0.231,  0.186,  0.152,  0.06 , -0.035,
             0.19 ,  0.367, -0.106,  0.09 , -0.089, -0.152,  0.204,  0.233,
             0.197,  0.13 ,  0.184,  0.28 , -0.22 , -0.045, -0.2  , -0.219,
             0.172,  0.144,  0.098, -0.008,  0.142,  0.275]])


```python
pca.explained_variance_ratio_, sum(pca.explained_variance_ratio_) # 각 주성분 요소들이 얼마나 데이터를
                                                                  # 잘 설명하는지 파악
```


    (array([0.44272026, 0.18971182]), 0.6324320765155942)


```python
cross_val_score(rfc, pca_result, y, cv=5).mean().round(4)
```


    0.9315

<br>

#### PCA로 6개의 차원만 사용하는 경우


```python
pca = PCA(n_components=6)
pca_result = pca.fit_transform(X_all)
cross_val_score(rfc, pca_result, y, cv=5).mean().round(4)
```


    0.949

**앞의 selectPercentile 보다 성능이 개선됨**

<br>

<br>

## tSNE

고차원의 데이터 분포와 비슷하도록(KL divergence가 최소가 되도록) 저차원의 데이터를 mapping. T-distribution 사용.

![image-20211101150125696](https://user-images.githubusercontent.com/70505378/139628607-ef4b3b63-4d68-4bb4-b0fa-5ffbc65cd8ff.png)

![image-20211101150217944](https://user-images.githubusercontent.com/70505378/139628608-1079d7e0-3959-4231-a345-0764f1b7fc87.png)

- `TSNE(n_components, perplexity, n_iter)`
- non-linear method
- 고차원의 데이터를 저차원으로 축소. 데이터 시각화에 주로 사용. 
- 고차원 공간에서 유클리드 거리를 데이터 포인트의 유사성을 표현하는 조건부 확률로 변환하는 방법

### tSNE visualization

- n_components: Dimension of the embedded space
- perplexity: float, optional (default: 30) : The perplexity is related to the number of nearest neighbors that is used in other manifold learning algorithms. Larger datasets usually require a larger perplexity. Consider selecting a value between 5 and 50. Different values can result in significanlty different results.(당혹, 곤혹), 데이터 점 xi의 유효한 근방의 개수의 척도


```python
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(cancer.data)
```

    [t-SNE] Computing 121 nearest neighbors...
    [t-SNE] Indexed 569 samples in 0.002s...
    [t-SNE] Computed neighbors for 569 samples in 0.019s...
    [t-SNE] Computed conditional probabilities for sample 569 / 569
    [t-SNE] Mean sigma: 33.679708
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 49.179726
    [t-SNE] KL divergence after 1000 iterations: 0.216705

```python
m = ['v','o']
c = ['r','b']
plt.figure(figsize=(8,6))
for i in range(len(y)):
    plt.scatter(tsne_results[:,0][i],tsne_results[:,1][i], marker=m[y[i]], c=c[y[i]], s=5)
plt.show()
```


![output_42_0](https://user-images.githubusercontent.com/70505378/139628669-25999dc3-7e5f-4824-bddf-7d61cdb75fd3.png)
    

<br>

### Scaling and tSNE visulaization


```python
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=1000)
tsne_results = tsne.fit_transform(X_all)
```

    [t-SNE] Computing 121 nearest neighbors...
    [t-SNE] Indexed 569 samples in 0.002s...
    [t-SNE] Computed neighbors for 569 samples in 0.043s...
    [t-SNE] Computed conditional probabilities for sample 569 / 569
    [t-SNE] Mean sigma: 1.522404
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 63.951508
    [t-SNE] KL divergence after 1000 iterations: 0.852838

```python
m = ['v','o']
c = ['r','b']
plt.figure(figsize=(8,6))
for i in range(len(y)):
    plt.scatter(tsne_results[:,0][i],tsne_results[:,1][i], marker=m[y[i]], c=c[y[i]], s=5)
plt.show()
```


![output_45_0](https://user-images.githubusercontent.com/70505378/139628671-d3c8e334-9c0a-45d0-a802-d50949eabc0d.png)
    

<br>

<br>

## MNIST dataset dimension reduction


```python
from tensorflow.keras.datasets import mnist
import numpy as np
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
```


```python
plt.imshow(x_train[0])
```




![output_48_1](https://user-images.githubusercontent.com/70505378/139628672-57a82e98-c786-4b01-8c07-b707e17a51b9.png)
    



```python
# 784차원 데이터
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
```

    (60000, 784) (10000, 784) (60000,) (10000,)

<br>

### PCA

```python
pca = PCA(n_components = 2) # 2차원으로 축소
pca_result = pca.fit_transform(x_train) 

plt.figure(figsize=(8,6))
plt.scatter(pca_result[:, 0], pca_result[:, 1], c=y_train, cmap='jet', alpha=0.5, s=3)
plt.colorbar()
plt.show()
```


![output_50_0](https://user-images.githubusercontent.com/70505378/139628673-46d49307-c834-4636-9d50-f42b5bf5eb33.png)
    

<br>

### TSNE

```python
x_train = x_train[:6000]   # too big to computer tSNE
y_train = y_train[:6000]

tsne = TSNE(n_components = 2, verbose=1, perplexity=40, n_iter=1000)
tsne_result = tsne.fit_transform(x_train)

plt.figure(figsize=(8,6))
plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c=y_train, s=5)
plt.colorbar()
plt.show()
```

    [t-SNE] Computing 121 nearest neighbors...
    [t-SNE] Indexed 6000 samples in 0.004s...
    [t-SNE] Computed neighbors for 6000 samples in 0.939s...
    [t-SNE] Computed conditional probabilities for sample 1000 / 6000
    [t-SNE] Computed conditional probabilities for sample 2000 / 6000
    [t-SNE] Computed conditional probabilities for sample 3000 / 6000
    [t-SNE] Computed conditional probabilities for sample 4000 / 6000
    [t-SNE] Computed conditional probabilities for sample 5000 / 6000
    [t-SNE] Computed conditional probabilities for sample 6000 / 6000
    [t-SNE] Mean sigma: 2.277370
    [t-SNE] KL divergence after 250 iterations with early exaggeration: 81.262482
    [t-SNE] KL divergence after 1000 iterations: 1.476199




![output_51_1](https://user-images.githubusercontent.com/70505378/139628676-29949837-2606-4ee0-8890-f3a8c6302cd5.png)
    

**PCA보다 시간은 훨씬 오래 걸리지만 그만큼 높은 성능을 보인다.**

