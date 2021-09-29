---
layout: single
title: "[Machine Learning] Performance Evaluation"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: ['Regression', 'Classification', 'Metrics']
---



## Setup


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC 
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score 
%matplotlib inline
```

<br>

## Regression Performance

- MAE (mean absolute error)
- MSE (mean square error)
- RMSE (root mean square error)
- R-squared 



![image-20210929161615375](https://user-images.githubusercontent.com/70505378/135221003-7d4185fd-9636-4b5e-984a-b00196823b11.png)

## Classification Performance 
- 분류 알고리즘 비교
  - 리지 규제, 라쏘 규제
  - 교차검증
  - 정적 성능평가 Confusion matrix  
  - 동적 성능평가 ROC, AUC

- Data
  - [포도주 품질 분류 데이터](https://goo.gl/Gyc8K7)
  - https://www.kaggle.com/vishalyo990/prediction-of-quality-of-wine/notebook

<br>

## Classification Example (포도주 품질 평가 데이터)

### Import dataset


```python
!curl -L https://goo.gl/Gyc8K7 -o winequality-red.csv
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
    
    100   144    0   144    0     0    200      0 --:--:-- --:--:-- --:--:--   200
    100   144    0   144    0     0    199      0 --:--:-- --:--:-- --:--:--     0
    
    100   318  100   318    0     0    279      0  0:00:01  0:00:01 --:--:--   279
    
     15   98k   15 15367    0     0   7828      0  0:00:12  0:00:01  0:00:11  7828
    100   98k  100   98k    0     0  43569      0  0:00:02  0:00:02 --:--:--  236k



```python
wine = pd.read_csv('./winequality-red.csv')
print(wine.shape)
wine.head(5)
```

    (1599, 12)





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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
      <th>quality</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
      <td>5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>11.2</td>
      <td>0.28</td>
      <td>0.56</td>
      <td>1.9</td>
      <td>0.075</td>
      <td>17.0</td>
      <td>60.0</td>
      <td>0.9980</td>
      <td>3.16</td>
      <td>0.58</td>
      <td>9.8</td>
      <td>6</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



- fixed acidity - 결합 산도 
- volatile acidity - 휘발성 산도 
- citric acid - 시트르산 
- residual sugar - 잔류 설탕 
- chlorides	 - 염화물 
- free sulfur dioxide - 자유 이산화황 
- total sulfur dioxide - 총 이산화황 
- density - 밀도 
- pH - pH 
- sulphates - 황산염 
- alcohol - 알코올 
- quality - 품질 (0 ~ 10 점)

<br>


```python
wine.info() # 데이터 정보
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1599 entries, 0 to 1598
    Data columns (total 12 columns):
     #   Column                Non-Null Count  Dtype  
    ---  ------                --------------  -----  
     0   fixed acidity         1599 non-null   float64
     1   volatile acidity      1599 non-null   float64
     2   citric acid           1599 non-null   float64
     3   residual sugar        1599 non-null   float64
     4   chlorides             1599 non-null   float64
     5   free sulfur dioxide   1599 non-null   float64
     6   total sulfur dioxide  1599 non-null   float64
     7   density               1599 non-null   float64
     8   pH                    1599 non-null   float64
     9   sulphates             1599 non-null   float64
     10  alcohol               1599 non-null   float64
     11  quality               1599 non-null   int64  
    dtypes: float64(11), int64(1)
    memory usage: 150.0 KB



```python
wine.columns
```




    Index(['fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
           'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
           'pH', 'sulphates', 'alcohol', 'quality'],
          dtype='object')

<br>

### Preprocessing (Label 만들기)


```python
wine['quality'].value_counts()
```




    5    681
    6    638
    7    199
    4     53
    8     18
    3     10
    Name: quality, dtype: int64

<br>

#### Make to binary dataset


```python
# 품질이 좋고 나쁜 것을 나누는 기준 설정
# 6.5를 기준으로 bad(0) good(1)으로 나눈다 (임의로 나눈 것)
my_bins = (2.5, 6.5, 8.5)
groups = [0, 1]
wine['qual'] = pd.cut(wine['quality'], bins = my_bins, labels = groups) 

wine['qual'].value_counts()
```




    0    1382
    1     217
    Name: qual, dtype: int64




```python
X = wine.drop(['quality', 'qual'], axis = 1) 
y = wine['qual'] 

y.value_counts()
```




    0    1382
    1     217
    Name: qual, dtype: int64




```python
X[:3]
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
      <th>fixed acidity</th>
      <th>volatile acidity</th>
      <th>citric acid</th>
      <th>residual sugar</th>
      <th>chlorides</th>
      <th>free sulfur dioxide</th>
      <th>total sulfur dioxide</th>
      <th>density</th>
      <th>pH</th>
      <th>sulphates</th>
      <th>alcohol</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>7.4</td>
      <td>0.70</td>
      <td>0.00</td>
      <td>1.9</td>
      <td>0.076</td>
      <td>11.0</td>
      <td>34.0</td>
      <td>0.9978</td>
      <td>3.51</td>
      <td>0.56</td>
      <td>9.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>7.8</td>
      <td>0.88</td>
      <td>0.00</td>
      <td>2.6</td>
      <td>0.098</td>
      <td>25.0</td>
      <td>67.0</td>
      <td>0.9968</td>
      <td>3.20</td>
      <td>0.68</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>2</th>
      <td>7.8</td>
      <td>0.76</td>
      <td>0.04</td>
      <td>2.3</td>
      <td>0.092</td>
      <td>15.0</td>
      <td>54.0</td>
      <td>0.9970</td>
      <td>3.26</td>
      <td>0.65</td>
      <td>9.8</td>
    </tr>
  </tbody>
</table>
</div>

<br>

### Standard Scaling (표준 스케일링)
- transform the dataset to Gaussian dist (0, 1) - numerical features only
- test dataset should also be scaled


```python
sc = StandardScaler()
X = sc.fit_transform(X)  # fit and transform
```


```python
X[:3]
```




    array([[-0.52835961,  0.96187667, -1.39147228, -0.45321841, -0.24370669,
            -0.46619252, -0.37913269,  0.55827446,  1.28864292, -0.57920652,
            -0.96024611],
           [-0.29854743,  1.96744245, -1.39147228,  0.04341614,  0.2238752 ,
             0.87263823,  0.62436323,  0.02826077, -0.7199333 ,  0.1289504 ,
            -0.58477711],
           [-0.29854743,  1.29706527, -1.18607043, -0.16942723,  0.09635286,
            -0.08366945,  0.22904665,  0.13426351, -0.33117661, -0.04808883,
            -0.58477711]])




```python
np.random.seed(11)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

X_train.shape, y_train.shape, X_test.shape, y_test.shape
```




    ((1279, 11), (1279,), (320, 11), (320,))

<br>

### Model scores

* Regression model - R<sup>2</sup> score
* Classification model - Accuracy

<br>

#### Linear model (Stochastic Gradient Descent method)


```python
sgd = SGDClassifier()
sgd.fit(X_train, y_train)
sgd.score(X_test,y_test)
```


    0.81875

<br>

#### Decesion Tree


```python
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(max_depth=5)
clf.fit(X_train, y_train)
clf.score(X_train,y_train), clf.score(X_test,y_test)
```


    (0.9335418295543393, 0.878125)

<br>

#### Random Forest Classifier


```python
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=300, max_depth=5) 
rfc.fit(X_train, y_train)
rfc.score(X_train,y_train), rfc.score(X_test,y_test)
```


    (0.9296325254104769, 0.88125)

<br>

#### Support Vector Classifier (SVC)


```python
svc = SVC()   # default: C=1.0, kernel='rbf', gamma='scale' 
svc.fit(X_train, y_train)
svc.score(X_train,y_train), svc.score(X_test,y_test)
```


    (0.8991399530883503, 0.88125)

<br>

#### Logistic Regression


```python
log = LogisticRegression()
log.fit(X_train, y_train)
log.score(X_train,y_train), log.score(X_test,y_test)
```


    (0.8819390148553558, 0.86875)

<br>

#### Cross validation(교차 검증)


```python
# estimator = 모델, cv는 분할 블록의 갯수
rfc_eval = cross_val_score(rfc, X, y, cv = 5)  
rfc_eval, rfc_eval.mean()
```


    (array([0.875     , 0.871875  , 0.875     , 0.86875   , 0.88401254]),
     0.8749275078369905)

<br>

<br>

### Performace metrics

####  Performance : 정적 평가, 혼돈 매트릭스 (confusion_matrix)


```python
y_pred = sgd.predict(X_test)
confusion_matrix(y_test, y_pred)
```


    array([[253,  16],
           [ 42,   9]], dtype=int64)


```python
print(classification_report(y_test, y_pred))
```

                  precision    recall  f1-score   support
    
               0       0.86      0.94      0.90       269
               1       0.36      0.18      0.24        51
    
        accuracy                           0.82       320
       macro avg       0.61      0.56      0.57       320
    weighted avg       0.78      0.82      0.79       320

<br>

#### Score (or Probability)


```python
y_score = sgd.decision_function(X_test)   # sgd 는 predict_proba() 가 없음
# decision_function(): The confidence score for a sample is the signed distance 
# of that sample to the hyperplane

y_score[:5]
```


    array([ 0.79259076, -2.95713556, -5.74014753, -1.21517746, -5.88022051])

<br>

#### Ranking (순서를 평가)


```python
result = pd.DataFrame(list(zip(y_score, y_pred, y_test)), 
                      columns=['score', 'predict', 'real'])
result['correct'] = (result.predict == result.real)
result.head()
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
      <th>score</th>
      <th>predict</th>
      <th>real</th>
      <th>correct</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.792591</td>
      <td>1</td>
      <td>0</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-2.957136</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-5.740148</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-1.215177</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-5.880221</td>
      <td>0</td>
      <td>0</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
</div>

<br>

#### ROC and AUC (맞춘 순서로 평가)


```python
fpr = dict()
tpr = dict()
roc_auc = dict()
fpr, tpr, _ = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC')
plt.legend(loc="lower right")
```




![output_45_1](https://user-images.githubusercontent.com/70505378/135220512-cb7fe749-c5dc-487f-a747-438ca0d89d77.png)
    

<br>

#### Precision-Recall curve


```python
from sklearn.metrics import average_precision_score
precision, recall, thresholds = precision_recall_curve(y_test, y_score)
auc_score = auc(recall, precision)
print(average_precision_score(y_test, y_score))
plt.plot(recall, precision, marker='.', label='area = %0.2f' % auc_score)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()
```

    0.36505616448154043




![output_47_2](https://user-images.githubusercontent.com/70505378/135220516-701c2e8d-172c-41fa-acd5-28c6fd488068.png)
    

