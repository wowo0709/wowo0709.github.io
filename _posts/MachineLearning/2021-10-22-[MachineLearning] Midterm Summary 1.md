---
layout: single
title: "[Machine Learning] Midterm Summary 1"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: []
---

<br>

## week2

### Lab 1: End to End Machine Learning Process

#### 데이터셋 분석

```python
import pandas as pd

housing = pd.read_csv(csv_path) # csv 파일을 읽어와서 데이터프레임으로 변환

housing.columns # 데이터프레임의 feature list
housing.shape # 데이터프레임 형상
housing.info() # 데이터프레임의 각 feature의 개수, dtype
housing["열 이름"].value_counts() # 해당 열의 어떤 값이 몇 개 있는지 반환
housing.describe() # 데이터프레임의 각 feature에 대한 수치적 정보(개수, 평균, 최소값 등)를 반환

housing["median_income"].hist() # feature의 value_count()를 히스토그램으로 plot
housing["income_cat"] = pd.cut(housing["median_income"], # feature의 값을 특정 구간으로 매핑
                              bins = [0., 1.5, 3.0, 4.5, 6. np.inf], 
                              labels = [1,2,3,4,5])

housing.corr() # 상관관계 행렬
housing.sort_values(by="열 이름", ascending) # 데이터프레임을 특정 열의 값을 기준으로 정렬

# 추가
data['diagnosis'] = data['diagnosis'].map({'M':1, 'B':0})
```

<br>

#### 훈련-테스트 데이터셋 분리

* `train_test_split`: dataset, test_size, random_state
* `StratifiedShuffleSplit`: n_splits, test_size, random_state

```python
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42) # 학습-테스트 데이터셋 분리

split = StratifiedShuffleSplit(n_splits=1,test_size=0.2,random_state=42) # 클래스 비율이 같도록 데이터셋 분리
for train_index, test_index in split.split(housing, housing["income_cat"]): # X, y
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

```

<br>

#### 데이터 시각화

```python
# dataframe.plot
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
plt.show()
```



![image-20211022155811569](https://user-images.githubusercontent.com/70505378/138478845-fca0c461-2949-496b-88c2-6f5f18ac436d.png)

```python
# seaborn
import seaborn as sns

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
sns.pairplot(housing[attributes])
```



![image-20211022155825315](https://user-images.githubusercontent.com/70505378/138478847-a2ec390f-89da-46c4-a339-d68a395b73d2.png)

```python
# pandas.plotting
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
```



![image-20211022155838080](https://user-images.githubusercontent.com/70505378/138478850-25ba652d-8b13-4224-8180-7812f63f5421.png)

<br>

#### 데이터 전처리

```python
housing.drop("열 이름", axis) # 0: 행 방향, 1: 열 방향
housing.isnull().any(0) # 각 feature 별로 null이 있는지 검사
housing.isnull().sum() # 각 feature 별로 null이 몇 개인지 검사
housing_num = housing.drop("ocean_proximity", axis=1)

# 누락값에 대한 처리
housing.dropna(subset=["열 이름"]) # null이 있는 샘플을 누락. subset은 null이 있는지 검사할 열을 지정. 
housing.drop("열 이름", axis=1) # null이 있는 feature를 누락. 
median = housing["열 이름"].median()
housing["열 이름"].fillna(median) # 다른 값으로 대체


# Imputer
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
imputer.fit(numerical_dataframe)
numerical_dataframe_without_nan = imputer.transform(numerical_dataframe_with_nan)


# Encoding
from sklearn.preprocessing import OrdinalEncoder, LebelEncoder, OneHotEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
'''
array([[3.],
       [0.],
       [0.],
       [1.],
       [1.],
       [1.],
       [1.],
       [0.],
       [1.],
       [1.]])
'''
ordinal_encoder.categories_
'''
[array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
       dtype=object)]
'''


cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
'''
<16512x5 sparse matrix of type '<class 'numpy.float64'>'
	with 16512 stored elements in Compressed Sparse Row format>
'''
housing_cat_1hot.toarray()
'''
array([[0., 0., 0., 1., 0.],
       [1., 0., 0., 0., 0.],
       [1., 0., 0., 0., 0.],
       ...,
       [0., 1., 0., 0., 0.],
       [0., 0., 0., 1., 0.],
       [1., 0., 0., 0., 0.]])
'''
cat_encoder.categories_
'''
array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
       dtype=object)]
'''


# Scaling
from sklearn.preprocessing import StandardScaler, MinMaxScaler

scaler = StandardScaler()
scaled_num_attribs = scaler.fit_transform(num_attribs)
scaler.mean_, scaler.std_
num_attribs = scaler.inverse_transform(scaled_num_attribs)


# Pipeline
col_names = ["total_rooms", "total_bedrooms", "population", "households"] # 추가할 특성
rooms_ix, bedrooms_ix, population_ix, households_ix = [ # 3, 4, 5, 6
    housing.columns.get_loc(c) for c in col_names] 
class CombinedAttributesAdder(): # 특성 추가 클래스 (fit과 transform 메서드 정의 필요)
    def fit(self, X, y=None):
        return self  
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]

        X = np.delete(X, [households_ix, rooms_ix, population_ix, bedrooms_ix], 1)

        return np.c_[X, rooms_per_household, population_per_household,
                    bedrooms_per_room]
    
from sklearn.pipeline import Pipeline
# 수치 데이터에 대한 파이프라인
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

from sklearn.compose import ColumnTransformer # 특성 별로 다른 변환 적용

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]
# 전체 파이프라인
full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs), # 수치 데이터는 수치 데이터에 대한 파이프라인 적용
        ("cat", OneHotEncoder(), cat_attribs), # 카테고리 데이터는 원-핫 인코딩 적용
    ])

housing_prepared = full_pipeline.fit_transform(housing)
```

<br>

#### 모델 선택, 훈련

```python
# 선형 회귀 모델
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# 성능 평가
from sklearn.metrics import mean_squared_error, r2_score

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_r2 = r2_score(housing_labels, housing_predictions)
lin_mse, np.sqrt(lin_mse), lin_r2, lin_reg.score(housing_prepared, housing_labels)

# 결정 트리 모델
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42, max_depth=20)
tree_reg.fit(housing_prepared, housing_labels)

housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_r2 = r2_score(housing_labels, housing_predictions)
tree_mse, np.sqrt(tree_mse), tree_r2, tree_reg.score(housing_prepared, housing_labels)
```

<br>

<br>

## week3

**<span style="color:red">선형 모델이나 SVM, 신경망</span>에서는 반드시 <span style="color:red">Scaling</span>을 해야 한다.**

### Gradient Descent Regression

#### 임시 데이터 생성

```python
n = 100
x = np.random.randn(n)                # batch size
y = x*20 + 10                         # w=20, b=10
y = y + np.random.randn(n) * 10       # add noise

plt.scatter(x,y)
```

![image-20211022174854187](https://user-images.githubusercontent.com/70505378/138478851-280b812a-4c50-4cf1-baa5-302f3dd52843.png)

#### 경사 하강 알고리즘

손코딩 나올 수도!!!

```python
w=np.random.randn()   
b=np.random.randn()

lr = 0.1          # learning rate
n_epoch = 200     # number of epoch
lossHistory = []  

# 1 feature
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

'''
epoch= 0 loss= 608.135326701648 w= 0.689784764278889 b= 0.9972609240805469
epoch= 10 loss= 207.53080423004252 w= 11.670716818468396 b= 7.227075715189502
epoch= 20 loss= 152.28938253673073 w= 15.930927964152566 b= 9.20608673708662
epoch= 30 loss= 144.4693105859433 w= 17.59153654618134 b= 9.821009291157926
epoch= 40 loss= 143.33405880761802 w= 18.24161978885116 b= 10.006080890111248
epoch= 50 loss= 143.16542027501967 w= 18.497100763621827 b= 10.059105160686649
epoch= 60 loss= 143.1398621531657 w= 18.597855042631664 b= 10.073064262692903
epoch= 70 loss= 143.13592303973158 w= 18.63771339903109 b= 10.076142655288594
epoch= 80 loss= 143.13530760578266 w= 18.653524866671727 b= 10.076507439041466
epoch= 90 loss= 143.13521041460535 w= 18.659812389462125 b= 10.076353219559355
epoch= 100 loss= 143.13519493822233 w= 18.66231798888442 b= 10.076187429013352
epoch= 110 loss= 143.1351924582818 w= 18.66331833668638 b= 10.076084942824329
epoch= 120 loss= 143.1351920590203 w= 18.663718366743243 b= 10.076031356750361
epoch= 130 loss= 143.13519199451582 w= 18.66387855956513 b= 10.076005529109846
epoch= 140 loss= 143.13519198406763 w= 18.663942786892253 b= 10.075993661364555
epoch= 150 loss= 143.1351919823721 w= 18.66396856497532 b= 10.0759883752038
epoch= 160 loss= 143.13519198209656 w= 18.663978920499822 b= 10.075986071025419
epoch= 170 loss= 143.13519198205177 w= 18.66398308371867 b= 10.075985082344856
epoch= 180 loss= 143.13519198204446 w= 18.663984758562417 b= 10.075984663108663
epoch= 190 loss= 143.1351919820433 w= 18.66398543272708 b= 10.075984486949109
---------------------------
epoch= 199 loss= 143.13519198204307 w= 18.663985686793474 b= 10.075984418251208
'''

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

# 2 features
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

'''
epoch= 0 loss= 143.13519198204307 w= 18.663985686793474 b= 5.527577031987033
epoch= 10 loss= 143.13519198204307 w= 18.663985686793474 b= 34.161569950771764
epoch= 20 loss= 143.13519198204307 w= 18.663985686793474 b= 42.63247728631368
epoch= 30 loss= 143.13519198204307 w= 18.663985686793474 b= 45.17114084654051
epoch= 40 loss= 143.13519198204307 w= 18.663985686793474 b= 45.94040900283646
epoch= 50 loss= 143.13519198204307 w= 18.663985686793474 b= 46.17526739749047
epoch= 60 loss= 143.13519198204307 w= 18.663985686793474 b= 46.247108444149255
epoch= 70 loss= 143.13519198204307 w= 18.663985686793474 b= 46.26894507259233
epoch= 80 loss= 143.13519198204307 w= 18.663985686793474 b= 46.27545836078856
epoch= 90 loss= 143.13519198204307 w= 18.663985686793474 b= 46.27732562044224
epoch= 100 loss= 143.13519198204307 w= 18.663985686793474 b= 46.277819716244096
epoch= 110 loss= 143.13519198204307 w= 18.663985686793474 b= 46.27792830496731
epoch= 120 loss= 143.13519198204307 w= 18.663985686793474 b= 46.27793959569699
epoch= 130 loss= 143.13519198204307 w= 18.663985686793474 b= 46.27793226596585
epoch= 140 loss= 143.13519198204307 w= 18.663985686793474 b= 46.27792489363772
epoch= 150 loss= 143.13519198204307 w= 18.663985686793474 b= 46.27792025712462
epoch= 160 loss= 143.13519198204307 w= 18.663985686793474 b= 46.27791775399147
epoch= 170 loss= 143.13519198204307 w= 18.663985686793474 b= 46.27791650126122
epoch= 180 loss= 143.13519198204307 w= 18.663985686793474 b= 46.27791590193356
epoch= 190 loss= 143.13519198204307 w= 18.663985686793474 b= 46.27791562359162
---------------------------
epoch= 199 error= 409.6591467434651 w1= 27.94 w2= 40.08 b= 46.28
'''
```

<br>

#### LinearRegression 모델 사용

```python
from sklearn.linear_model import LinearRegression

# Make it to matrix(two features)
X = np.concatenate([x1.reshape(n,1), x2.reshape(n,1)], axis=1)

model = LinearRegression()        # create model
model.fit(X,y)                    # train model
print("score: ",model.score(X,y))
print('w1=', model.coef_[0], 'w2=', model.coef_[1], 'b=', model.intercept_)

# prediction
new_X=[1,3] # x1, x2
print('Real Value: ', 1*30 + 3*40 + 50)        # y 
print('Predicted Value', *model.predict([new_X]))  # model predict(inference)

'''
score:  0.8465475643687691
w1= 27.93626338823067 w2= 40.08110377245416 b= 46.27791539580053
Real Value:  200
Predicted Value 194.4574901013937
'''
```

<br>

#### 선형 회귀 데이터 만들기

* `make_regression`: n_samples, n_features, noise, random_state

```python
from sklearn.datasets import make_regression 

X, y = make_regression(n_samples=2000, n_features=2, noise=1.5, random_state=1)
```

![image-20211022180323986](https://user-images.githubusercontent.com/70505378/138478853-d2811c12-1746-48cc-8940-af7dccf2f91b.png)

<br>

### Gradient Descent Classification

#### 선형 분류 데이터셋 만들기

* `make_blobs`: n_samples, n_features, centers, cluster_std, random_state

```python
from sklearn.datasets import make_blobs

N = 500
(X, y) = make_blobs(n_samples=N, n_features=2, centers=2, cluster_std=2.0, random_state=17)
```

![image-20211022180352475](https://user-images.githubusercontent.com/70505378/138478856-552c044a-2117-42d4-8279-9617c5f7885c.png)

#### 경사 하강 알고리즘

이진 분류는 경사 하강 알고리즘 + **시그모이드 함수** 사용

손실 함수로 **cross entropy** 사용

![image-20211022181544852](https://user-images.githubusercontent.com/70505378/138478860-09fb6e5c-9071-45a0-a1bc-92d61164a2e4.png)

```python
w1 = np.random.randn()
w2 = np.random.randn()
b  = np.random.randn() 

def sigmoid_activation(z):
    return 1.0 / (1 + np.exp(-z))

lossHistory = []
epochs = 500
alpha = 0.03

for epoch in np.arange(epochs):
    # 예측
    z = w1*x1 + w2*x2 + b
    # 활성화 함수
    y_hat = sigmoid_activation(z)       # prediction
	# 손실 함수
    loss = -((y*np.log(y_hat) + (1-y)*np.log(1-y_hat))).mean()  # loss = cross entropy
    lossHistory.append(loss)
    # 미분치 계산
    dloss_dz = y_hat - y
    w1_deriv = dloss_dz * x1        # d(loss)/dw1 = d(loss)/dz * dz/dw1
    w2_deriv = dloss_dz * x2
    b_deriv = dloss_dz * 1
    # 가중치 갱신
    w1 = w1 - (alpha * w1_deriv).mean()
    w2 = w2 - (alpha * w2_deriv).mean()
    b  = b  - (alpha * b_deriv).mean()
```

<br>

#### Hinge Loss

분류에서 활성화 함수를 지난 값을 이용해 가중치를 갱신하는 대신, 활성화 함수를 지나기 전의 값(예측값)을 이용해 가중치를 갱신할 수 있다. 

이를 **Hinge Loss**라 한다. 

- Hinge loss for input-output pair (x,y) is given as:
  - L = max(0, 1 - yf(x))
  - L = 0 (if y*f(x) >= 1), 1-y*f(x) (otherwise)
  - dL/dw1 = 0 (if y*f(x) >= 1), -y*x1 (otherwise)

![image-20211022181640847](https://user-images.githubusercontent.com/70505378/138479322-d282bc45-357d-4a29-86b4-3f513845896c.png)

**Hinge Loss**는 분류에서 데이터들을 **1(제대로 예측), -1(잘못 예측)**로 구분한다. 

```python
# 데이터셋 변환
N = 500
(X, y_org) = make_blobs(n_samples=N, n_features=2, centers=2, cluster_std=2.0, random_state=17)
x1, x2 = X[:,0], X[:,1]
y = y_org.copy()
y[y==0] = -1 # 값이 0인 라벨 데이터를 -1로 변경 (Hinge Loss)

# 가중치 및 하이퍼파라미터 초기화
w1, w2, b = np.random.randn(), np.random.randn(), np.random.randn()
lossHistory = []
epochs = 500
alpha = 0.03
N = len(x1)

# 경사 하강 알고리즘
for epoch in np.arange(epochs):

    w1_deriv, w2_deriv, b_deriv, loss = 0., 0., 0., 0.
    for i in range(N):
        score = y[i]*(w1*x1[i] + w2*x2[i] + b) # y * y_hat
        if score <= 1: # Loss 발생 (y*y_hat <= 1, 1 - y*y_hat >= 0)
            w1_deriv = w1_deriv - x1[i]*y[i]
            w2_deriv = w2_deriv - x2[i]*y[i]
            b_deriv = b_deriv - y[i]
            loss = loss + (1 - score)
        # else : derivatives are zero. loss is 0
    
    # mean
    w1_deriv /= float(N)
    w2_deriv /= float(N)
    b_deriv  /= float(N)
    loss /= float(N)
    # update parameters
    w1 = w1 - alpha * w1_deriv
    w2 = w2 - alpha * w2_deriv
    b  =  b - alpha *  b_deriv
```

<br>

#### SGDClassifier/LogisticRegression 모델 사용

* `SGDClassifier` 모델은 기본 손실 함수로 **Hinge Loss**를 사용
* `LogisticRegressor` 모델은 기본 손실 함수로 **Cross Entropy**를 사용

```python
from sklearn.linear_model import SGDClassifier, LogisticRegression

N = 500
(X, y) = make_blobs(n_samples=N, n_features=2, centers=2, cluster_std=2.0, random_state=17)

clf = SGDClassifier()      
clf.fit(X[:,:2], y)    
print("SGDClassifier: ", clf.score(X[:, :2],y))
print(clf.coef_, clf.intercept_)

log = LogisticRegression()      
log.fit(X[:,:2], y)    
print("Logistic Regression: ", log.score(X[:, :2],y))
print(log.coef_, log.intercept_)

'''
SGDClassifier:  0.984
[[ -8.56625092 -34.52822982]] [-211.3932073]
Logistic Regression:  0.994
[[-0.46621709 -1.9549905 ]] [-10.32380395]
'''
```

<br>

<br>

## week 4

### KFold Validation

* `KFold(StratifiedKFold)`: n_splits, shuffle, random_state
* `cross_val_score`: model, X, y, cv object

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

X = X_all[:,0]
y = X_all[:,2]

cv = KFold(n_splits=5,shuffle=True) # Returns the number of splitting iterations in the cross-validator.
# cv = StratifiedKFold(n_splits=5, shuffle=True)
score = cross_val_score(LinearRegression(), X.reshape(-1,1), y, cv=cv)

print(score.round(2))
print(score.mean().round(2))

'''
[0.83 0.71 0.8  0.72 0.62]
0.74
'''
```

#### cv란?

```python
for train_index, test_index in cv.split(X):
    print("TRAIN:\n", train_index,'\n', "TEST:\n", test_index)
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]
    
''' 총 5개
TRAIN:
 [  0   1   2   3   4   5   6   7   8   9  10  11  12  13  14  17  18  19
  21  22  23  24  27  28  30  31  32  33  34  35  36  37  38  39  40  43
  44  45  46  47  48  50  51  58  59  60  61  62  63  64  65  66  67  68
  70  71  74  75  76  78  79  80  83  84  85  88  89  90  91  92  93  94
  95  96  97  98  99 100 101 102 103 104 105 106 107 108 109 111 112 113
 114 115 116 117 118 119 120 121 122 123 124 125 126 128 129 130 131 132
 133 134 135 136 137 138 140 141 142 143 145 146] 
 TEST:
 [ 15  16  20  25  26  29  41  42  49  52  53  54  55  56  57  69  72  73
  77  81  82  86  87 110 127 139 144 147 148 149]
...
TRAIN:
 [  0   2   3   4   6   7   8   9  10  11  13  14  15  16  18  19  20  21
  22  23  24  25  26  27  28  29  31  32  33  34  35  36  37  38  41  42
  43  44  45  46  47  49  50  51  52  53  54  55  56  57  58  59  60  61
  63  65  67  69  70  71  72  73  76  77  78  79  80  81  82  83  84  85
  86  87  88  89  91  92  95  96  97  98  99 100 101 102 103 105 106 107
 108 109 110 111 112 116 118 120 121 122 124 127 128 129 130 132 133 134
 135 136 137 138 139 142 144 145 146 147 148 149] 
 TEST:
 [  1   5  12  17  30  39  40  48  62  64  66  68  74  75  90  93  94 104
 113 114 115 117 119 123 125 126 131 140 141 143]
'''
```

<br>

### Regualarization

* **alpha** 하이퍼파라미터를 사용하고, 값이 클수록 규제의 효과가 크다.  

* `Ridge`: alpha

  * L2 규제로, 가중치의 제곱항을 줄인다. 따라서 값이 큰 가중치들이 많이 줄어든다.
  * overfitting 을 완화(일반화)시키는 효과가 있다. 

  ![image-20211022231435458](https://user-images.githubusercontent.com/70505378/138479335-17d3c217-21eb-4145-badc-22866e0b0e4f.png)

  ![image-20211022231516937](https://user-images.githubusercontent.com/70505378/138479339-bf54849c-5cc9-49d7-9906-bbb89b957a2c.png)

* `Lasso`: alpha

  * L1 규제로, 가중치를 전체적으로 줄인다. 따라서 값이 작은 가중치들이 먼저 사라지고 값이 큰 가중치들이 살아남는다. 
  * feature selection 효과가 있다. 

  ![image-20211022231304308](https://user-images.githubusercontent.com/70505378/138479329-113565a3-bbca-4062-8103-13977378810d.png)

  ![image-20211022231536570](https://user-images.githubusercontent.com/70505378/138479342-5dece631-beaa-4a88-82af-ea3ef22f6e03.png)

```python
# Ridge(L2)
from sklearn.linear_model import Ridge

ridge = Ridge(alpha=1000)
ridge.fit(X, y_out)
y_ridge_pred = ridge.predict(X)
```

<br>

### Multi-class Classification

* `LogisticRegression`: multi_class, C, random_state
  * **multi_class = "multinomial"**로 지정하면 **Softmax 함수(+Cross Entropy)**를 사용하여 다중 분류를 수행한다. 
  * **multi_class = "ovr"**로 지정하면 **One vs Rest** 방법을 사용하여 다중 분류를 수행한다. 
  * **C**는 모델의 규제 강도의 역수로, 작을수록 규제가 강해진다(모델의 복잡도가 줄어든다). 

```python
softmax_reg = LogisticRegression(multi_class="multinomial", C=10, random_state=42)
ovr_clf = LogisticRegression(multi_class="ovr", C=10, random_state=42)

# 예측 수행 (특성 2개로 훈련된 모델)
softmax_reg.predict([[5, 2]]) # array([2])
softmax_reg.predict_proba([[5, 2]]) # array([[6.38014896e-07, 5.74929995e-02, 9.42506362e-01]])
```

<br>

<br>

## week 5

### Classification Performance

#### Static performance and Confusion matrix

**Confusion matrix** - accuracy(**model.score()**), precision, recall(sensitivity), f1-score

![image-20211022233716443](https://user-images.githubusercontent.com/70505378/138479345-99c01d80-c564-49e7-8612-63864639e8a6.png)

![image-20211022233735058](https://user-images.githubusercontent.com/70505378/138479348-df2dc8fd-7a2e-4f23-8ab1-139a9e59eef8.png)

* `confusion_matrix`: y_test, y_pred
* `classification_report`: y_test, y_pred

```python
y_pred=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0]
y_test=[1,1,0,1,0,1,1,1,0,0,1,0,1,1,0,1,0,0,0,0]

from sklearn.metrics import confusion_matrix, classification_report

confusion_matrix(y_test, y_pred)
'''
# x축(예측값): 0, 1
# y축(실제값): 0, 1
array([[5, 5],
       [1, 9]], dtype=int64)
'''
classification_report(y_test, y_pred)
'''
              precision    recall  f1-score   support

           0       0.83      0.50      0.62        10
           1       0.64      0.90      0.75        10

    accuracy                           0.70        20
   macro avg       0.74      0.70      0.69        20
weighted avg       0.74      0.70      0.69        20
'''
```

<br>

#### Dynamic Performance and ROC, AUC

**Ranking-based or Score-based**

![image-20211022234557761](https://user-images.githubusercontent.com/70505378/138479618-7de7786b-4a12-40eb-aaf4-69af3f8ffdb0.png)

* `fpr, tpr, threshold = roc_curve(y_test, y_score)`
* `auc(fpr, tpr)`

```python
y_score = np.linspace(99, 60, 20).round(1)
result = pd.DataFrame(list(zip(y_score, y_pred, y_test)), 
                      columns=['score', 'predict', 'real'])
result['correct'] = (result.predict == result.real)
'''

   score predict real correct
0	99.0	1	1		True
1	96.9	1	1		True
2	94.9	1	0		False
3	92.8	1	1		True
4	90.8	1	0		False
5	88.7	1	1		True
6	86.7	1	1		True
7	84.6	1	1		True
8	82.6	1	0		False
9	80.5	1	0		False
10	78.5	1	1		True
11	76.4	1	0		False
12	74.4	1	1		True
13	72.3	1	1		True
14	70.3	0	0		True
15	68.2	0	1		False
16	66.2	0	0		True
17	64.1	0	0		True
18	62.1	0	0		True
19	60.0	0	0		True
'''

fpr, tpr, thresholds1 = roc_curve(y_test, y_score)
roc_auc = auc(fpr, tpr)

pd.DataFrame([thresholds1, tpr, fpr], index=['threshold','tpr','fpr'])
```

![image-20211022234935885](https://user-images.githubusercontent.com/70505378/138479625-17ab9a89-edb0-4595-9ebb-2fc359d42386.png)

```python
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

![image-20211022235006684](https://user-images.githubusercontent.com/70505378/138479627-6eca4469-de88-4f63-b86f-9ea22263f2c9.png)

#### Precision-Recall Curve

* `precision, recall, threshold = precision_recall_curve(y_test, y_score)`

```python
from sklearn.metrics import precision_recall_curve, plot_precision_recall_curve

precision, recall, thresholds2 = precision_recall_curve(y_test, y_score)

auc_score = auc(recall, precision)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % auc_score)
plt.legend(loc="upper right")
```

![image-20211022235456187](https://user-images.githubusercontent.com/70505378/138479632-7a85e6d4-a6d4-45d7-8928-6547891c2e68.png)

<br>

대부분의 경우에 높은 AUC 값이 더 나은 모델 성능을 나타내지만, 데이터셋의 **불균형이 심한 경우** 신뢰할 수 없다. 

<br>

### Regression Performance

* MAE (mean absolute error): y_true, y_pred

* MSE (mean square error): y_true, y_pred

* RMSE (root mean square error)

* R-squared score (**model.score()**): y_true, y_pred

  ![image-20211023000237963](https://user-images.githubusercontent.com/70505378/138479634-473dbc6c-2deb-48af-ad8b-90d2e3987ec2.png)

```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```

<br>

<br>

