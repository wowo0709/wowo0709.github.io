---
layout: single
title: "[Machine Learning]end-to-end machine learning process"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: ['Regression']
---





<br>

<span style="color:red">**The full code is at the bottom!!!**</span>

<br>

## Setup

Import necessary libraries. 


```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import urllib.request
import tarfile
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import train_test_split
# from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
```

<br>

## Import dataset

Import [California housing dataset](https://www.kaggle.com/camnugent/california-housing-prices) and convert it to pandas dataframe format


```python
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
urllib.request.urlretrieve(url, "housing.tgz")
tar = tarfile.open("housing.tgz")
tar.extractall()
tar.close()
housing = pd.read_csv("housing.csv")

housing
```

<br>


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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>20635</th>
      <td>-121.09</td>
      <td>39.48</td>
      <td>25.0</td>
      <td>1665.0</td>
      <td>374.0</td>
      <td>845.0</td>
      <td>330.0</td>
      <td>1.5603</td>
      <td>78100.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20636</th>
      <td>-121.21</td>
      <td>39.49</td>
      <td>18.0</td>
      <td>697.0</td>
      <td>150.0</td>
      <td>356.0</td>
      <td>114.0</td>
      <td>2.5568</td>
      <td>77100.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20637</th>
      <td>-121.22</td>
      <td>39.43</td>
      <td>17.0</td>
      <td>2254.0</td>
      <td>485.0</td>
      <td>1007.0</td>
      <td>433.0</td>
      <td>1.7000</td>
      <td>92300.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20638</th>
      <td>-121.32</td>
      <td>39.43</td>
      <td>18.0</td>
      <td>1860.0</td>
      <td>409.0</td>
      <td>741.0</td>
      <td>349.0</td>
      <td>1.8672</td>
      <td>84700.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>20639</th>
      <td>-121.24</td>
      <td>39.37</td>
      <td>16.0</td>
      <td>2785.0</td>
      <td>616.0</td>
      <td>1387.0</td>
      <td>530.0</td>
      <td>2.3886</td>
      <td>89400.0</td>
      <td>INLAND</td>
    </tr>
  </tbody>
</table>
<p>20640 rows × 10 columns</p>
</div>

<br>

## Data Analysis


```python
housing.describe()
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>

<br>


```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB

<br>

<br>

## Split dataset to Training/Test Set


```python
X = housing.drop('median_house_value',axis=1)
y = housing['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    shuffle=True, 
                                                    random_state=42)

    
```


```python
X_train.head(10)
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14196</th>
      <td>-117.03</td>
      <td>32.71</td>
      <td>33.0</td>
      <td>3126.0</td>
      <td>627.0</td>
      <td>2300.0</td>
      <td>623.0</td>
      <td>3.2596</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>8267</th>
      <td>-118.16</td>
      <td>33.77</td>
      <td>49.0</td>
      <td>3382.0</td>
      <td>787.0</td>
      <td>1314.0</td>
      <td>756.0</td>
      <td>3.8125</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>17445</th>
      <td>-120.48</td>
      <td>34.66</td>
      <td>4.0</td>
      <td>1897.0</td>
      <td>331.0</td>
      <td>915.0</td>
      <td>336.0</td>
      <td>4.1563</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>14265</th>
      <td>-117.11</td>
      <td>32.69</td>
      <td>36.0</td>
      <td>1421.0</td>
      <td>367.0</td>
      <td>1418.0</td>
      <td>355.0</td>
      <td>1.9425</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>2271</th>
      <td>-119.80</td>
      <td>36.78</td>
      <td>43.0</td>
      <td>2382.0</td>
      <td>431.0</td>
      <td>874.0</td>
      <td>380.0</td>
      <td>3.5542</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>17848</th>
      <td>-121.86</td>
      <td>37.42</td>
      <td>20.0</td>
      <td>5032.0</td>
      <td>808.0</td>
      <td>2695.0</td>
      <td>801.0</td>
      <td>6.6227</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>6252</th>
      <td>-117.97</td>
      <td>34.04</td>
      <td>28.0</td>
      <td>1686.0</td>
      <td>417.0</td>
      <td>1355.0</td>
      <td>388.0</td>
      <td>2.5192</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>9389</th>
      <td>-122.53</td>
      <td>37.91</td>
      <td>37.0</td>
      <td>2524.0</td>
      <td>398.0</td>
      <td>999.0</td>
      <td>417.0</td>
      <td>7.9892</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>6113</th>
      <td>-117.90</td>
      <td>34.13</td>
      <td>5.0</td>
      <td>1126.0</td>
      <td>316.0</td>
      <td>819.0</td>
      <td>311.0</td>
      <td>1.5000</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>6061</th>
      <td>-117.79</td>
      <td>34.02</td>
      <td>5.0</td>
      <td>18690.0</td>
      <td>2862.0</td>
      <td>9427.0</td>
      <td>2777.0</td>
      <td>6.4266</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>

<br>

## Data Preprocessing

강의 내용과 다르게, 하나의 'district'에 대한 데이터를 하나의 'house'에 대한 데이터로 변환하지 않는다. 

이는 샘플이 하나의 'district' 단위이기 때문에 가구 당 평균치로 환산하는 것이 오히려 이 데이터에서는 성능을 떨어트릴 수 있기 때문이다. 

대신, NaN 값이 있는 'total bedroom' feature의 값을 대체한 뒤에 numerical data는 scaling을,  categorical data인 'ocean proximity' 컬럼은 one-hot encoding한다. 

### Imputing

'total_bedrooms' feature가 결측치를 갖기 때문에 상관도가 높고 결측치가 없는 다른 특성을 이용해 결측치를 채운다. 


```python
X_train.corr()['total_bedrooms'].sort_values(ascending=False)
```




    total_bedrooms        1.000000
    households            0.980255
    total_rooms           0.930489
    population            0.878932
    longitude             0.063064
    median_income        -0.009141
    latitude             -0.059998
    housing_median_age   -0.320624
    Name: total_bedrooms, dtype: float64




```python
class MyImputer():
    def __init__(self):
        self.proportion = 0
        
    def fit(self,features,labels,reset=True):
        tot_feature, tot_label = 0, 0
        for feature,label in zip(features,labels):
            if not np.isnan(feature) and not np.isnan(label):  
                tot_feature += feature
                tot_label += label
                
        if reset: self.proportion = tot_feature / tot_label
        else: self.proportion = (tot_feature / tot_label + self.proportion) / 2
        
        return
        
    def transform(self,features,labels):
        imputed_features = []
        for feature,label in zip(features,labels):
            if np.isnan(feature) and not np.isnan(label):
                imputed_features.append(round(label * self.proportion))
            else:
                imputed_features.append(feature)
                
        return imputed_features
                
    def fit_transform(self,features,labels,reset=True):
        self.fit(features,labels,reset)
        return self.transform(features,labels)
```


```python
imputer = MyImputer()

# correlation이 높은 'households' feature를 사용해 결측치 보간
X_train['total_bedrooms'] = imputer.fit_transform(X_train['total_bedrooms'], X_train['households'])
```

```python
X_train.head(10)
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14196</th>
      <td>-117.03</td>
      <td>32.71</td>
      <td>33.0</td>
      <td>3126.0</td>
      <td>627.0</td>
      <td>2300.0</td>
      <td>623.0</td>
      <td>3.2596</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>8267</th>
      <td>-118.16</td>
      <td>33.77</td>
      <td>49.0</td>
      <td>3382.0</td>
      <td>787.0</td>
      <td>1314.0</td>
      <td>756.0</td>
      <td>3.8125</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>17445</th>
      <td>-120.48</td>
      <td>34.66</td>
      <td>4.0</td>
      <td>1897.0</td>
      <td>331.0</td>
      <td>915.0</td>
      <td>336.0</td>
      <td>4.1563</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>14265</th>
      <td>-117.11</td>
      <td>32.69</td>
      <td>36.0</td>
      <td>1421.0</td>
      <td>367.0</td>
      <td>1418.0</td>
      <td>355.0</td>
      <td>1.9425</td>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>2271</th>
      <td>-119.80</td>
      <td>36.78</td>
      <td>43.0</td>
      <td>2382.0</td>
      <td>431.0</td>
      <td>874.0</td>
      <td>380.0</td>
      <td>3.5542</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>17848</th>
      <td>-121.86</td>
      <td>37.42</td>
      <td>20.0</td>
      <td>5032.0</td>
      <td>808.0</td>
      <td>2695.0</td>
      <td>801.0</td>
      <td>6.6227</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>6252</th>
      <td>-117.97</td>
      <td>34.04</td>
      <td>28.0</td>
      <td>1686.0</td>
      <td>417.0</td>
      <td>1355.0</td>
      <td>388.0</td>
      <td>2.5192</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>9389</th>
      <td>-122.53</td>
      <td>37.91</td>
      <td>37.0</td>
      <td>2524.0</td>
      <td>398.0</td>
      <td>999.0</td>
      <td>417.0</td>
      <td>7.9892</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>6113</th>
      <td>-117.90</td>
      <td>34.13</td>
      <td>5.0</td>
      <td>1126.0</td>
      <td>316.0</td>
      <td>819.0</td>
      <td>311.0</td>
      <td>1.5000</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>6061</th>
      <td>-117.79</td>
      <td>34.02</td>
      <td>5.0</td>
      <td>18690.0</td>
      <td>2862.0</td>
      <td>9427.0</td>
      <td>2777.0</td>
      <td>6.4266</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>

<br>

### Scaling

각 특성의 값이 0과 1 사이에 오도록 스케일링


```python
num_columns = list(X_train.columns[:-1])
print(num_columns)
num_attribs = X_train.drop('ocean_proximity', axis=1)
```

    ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income']



```python
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
scaled_num_attribs = scaler.fit_transform(num_attribs)

scaled_num_attribs[:10]
```




    array([[0.72908367, 0.01702128, 0.62745098, 0.0794547 , 0.09714463,
            0.06437961, 0.10228581, 0.19032151],
           [0.61653386, 0.12978723, 0.94117647, 0.08596572, 0.12197393,
            0.0367443 , 0.12415721, 0.22845202],
           [0.38545817, 0.22446809, 0.05882353, 0.04819675, 0.05121043,
            0.02556125, 0.05508962, 0.25216204],
           [0.72111554, 0.01489362, 0.68627451, 0.03609034, 0.05679702,
            0.03965918, 0.05821411, 0.09948828],
           [0.45318725, 0.45      , 0.82352941, 0.06053207, 0.06672874,
            0.02441212, 0.06232528, 0.21063847],
           [0.24800797, 0.51808511, 0.37254902, 0.12793123, 0.12523277,
            0.07545055, 0.13155731, 0.42225624],
           [0.63545817, 0.15851064, 0.52941176, 0.04283026, 0.06455618,
            0.03789344, 0.06364085, 0.13926015],
           [0.1812749 , 0.57021277, 0.70588235, 0.06414365, 0.0616077 ,
            0.02791558, 0.0684098 , 0.51649632],
           [0.64243028, 0.16808511, 0.07843137, 0.02858742, 0.04888268,
            0.0228706 , 0.05097846, 0.06897146],
           [0.65338645, 0.15638298, 0.07843137, 0.47530393, 0.4439789 ,
            0.26413296, 0.45650386, 0.40873229]])

<br>

### Encoding

Categorical data에 대해 one-hot encoding 수행


```python
cat_attribes = X_train['ocean_proximity']
```


```python
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder()
encoded_cat_attribs = encoder.fit_transform(X_train['ocean_proximity'].values.reshape(-1,1)).toarray()

one_hot_columns = list(*encoder.categories_)
print(one_hot_columns)
encoded_cat_attribs[:10]
```

    ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN']
    
    array([[0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 1.],
           [0., 0., 0., 0., 1.],
           [0., 1., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0.],
           [1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.]])

<br>

### Put together


```python
preprocessed_X_train_array = np.hstack([scaled_num_attribs,encoded_cat_attribs])

columns = num_columns + one_hot_columns
preprocessed_X_train = pd.DataFrame(preprocessed_X_train_array, columns=columns)   

preprocessed_X_train[:10]
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
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>&lt;1H OCEAN</th>
      <th>INLAND</th>
      <th>ISLAND</th>
      <th>NEAR BAY</th>
      <th>NEAR OCEAN</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.729084</td>
      <td>0.017021</td>
      <td>0.627451</td>
      <td>0.079455</td>
      <td>0.097145</td>
      <td>0.064380</td>
      <td>0.102286</td>
      <td>0.190322</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.616534</td>
      <td>0.129787</td>
      <td>0.941176</td>
      <td>0.085966</td>
      <td>0.121974</td>
      <td>0.036744</td>
      <td>0.124157</td>
      <td>0.228452</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.385458</td>
      <td>0.224468</td>
      <td>0.058824</td>
      <td>0.048197</td>
      <td>0.051210</td>
      <td>0.025561</td>
      <td>0.055090</td>
      <td>0.252162</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.721116</td>
      <td>0.014894</td>
      <td>0.686275</td>
      <td>0.036090</td>
      <td>0.056797</td>
      <td>0.039659</td>
      <td>0.058214</td>
      <td>0.099488</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.453187</td>
      <td>0.450000</td>
      <td>0.823529</td>
      <td>0.060532</td>
      <td>0.066729</td>
      <td>0.024412</td>
      <td>0.062325</td>
      <td>0.210638</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.248008</td>
      <td>0.518085</td>
      <td>0.372549</td>
      <td>0.127931</td>
      <td>0.125233</td>
      <td>0.075451</td>
      <td>0.131557</td>
      <td>0.422256</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.635458</td>
      <td>0.158511</td>
      <td>0.529412</td>
      <td>0.042830</td>
      <td>0.064556</td>
      <td>0.037893</td>
      <td>0.063641</td>
      <td>0.139260</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.181275</td>
      <td>0.570213</td>
      <td>0.705882</td>
      <td>0.064144</td>
      <td>0.061608</td>
      <td>0.027916</td>
      <td>0.068410</td>
      <td>0.516496</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.642430</td>
      <td>0.168085</td>
      <td>0.078431</td>
      <td>0.028587</td>
      <td>0.048883</td>
      <td>0.022871</td>
      <td>0.050978</td>
      <td>0.068971</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.653386</td>
      <td>0.156383</td>
      <td>0.078431</td>
      <td>0.475304</td>
      <td>0.443979</td>
      <td>0.264133</td>
      <td>0.456504</td>
      <td>0.408732</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>

<br>

<br>

## Model Training


```python
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor()
model.fit(preprocessed_X_train, y_train)

housing_predictions = model.predict(preprocessed_X_train)
train_rmse = mean_squared_error(y_train, housing_predictions)**(1/2)
train_r2 = r2_score(y_train, housing_predictions)
train_score = model.score(preprocessed_X_train, housing_predictions)

train_rmse, train_r2, train_score
```




    (18040.92581263233, 0.975652280886504, 1.0)

<br>

## Model Evaluation


```python
# correlation이 높은 'households' feature를 사용해 결측치 보간
X_test['total_bedrooms'] = imputer.transform(X_test['total_bedrooms'], X_test['households'])


columns = X_test.columns[:-1]
num_attribs, cat_attribs = X_test.drop('ocean_proximity', axis=1), X_test['ocean_proximity']


scaled_num_attribs = scaler.transform(num_attribs)


one_hot_columns = list(*encoder.categories_)
encoded_cat_attribs = encoder.transform(X_test['ocean_proximity'].values.reshape(-1,1)).toarray()


preprocessed_X_test = pd.DataFrame(np.hstack([scaled_num_attribs,encoded_cat_attribs]),columns=list(columns)+list(one_hot_columns))


final_predictions = model.predict(preprocessed_X_test)

test_rmse = mean_squared_error(y_test, final_predictions)**(1/2)
test_r2 = r2_score(y_test, final_predictions)
test_score = model.score(preprocessed_X_test, y_test)

test_rmse, test_r2, test_score
```

    (48969.517591501055, 0.8170026539070696, 0.8170026539070696)

<br>

<br>

## Full code


```python
'''1. Setup'''
import numpy as np
import pandas as pd
%matplotlib inline
import urllib.request
import tarfile
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor


'''2. Import dataset'''
url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz"
urllib.request.urlretrieve(url, "housing.tgz")
tar = tarfile.open("housing.tgz")
tar.extractall()
tar.close()
housing = pd.read_csv("housing.csv")


'''3. Split Train/Test dataset'''
X = housing.drop('median_house_value',axis=1)
y = housing['median_house_value']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    shuffle=True, 
                                                    random_state=42)


'''4. Data Preprocessing'''
# Imputing
class MyImputer():
    def __init__(self):
        self.proportion = 0
        
    def fit(self,features,labels,reset=True):
        tot_feature, tot_label = 0, 0
        for feature,label in zip(features,labels):
            if not np.isnan(feature) and not np.isnan(label):  
                tot_feature += feature
                tot_label += label
                
        if reset: self.proportion = tot_feature / tot_label
        else: self.proportion = (tot_feature / tot_label + self.proportion) / 2
        
        return
        
    def transform(self,features,labels):
        imputed_features = []
        for feature,label in zip(features,labels):
            if np.isnan(feature) and not np.isnan(label):
                imputed_features.append(round(label * self.proportion))
            else:
                imputed_features.append(feature)
                
        return imputed_features
                
    def fit_transform(self,features,labels,reset=True):
        self.fit(features,labels,reset)
        return self.transform(features,labels)
    
imputer = MyImputer()
X_train['total_bedrooms'] = imputer.fit_transform(X_train['total_bedrooms'], X_train['households'])

# Scaling
num_columns = list(X_train.columns[:-1])
num_attribs = X_train.drop('ocean_proximity', axis=1)

scaler = MinMaxScaler()
scaled_num_attribs = scaler.fit_transform(num_attribs)

# Encoding
cat_attribes = X_train['ocean_proximity']

encoder = OneHotEncoder()
encoded_cat_attribs = encoder.fit_transform(X_train['ocean_proximity'].values.reshape(-1,1)).toarray()

one_hot_columns = list(*encoder.categories_)

# Put together
preprocessed_X_train_array = np.hstack([scaled_num_attribs,encoded_cat_attribs])

columns = num_columns + one_hot_columns
preprocessed_X_train = pd.DataFrame(preprocessed_X_train_array, columns=columns) 



'''5. Model Training'''
model = RandomForestRegressor()
model.fit(preprocessed_X_train, y_train)



'''6. Model Evaluation'''
# Imputing
X_test['total_bedrooms'] = imputer.transform(X_test['total_bedrooms'], X_test['households'])

# Scaling/Encoding
num_columns = list(X_test.columns[:-1])
num_attribs, cat_attribs = X_test.drop('ocean_proximity', axis=1), X_test['ocean_proximity']

scaled_num_attribs = scaler.transform(num_attribs)

encoded_cat_attribs = encoder.transform(X_test['ocean_proximity'].values.reshape(-1,1)).toarray()
one_hot_columns = list(*encoder.categories_)

columns = num_columns + one_hot_columns
preprocessed_X_test = pd.DataFrame(np.hstack([scaled_num_attribs,encoded_cat_attribs]),columns=columns)

# Prediction
final_predictions = model.predict(preprocessed_X_test)

test_rmse = mean_squared_error(y_test, final_predictions)**(1/2)
test_score = model.score(preprocessed_X_test, y_test)

print("rmse: {}\ttest score: {}%".format(round(test_rmse,2), round(test_score*100,2)))
```


    rmse: 48841.13	test score: 81.8%

<br>

## Discussion

간단하게 전체 코드의 흐름을 설명하겠습니다. 

### 1. Setup
필요한 모듈/클래스들을 **import** 합니다. 

* <span style="color:black">_Basics_</span>: **numpy, pandas**
* <span style="color:blue">_Import dataset_</span>: **urllib.request, tarfile**
* <span style="color:skyblue">_Spliting_</span>: **train_test_split**
* <span style="color:green">_Preprocessing_</span>
    * _Imputing_: **(Customized)**
    * _Scaling_: **MinMaxScaler**
    * _Encoding_: **OntHotEncoder**
* <span style="color:red">_Model_</span>: **RandomForestRegressor**
* <span style="color:purple">_Evaluation_</span>: **mean_squared_error**

<br>

### <span style="color:blue">_2. Import dataset_</span>
**california housing dataset**을 가져옵니다. 

<br>

### <span style="color:skyblue">_3. Split Train/Test dataset_</span>
가져온 housing 데이터셋을 **train/test 데이터셋으로 분리**합니다. 
Train set과 Test set의 분포를 비슷하게 가져가기 위해 `shuffle=True`로 설정합니다. 

`test_size` 는 전체 데이터의 20%로 설정합니다. 

<br>

### <span style="color:green">_4. Data Preprocessing_</span>
#### _4.1 Imputing_
기존에 존재하는 Imputing 클래스를 사용하지 않고 **직접 구현**했습니다. 

1. 먼저 `corr()` 메서드로 결측치를 보간할 feature와 correlation이 가장 높은 feature를 찾습니다. 
2. 결측치가 있는 feature와 앞서 구한 feature를 이용해 두 feature 사이의 proportion을 구합니다. 
3. 앞서 구한 proportion과 feature를 이용해 결측치가 있는 feature의 값을 보간합니다.

이로써 결측치를 가치있는 값으로 대체할 수 있습니다. 


#### _4.2 Scaling_
Standard scaling과 Min-max scaling 중 회귀에 조금 더 적합한 **`Min-max scaling`**을 선택했습니다. 


#### _4.3 Encoding_
categorical feature인 'ocean_proximity' 를 **원-핫 인코딩**합니다. 

<br>

### <span style="color:red">_5. Model Training_</span>
**모델을 선택**하고 `fit()` 메서드로 **훈련**합니다. 

모델로는 `RandomForestRegressor` 를 선택했습니다. 

<br>

### <span style="color:purple">_6. Model Evaluation_</span>
Test set에 대해 앞에서 수행했던 전처리를 똑같이 수행하고, 훈련된 모델로 **예측 및 평가**를 진행합니다. 

