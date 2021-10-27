---
layout: single
title: "[Machine Learning] Clustering"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: ['Agglomerative', 'K-Means', 'DBSCAN']
---

<br>

## Clustering (군집화)

- data from 전력거래소
- 전력판매량(시도별/용도별) 액셀 파일: https://goo.gl/Cx8Rzw
- Agglomerative, KMeans, DBSCAN


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
```

### data read


```python
!curl -L "https://goo.gl/Cx8Rzw" -o power_data.xls
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
      0     0    0     0    0     0      0      0 --:--:-- --:--:-- --:--:--     0
    
    100   184    0   184    0     0    159      0 --:--:--  0:00:01 --:--:--   159
    
    100   318  100   318    0     0    219      0  0:00:01  0:00:01 --:--:--   219
    
      0     0    0     0    0     0      0      0 --:--:--  0:00:01 --:--:--     0
    100  1076    0  1076    0     0    455      0 --:--:--  0:00:02 --:--:--  1259
    
      0     0    0     0    0     0      0      0 --:--:--  0:00:03 --:--:--     0
    100 17920  100 17920    0     0   5619      0  0:00:03  0:00:03 --:--:-- 1750k

<br>

```python
df = pd.read_excel('power_data.xls')
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
      <th>구분</th>
      <th>주거용</th>
      <th>공공용</th>
      <th>서비스업</th>
      <th>업무용합계</th>
      <th>농림어업</th>
      <th>광업</th>
      <th>제조업</th>
      <th>식료품제조</th>
      <th>섬유,의류</th>
      <th>...</th>
      <th>기타기계</th>
      <th>사무기기</th>
      <th>전기기기</th>
      <th>영상,음향</th>
      <th>자동차</th>
      <th>기타운송</th>
      <th>가구및기타</th>
      <th>재생재료</th>
      <th>산업용합계</th>
      <th>합계</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>강원</td>
      <td>1940933</td>
      <td>1400421</td>
      <td>6203749</td>
      <td>7604170</td>
      <td>607139</td>
      <td>398287</td>
      <td>6002286</td>
      <td>546621</td>
      <td>13027</td>
      <td>...</td>
      <td>35063</td>
      <td>2019</td>
      <td>38062</td>
      <td>43986</td>
      <td>113448</td>
      <td>108629</td>
      <td>12872</td>
      <td>3418</td>
      <td>7007712</td>
      <td>16552816</td>
    </tr>
    <tr>
      <th>1</th>
      <td>개성</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>경기</td>
      <td>16587710</td>
      <td>5533662</td>
      <td>33434551</td>
      <td>38968213</td>
      <td>2371347</td>
      <td>317263</td>
      <td>56603327</td>
      <td>2544420</td>
      <td>2109963</td>
      <td>...</td>
      <td>3613798</td>
      <td>317244</td>
      <td>1040171</td>
      <td>24519644</td>
      <td>2977165</td>
      <td>67594</td>
      <td>1833112</td>
      <td>133041</td>
      <td>59291937</td>
      <td>114847859</td>
    </tr>
    <tr>
      <th>3</th>
      <td>경남</td>
      <td>4260988</td>
      <td>1427560</td>
      <td>8667737</td>
      <td>10095297</td>
      <td>2141813</td>
      <td>95989</td>
      <td>18053778</td>
      <td>932743</td>
      <td>346974</td>
      <td>...</td>
      <td>1902913</td>
      <td>8070</td>
      <td>924235</td>
      <td>534196</td>
      <td>2156059</td>
      <td>2048646</td>
      <td>262523</td>
      <td>47662</td>
      <td>20291580</td>
      <td>34647864</td>
    </tr>
    <tr>
      <th>4</th>
      <td>경북</td>
      <td>3302463</td>
      <td>1578115</td>
      <td>8487402</td>
      <td>10065517</td>
      <td>1747462</td>
      <td>224568</td>
      <td>30115601</td>
      <td>566071</td>
      <td>3780171</td>
      <td>...</td>
      <td>782570</td>
      <td>14468</td>
      <td>750786</td>
      <td>4174971</td>
      <td>2356890</td>
      <td>123935</td>
      <td>60280</td>
      <td>77104</td>
      <td>32087631</td>
      <td>45455611</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>

<br>


```python
print(df.columns)
print(df.index)
```

    Index(['구분', '주거용', '공공용', '서비스업', '업무용합계', '농림어업', '광업', '제조업', '식료품제조',
           '섬유,의류', '목재,나무', '펄프,종이', '출판,인쇄', '석유,화확', '의료,광학', '요업', '1차금속',
           '조립금속', '기타기계', '사무기기', '전기기기', '영상,음향', '자동차', '기타운송', '가구및기타', '재생재료',
           '산업용합계', '합계'],
          dtype='object')
    RangeIndex(start=0, stop=19, step=1)

```python
df.shape
```


    (19, 28)


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 19 entries, 0 to 18
    Data columns (total 28 columns):
     #   Column  Non-Null Count  Dtype 
    ---  ------  --------------  ----- 
     0   구분      19 non-null     object
     1   주거용     19 non-null     int64 
     2   공공용     19 non-null     int64 
     3   서비스업    19 non-null     int64 
     4   업무용합계   19 non-null     int64 
     5   농림어업    19 non-null     int64 
     6   광업      19 non-null     int64 
     7   제조업     19 non-null     int64 
     8   식료품제조   19 non-null     int64 
     9   섬유,의류   19 non-null     int64 
     10  목재,나무   19 non-null     int64 
     11  펄프,종이   19 non-null     int64 
     12  출판,인쇄   19 non-null     int64 
     13  석유,화확   19 non-null     int64 
     14  의료,광학   19 non-null     int64 
     15  요업      19 non-null     int64 
     16  1차금속    19 non-null     int64 
     17  조립금속    19 non-null     int64 
     18  기타기계    19 non-null     int64 
     19  사무기기    19 non-null     int64 
     20  전기기기    19 non-null     int64 
     21  영상,음향   19 non-null     int64 
     22  자동차     19 non-null     int64 
     23  기타운송    19 non-null     int64 
     24  가구및기타   19 non-null     int64 
     25  재생재료    19 non-null     int64 
     26  산업용합계   19 non-null     int64 
     27  합계      19 non-null     int64 
    dtypes: int64(27), object(1)
    memory usage: 4.3+ KB




```python
df.describe()
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
      <th>주거용</th>
      <th>공공용</th>
      <th>서비스업</th>
      <th>업무용합계</th>
      <th>농림어업</th>
      <th>광업</th>
      <th>제조업</th>
      <th>식료품제조</th>
      <th>섬유,의류</th>
      <th>목재,나무</th>
      <th>...</th>
      <th>기타기계</th>
      <th>사무기기</th>
      <th>전기기기</th>
      <th>영상,음향</th>
      <th>자동차</th>
      <th>기타운송</th>
      <th>가구및기타</th>
      <th>재생재료</th>
      <th>산업용합계</th>
      <th>합계</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
      <td>...</td>
      <td>1.900000e+01</td>
      <td>19.000000</td>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
      <td>19.000000</td>
      <td>1.900000e+01</td>
      <td>1.900000e+01</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.899673e+06</td>
      <td>2.410981e+06</td>
      <td>1.451057e+07</td>
      <td>1.692155e+07</td>
      <td>1.650270e+06</td>
      <td>1.628526e+05</td>
      <td>2.694144e+07</td>
      <td>1.158857e+06</td>
      <td>1.184641e+06</td>
      <td>2.016269e+05</td>
      <td>...</td>
      <td>1.107597e+06</td>
      <td>51397.000000</td>
      <td>6.087239e+05</td>
      <td>5.018716e+06</td>
      <td>1.878618e+06</td>
      <td>4.595993e+05</td>
      <td>3.581517e+05</td>
      <td>59117.789474</td>
      <td>2.875456e+07</td>
      <td>5.257579e+07</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.457381e+07</td>
      <td>4.957221e+06</td>
      <td>3.031208e+07</td>
      <td>3.526148e+07</td>
      <td>3.464035e+06</td>
      <td>3.102484e+05</td>
      <td>5.669154e+07</td>
      <td>2.399623e+06</td>
      <td>2.604338e+06</td>
      <td>4.514169e+05</td>
      <td>...</td>
      <td>2.437518e+06</td>
      <td>127516.047494</td>
      <td>1.300814e+06</td>
      <td>1.190370e+07</td>
      <td>3.981244e+06</td>
      <td>1.058183e+06</td>
      <td>8.481672e+05</td>
      <td>126709.246048</td>
      <td>6.037040e+07</td>
      <td>1.092609e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>...</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>1.906912e+06</td>
      <td>6.959615e+05</td>
      <td>3.802654e+06</td>
      <td>4.524926e+06</td>
      <td>7.203850e+04</td>
      <td>9.938500e+03</td>
      <td>2.759556e+06</td>
      <td>1.959545e+05</td>
      <td>7.325600e+04</td>
      <td>5.590500e+03</td>
      <td>...</td>
      <td>7.220050e+04</td>
      <td>3672.000000</td>
      <td>6.083550e+04</td>
      <td>4.510550e+04</td>
      <td>9.622850e+04</td>
      <td>1.154750e+04</td>
      <td>1.313200e+04</td>
      <td>2739.000000</td>
      <td>2.814293e+06</td>
      <td>1.240509e+07</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2.326183e+06</td>
      <td>1.089613e+06</td>
      <td>5.690659e+06</td>
      <td>6.654683e+06</td>
      <td>6.071390e+05</td>
      <td>7.152900e+04</td>
      <td>1.236782e+07</td>
      <td>5.329430e+05</td>
      <td>3.338460e+05</td>
      <td>2.799800e+04</td>
      <td>...</td>
      <td>1.988470e+05</td>
      <td>7240.000000</td>
      <td>1.785020e+05</td>
      <td>4.200050e+05</td>
      <td>6.128980e+05</td>
      <td>6.812700e+04</td>
      <td>4.181400e+04</td>
      <td>19725.000000</td>
      <td>1.258230e+07</td>
      <td>2.451531e+07</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.058920e+06</td>
      <td>1.413990e+06</td>
      <td>8.034786e+06</td>
      <td>9.476781e+06</td>
      <td>1.837764e+06</td>
      <td>1.822120e+05</td>
      <td>2.366853e+07</td>
      <td>1.034889e+06</td>
      <td>8.374750e+05</td>
      <td>1.033945e+05</td>
      <td>...</td>
      <td>8.433595e+05</td>
      <td>14393.500000</td>
      <td>5.898460e+05</td>
      <td>2.614198e+06</td>
      <td>2.256474e+06</td>
      <td>1.775380e+05</td>
      <td>1.976150e+05</td>
      <td>46850.000000</td>
      <td>2.530336e+07</td>
      <td>4.005174e+07</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.457642e+07</td>
      <td>2.220411e+07</td>
      <td>1.347485e+08</td>
      <td>1.569527e+08</td>
      <td>1.537399e+07</td>
      <td>1.347957e+06</td>
      <td>2.529425e+08</td>
      <td>1.073583e+07</td>
      <td>1.124758e+07</td>
      <td>1.905882e+06</td>
      <td>...</td>
      <td>1.050464e+07</td>
      <td>487262.000000</td>
      <td>5.763846e+06</td>
      <td>4.765581e+07</td>
      <td>1.779015e+07</td>
      <td>4.311878e+06</td>
      <td>3.396006e+06</td>
      <td>559909.000000</td>
      <td>2.696645e+08</td>
      <td>4.911936e+08</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 27 columns</p>

</div>

<br>

```python
df.tail()
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
      <th>구분</th>
      <th>주거용</th>
      <th>공공용</th>
      <th>서비스업</th>
      <th>업무용합계</th>
      <th>농림어업</th>
      <th>광업</th>
      <th>제조업</th>
      <th>식료품제조</th>
      <th>섬유,의류</th>
      <th>...</th>
      <th>기타기계</th>
      <th>사무기기</th>
      <th>전기기기</th>
      <th>영상,음향</th>
      <th>자동차</th>
      <th>기타운송</th>
      <th>가구및기타</th>
      <th>재생재료</th>
      <th>산업용합계</th>
      <th>합계</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14</th>
      <td>전북</td>
      <td>2326183</td>
      <td>1096968</td>
      <td>4910318</td>
      <td>6007286</td>
      <td>1415004</td>
      <td>85300</td>
      <td>12965875</td>
      <td>1459217</td>
      <td>731651</td>
      <td>...</td>
      <td>159699</td>
      <td>7240</td>
      <td>130692</td>
      <td>420005</td>
      <td>859741</td>
      <td>70980</td>
      <td>16175</td>
      <td>99003</td>
      <td>14466179</td>
      <td>22799647</td>
    </tr>
    <tr>
      <th>15</th>
      <td>제주</td>
      <td>782601</td>
      <td>301727</td>
      <td>2308732</td>
      <td>2610459</td>
      <td>1364930</td>
      <td>14019</td>
      <td>241537</td>
      <td>155987</td>
      <td>3497</td>
      <td>...</td>
      <td>1167</td>
      <td>0</td>
      <td>771</td>
      <td>0</td>
      <td>773</td>
      <td>532</td>
      <td>1743</td>
      <td>743</td>
      <td>1620486</td>
      <td>5013545</td>
    </tr>
    <tr>
      <th>16</th>
      <td>충남</td>
      <td>2691823</td>
      <td>1089613</td>
      <td>7164439</td>
      <td>8254052</td>
      <td>1928066</td>
      <td>248313</td>
      <td>37057955</td>
      <td>1137035</td>
      <td>269998</td>
      <td>...</td>
      <td>611925</td>
      <td>12208</td>
      <td>428906</td>
      <td>10953811</td>
      <td>2526658</td>
      <td>33766</td>
      <td>53804</td>
      <td>19725</td>
      <td>39234334</td>
      <td>50180209</td>
    </tr>
    <tr>
      <th>17</th>
      <td>충북</td>
      <td>2027281</td>
      <td>1267140</td>
      <td>4804638</td>
      <td>6071778</td>
      <td>721131</td>
      <td>139856</td>
      <td>15883448</td>
      <td>1152073</td>
      <td>333846</td>
      <td>...</td>
      <td>366871</td>
      <td>23076</td>
      <td>1125141</td>
      <td>4103832</td>
      <td>603349</td>
      <td>82496</td>
      <td>513501</td>
      <td>46038</td>
      <td>16744435</td>
      <td>24843494</td>
    </tr>
    <tr>
      <th>18</th>
      <td>합계</td>
      <td>64576423</td>
      <td>22204112</td>
      <td>134748546</td>
      <td>156952658</td>
      <td>15373994</td>
      <td>1347957</td>
      <td>252942540</td>
      <td>10735833</td>
      <td>11247578</td>
      <td>...</td>
      <td>10504640</td>
      <td>487262</td>
      <td>5763846</td>
      <td>47655808</td>
      <td>17790147</td>
      <td>4311878</td>
      <td>3396006</td>
      <td>559909</td>
      <td>269664491</td>
      <td>491193571</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>


```python
df = df.set_index('구분')
df = df.drop(['합계', '개성'], errors='ignore')
df.shape
```


    (17, 27)


```python
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
      <th>주거용</th>
      <th>공공용</th>
      <th>서비스업</th>
      <th>업무용합계</th>
      <th>농림어업</th>
      <th>광업</th>
      <th>제조업</th>
      <th>식료품제조</th>
      <th>섬유,의류</th>
      <th>목재,나무</th>
      <th>...</th>
      <th>기타기계</th>
      <th>사무기기</th>
      <th>전기기기</th>
      <th>영상,음향</th>
      <th>자동차</th>
      <th>기타운송</th>
      <th>가구및기타</th>
      <th>재생재료</th>
      <th>산업용합계</th>
      <th>합계</th>
    </tr>
    <tr>
      <th>구분</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>강원</th>
      <td>1940933</td>
      <td>1400421</td>
      <td>6203749</td>
      <td>7604170</td>
      <td>607139</td>
      <td>398287</td>
      <td>6002286</td>
      <td>546621</td>
      <td>13027</td>
      <td>19147</td>
      <td>...</td>
      <td>35063</td>
      <td>2019</td>
      <td>38062</td>
      <td>43986</td>
      <td>113448</td>
      <td>108629</td>
      <td>12872</td>
      <td>3418</td>
      <td>7007712</td>
      <td>16552816</td>
    </tr>
    <tr>
      <th>경기</th>
      <td>16587710</td>
      <td>5533662</td>
      <td>33434551</td>
      <td>38968213</td>
      <td>2371347</td>
      <td>317263</td>
      <td>56603327</td>
      <td>2544420</td>
      <td>2109963</td>
      <td>529274</td>
      <td>...</td>
      <td>3613798</td>
      <td>317244</td>
      <td>1040171</td>
      <td>24519644</td>
      <td>2977165</td>
      <td>67594</td>
      <td>1833112</td>
      <td>133041</td>
      <td>59291937</td>
      <td>114847859</td>
    </tr>
    <tr>
      <th>경남</th>
      <td>4260988</td>
      <td>1427560</td>
      <td>8667737</td>
      <td>10095297</td>
      <td>2141813</td>
      <td>95989</td>
      <td>18053778</td>
      <td>932743</td>
      <td>346974</td>
      <td>60160</td>
      <td>...</td>
      <td>1902913</td>
      <td>8070</td>
      <td>924235</td>
      <td>534196</td>
      <td>2156059</td>
      <td>2048646</td>
      <td>262523</td>
      <td>47662</td>
      <td>20291580</td>
      <td>34647864</td>
    </tr>
    <tr>
      <th>경북</th>
      <td>3302463</td>
      <td>1578115</td>
      <td>8487402</td>
      <td>10065517</td>
      <td>1747462</td>
      <td>224568</td>
      <td>30115601</td>
      <td>566071</td>
      <td>3780171</td>
      <td>72680</td>
      <td>...</td>
      <td>782570</td>
      <td>14468</td>
      <td>750786</td>
      <td>4174971</td>
      <td>2356890</td>
      <td>123935</td>
      <td>60280</td>
      <td>77104</td>
      <td>32087631</td>
      <td>45455611</td>
    </tr>
    <tr>
      <th>광주</th>
      <td>1954876</td>
      <td>565527</td>
      <td>3174973</td>
      <td>3740500</td>
      <td>74608</td>
      <td>2898</td>
      <td>2910768</td>
      <td>161072</td>
      <td>295922</td>
      <td>6782</td>
      <td>...</td>
      <td>198847</td>
      <td>5967</td>
      <td>236622</td>
      <td>723764</td>
      <td>512148</td>
      <td>5140</td>
      <td>13392</td>
      <td>16049</td>
      <td>2988274</td>
      <td>8683649</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>

<br>

### Fonts for Korean Letters (한글폰트) 


```python
# Colab 에서 한글 폰트 설정 - 설중 후에 꼭 다시 runtime restart 해 주어야 함
import matplotlib as mpl
import matplotlib.pyplot as plt
 
%config InlineBackend.figure_format = 'retina'
 
!apt -qq -y install fonts-nanum
 
import matplotlib.font_manager as fm
fontpath = '/usr/share/fonts/truetype/nanum/NanumBarunGothic.ttf'
font = fm.FontProperties(fname=fontpath, size=9)
plt.rc('font', family='NanumBarunGothic') 
mpl.font_manager._rebuild()

```

    'apt'은(는) 내부 또는 외부 명령, 실행할 수 있는 프로그램, 또는
    배치 파일이 아닙니다.

```python
# '-' 기호 보이게 하기
import platform
import matplotlib
from matplotlib import font_manager, rc
matplotlib.rcParams['axes.unicode_minus'] = False
```


```python
import platform
import matplotlib
from matplotlib import font_manager, rc

# '-' 기호 보이게 하기
matplotlib.rcParams['axes.unicode_minus'] = False

# 운영 체제마다 한글이 보이게 하는 설정
if platform.system() == 'Windows':
    path = "c:\Windows\Fonts\malgun.ttf"
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font', family=font_name)
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
elif platform.system() == 'Linux':
    rc('font', family='NanumBarunGothic')
```


```python
df.head(2)
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
      <th>주거용</th>
      <th>공공용</th>
      <th>서비스업</th>
      <th>업무용합계</th>
      <th>농림어업</th>
      <th>광업</th>
      <th>제조업</th>
      <th>식료품제조</th>
      <th>섬유,의류</th>
      <th>목재,나무</th>
      <th>...</th>
      <th>기타기계</th>
      <th>사무기기</th>
      <th>전기기기</th>
      <th>영상,음향</th>
      <th>자동차</th>
      <th>기타운송</th>
      <th>가구및기타</th>
      <th>재생재료</th>
      <th>산업용합계</th>
      <th>합계</th>
    </tr>
    <tr>
      <th>구분</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>강원</th>
      <td>1940933</td>
      <td>1400421</td>
      <td>6203749</td>
      <td>7604170</td>
      <td>607139</td>
      <td>398287</td>
      <td>6002286</td>
      <td>546621</td>
      <td>13027</td>
      <td>19147</td>
      <td>...</td>
      <td>35063</td>
      <td>2019</td>
      <td>38062</td>
      <td>43986</td>
      <td>113448</td>
      <td>108629</td>
      <td>12872</td>
      <td>3418</td>
      <td>7007712</td>
      <td>16552816</td>
    </tr>
    <tr>
      <th>경기</th>
      <td>16587710</td>
      <td>5533662</td>
      <td>33434551</td>
      <td>38968213</td>
      <td>2371347</td>
      <td>317263</td>
      <td>56603327</td>
      <td>2544420</td>
      <td>2109963</td>
      <td>529274</td>
      <td>...</td>
      <td>3613798</td>
      <td>317244</td>
      <td>1040171</td>
      <td>24519644</td>
      <td>2977165</td>
      <td>67594</td>
      <td>1833112</td>
      <td>133041</td>
      <td>59291937</td>
      <td>114847859</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 27 columns</p>
</div>

<br>


```python
df = df.drop("합계",axis=1)
```


```python
df.plot(kind='barh', figsize=(10,6), stacked=True)
```




<img width="614" alt="output_18_1" src="https://user-images.githubusercontent.com/70505378/139006567-ed8ebc39-c008-486b-b60f-e8850b83dedf.png">
    



```python
see_c = ['서비스업','제조업']
df[see_c].plot(kind='barh', figsize=(10,6), stacked=True)
```




<img width="614" alt="output_19_1" src="https://user-images.githubusercontent.com/70505378/139006570-99dbda43-9a53-4ec2-99b4-43e6b82dcfb3.png">
    

<br>

```python
df2 = df[see_c]
df2.head(5)
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
      <th>서비스업</th>
      <th>제조업</th>
    </tr>
    <tr>
      <th>구분</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>강원</th>
      <td>6203749</td>
      <td>6002286</td>
    </tr>
    <tr>
      <th>경기</th>
      <td>33434551</td>
      <td>56603327</td>
    </tr>
    <tr>
      <th>경남</th>
      <td>8667737</td>
      <td>18053778</td>
    </tr>
    <tr>
      <th>경북</th>
      <td>8487402</td>
      <td>30115601</td>
    </tr>
    <tr>
      <th>광주</th>
      <td>3174973</td>
      <td>2910768</td>
    </tr>
  </tbody>
</table>
</div>

<br>

### scatter plot


```python
plt.figure(figsize=(6,6))
plt.scatter(df2['서비스업'], df2['제조업'],c='k',marker='o')
plt.xlabel('서비스업')
plt.ylabel('제조업')

for n in range(df2.shape[0]):
    plt.text(df2['서비스업'][n]*1.03, df2['제조업'][n]*0.98, df2.index[n])
```


<img width="390" alt="output_22_0" src="https://user-images.githubusercontent.com/70505378/139006573-e09a7429-915f-4892-a99d-616a5a3aee05.png">
    



```python
# drop outlier
df2 = df2.drop(['경기', '서울'])
df2.shape
```


    (15, 2)


```python
plt.figure(figsize=(6,6))
plt.scatter(df2['서비스업'], df2['제조업'],c='k',marker='o')
plt.xlabel('서비스업')
plt.ylabel('제조업')

for n in range(df2.shape[0]):
    plt.text(df2['서비스업'][n]*1.03, df2['제조업'][n]*0.98, df2.index[n])
```


<img width="399" alt="output_24_0" src="https://user-images.githubusercontent.com/70505378/139006575-b7ffdd07-d90a-441b-b85a-8dd4b2302e78.png">
    

<br>

<br>

## Agglomerative clustering (Hierarchical) and Dendrogram

- `linkage`: dataframe, metric, method


```python
from scipy.cluster.hierarchy import dendrogram, linkage

plt.figure(figsize=(10, 5))
link_dist = linkage(df2, metric='euclidean', method='centroid')
link_dist
```


    array([[0.00000000e+00, 4.00000000e+00, 7.46490444e+05, 2.00000000e+00],
           [3.00000000e+00, 5.00000000e+00, 8.37460840e+05, 2.00000000e+00],
           [7.00000000e+00, 1.20000000e+01, 2.08750703e+06, 2.00000000e+00],
           [9.00000000e+00, 1.10000000e+01, 2.32242339e+06, 2.00000000e+00],
           [6.00000000e+00, 1.50000000e+01, 2.35416537e+06, 3.00000000e+00],
           [1.60000000e+01, 1.70000000e+01, 2.81483295e+06, 4.00000000e+00],
           [1.40000000e+01, 1.80000000e+01, 3.44294208e+06, 3.00000000e+00],
           [1.00000000e+00, 1.00000000e+01, 4.51929196e+06, 2.00000000e+00],
           [1.90000000e+01, 2.00000000e+01, 6.06223563e+06, 7.00000000e+00],
           [2.10000000e+01, 2.20000000e+01, 6.21282975e+06, 5.00000000e+00],
           [2.00000000e+00, 8.00000000e+00, 6.42807846e+06, 2.00000000e+00],
           [1.30000000e+01, 2.50000000e+01, 9.12465562e+06, 3.00000000e+00],
           [2.30000000e+01, 2.40000000e+01, 1.25088770e+07, 1.20000000e+01],
           [2.60000000e+01, 2.70000000e+01, 2.21152298e+07, 1.50000000e+01]])

<br>

```python
# method = ward, median, centroid, average, weightd, complete, single
dendrogram(link_dist, labels=list(df2.index))
plt.show()
```


<img width="370" alt="output_27_0" src="https://user-images.githubusercontent.com/70505378/139006579-e09099e5-5cac-4978-af1c-2b0b6bb51da5.png">
    

<br>

### Clustering after Scaling (Don't forget!!)


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df2[['서비스업', '제조업']] = scaler.fit_transform(df2[['서비스업', '제조업']])

plt.figure(figsize=(6,6))
plt.scatter(df2['서비스업'], df2['제조업'],c='k',marker='o')
plt.xlabel('서비스업')
plt.ylabel('제조업')

for n in range(df2.shape[0]):
    plt.text(df2['서비스업'][n]*1.03, df2['제조업'][n]*0.98, df2.index[n])

Z = linkage(df2, metric='euclidean', method='centroid')
plt.figure(figsize=(10, 5))
plt.title('Dendrogram')
dendrogram(Z, labels=df2.index)
plt.show()
```


<img width="397" alt="output_29_0" src="https://user-images.githubusercontent.com/70505378/139006581-9377508a-031a-42bd-b812-b2a87fe9eb70.png">
    



<img width="593" alt="output_29_1" src="https://user-images.githubusercontent.com/70505378/139006582-df66b9fb-4a34-4595-a43a-a969c299cd9b.png">
    

<br>

<br>

## KMeans

* `KMeans`: n_clusters

1. Initialize k centroids.
2. Assign each data to the nearest centroid, these step will create clusters.
3. Recalculate centroid - which is mean of all data belonging to same cluster.
4. Repeat steps 2 & 3, till there is no data to reassign a different centroid.

Animation to explain algo - http://tech.nitoyon.com/en/blog/2013/11/07/k-means/


```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters = 3).fit(df2)
print(km.n_clusters)
```

    3



```python
km.labels_, km.cluster_centers_
```




    (array([1, 0, 2, 1, 1, 1, 1, 1, 2, 0, 0, 0, 1, 2, 0]),
     array([[ 6245553.6       , 16144968.6       ],
            [ 4191629.42857143,  3805868.14285714],
            [ 6433742.33333333, 31018896.        ]]))

<br>


```python
km_labels_two = km.labels_       # for compare later on
```


```python
df2['클러스터'] = km.labels_
df2
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
      <th>서비스업</th>
      <th>제조업</th>
      <th>클러스터</th>
    </tr>
    <tr>
      <th>구분</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>강원</th>
      <td>6203749</td>
      <td>6002286</td>
      <td>1</td>
    </tr>
    <tr>
      <th>경남</th>
      <td>8667737</td>
      <td>18053778</td>
      <td>0</td>
    </tr>
    <tr>
      <th>경북</th>
      <td>8487402</td>
      <td>30115601</td>
      <td>2</td>
    </tr>
    <tr>
      <th>광주</th>
      <td>3174973</td>
      <td>2910768</td>
      <td>1</td>
    </tr>
    <tr>
      <th>대구</th>
      <td>5470438</td>
      <td>5862633</td>
      <td>1</td>
    </tr>
    <tr>
      <th>대전</th>
      <td>3955921</td>
      <td>2608343</td>
      <td>1</td>
    </tr>
    <tr>
      <th>부산</th>
      <td>7582169</td>
      <td>7512588</td>
      <td>1</td>
    </tr>
    <tr>
      <th>세종</th>
      <td>645424</td>
      <td>1502922</td>
      <td>1</td>
    </tr>
    <tr>
      <th>울산</th>
      <td>3649386</td>
      <td>25883132</td>
      <td>2</td>
    </tr>
    <tr>
      <th>인천</th>
      <td>7154416</td>
      <td>12367816</td>
      <td>0</td>
    </tr>
    <tr>
      <th>전남</th>
      <td>5690659</td>
      <td>21453926</td>
      <td>0</td>
    </tr>
    <tr>
      <th>전북</th>
      <td>4910318</td>
      <td>12965875</td>
      <td>0</td>
    </tr>
    <tr>
      <th>제주</th>
      <td>2308732</td>
      <td>241537</td>
      <td>1</td>
    </tr>
    <tr>
      <th>충남</th>
      <td>7164439</td>
      <td>37057955</td>
      <td>2</td>
    </tr>
    <tr>
      <th>충북</th>
      <td>4804638</td>
      <td>15883448</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>

<br>


```python
df2.drop('클러스터', axis = 1, inplace=True) ; df2.head()
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
      <th>서비스업</th>
      <th>제조업</th>
    </tr>
    <tr>
      <th>구분</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>강원</th>
      <td>6203749</td>
      <td>6002286</td>
    </tr>
    <tr>
      <th>경남</th>
      <td>8667737</td>
      <td>18053778</td>
    </tr>
    <tr>
      <th>경북</th>
      <td>8487402</td>
      <td>30115601</td>
    </tr>
    <tr>
      <th>광주</th>
      <td>3174973</td>
      <td>2910768</td>
    </tr>
    <tr>
      <th>대구</th>
      <td>5470438</td>
      <td>5862633</td>
    </tr>
  </tbody>
</table>
</div>




```python
centers = km.cluster_centers_ ; centers
```


    array([[ 6245553.6       , 16144968.6       ],
           [ 4191629.42857143,  3805868.14285714],
           [ 6433742.33333333, 31018896.        ]])

<br>


```python
my_markers=['*','^', 'o','^','.',',','1','2']
my_color =['r','c','g','b','g','k','r','y']

plt.figure(figsize=(10, 8))
plt.xlabel('서비스업')
plt.ylabel('제조업')
for n in range(df2.shape[0]):
    label = km.labels_[n]
    plt.scatter(df2['서비스업'][n], df2['제조업'][n], c=my_color[label], marker=my_markers[label], s=100)
    plt.text(df2['서비스업'][n]*1.03, df2['제조업'][n]*0.98, df2.index[n])
    
for i in range(km.n_clusters):
    plt.scatter(centers[i][0], centers[i][1], c = 'b', s= 50)
```


<img width="618" alt="output_37_0" src="https://user-images.githubusercontent.com/70505378/139006583-73e08de0-edba-4138-816f-f296b3d68c8d.png">
    

<br>

### Clustering after Scaling (Don't forget!!)


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df2[['서비스업', '제조업']] = scaler.fit_transform(df2[['서비스업', '제조업']])
df2.head(3)
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
      <th>서비스업</th>
      <th>제조업</th>
    </tr>
    <tr>
      <th>구분</th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>강원</th>
      <td>0.393992</td>
      <td>-0.676282</td>
    </tr>
    <tr>
      <th>경남</th>
      <td>1.498349</td>
      <td>0.431200</td>
    </tr>
    <tr>
      <th>경북</th>
      <td>1.417523</td>
      <td>1.539632</td>
    </tr>
  </tbody>
</table>
</div>

<br>


```python
km = KMeans(n_clusters= 3).fit(df2)
```


```python
centers = km.cluster_centers_
km_labels_two_scaled = km.labels_        # for compare later on
```


```python
plt.clf()
plt.figure(figsize=(8, 6))
plt.xlabel('서비스업')
plt.ylabel('제조업')

for n in range(df2.shape[0]):
    label = km.labels_[n]
    plt.scatter(df2['서비스업'][n], df2['제조업'][n], c=my_color[label], marker=my_markers[label], s=100)
    plt.text(df2['서비스업'][n]*1.05, df2['제조업'][n]*0.99, df2.index[n])
    
for i in range(km.n_clusters):
    plt.scatter(centers[i][0], centers[i][1], c = 'k', s= 50)
```



<img width="509" alt="output_42_1" src="https://user-images.githubusercontent.com/70505378/139006584-641093a9-869b-4c3f-9d24-0ea4150406fe.png">
    

<br>

<br>

## Let's use all features for clustering (instead of two)


```python
df.head().T
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
      <th>구분</th>
      <th>강원</th>
      <th>경기</th>
      <th>경남</th>
      <th>경북</th>
      <th>광주</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>주거용</th>
      <td>1940933</td>
      <td>16587710</td>
      <td>4260988</td>
      <td>3302463</td>
      <td>1954876</td>
    </tr>
    <tr>
      <th>공공용</th>
      <td>1400421</td>
      <td>5533662</td>
      <td>1427560</td>
      <td>1578115</td>
      <td>565527</td>
    </tr>
    <tr>
      <th>서비스업</th>
      <td>6203749</td>
      <td>33434551</td>
      <td>8667737</td>
      <td>8487402</td>
      <td>3174973</td>
    </tr>
    <tr>
      <th>업무용합계</th>
      <td>7604170</td>
      <td>38968213</td>
      <td>10095297</td>
      <td>10065517</td>
      <td>3740500</td>
    </tr>
    <tr>
      <th>농림어업</th>
      <td>607139</td>
      <td>2371347</td>
      <td>2141813</td>
      <td>1747462</td>
      <td>74608</td>
    </tr>
    <tr>
      <th>광업</th>
      <td>398287</td>
      <td>317263</td>
      <td>95989</td>
      <td>224568</td>
      <td>2898</td>
    </tr>
    <tr>
      <th>제조업</th>
      <td>6002286</td>
      <td>56603327</td>
      <td>18053778</td>
      <td>30115601</td>
      <td>2910768</td>
    </tr>
    <tr>
      <th>식료품제조</th>
      <td>546621</td>
      <td>2544420</td>
      <td>932743</td>
      <td>566071</td>
      <td>161072</td>
    </tr>
    <tr>
      <th>섬유,의류</th>
      <td>13027</td>
      <td>2109963</td>
      <td>346974</td>
      <td>3780171</td>
      <td>295922</td>
    </tr>
    <tr>
      <th>목재,나무</th>
      <td>19147</td>
      <td>529274</td>
      <td>60160</td>
      <td>72680</td>
      <td>6782</td>
    </tr>
    <tr>
      <th>펄프,종이</th>
      <td>24382</td>
      <td>1917458</td>
      <td>817685</td>
      <td>361772</td>
      <td>41827</td>
    </tr>
    <tr>
      <th>출판,인쇄</th>
      <td>7727</td>
      <td>731348</td>
      <td>28486</td>
      <td>44402</td>
      <td>22038</td>
    </tr>
    <tr>
      <th>석유,화확</th>
      <td>175323</td>
      <td>6881775</td>
      <td>1865583</td>
      <td>3653665</td>
      <td>391151</td>
    </tr>
    <tr>
      <th>의료,광학</th>
      <td>84397</td>
      <td>1336390</td>
      <td>178498</td>
      <td>217771</td>
      <td>27221</td>
    </tr>
    <tr>
      <th>요업</th>
      <td>3695776</td>
      <td>1728379</td>
      <td>429920</td>
      <td>1269917</td>
      <td>20728</td>
    </tr>
    <tr>
      <th>1차금속</th>
      <td>1038913</td>
      <td>2020196</td>
      <td>3809547</td>
      <td>10874970</td>
      <td>75702</td>
    </tr>
    <tr>
      <th>조립금속</th>
      <td>39477</td>
      <td>2302355</td>
      <td>1699879</td>
      <td>933178</td>
      <td>156396</td>
    </tr>
    <tr>
      <th>기타기계</th>
      <td>35063</td>
      <td>3613798</td>
      <td>1902913</td>
      <td>782570</td>
      <td>198847</td>
    </tr>
    <tr>
      <th>사무기기</th>
      <td>2019</td>
      <td>317244</td>
      <td>8070</td>
      <td>14468</td>
      <td>5967</td>
    </tr>
    <tr>
      <th>전기기기</th>
      <td>38062</td>
      <td>1040171</td>
      <td>924235</td>
      <td>750786</td>
      <td>236622</td>
    </tr>
    <tr>
      <th>영상,음향</th>
      <td>43986</td>
      <td>24519644</td>
      <td>534196</td>
      <td>4174971</td>
      <td>723764</td>
    </tr>
    <tr>
      <th>자동차</th>
      <td>113448</td>
      <td>2977165</td>
      <td>2156059</td>
      <td>2356890</td>
      <td>512148</td>
    </tr>
    <tr>
      <th>기타운송</th>
      <td>108629</td>
      <td>67594</td>
      <td>2048646</td>
      <td>123935</td>
      <td>5140</td>
    </tr>
    <tr>
      <th>가구및기타</th>
      <td>12872</td>
      <td>1833112</td>
      <td>262523</td>
      <td>60280</td>
      <td>13392</td>
    </tr>
    <tr>
      <th>재생재료</th>
      <td>3418</td>
      <td>133041</td>
      <td>47662</td>
      <td>77104</td>
      <td>16049</td>
    </tr>
    <tr>
      <th>산업용합계</th>
      <td>7007712</td>
      <td>59291937</td>
      <td>20291580</td>
      <td>32087631</td>
      <td>2988274</td>
    </tr>
  </tbody>
</table>
</div>

<br>


```python
df.drop(['업무용합계', '산업용합계', '합계'], axis=1, inplace=True, errors='ignore')
df.drop(['경기','서울'], inplace=True,errors='ignore')
```


```python
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
      <th>주거용</th>
      <th>공공용</th>
      <th>서비스업</th>
      <th>농림어업</th>
      <th>광업</th>
      <th>제조업</th>
      <th>식료품제조</th>
      <th>섬유,의류</th>
      <th>목재,나무</th>
      <th>펄프,종이</th>
      <th>출판,인쇄</th>
      <th>석유,화확</th>
      <th>의료,광학</th>
      <th>요업</th>
      <th>1차금속</th>
      <th>조립금속</th>
      <th>기타기계</th>
      <th>사무기기</th>
      <th>전기기기</th>
      <th>영상,음향</th>
      <th>자동차</th>
      <th>기타운송</th>
      <th>가구및기타</th>
      <th>재생재료</th>
    </tr>
    <tr>
      <th>구분</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>강원</th>
      <td>1940933</td>
      <td>1400421</td>
      <td>6203749</td>
      <td>607139</td>
      <td>398287</td>
      <td>6002286</td>
      <td>546621</td>
      <td>13027</td>
      <td>19147</td>
      <td>24382</td>
      <td>7727</td>
      <td>175323</td>
      <td>84397</td>
      <td>3695776</td>
      <td>1038913</td>
      <td>39477</td>
      <td>35063</td>
      <td>2019</td>
      <td>38062</td>
      <td>43986</td>
      <td>113448</td>
      <td>108629</td>
      <td>12872</td>
      <td>3418</td>
    </tr>
    <tr>
      <th>경남</th>
      <td>4260988</td>
      <td>1427560</td>
      <td>8667737</td>
      <td>2141813</td>
      <td>95989</td>
      <td>18053778</td>
      <td>932743</td>
      <td>346974</td>
      <td>60160</td>
      <td>817685</td>
      <td>28486</td>
      <td>1865583</td>
      <td>178498</td>
      <td>429920</td>
      <td>3809547</td>
      <td>1699879</td>
      <td>1902913</td>
      <td>8070</td>
      <td>924235</td>
      <td>534196</td>
      <td>2156059</td>
      <td>2048646</td>
      <td>262523</td>
      <td>47662</td>
    </tr>
    <tr>
      <th>경북</th>
      <td>3302463</td>
      <td>1578115</td>
      <td>8487402</td>
      <td>1747462</td>
      <td>224568</td>
      <td>30115601</td>
      <td>566071</td>
      <td>3780171</td>
      <td>72680</td>
      <td>361772</td>
      <td>44402</td>
      <td>3653665</td>
      <td>217771</td>
      <td>1269917</td>
      <td>10874970</td>
      <td>933178</td>
      <td>782570</td>
      <td>14468</td>
      <td>750786</td>
      <td>4174971</td>
      <td>2356890</td>
      <td>123935</td>
      <td>60280</td>
      <td>77104</td>
    </tr>
    <tr>
      <th>광주</th>
      <td>1954876</td>
      <td>565527</td>
      <td>3174973</td>
      <td>74608</td>
      <td>2898</td>
      <td>2910768</td>
      <td>161072</td>
      <td>295922</td>
      <td>6782</td>
      <td>41827</td>
      <td>22038</td>
      <td>391151</td>
      <td>27221</td>
      <td>20728</td>
      <td>75702</td>
      <td>156396</td>
      <td>198847</td>
      <td>5967</td>
      <td>236622</td>
      <td>723764</td>
      <td>512148</td>
      <td>5140</td>
      <td>13392</td>
      <td>16049</td>
    </tr>
    <tr>
      <th>대구</th>
      <td>3151904</td>
      <td>826396</td>
      <td>5470438</td>
      <td>69142</td>
      <td>5858</td>
      <td>5862633</td>
      <td>212626</td>
      <td>1057342</td>
      <td>16215</td>
      <td>445646</td>
      <td>46804</td>
      <td>418485</td>
      <td>85871</td>
      <td>68137</td>
      <td>317580</td>
      <td>661307</td>
      <td>516493</td>
      <td>58446</td>
      <td>180189</td>
      <td>252662</td>
      <td>1381273</td>
      <td>68127</td>
      <td>41814</td>
      <td>33616</td>
    </tr>
  </tbody>
</table>
</div>

<br>


```python
index_ = df.index
column_ = df.columns
```


```python
index_
```


    Index(['강원', '경남', '경북', '광주', '대구', '대전', '부산', '세종', '울산', '인천', '전남', '전북',
           '제주', '충남', '충북'],
          dtype='object', name='구분')


```python
column_
```


    Index(['주거용', '공공용', '서비스업', '농림어업', '광업', '제조업', '식료품제조', '섬유,의류', '목재,나무',
           '펄프,종이', '출판,인쇄', '석유,화확', '의료,광학', '요업', '1차금속', '조립금속', '기타기계',
           '사무기기', '전기기기', '영상,음향', '자동차', '기타운송', '가구및기타', '재생재료'],
          dtype='object')


```python
type(column_)
```


    pandas.core.indexes.base.Index


```python
list(column_)
```


    ['주거용',
     '공공용',
     '서비스업',
     '농림어업',
     '광업',
     '제조업',
     '식료품제조',
     '섬유,의류',
     '목재,나무',
     '펄프,종이',
     '출판,인쇄',
     '석유,화확',
     '의료,광학',
     '요업',
     '1차금속',
     '조립금속',
     '기타기계',
     '사무기기',
     '전기기기',
     '영상,음향',
     '자동차',
     '기타운송',
     '가구및기타',
     '재생재료']


```python
list(column_).index('제조업'), list(column_).index('서비스업')
```


    (5, 2)

<br>

### Scaling


```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)    # returns an array (not dataframe)
```


```python
print(type(df), type(df_scaled))
```

    <class 'pandas.core.frame.DataFrame'> <class 'numpy.ndarray'>

<br>

### Agglomerative

```python
Z = linkage(df_scaled, metric='euclidean', method='centroid')
plt.figure(figsize=(10, 5))
plt.title('Dendrogram for all features (scaled)')
dendrogram(Z, labels=index_)   # version 에 따라 list(index_) 로 해야 할 수도 있음
plt.show()
```


<img width="585" alt="output_55_0" src="https://user-images.githubusercontent.com/70505378/139006588-515638a3-8f61-4a16-aee0-a7ed21890f64.png">
    

<br>

### KMeans

```python
km = KMeans(n_clusters=3).fit(df_scaled)
print(km.cluster_centers_)
print(km.labels_)
```

    [[ 0.78279298  0.44601929  0.49612314 -0.14792601 -0.2202511  -0.11522065
       0.61689344 -0.07050385  0.56693391  0.41413847  0.54521244 -0.34444738
       0.42201991 -0.08911986 -0.28490127  0.72185836  0.69386845  0.50846817
       0.37057962 -0.15995802  0.12890552  0.25516914  0.7424765   0.59481401]
     [-0.80670038 -0.65022296 -0.74554771 -0.15444205 -0.1766458  -0.43227338
      -0.72564872 -0.39191158 -0.50458273 -0.47379294 -0.79242599  0.16406657
      -0.45929592 -0.02806164 -0.39192638 -0.78715114 -0.73512937 -0.48012602
      -0.5501468  -0.46830075 -0.56935008 -0.10680832 -0.56171066 -0.71096136]
     [ 0.47507237  0.93772248  1.12104758  0.98432519  1.2790136   1.85861879
       0.68909021  1.58320209  0.06523783  0.4158599   1.13785365  0.45910913
       0.34147599  0.3655753   2.22644616  0.5894539   0.49134744  0.15503658
       0.81377495  2.11892669  1.6060087  -0.39167828 -0.26144221  0.70392275]]
    [1 0 2 1 0 1 0 1 1 0 1 0 1 2 0]



```python
print("all features, scaled: ", km.labels_)
print("two features:         ", km_labels_two)
print("two features, scaled: ", km_labels_two_scaled)
```

    all features, scaled:  [1 0 2 1 0 1 0 1 1 0 1 0 1 2 0]
    two features:          [2 0 1 2 2 2 2 2 1 0 0 0 2 1 0]
    two features, scaled:  [0 1 1 2 0 2 0 2 0 0 0 0 2 1 0]

<br>

<br>

## DBSCAN

* `DBSCAN`: eps, min_samples, metric

- eps: The maximum distance between two samples for one to be considered as in the neighborhood of the other. This is the most important DBSCAN parameter to choose appropriately for your data set and distance function.
- min_samples: The number of samples (or total weight) in a neighborhood for a point to be considered as a core point. This includes the point itself.
- metric: The metric to use when calculating distance between instances in a feature array. 


```python
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
X, y = make_moons(n_samples=300, noise=0.1, random_state=11)   # X :samples, y: label

X[:5],y[:5]
```


    (array([[ 0.07466556, -0.13809597],
            [-0.96718677,  0.77131154],
            [-0.211739  ,  1.01774394],
            [ 1.22822953, -0.34209166],
            [ 0.99307628,  0.13267972]]), array([1, 0, 0, 1, 0]))


```python
plt.scatter(X[:,0], X[:,1], c='b')
plt.show()
```


<img width="383" alt="output_60_0" src="https://user-images.githubusercontent.com/70505378/139006590-047cb5ec-a82e-4107-91a4-eefb391b0bc8.png">
    

<br>

```python
kmeans = KMeans(n_clusters=2)
predict = kmeans.fit_predict(X)   # returns a predicted array

predict
```


    array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1,
           1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 0, 0,
           1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 1,
           1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 1,
           0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1,
           0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 0,
           1, 0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 1,
           0, 1, 1, 0, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 1,
           1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 0,
           1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 0,
           0, 1, 1, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1,
           1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1,
           1, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 0,
           1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0], dtype=int32)


```python
plt.scatter(X[:,0], X[:,1],c=predict)
```




<img width="383" alt="output_62_1" src="https://user-images.githubusercontent.com/70505378/139006591-5867fad9-fe5d-4874-a612-99c507615b72.png">
    

<br>

```python
dbscan = DBSCAN(eps=0.2, min_samples=5, metric='euclidean')
predict = dbscan.fit_predict(X)
plt.scatter(X[:,0], X[:,1],c=predict)
```




<img width="383" alt="output_63_1" src="https://user-images.githubusercontent.com/70505378/139006596-fd194439-8711-4629-94f0-366318efab37.png">
    

<br>

```python
predict
```


    array([0, 1, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 0,
           1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1,
           0, 0, 1, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0,
           1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1,
           0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0,
           1, 0, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 0, 0, 0, 1,
           0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 0, 0,
           0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1,
           0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1,
           0, 1, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1,
           1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 1, 1, 1,
           0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0,
           0, 0, 0, 1, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1,
           1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0])

<br>


```python
dbscan = DBSCAN(eps=0.1, min_samples=5, metric='euclidean')
predict = dbscan.fit_predict(X)
plt.scatter(X[:,0], X[:,1],c=predict)
```




<img width="383" alt="output_65_1" src="https://user-images.githubusercontent.com/70505378/139006598-a89166dc-bd33-4b5b-af2c-e315b880488d.png">
    

<br>

```python
dbscan = DBSCAN(eps=0.1, min_samples=10, metric='euclidean')
predict = dbscan.fit_predict(X)
print(predict)
plt.scatter(X[:,0], X[:,1],c=predict)
```

    [-1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
     -1  0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
     -1 -1 -1 -1 -1  0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 -1 -1 -1
     -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
     -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
     -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 -1 -1
     -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
     -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
     -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 -1 -1 -1 -1 -1 -1 -1 -1
     -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1
     -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 -1 -1 -1 -1 -1 -1  0 -1
     -1 -1 -1 -1 -1 -1  0 -1 -1 -1 -1 -1 -1 -1 -1 -1  0 -1 -1 -1 -1 -1 -1 -1
     -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1 -1]




<img width="383" alt="output_66_2" src="https://user-images.githubusercontent.com/70505378/139006600-5caf3df3-d3e3-4e39-b419-86b48d25b094.png">
    

<br>

```python
dbscan = DBSCAN(eps=0.5, min_samples=10, metric='euclidean')
predict = dbscan.fit_predict(X)
plt.scatter(X[:,0], X[:,1],c=predict)
```




<img width="383" alt="output_67_1" src="https://user-images.githubusercontent.com/70505378/139006603-f3507f1f-a291-4e3e-a562-92b574deba3c.png">
    

<br>

```python
dbscan = DBSCAN(eps=0.2, min_samples=6, metric='euclidean')
predict = dbscan.fit_predict(X)
plt.scatter(X[:,0], X[:,1],c=predict)
```




<img width="383" alt="output_68_1" src="https://user-images.githubusercontent.com/70505378/139006604-c2f12e4f-60c6-42de-9332-d49dece84b0d.png">
    

<br>

```python
dbscan = DBSCAN(eps=0.2, min_samples=15, metric='euclidean')
predict = dbscan.fit_predict(X)
plt.scatter(X[:,0], X[:,1],c=predict)
```




<img width="383" alt="output_69_1" src="https://user-images.githubusercontent.com/70505378/139006607-e44fbfde-4258-4cef-b247-1dcb1b1dcd20.png">
    

<br>

<br>
