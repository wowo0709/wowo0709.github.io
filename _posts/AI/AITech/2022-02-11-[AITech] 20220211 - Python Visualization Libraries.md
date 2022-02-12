---

layout: single
title: "[AITech] 20220211 - Python Visualization Libraries"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['missingNo', 'squarify', 'pyWaffle', 'matplotlib_venn']
---



<br>

# 학습 내용

## 라이브러리 다운로드 및 버전 확인

```python
# !pip install --upgrade pip
# !pip install missingno squarify pywaffle matplotlib_venn

import missingno as msno
import squarify
import pywaffle
import matplotlib_venn
```









<br>

## MissingNo

`missingno` 라이브러리는 결측치(NaN)를 시각화하는 파이썬 시각화 라이브러리입니다. 

* [ResidentMario/missingno](https://github.com/ResidentMario/missingno/blob/master/CONFIGURATION.md)

missingno 실습을 위해 결측치가 있는 titanic dataset를 사용하겠습니다. 


```python
titanic = sns.load_dataset('titanic')
titanic.head()
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
      <th>survived</th>
      <th>pclass</th>
      <th>sex</th>
      <th>age</th>
      <th>sibsp</th>
      <th>parch</th>
      <th>fare</th>
      <th>embarked</th>
      <th>class</th>
      <th>who</th>
      <th>adult_male</th>
      <th>deck</th>
      <th>embark_town</th>
      <th>alive</th>
      <th>alone</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>22.0</td>
      <td>1</td>
      <td>0</td>
      <td>7.2500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>False</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>38.0</td>
      <td>1</td>
      <td>0</td>
      <td>71.2833</td>
      <td>C</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Cherbourg</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1</td>
      <td>3</td>
      <td>female</td>
      <td>26.0</td>
      <td>0</td>
      <td>0</td>
      <td>7.9250</td>
      <td>S</td>
      <td>Third</td>
      <td>woman</td>
      <td>False</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>True</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>1</td>
      <td>female</td>
      <td>35.0</td>
      <td>1</td>
      <td>0</td>
      <td>53.1000</td>
      <td>S</td>
      <td>First</td>
      <td>woman</td>
      <td>False</td>
      <td>C</td>
      <td>Southampton</td>
      <td>yes</td>
      <td>False</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0</td>
      <td>3</td>
      <td>male</td>
      <td>35.0</td>
      <td>0</td>
      <td>0</td>
      <td>8.0500</td>
      <td>S</td>
      <td>Third</td>
      <td>man</td>
      <td>True</td>
      <td>NaN</td>
      <td>Southampton</td>
      <td>no</td>
      <td>True</td>
    </tr>
  </tbody>
</table>

</div>


```python
titanic.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 891 entries, 0 to 890
    Data columns (total 15 columns):
     #   Column       Non-Null Count  Dtype   
    ---  ------       --------------  -----   
     0   survived     891 non-null    int64   
     1   pclass       891 non-null    int64   
     2   sex          891 non-null    object  
     3   age          714 non-null    float64 
     4   sibsp        891 non-null    int64   
     5   parch        891 non-null    int64   
     6   fare         891 non-null    float64 
     7   embarked     889 non-null    object  
     8   class        891 non-null    category
     9   who          891 non-null    object  
     10  adult_male   891 non-null    bool    
     11  deck         203 non-null    category
     12  embark_town  889 non-null    object  
     13  alive        891 non-null    object  
     14  alone        891 non-null    bool    
    dtypes: bool(2), category(2), float64(2), int64(4), object(5)
    memory usage: 80.6+ KB

<br>


```python
import missingno as msno
```

`missingno`는 결측치를 matrix로 나타내어 흰 부분으로 표시합니다.


```python
msno.matrix(titanic)
```


![output_11_1](https://user-images.githubusercontent.com/70505378/153718128-8fc6114f-aefb-47cc-9517-5da2f4fc7a7c.png)
    


row당 결측치의 개수가 다르기 때문에 다음과 같이 정렬을 진행할 수 있습니다.


```python
msno.matrix(titanic, 
            sort='descending', # ascending
           ) 
```




![output_13_1](https://user-images.githubusercontent.com/70505378/153718130-ec052907-5270-41cd-b7b7-c58c2ffc51de.png)
    


위의 방법 외에는 개수를 직접적으로 bar chart를 그려주는 방법이 있습니다.


```python
msno.bar(titanic)
```




![output_15_1](https://user-images.githubusercontent.com/70505378/153718132-21dc219a-12e2-461e-9296-8b4b875af9c4.png)
    



<br>

## Squarify

`Squarify`는 계층적 데이터를 표현하는 시각화인 **Treemap**을 위한 라이브러리입니다. 

* [laserson/squarify](https://github.com/laserson/squarify)

![image-20220213003449103](https://user-images.githubusercontent.com/70505378/153718141-1de2ae11-211b-4280-aa83-a85186ddacaa.png)

<br>

```python
import squarify

values = [100, 200, 300, 400]
squarify.plot(values)
```

![image-20220213004212911](https://user-images.githubusercontent.com/70505378/153718142-3c5074d8-44bd-4c91-94f2-2bec855313db.png)

다음과 같은 파라미터들로 커스텀할 수 있습니다. 

* `label` : 텍스트 라벨을 달아줍니다. (Pie차트와 유사)
* `color` : 색을 개별적으로 지정 가능
* `pad` 
* `text_kwargs` : 텍스트 요소를 딕셔너리로 전달

```python
fig, ax = plt.subplots()
values = [100, 200, 300, 400]
label = list('ABCD')
color = ['#4285F4', '#DB4437', '#F4B400', '#0F9D58']

squarify.plot(values, label=label, color=color, pad=0.2, 
               text_kwargs={'color':'white', 'weight':'bold'}, ax=ax)

ax.axis('off')
plt.show()
```

![image-20220213004338568](https://user-images.githubusercontent.com/70505378/153718138-39470b85-f871-4982-ae80-567365339822.png)





<br>

## PyWaffle

PyWaffle이 만들어주는 `Waffle Chart`는 와플 형태로 discrete하게 값을 나타내는 차트로, 기본적인 형태는 정사각형이나 원하는 벡터 이미지로도 사용이 가능합니다. 다양한 Icon을 활용하여 인포그래픽에서 유용하게 사용할 수 있습니다. 

![image-20220213004707088](https://user-images.githubusercontent.com/70505378/153718139-3ad52ec2-2bf3-497a-8477-942726fb317d.png)

* [gyli/PyWaffle](https://github.com/gyli/PyWaffle)

### 기본 Waffle

- `rows`와 `coloums`로 사각형의 전체 형태를 지정할 수 있습니다. 
- `values`로 값 전달


```python
from pywaffle import Waffle
```


```python
fig = plt.figure(
    FigureClass=Waffle, 
    rows=5, 
    columns=10, 
    values=[48, 46, 6],
    figsize=(5, 3)
)
plt.show()
```


![output_31_0](https://user-images.githubusercontent.com/70505378/153718156-729f3b7f-b462-41ef-a334-b5def525e5f6.png)
    


### legend

- legend는 딕셔너리로 전달합니다. 우측 상단 또는 중앙 하단을 추천합니다.


```python
data = {'A': 50, 'B': 45, 'C': 15}

fig = plt.figure(
    FigureClass=Waffle, 
    rows=5, 
    values=data, 
    legend={'loc': 'upper left', 'bbox_to_anchor': (1.1, 1)}
)
plt.show()
```


![output_33_0](https://user-images.githubusercontent.com/70505378/153718158-15155d1b-dc8f-4e2a-ac87-337636afc42e.png)
    


### Color

- `cmap_name` : 컬러맵을 전달해서 색을 변경할 수 있습니다.


```python
data = {'A': 50, 'B': 45, 'C': 15}

fig = plt.figure(
    FigureClass=Waffle, 
    rows=5, 
    values=data, 
    cmap_name='tab10',
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(data), 'framealpha': 0},
)

plt.show()
```


 ![output_35_0](https://user-images.githubusercontent.com/70505378/153718159-9ae6b48b-bfa4-422f-857d-835dd0352c20.png)
    


- `colors` : 각 범주의 색을 전달할 수도 있습니다.


```python
data = {'A': 50, 'B': 45, 'C': 15}

fig = plt.figure(
    FigureClass=Waffle, 
    rows=5, 
    values=data, 
    colors=["#232066", "#983D3D", "#DCB732"],
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(data), 'framealpha': 0},
)

plt.show()
```


![output_37_0](https://user-images.githubusercontent.com/70505378/153718160-509939c4-e140-4c0d-93eb-99e574e3da5b.png)
    


### Block Arraging Style

- `starting_location` : 네 꼭지점을 기준으로 시작점을 잡을 수 있습니다.


```python
data = {'A': 50, 'B': 45, 'C': 15}

fig = plt.figure(
    FigureClass=Waffle, 
    rows=5, 
    values=data, 
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(data), 'framealpha': 0},
    starting_location='SE' # NW, SW, NE and SE
)

plt.show()
```


![output_39_0](https://user-images.githubusercontent.com/70505378/153718161-5f8639d8-9f55-4325-a867-6999ad14f012.png)
    


- `vertical` : 기본적으로는 가로로 진행합니다. 세로로 진행하고 싶다면 True를 전달하면 됩니다.


```python
data = {'A': 50, 'B': 45, 'C': 15}

fig = plt.figure(
    FigureClass=Waffle, 
    rows=5, 
    values=data, 
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(data), 'framealpha': 0},
    vertical=True
)

plt.show()
```


![output_41_0](https://user-images.githubusercontent.com/70505378/153718162-34d7d41a-d998-451b-bf48-c35a767caac8.png)
    


- `block_arranging_style` : 어떤 식으로 나열 할지 정할 수 있습니다. 기본은 snake 방식입니다.


```python
fig = plt.figure(
    FigureClass=Waffle,
    rows=7,
    values=data, 
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(data), 'framealpha': 0},
    block_arranging_style= 'new-line',
)

```


![output_43_0](https://user-images.githubusercontent.com/70505378/153718164-0e147839-bad6-4fa4-aa59-12b1303b6465.png)
    


### Icon

[Font Awesome](https://fontawesome.com/)의 아이콘을 사용할 수 있습니다.

- `icons` : 아이콘 명칭
- `icon_legend` : 아이콘을 범례로 사용할 것인가
- `font_size` : 아이콘 사이즈


```python
fig = plt.figure(
    FigureClass=Waffle, 
    rows=10,     
    values=data, 
    legend={'loc': 'lower left', 'bbox_to_anchor': (0, -0.4), 'ncol': len(data), 'framealpha': 0},
    icons='child',
    icon_legend=True,
    font_size=15,
)
plt.show()
```


![output_45_0](https://user-images.githubusercontent.com/70505378/153718165-0b5455de-9aae-471e-a83d-9081eee7f14d.png)
    











<br>

## Matplotlib_venn

`Venn`은 집합 등에서 사용하는 익숙한 벤 다이어그램을 나타냅니다. EDA보다는 출판 및 프레젠테이션에 사용하고, 디테일한 사용이 draw.io나 ppt에 비해 어렵습니다. 

* [konstantint/matplotlib-venn](https://github.com/konstantint/matplotlib-venn)

### 2개의 Subset

이진법을 사용하여 각각에 들어갈 값을 정할 수 있습니다.

- 01 : 1번째 Set에 들어갈 내용
- 10 : 2번째 Set에 들어갈 내용
- 11 : 교집합에 들어갈 내용


```python
from matplotlib_venn import venn2
venn2(subsets = (3, 2, 1))
```




    <matplotlib_venn._common.VennDiagram at 0x243d2beeb50>




![output_47_1](https://user-images.githubusercontent.com/70505378/153718166-7746c076-b687-43b9-924b-e25be26b8c9f.png)
    


### 3개의 서브셋

- 1개만 포함되는 인덱스
  - 1 : 001
  - 2 : 010
  - 4 : 100
- 2개가 포함되는 인덱스
  - 3 : 011
  - 5 : 101
  - 6 : 110
- 3개가 포함되는 인덱스
  - 7 : 111


```python
from matplotlib_venn import venn3
venn3(subsets = (1, 2, 3, 4, 5, 6, 7), set_labels = ('Set1', 'Set2', 'Set3'))
```




    <matplotlib_venn._common.VennDiagram at 0x243d2335e20>




![output_49_1](https://user-images.githubusercontent.com/70505378/153718168-42083204-7c1f-4641-b805-29e340aa69f7.png)
    


### Set으로 전달하기

set을 전달하면 자동적으로 counting 하여 표현해줍니다.


```python
set1 = set(['A', 'B', 'C', 'D'])
set2 = set(['B', 'C', 'D', 'E'])
set3 = set(['C', 'D',' E', 'F', 'G'])

venn3([set1, set2, set3], ('Set1', 'Set2', 'Set3'))
plt.show()
```


![output_51_0](https://user-images.githubusercontent.com/70505378/153718170-ab57dda1-2e3f-4cca-9530-cc7c1cf3c4f3.png)
    



























<br>

<br>

# 참고 자료

* [ResidentMario/missingno](https://github.com/ResidentMario/missingno/blob/master/CONFIGURATION.md)
* [laserson/squarify](https://github.com/laserson/squarify)
* [gyli/PyWaffle](https://github.com/gyli/PyWaffle)
* [konstantint/matplotlib-venn](https://github.com/konstantint/matplotlib-venn)















<br>
