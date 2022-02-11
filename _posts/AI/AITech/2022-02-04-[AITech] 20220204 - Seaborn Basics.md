---

layout: single
title: "[AITech] 20220204 - Seaborn Basics"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Count/Box/Violin plot', 'Hist/Kde/Ecdf/Rug plot', 'Scatter/Line/Reg plot', 'Heap map']
---



<br>

# 학습 내용

## Seaborn 소개

`Seaborn`은 Matplotlib 기반 통계 시각화 라이브러리로, **쉬운 문법**과 **깔끔한 디자인**이 특징이며 matplotlib 기반이라 **Matplotlib으로 커스텀**이 가능해서 파이썬 데이터분석에서는 꼭 사용됩니다. 

![image-20220211113025400](https://user-images.githubusercontent.com/70505378/153542037-74ee9c30-27f4-4632-8cd5-2e7aed9937a9.png)

현재 Seaborn의 최신 버전은 0.11 버전이며, seaborn을 import해서 사용할 때는 sns로 사용합니다. 

```python
!pip install seaborn==0.11
import seaborn as sns
```

Seaborn은 시각화의 목적과 방법에 따라 API를 분류하여 제공합니다. 





<br>

## Import Modules&Data

```python
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

print('seaborn version : ', sns.__version__)
# seaborn version :  0.11.0

student = pd.read_csv('./StudentsPerformance.csv')
student.head()
```

![image-20220211113924482](https://user-images.githubusercontent.com/70505378/153542043-d1cf48d3-5efd-40ee-8b63-aef4ac85bb36.png)

```python
student.describe()
```

![image-20220211115922523](https://user-images.githubusercontent.com/70505378/153542046-c34a8896-179c-4584-b5d2-f98325d6918f.png)











<br>

## Categorical API



### Count Plot

Categorical API의 대표적인 시각화로 `Count Plot`이 있습니다. Count plot은 범주를 이산적으로 세서 막대 그래프로 그려주는 함수입니다. 

기본적으로 다음과 같은 파라미터들이 있습니다. Count plot에서 사용되는 파라미터들은 다른 categorial API plots에도 사용되니 이를 잘 알아두는 것이 좋습니다. 

* `x(y), data, order, hue(hue_order), palette, color, saturation, ax, ...`

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.countplot(y='race/ethnicity',                                  # y: 가로로 그림
              data=student,                                        # data: 사용할 데이터
              order=sorted(student['race/ethnicity'].unique()),    # order: 막대 순서
              hue='gender',                                        # hue: 색으로 구분할 feature
              palette='Set2',                                      # palette: 색깔 팔레트
              saturation=0.3,                                      # saturation: 채도(0~1)
              ax=axes[0]                                           # ax: 그릴 subplot
             )

sns.countplot(x='gender',                                          # x: 세로로 그림
              data=student,                                        # data: 사용할 데이터
              order=sorted(student['gender'].unique()),            # order: 막대 순서
              hue='race/ethnicity',                                # hue: 색으로 구분할 feature
              hue_order=sorted(student['race/ethnicity'].unique()),# hue_order: 색깔을 매칭할 순서
              color='red',                                         # color: 사용할 단일 색상 지정(그래디언트하게 표현)
              saturation=1,                                        # saturation: 채도(0~1)
              ax=axes[1]                                           # ax: 그릴 subplot
             )

plt.show()
```

![image-20220211115703849](https://user-images.githubusercontent.com/70505378/153542044-a08d56a0-80cd-4cf4-b55c-1289da02494b.png)

### Box Plot

* [Understanding Boxplots](https://towardsdatascience.com/understanding-boxplots-5e2df7bcbd51)

`Box plot`은 분포를 살피는 대표적인 시각화 방법으로 중간의 사각형은 왼쪽부터 25%, 50%, 75% 값을 의미합니다. 이 때 사각형의 길이를 IQR이라고 하고 양 쪽의 선의 길이는 `min(1.5*IQR, min(max) value)`를 나타냅니다. 

![image-20220211125730377](https://user-images.githubusercontent.com/70505378/153542048-5c863070-934d-4abb-a5c6-ea904781013e.png)

Box plot은 기본적으로 count plot에 있었던 파라미터 외에 다음과 같은 파라미터들을 가집니다. 

* `width, linewidth, fliersize `

```python
fig, ax = plt.subplots(1,1, figsize=(10, 5))

sns.boxplot(x='race/ethnicity', y='math score', data=student,
            hue='gender', 
            order=sorted(student['race/ethnicity'].unique()),
            width=0.3,    # width: 가운데 박스의 너비
            linewidth=2,  # linewidth: 박스플롯 외부 선들의 굵기
            fliersize=10, # outlier를 나타내는 범위 밖 점들의 크기
            ax=ax)

plt.show()
```

![image-20220211130553298](https://user-images.githubusercontent.com/70505378/153542049-6986456a-815e-4296-b2ce-2601a8141724.png)







### Violin Plot

Box plot은 대푯값을 잘 보여주지만 실제 분포를 표현하기에는 부족합니다. `Violin plot`은 이런 분포에 대한 정보를 더 제공해주기에 적합한 방식입니다. 

가운데 흰 점이 50%를 나타내고 중간의 굵은 검은색 막대가 IQR 범위를 의미합니다. 

```python
fig, ax = plt.subplots(1,1, figsize=(12, 5))
sns.violinplot(x='math score', data=student, ax=ax)
plt.show()
```

![image-20220211130944012](https://user-images.githubusercontent.com/70505378/153542053-17ca2250-a984-47b3-a1d7-8dc12ea4677f.png)

그러나 violin plot은 오해가 생기기 충분한 표현 방식입니다. 

* 데이터는 연속적이지 않습니다. (kernel density estimate를 사용합니다.)
- 또한 연속적 표현에서 생기는 데이터의 손실과 오차가 존재합니다.
- 데이터의 범위가 없는 데이터까지 표시됩니다.

이런 오해를 줄이고 정보량을 높이는 방법은 다음과 같은 방법이 있습니다.

- `bw` : 분포 표현을 얼마나 자세하게 보여줄 것인가(0~1)
    - ‘scott’, ‘silverman’, float
- `cut` : 끝부분을 얼마나 자를 것인가?
    - float
- `inner` : 내부를 어떻게 표현할 것인가 
    - “box”, “quartile”, “point”, “stick”, None

```python
fig, ax = plt.subplots(1,1, figsize=(12, 5))
sns.violinplot(x='math score', data=student, ax=ax,
               bw=0.1,
               cut=0,
               inner='quartile'
              )

plt.show()
```

![image-20220211132113472](https://user-images.githubusercontent.com/70505378/153542055-39262a04-9979-4a4a-b6d8-460d8d406914.png)

그리고 이외에도 다음과 같은 파라미터들이 있습니다. 

* `scale` : 각 바이올린의 종류
  - “area”, “count”, “width”
- `split` : 동시에 비교

```python
fig, ax = plt.subplots(1,1, figsize=(12, 7))
sns.violinplot(x='race/ethnicity', y='math score', data=student, ax=ax,
               order=sorted(student['race/ethnicity'].unique()),
               hue='gender', 
               bw=0.2, 
               cut=0, 
               inner="quartile", 
               scale="area",
               split=True
              )

plt.show()
```

![image-20220211132647227](https://user-images.githubusercontent.com/70505378/153542057-ee2f273c-3ae8-49ab-866a-249fc90630e4.png)







### ETC

`boxen plot`, `swarmplot`, `stripplot` 등의 바리에이션들이 있습니다. 

```python
fig, axes = plt.subplots(3,1, figsize=(12, 21))

sns.boxenplot(x='race/ethnicity', y='math score', data=student, ax=axes[0],
               order=sorted(student['race/ethnicity'].unique()))

sns.swarmplot(x='race/ethnicity', y='math score', data=student, ax=axes[1],
               order=sorted(student['race/ethnicity'].unique()))

sns.stripplot(x='race/ethnicity', y='math score', data=student, ax=axes[2],
               order=sorted(student['race/ethnicity'].unique()))

plt.show()
```

![image-20220211132906555](https://user-images.githubusercontent.com/70505378/153542058-6d71ee7d-9d73-42a9-a14e-93c15bb8068a.png)











<br>

## Distribution API

Distribution API는 범주형/연속형 모두 살펴볼 수 있습니다. 

### Univariate Distribution

하나의 feature에 대한 분포를 보여주는 plot입니다. 

* `histplot` : 히스토그램
- `kdeplot` : Kernel Density Estimate
- `ecdfplot` : 누적 밀도 함수
- `rugplot` : 선을 사용한 밀도함수

#### hist plot

* `binwidth`: 막대 하나의 너비
* `bins`: 총 생성할 막대 개수
  * binwidth와 bins는 동시에 사용할 수 없음
* `element`: 히스토그램 표현 방식
  * bars, step, poly
* `multiple`: feature 분포 내에서 다른 feature의 분포도 표현하고 싶을 때 사용할 방법
  * hue 등을 지정 시 사용

```python
fig, ax = plt.subplots(figsize=(12, 7))

sns.histplot(x='math score', data=student, ax=ax,
             hue='gender', 
             bins=20,
             element='step', # bars, step, poly
             multiple='stack', # layer, dodge, stack, fill
            )

plt.show()
```

![image-20220211133857220](https://user-images.githubusercontent.com/70505378/153542060-1990ce7b-15cf-4202-8f82-0ed9b6353304.png)

#### kde plot

kdeplot은 연속확률밀도를 보여주는 함수로 앞서 살펴봤던 violin plot의 반쪽이라고 생각할 수 있습니다. Seaborn의 다양한 smoothing 및 분포 시각화에 보조 정보로도 많이 사용됩니다. 

* `fill`: kde plot 내부를 채워줍니다. 
* `bw_method`: violin plot의 bw 파라미터와 동일합니다. 
* `multiple`: 여러 개의 분포를 나타낼 때 사용할 방식
  * layer, stack, fill
  * layer를 사용하는 것이 가장 권장됩니다. 
* `cumalative`: 값의 누적을 보여줍니다. 

```python
fig, ax = plt.subplots(figsize=(12, 7))

sns.kdeplot(x='math score', data=student, ax=ax,
            fill=True, 
            hue='race/ethnicity', 
            hue_order=sorted(student['race/ethnicity'].unique()),
            multiple="layer", # layer, stack, fill
            cumulative=False,
            cut=0
           )

plt.show()
```

![image-20220211134541350](https://user-images.githubusercontent.com/70505378/153542062-1e7d6a04-4325-46ef-b454-f04a6e3ad716.png)

#### ecdf plot

ecdf plot은 누적되는 양을 표현합니다. kde plot에서 cumulative=True로 설정한 것과 같습니다. 

```python
fig, ax = plt.subplots(figsize=(12, 7))
sns.ecdfplot(x='math score', data=student, ax=ax,
             hue='gender',
             stat='count', # count, proportion
             # complementary=True
            )
plt.show()
```

![image-20220211135115416](https://user-images.githubusercontent.com/70505378/153542064-bf28338b-bcf1-4201-8f7c-be1538468721.png)

#### rugplot

rugplot은 조밀한 정도를 통해 밀도를 표현합니다. 많이 사용하지는 않지만 한정된 공간 내에서 분포를 표현하기에 적합합니다. 

```python
fig, ax = plt.subplots(figsize=(12, 7))
sns.rugplot(x='math score', data=student, ax=ax)
plt.show()
```

![image-20220211135215665](https://user-images.githubusercontent.com/70505378/153542066-efdcaac0-4a84-4c22-ae64-75824c866a6d.png)



### Bivariate Distribution

두 변수 간의 분포를 보여줄 수 있습니다. 

함수는 histplot과 kdeplot을 사용하고, 입력에 1개의 축만 넣는 게 아닌 2개의 축 모두 입력을 넣어주는 것이 특징입니다.

#### hist plot

일반 scatter plot의 alpha 값을 조정한 분포와 2차원 hist plot을 비교해보겠습니다. 

```python
fig, axes = plt.subplots(1,2, figsize=(12, 7))

axes[0].scatter(student['math score'], student['reading score'], alpha=0.2)

sns.histplot(x='math score', y='reading score', 
             data=student, ax=axes[1],
             cbar=False,
             bins=(10, 20), 
            )

plt.show()
```

![image-20220211135608420](https://user-images.githubusercontent.com/70505378/153542067-0cb4a166-63ef-4a5e-8e14-326bb8de3d90.png)

#### kde plot

2차원 kde plot을 활용하면 다음과 같이 등고선 형태의 분포를 나타낼 수 있습니다. 

```python
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_aspect(1)

sns.kdeplot(x='math score', y='reading score', 
             data=student, ax=ax,
            fill=True,
#             bw_method=0.1
            )

plt.show()
```

![image-20220211135717659](https://user-images.githubusercontent.com/70505378/153542071-e975afde-bbb1-4d1d-b276-73893fc8a579.png)



<br>

## Relation&Regression API

### Scatter plot

산점도는 다음과 같은 요소를 사용할 수 있습니다.

- `style`
- `hue`
- `size`

앞서 차트의 요소에서 다루었기에 가볍게만 살펴보고 넘어가겠습니다.

`style, hue, size`에 대한 순서는 각각 `style_order, hue_order, size_order`로 전달할 수 있습니다.

```python
fig, ax = plt.subplots(figsize=(7, 7))
sns.scatterplot(x='math score', y='reading score', data=student,
                style='gender', markers={'male':'s', 'female':'o'},
                hue='race/ethnicity', 
                size='writing score',
               )
plt.show()
```

![image-20220211140014556](https://user-images.githubusercontent.com/70505378/153542072-76768536-7463-4a89-9e0e-4311f7bf8a78.png)





### Line plot

선 그래프에서는 다른 데이터셋을 사용해보겠습니다. 

```python
flights = sns.load_dataset("flights")
flights_wide = flights.pivot("year", "month", "passengers")
flights_wide.head()
```

![image-20220211140123492](https://user-images.githubusercontent.com/70505378/153542074-62654299-7147-4042-9af6-54249ba166c9.png)

자동으로 평균과 표준편차로 오차범위를 시각화해줍니다. 

```python
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
sns.lineplot(data=flights, x="year", y="passengers", ax=ax)
plt.show()
```

![image-20220211140525777](https://user-images.githubusercontent.com/70505378/153542075-b21e9590-7be1-4074-a259-93055acff6b6.png)

사용할 수 있는 파라미터로는 다음의 것들이 있습니다. 

* `style`: 어떤 feature에 따라 스타일을 구분할 지 지정합니다. 
* `markers`: 값이 있는 지점마다 marker를 찍어줄 지 지정합니다. 
* `dashes`: dash를 사용할 지 여부(False면 line 사용)

```python
fig, ax = plt.subplots(1, 1, figsize=(12, 7))
sns.lineplot(data=flights, x="year", y="passengers", hue='month', 
             style='month', markers=True, # dashes=False,
             ax=ax)
plt.show()
```

![image-20220211140921108](https://user-images.githubusercontent.com/70505378/153542077-ada50a5c-bfe6-40c2-bee5-e481c6b5a197.png)





### Reg plot

회귀선을 추가한 scatter plot입니다. 

```python
fig, ax = plt.subplots(figsize=(7, 7))
sns.regplot(x='math score', y='reading score', data=student,
               )
plt.show()
```

![image-20220211141105305](https://user-images.githubusercontent.com/70505378/153542079-9e513739-c118-41f7-8818-88f20d75d2b9.png)

다음의 파라미터들을 사용할 수 있습니다. 

* `x_estimators`: 한 축에 특정 값 하나만 보여주도록 설정할 수 있습니다.  
* `x_bins`: 축의 개수도 설정할 수 있습니다. 
* `order`: 다차원 회귀선을 추가할 수 있습니다. 
* `logx`: 로그 그래프를 그릴 수도 있습니다.

```python
fig, ax = plt.subplots(figsize=(7, 7))
sns.regplot(x='math score', y='reading score', data=student,
            x_estimator=np.mean, 
            x_bins=20
           )
plt.show()
```

![image-20220211141407917](https://user-images.githubusercontent.com/70505378/153542082-fc7c7744-0ff2-4bba-9c85-8d95ef9bcebe.png)



<br>

## Matrix API

### Heat map

히트맵은 다양한 방식으로 사용될 수 있는데요, 대표적으로는 상관관계(correlation) 시각화에 많이 사용됩니다. 

성적 데이터는 선형성이 강하므로 다른 데이터셋을 이용해보겠습니다. 

```python
heart = pd.read_csv('./heart.csv')
heart.corr()
```

![image-20220211141805401](https://user-images.githubusercontent.com/70505378/153542085-e836ccc5-0db4-4593-af1c-8c97d26a4974.png)

가장 기본적인 heapmap은 다음과 같습니다. 

```python
fig, ax = plt.subplots(1,1 ,figsize=(7, 6))
sns.heatmap(heart.corr(), ax=ax)
plt.show()
```

![image-20220211141848328](https://user-images.githubusercontent.com/70505378/153542087-7fce24fe-63ae-4353-8ab8-1af918d8f6a3.png)

heapmap에서 사용할 수 있는 파라미터에는 다음의 것들이 있습니다. 

* `vmin`, `vmax`: 값의 범위 조정
* `center` 색상이 발산할 값의 기준을 정해줄 수 있습니다. 
* `cmap`: 사용할 color map을 지정할 수 있습니다. 
* `annot`: 각각의 박스 안에 값을 표시합니다. 
* `fmt`: annot=True 시 값을 표시할 때 사용할 포맷을 지정합니다. 
  * '.2f', 'd' 등
* `linewidth`: 칸 사이를 나눌 선을 그을 수 있습니다. 
* `square`: True이면 히트맵을 정사각형으로 그립니다. 
* `mask`: boolean list를 전달하여 필요없는 부분을 지울 수 있습니다. 

```python
fig, ax = plt.subplots(1,1 ,figsize=(12, 9))
sns.heatmap(heart.corr(), ax=ax,
           vmin=-1, vmax=1, center=0,
            cmap='coolwarm',
            annot=True, fmt='.2f',
            linewidth=0.1, square=True
           )
plt.show()
```

![image-20220211142521896](https://user-images.githubusercontent.com/70505378/153542090-7f66b84f-474b-438b-a5b6-d2788ff33af9.png)

```python
fig, ax = plt.subplots(1,1 ,figsize=(10, 9))
# masking
mask = np.zeros_like(heart.corr())
mask[np.triu_indices_from(mask)] = True

sns.heatmap(heart.corr(), ax=ax,
           vmin=-1, vmax=1, center=0,
            cmap='coolwarm',
            annot=True, fmt='.2f',
            linewidth=0.1, square=True, cbar=False,
            mask=mask
           )
plt.show()
```

![image-20220211142555322](https://user-images.githubusercontent.com/70505378/153542091-5f08c1c3-57ed-4c4a-a59a-4a9212d179b6.png)



















<br>

<br>

# 참고 자료

* 

















<br>
