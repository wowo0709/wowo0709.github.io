---

layout: single
title: "[AITech][Visualization] 20220214 - Plotly Express"
categories: ['AI', 'AITech', 'Visualization']
toc: true
toc_sticky: true
tag: []
---



<br>

# 학습 내용

## Interactive를 사용하는 이유

`Interactive visualization`을 사용하는 이유는 무엇일까요? 여러가지가 있겠지만, 대표적인 이유로 **공간의 활용**을 들 수 있습니다. 먄약 10개 feature간의 상관관계를 보고 싶다면, 정적 시각화의 경우 총 **10x9/2=45개**의 그래프를 그려야 하겠죠. 이는 너무 공간적 낭비가 크며, 이런 경우에 동적 시각화를 사용한다면 하나의 그래프에 사용자가 원하는 정보만 띄우도록 설정할 수 있습니다. 

물론, 항상 동적 시각화가 좋은 것은 아닙니다. 가령 다른 사람들에게 **발표/설득을 하는 자리에서 본인의 메세지를 함축해서 담는 것은 정적 시각화의 장점**입니다. 

**Interactive**의 종류에는 Select, Explore, Reconfigure... 등의 많은 종류가 있는데, 여기서 다루지는 않고 더 많은 정보가 궁금하신 분들은 맨 아래 '참고 자료'를 참조하시기 바랍니다. 

Interactive visualization을 실현시킬 수 있는 라이브러리로는 크게 `Plotly`, `Bokeh`, `Altair`가 있으며, 여기서는 현재 그 사용성이 가장 높은 Plotly, 그 중에서도 seaborn과 유사하게 만들어져 문법이 익숙하고 다양한 함수를 제공하는 **Plotly Express**에 대해 실습 코드와 함께 살펴보겠습니다. 







<br>

## Plotly

`Plotly`는 인터랙티브 시각화에 가장 많이 사용되는 시각화 라이브러리입니다. Python 뿐만이 아니라 R, JS에서도 제공되어 그 호환성이 높죠. 

Plotly의 장점으로는 다음과 같은 것들이 있습니다. 

* 예시+문서화가 잘 되어 있음
* 통계 시각화 외에도 지리 시각화 + 3D 시각화 + 금융 시각화 등 다양한 시각화 기능 제공
* JS 시각화 라이브러리 D3js를 기반으로 만들어져 웹에서 사용 가능
* 형광 Color가 인상적

### Plotly Express

`Ployly Express`에 대한 설명으로는 다음과 같은 것들이 있습니다. 

- plotly의 단순화 버전
    - 간단한 interactive
- 유연한 input (list, dict, DataFrame, GeoDataFrame)
- 적절한 color encoding
- Facet 등 seaborn의 기능 대다수 제공
- 그 외 다수 plotly에서 제공하는 유용한 시각화 제공
- 3D, animation 제공

Plotly를 설치해주고 Plotly Express를 px로 import합니다. 

```python
# !pip install plotly statsmodels
import plotly
import plotly.express as px 
import seaborn as sns
```





### Scatter, Bar, Line

먼저 기본 플롯들인 Scatter, Bar, Line plot에 대해 알아보겠습니다. 데이터는 iris 데이터를 사용합니다. 

```python
iris = px.data.iris() 
iris.head()
```

![image-20220214172018765](https://user-images.githubusercontent.com/70505378/153831253-f2a1a137-cb6b-47c5-85d2-4751596e28ec.png)

**Scatter**

우선, seaborn과 마찬가지로 기본요소 4가지를 사용할 수 있습니다. 

* `x`
* `y`
* `size`
* `color`

```python
fig = px.scatter(iris, 
                 x='sepal_length',
                 y='petal_length',
                 size='sepal_length',
                 color='species',             
            )

fig.show()
```

![image-20220214172720997](https://user-images.githubusercontent.com/70505378/153831257-6d1050e1-d0b6-4dd4-a19d-332c6bf303e5.png)

그리고 다음의 인자들로 커스텀할 수 있습니다. 

* `range_x`, `range_y`: x, y 축의 범위
* `marginal_x`, `marginal_y`: 각 축에 대한 분포
  * seaborn의 jointplot에서 사용한 것과 동일
* `hover_data`, `hover_name`: 마우스를 데이터 위에 올려놨을 때 x, y 외에 추가로 보여줄 데이터와 hover 이름 지정
* `trendline`: 회귀선
* `facet_col`, `facet_row`: Facet grid 기능

```python
fig = px.scatter(iris,
                 x='sepal_length', y='petal_length',
                 color='species',
                 range_x=[4, 8], range_y=[0, 7],
                 marginal_x="box",
                 # marginal_y="violin",
                 hover_data=['sepal_width', 'petal_width'],
                 hover_name='species',
                 trendline='ols',
                 # facet_row='species'
                 facet_col='species'
                 )

fig.show()
```

![image-20220214173342411](https://user-images.githubusercontent.com/70505378/153831261-51c13517-dcb1-4010-a90c-4785cfe80b91.png)



**Bar**

Bar plot에서는 아래와 같은 데이터를 사용합니다. 

```python
medals = px.data.medals_long()
medals
```

![image-20220214173710678](https://user-images.githubusercontent.com/70505378/153831264-c6495163-ce67-4071-805e-2f3c09a16e05.png)

기본적으로 아래와 같이 사용합니다. 

```python
fig = px.bar(medals, 
             x="nation", 
             y="count", 
             color="medal")

fig.show()
```

![image-20220214173829432](https://user-images.githubusercontent.com/70505378/153831266-ffdfea91-6785-4248-a0fa-98955258a2f8.png)

다른 데이터를 사용해보겠습니다. 

```python
medals_wide = px.data.medals_wide()
medals_wide
```

![image-20220214173855408](https://user-images.githubusercontent.com/70505378/153831268-9c54ffb8-359e-4a6f-ab0e-284186aca6f5.png)

```python
fig = px.bar(medals_wide, 
             x="nation", 
             y=["gold", "silver", "bronze"], 
             )

fig.show()
```

![image-20220214173921674](https://user-images.githubusercontent.com/70505378/153831272-3a709515-573d-44ae-93bf-68812df7e60a.png)

기본은 Stacked bar plot이고, `barmode` 인자를 통해 다른 형태로도 그릴 수 있습니다. 아래는 Grouped bar plot을 그린 것입니다. 

```python
fig = px.bar(medals, 
             x="nation", 
             y="count", 
             color="medal",
             barmode="group",
            )

fig.show()
```

![image-20220214174016899](https://user-images.githubusercontent.com/70505378/153831275-9cd9d54e-631d-43cd-9311-1ae2bc16f173.png)





**Line**

```python
flights = sns.load_dataset('flights')
flights.head()
```

![image-20220214174045047](https://user-images.githubusercontent.com/70505378/153831278-61b86e48-a95d-4e87-9a87-427f7e3b769f.png)

```python
fig = px.line(flights, 
              x='year',
              y='passengers',
              color='month',
            )

fig.show()
```



![image-20220214174101362](https://user-images.githubusercontent.com/70505378/153831282-f28e2e84-6d0c-4ed5-837d-bd1805fc91f7.png)



### Various Charts

seaborn의 다양한 내용과 겹치고. 필요에 따라 보면 좋습니다.

- `hist` : `histogram`, `density_heatmap`
- `kdeplot` : `density_contour`
- `boxplot` : `box`
- `violinplot` : `violin`
- `stripplot` : `strip`
- `heatmap` : `imshow`
- `pairplot` : `scatter_matrix`

plotly에서만 제공하거나, 사용하기 편리한 내용을 살펴보겠습니다.

**Part-of-Whole**

데이터를 트리 구조로 살필 때 유용한 시각화 방법론입니다.

- Sunburst
- Treemap

```python
tips = px.data.tips()
tips.head()
```



|      | total_bill |  tip |    sex | smoker |  day |   time | size |
| ---: | ---------: | ---: | -----: | -----: | ---: | -----: | ---: |
|    0 |      16.99 | 1.01 | Female |     No |  Sun | Dinner |    2 |
|    1 |      10.34 | 1.66 |   Male |     No |  Sun | Dinner |    3 |
|    2 |      21.01 | 3.50 |   Male |     No |  Sun | Dinner |    3 |
|    3 |      23.68 | 3.31 |   Male |     No |  Sun | Dinner |    2 |
|    4 |      24.59 | 3.61 | Female |     No |  Sun | Dinner |    4 |



```python
fig = px.sunburst(tips, 
                  path=['day', 'time', 'sex'], 
                  values='total_bill')
fig.show()
```

![image-20220214174625207](https://user-images.githubusercontent.com/70505378/153831286-0fc1fa4d-ca16-4ca9-910b-3ed41a95c88d.png)











```python
fig = px.treemap(tips, 
                  path=['day', 'time', 'sex'], 
                  values='total_bill')
fig.show()
```

![image-20220214174654447](https://user-images.githubusercontent.com/70505378/153831288-05e5d795-bade-4e48-9b6d-008d737bb274.png)



**3-Dimensional**



```python
fig = px.scatter_3d(iris, 
                    x='sepal_length',
                    y='sepal_width', 
                    z='petal_width',
                    symbol='species',
                    color='species')
fig.show()
```

![image-20220214174730255](https://user-images.githubusercontent.com/70505378/153831289-b3b222d1-31d3-42d1-afcd-1438440fa61b.png)





**Multidimensional**

다차원 데이터를 시각화하는 또 다른 방법론입니다.

- parallel_coordinates
- parallel_categories



```python
fig = px.parallel_coordinates(iris, 
                              color="species_id", 
#                              color_continuous_scale=px.colors.diverging.Tealrose,
                             )
fig.show()

```

![image-20220214174809440](https://user-images.githubusercontent.com/70505378/153831291-62bb1223-e4cf-41b5-959a-c27a201f7b4e.png)









```python
tips = px.data.tips()
tips['sex'] = tips['sex'].apply(lambda x : 'red' if x=='Female' else 'gray')
fig = px.parallel_categories(tips, color='sex')
fig.show()
```

![image-20220214174819714](https://user-images.githubusercontent.com/70505378/153831295-bb555a28-5e2e-4376-9011-7083c5722746.png)

**Geo**

- scatter_geo
- choropleth



```python
geo = px.data.gapminder()#.query("year == 2007")
geo.head()
```



|      |     country | continent | year | lifeExp |      pop |  gdpPercap | iso_alpha | iso_num |
| ---: | ----------: | --------: | ---: | ------: | -------: | ---------: | --------: | ------: |
|    0 | Afghanistan |      Asia | 1952 |  28.801 |  8425333 | 779.445314 |       AFG |       4 |
|    1 | Afghanistan |      Asia | 1957 |  30.332 |  9240934 | 820.853030 |       AFG |       4 |
|    2 | Afghanistan |      Asia | 1962 |  31.997 | 10267083 | 853.100710 |       AFG |       4 |
|    3 | Afghanistan |      Asia | 1967 |  34.020 | 11537966 | 836.197138 |       AFG |       4 |
|    4 | Afghanistan |      Asia | 1972 |  36.088 | 13079460 | 739.981106 |       AFG |       4 |



```python
fig = px.scatter_geo(geo, 
                     locations="iso_alpha",
                     color="continent", 
                     size="pop",
                     animation_frame="year",
                     projection="natural earth")
fig.show()
```

![image-20220214174851871](https://user-images.githubusercontent.com/70505378/153831300-68d82d28-9048-4b5e-80c2-d13f7a7af92d.png)













```python
fig = px.choropleth(geo, 
                     locations="iso_alpha",
                     color="continent", 
                     projection="natural earth")
fig.show()
```

![image-20220214174918548](https://user-images.githubusercontent.com/70505378/153831301-4337cad6-e490-4203-8799-dd881d8601e2.png)





































<br>

<br>

# 참고 자료

* [Toward a Deeper Understanding of the Role of Interaction in Information Visualization](https://www.cc.gatech.edu/~stasko/papers/infovis07-interaction.pdf)
* [Plotly](https://plotly.com/python/)
* [Bokeh](https://docs.bokeh.org/en/latest/index.html)
* [Altair](https://altair-viz.github.io/)









<br>
