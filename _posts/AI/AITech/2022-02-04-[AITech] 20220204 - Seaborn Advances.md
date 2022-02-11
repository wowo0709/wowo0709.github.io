---

layout: single
title: "[AITech] 20220204 - Seaborn Advances"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Joint plot', 'Pair plot', 'Facet grid']
---



<br>

# 학습 내용

이번 포스팅의 내용은 여러 차트를 사용하여 정보량을 높이는 방법입니다. 

이전에는 ax에 하나를 그리는 방법이었다면, 이제는 Figure-level로 전체적인 시각화를 그리는 API입니다. 

```python
student = pd.read_csv('./StudentsPerformance.csv')
'''
   gender race/ethnicity parental level of education         lunch  \
0  female        group B           bachelor's degree      standard   
1  female        group C                some college      standard   
2  female        group B             master's degree      standard   
3    male        group A          associate's degree  free/reduced   
4    male        group C                some college      standard  
  test preparation course  math score  reading score  writing score  
0                    none          72             72             74  
1               completed          69             90             88  
2                    none          90             95             93  
3                    none          47             57             44  
4                    none          76             78             75 
'''
iris = pd.read_csv('./iris.csv')
'''
   Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm      Species
0   1            5.1           3.5            1.4           0.2  Iris-setosa
1   2            4.9           3.0            1.4           0.2  Iris-setosa
2   3            4.7           3.2            1.3           0.2  Iris-setosa
3   4            4.6           3.1            1.5           0.2  Iris-setosa
4   5            5.0           3.6            1.4           0.2  Iris-setosa
'''
```



## Joint Plot

`Joint Plot`은 distribution api에서 살펴봤던 2개 feature의 결합 확률분포와 더불어 각각의 분포도 함께 볼 수 있는 시각화를 제공합니다. 

```python
sns.jointplot(x='math score', y='reading score',data=student,
             height=7)
```

 ![image-20220211181325251](https://user-images.githubusercontent.com/70505378/153569359-95aa9833-185e-4c99-b820-adc9c582b7b3.png)

```python
sns.jointplot(x='math score', y='reading score',data=student,
              hue='gender'
             )
```

![image-20220211181532521](https://user-images.githubusercontent.com/70505378/153569363-c644aa8c-5223-4dac-95b2-a48ede2b1a0f.png)

```python
sns.jointplot(x='math score', y='reading score',data=student,
              kind='reg', # { “scatter” | “kde” | “hist” | “hex” | “reg” | “resid” }, 
             )
```

![image-20220211181558634](https://user-images.githubusercontent.com/70505378/153569364-e4a72324-8830-4e5d-81b0-2b1084bfbd7d.png)





<br>

## Pair Plot

데이터 셋의 pair-wise 관계를 시각화하는 함수입니다. 

```python
sns.pairplot(data=iris)
```

![image-20220211181824493](https://user-images.githubusercontent.com/70505378/153569367-a77aa383-26d4-43e8-8d1e-597fa6524a56.png)

다음과 같은 파라미터들로 커스터마이징할 수 있습니다. 

* `hue`
* `kind`: 전체 서브플롯의 그래프 종류를 지정
  * scatter, kde, hist, reg
* `diag_kind`: 대각선에 있는 서브플롯의 그래프 종류를 지정
  * auto, hist, kde, None
* `corner`: 기본적으로 pair plot은 그래프가 대각선을 기준으로 대칭이기 때문에 대각선 아래쪽의 plot만 보도록 지정합니다. 

```python
sns.pairplot(data=iris, hue='Species', 
             kind='hist',
             diag_kind='kde',
             corner=True)
```

![image-20220211182324537](https://user-images.githubusercontent.com/70505378/153569372-7286b061-53eb-44a3-ae2a-9f419657d144.png)

<br>

## Facet Grid 사용하기

pairplot과 같이 다중 패널을 사용하는 시각화를 의미합니다.

다만 pairplot은 feature-feature 사이를 살폈다면, Facet Grid는 feature-feature 뿐만이 아니라 **feature's category-feature's category의 관계**도 살펴볼 수 있습니다.

단일 시각화도 가능하지만, 여기서는 최대한 여러 pair를 보며 관계를 살피는 것을 위주로 보면 좋습니다.

총 4개의 큰 함수가 Facet Grid를 기반으로 만들어졌습니다.

- `catplot` : Categorical 
- `displot` : Distribution
- `relplot` : Relational
- `lmplot` : Regression

### catplot

이미 수 많은 방법을 앞에서 살펴보았기에 각각에 대한 설명은 생략하도록 하겠습니다.
`catplot`은 다음 방법론을 사용할 수 있습니다.

- Categorical scatterplots:
    - `stripplot()` (with `kind="strip"`; the default)
    - `swarmplot()` (with `kind="swarm"`)

- Categorical distribution plots:
    - `boxplot()` (with `kind="box"`)
    - `violinplot()` (with `kind="violin"`)
    - `boxenplot()` (with `kind="boxen"`)

- Categorical estimate plots:
    - `pointplot()` (with `kind="point"`)
    - `barplot()` (with `kind="bar"`)
    - `countplot()` (with `kind="count"`)

```python
sns.catplot(x="race/ethnicity", y="math score", hue="gender", data=student,
            kind='box', 
            col='lunch', row='test preparation course'
           )
```



![image-20220211182933354](https://user-images.githubusercontent.com/70505378/153569376-4e3f9752-1e15-4d25-b1eb-50cbdce39b20.png)





### displot

`displot`은 다음 방법론을 사용할 수 있습니다.

- `histplot()` (with `kind="hist"`; the default)
- `kdeplot()` (with `kind="kde"`)
- `ecdfplot()` (with `kind="ecdf"`; univariate-only)

```python
sns.displot(x="math score", hue="gender", data=student,
           col='race/ethnicity', kind='kde', fill=True,
            col_order=sorted(student['race/ethnicity'].unique())
           )
```

![image-20220211183306815](https://user-images.githubusercontent.com/70505378/153569377-7ae084cd-c4e2-4493-a281-96e0d2f05d18.png)





### relplot

`relplot`은 다음 방법론을 사용할 수 있습니다.

- `scatterplot()` (with `kind="scatter"`; the default)
- `lineplot()` (with `kind="line"`)

```python
sns.relplot(x="math score", y='reading score', hue="gender", data=student,
           col='lunch')
```

![image-20220211183340445](https://user-images.githubusercontent.com/70505378/153569381-01fa0882-bde0-4523-98df-fe2e4da14424.png)

### lmplot

`lmplot`은 다음 방법론을 사용할 수 있습니다.

- `regplot()`

```python
sns.lmplot(x="math score", y='reading score', hue="gender", data=student)
```

![image-20220211183445287](https://user-images.githubusercontent.com/70505378/153569382-cba4dafb-96f2-4243-a96c-8a0c63dd4246.png)











<br>

<br>

# 참고 자료

* 

















<br>
