---

layout: single
title: "[AITech] 20220211 - Polar Coordination&Radar Chart"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Polar Coordination', 'Radar Chart']
---



<br>

# 학습 내용

이번 포스팅에서는 matplotlib을 이용해 **극 좌표계**를 다루는 방법에 대해 알아보겠습니다. 

![image-20220212215452319](https://user-images.githubusercontent.com/70505378/153712757-d9d52146-d78f-4cef-8181-4e0213071def.png)

## Polar Coordination

### Polar Coordinate 만들기

서브플롯 `ax`를 만들 때 `projection='polar'` 또는 `polar=True` 파라미터를 전달하면 다음과 같이 극좌표계를 사용할 수 있습니다.

```python
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar') # polar=True
plt.show()
```

![image-20220212213922887](https://user-images.githubusercontent.com/70505378/153712763-b122bb8e-b06c-4840-b3d9-0dd7f588a44b.png)

### Polar Coordinate 조정하기

- `set_rmax` : 반지름 조정 
- `set_rmin`을 조정한다면? 도넛형태가 될 수 있을까??
    - 중앙에서 r=1로 시작할 뿐!!
- `set_thetamax()` : 각도의 max값
- `set_thetamin()` : 각도의 min값
- `set_rticks` : 반지름 표기 grid 조정
- `set_rlabel_position`: 반지름 label이 적히는 위치의 각도 조정

```python
fig = plt.figure()
ax = fig.add_subplot(111, polar=True)

ax.set_rmax(2)
ax.set_rmin(1)
ax.set_thetamin(45)
ax.set_thetamax(135)
ax.set_rticks([1, 1.2, 1.4, 1.6, 1.8, 2.0])
# ax.set_rlabel_position(90)

plt.show()
```

![image-20220212215006796](https://user-images.githubusercontent.com/70505378/153712766-c2d0b031-345c-444d-a689-9bc525e7ed73.png)

### Polar 기본 차트

- `scatter()` : 기존 산점도와 같음 (theta, r 순서)

```python
np.random.seed(19680801)

N = 100
r = 2 * np.random.rand(N)
theta = 2 * np.pi * np.random.rand(N)
area = 200 * r**2
colors = theta

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
c = ax.scatter(theta, r, c=colors, s=area, cmap='hsv', alpha=0.75)
plt.show()
```

![image-20220212215158548](https://user-images.githubusercontent.com/70505378/153712767-133d6a12-c425-4da0-903e-845db8c27466.png)

- `bar()`
    - Polar coordinate에서의 bar plot은 실질적인 bar 간 크기 비교가 어려우므로 목적에 따라 잘 사용해야 함

```python
np.random.seed(19680801)

N = 6
r = np.random.rand(N)
theta = np.linspace(0, 2*np.pi, N, endpoint=False)

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.bar(theta, r, width=0.5, alpha=0.5)
plt.show()
```

![image-20220212215211144](https://user-images.githubusercontent.com/70505378/153712751-a0634b75-5923-4c59-8f41-2b018e59d427.png)

- `plot()`

```python
np.random.seed(19680801)

N = 1000
r = np.linspace(0, 1, N)
theta = np.linspace(0, 2*np.pi, N)

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.plot(theta, r)

plt.show()
```

![image-20220212215239493](https://user-images.githubusercontent.com/70505378/153712753-b1290764-0dbe-42d6-bd27-2e1066c23c20.png)

- `fill()`

```python
np.random.seed(19680801)

N = 1000
r = np.linspace(0, 1, N)
theta = np.linspace(0, 2*np.pi, N)

fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')
ax.fill(theta, r)
plt.show()
```



![image-20220212215257641](https://user-images.githubusercontent.com/70505378/153712754-740077fe-d891-46d1-baab-6b8b5cf692f3.png)











<br>

## Radar Chart

`Radar Chart`는 극 좌표계를 사용하는 대표적인 차트로, 중심점을 기준으로 N개의 변수 값을 표현하며 데이터의 Quality를 표현하기에 좋습니다. 

캐릭터 등의 능력치를 표현할 때 자주 사용하죠. 

![image-20220212215650682](https://user-images.githubusercontent.com/70505378/153712758-8914b48e-02db-4cdc-8831-8ed959a6781c.png)

Radar chart의 주의점으로는 다음과 같은 것들이 있습니다. 

* 각 feature는 독립적이며, 척도가 같아야 합니다. 
* 다각형의 면적이 중요하지만, 이는 feature의 순서에 따라 많이 달라집니다. 
* feature의 개수가 많을수록 가독성이 떨어집니다. 

여기서는 아래와 같은 [Pokemon with Stat](https://www.kaggle.com/abcsds/pokemon) 데이터셋을 사용하여 만들어보겠습니다.

```python
pokemon = pd.read_csv('./pokemon.csv')
pokemon.head()
```

![image-20220212215903269](https://user-images.githubusercontent.com/70505378/153712759-5cd43709-9296-446e-acc1-f4cc59872ba8.png)

### Radar Chart 기본 구성

```python
# 나타낼 feature 선택
stats = ["HP", "Attack", "Defense", "Sp. Atk", "Sp. Def", "Speed"]
values = pokemon.iloc[0][stats].to_list() # [45, 49, 49, 65, 65, 45]

# 각은 2𝜋 를 6등분
theta = np.linspace(0, 2*np.pi, 6, endpoint=False) # [0.         1.04719755 2.0943951  3.14159265 4.1887902  5.23598776]

# 끝 점을 포함하기 위해 마지막 데이터를 포함
fig = plt.figure()
ax = fig.add_subplot(111, projection='polar')

values.append(values[0]) # [45, 49, 49, 65, 65, 45, 45]
theta = theta.tolist() + [theta[0]] # [0.0, 1.0471975511965976, 2.0943951023931953, 3.141592653589793, 4.1887902047863905, 5.235987755982988, 0.0]

ax.plot(theta, values)
ax.fill(theta, values, alpha=0.5)

plt.show()
```

![image-20220212220226862](https://user-images.githubusercontent.com/70505378/153712760-964b732c-a359-4dd9-ba9e-1a189967c8d7.png)

### 커스텀 및 조정

- `set_thetagrids` : 각도에 따른 그리드 및 ticklabels 변경
- `set_theta_offset` : 시작 각도 변경 

```python
fig = plt.figure(figsize=(4, 4))
ax = fig.add_subplot(111, projection='polar')

values = pokemon.iloc[0][stats].to_list()
values.append(values[0])
theta = theta

ax.plot(theta, values)
ax.fill(theta, values, alpha=0.5)

ax.set_thetagrids([n*60 for n in range(6)], stats)
ax.set_theta_offset(np.pi/2)
ax.set_rmax(100)

plt.show()
```

![image-20220212220730160](https://user-images.githubusercontent.com/70505378/153712761-69341b19-3a2d-4baf-bb0a-cca68e4437ff.png)

```python
fig = plt.figure(figsize=(14, 4))

for idx in range(3):
    ax = fig.add_subplot(1,3,idx+1, projection='polar')

    values = pokemon.iloc[idx][stats].to_list()
    values.append(values[0])

    ax.plot(theta, values, color='forestgreen')
    ax.fill(theta, values, color='forestgreen', alpha=0.3)
    
    ax.set_rmax(100)
    ax.set_thetagrids([n*60 for n in range(6)], stats)
    ax.set_theta_offset(np.pi/2)
    
plt.show()
```

![image-20220212220753455](https://user-images.githubusercontent.com/70505378/153712762-df40ce7d-de05-4f29-a03c-25b6887fa3dd.png)























<br>

<br>

# 참고 자료

* [Radar Chart](https://matplotlib.org/stable/gallery/specialty_plots/radar_chart.html)















<br>
