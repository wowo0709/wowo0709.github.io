---
layout: single
title: "[AITech] 20220203 - Matplotlib Line Plot"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 학습 내용 정리

### 기본 Line Plot

**Line Plot**은 연속적으로 변화하는 값을 순서대로 **점**으로 나타내고, **점들을 선으로 잇습니다.**

따라서 시간/순서에 대한 변화를 나타내기에 적합하여 추세를 살피기 위한 용도로 사용됩니다. 

Line plot을 그릴 대는 `ax.plot()` 메서드를 사용합니다. 

```python
fig, axes = plt.subplots(1, 2, figsize=(12, 7))

x1 = [1, 2, 3, 4, 5]
x2 = [1, 3, 2, 4, 5] # x축 순서 엉망
y = [1, 3, 2, 1, 5]

axes[0].plot(x1, y)
axes[1].plot(x2, y)

plt.show()
```

![image-20220204143349802](https://user-images.githubusercontent.com/70505378/152480669-bf6bf232-a99d-447a-bab0-90f34d9f4c7d.png)

위의 두 그래프를 대조하면 선을 앞에서부터 순서대로 긋는 것이 아니라, 전달된 점을 순서대로 잇는 방식으로 동작한다는 것을 알 수 있습니다. 

#### Line Plot의 요소

Line plot에서 변주를 줄 수 있는 요소로는 다음의 것들이 있습니다. 

* **색상**: `color`
* **마커**: `marker`, `markersize`
  * [마커의 종류](https://matplotlib.org/stable/api/markers_api.html)
* **선의 종류**: `linestyle`, `linewidth`
  * **solid(default), dashed(--), dashdot(-.), dotted(:), None**

```python
fig, ax = plt.subplots(1, 1, figsize=(5, 5))

np.random.seed(97)
x = np.arange(7)
y = np.random.rand(7)

ax.plot(x, y,
        color='black',
        marker='*',
        linestyle='solid', 
       )

plt.show()
```

![image-20220204143737971](https://user-images.githubusercontent.com/70505378/152480671-7e1dd731-ab85-4c10-83a9-fd1ccf4f33cc.png)

#### Line Plot을 위한 전처리

매우 촘촘하고 변동 폭이 큰 데이터의 경우, Noise로 인해 패턴 및 추세 파악이 어려울 수 있습니다. 

Noise의 방해를 줄이기 위해 **smoothing**을 사용할 수 있는데요, 그래프를 그릴 때 적용하는 방법도 있지만 여기서는 데이터프레임 자체에 **이동평균**을 적용하는 방법을 살펴보겠습니다. 

```python
# 데이터셋
google = stock[stock['symbol']=='GOOGL']
google_rolling = google.rolling(window=20).mean() # 이동평균
# 그래프
fig, axes = plt.subplots(2, 1, figsize=(12, 7), dpi=300, sharex=True)

axes[0].plot(google.index,google['close'])
axes[1].plot(google_rolling.index,google_rolling['close'])

plt.show()
```

![image-20220204144105142](https://user-images.githubusercontent.com/70505378/152480654-3dc2ffd0-b33e-4de3-9e29-722bf01f0e0e.png)









<br>

### 더 정확한 Line Plot

#### 추세에 집중

**Line Plot**의 가장 큰 목적은 **추세**의 확인이므로, bar plot과 다르게 꼭 축을 0에서 시작할 필요는 없습니다. 

같은 맥락으로, 너무 구체적인 line plot보다는 어느정도 생략된 line plot이 더 나을 수 있습니다. 

* Grid, Annotate 등 모두 제거
* 디테일한 정보는 표로 제공하는 것을 추천

```python
from matplotlib.ticker import MultipleLocator # 축의 단위를 지정

fig = plt.figure(figsize=(12, 5))


np.random.seed(970725)

x = np.arange(20)
y = np.random.rand(20)


# Ax1(복잡, 많은 정보 -> 추세와 함께 정확한 수치값)
ax1 = fig.add_subplot(121)
ax1.plot(x, y,
         marker='o',
         linewidth=2)

ax1.xaxis.set_major_locator(MultipleLocator(1))
ax1.yaxis.set_major_locator(MultipleLocator(0.05))    
ax1.grid(linewidth=0.3)    


# Ax2(단순, 적은 정보 -> 추세만)
ax2 = fig.add_subplot(122)
ax2.plot(x, y,
       linewidth=2,)

ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)




ax1.set_title(f"Line Plot (information)", loc='left', fontsize=12, va= 'bottom', fontweight='semibold')
ax2.set_title(f"Line Plot (clean)", loc='left', fontsize=12, va= 'bottom', fontweight='semibold')


plt.show()
```

![image-20220204144336247](https://user-images.githubusercontent.com/70505378/152480655-13ac3979-08f3-43f8-b7ed-3c46d978bfb7.png)

#### 간격

Line plot에서는 **간격이 오해를 불러일으킬 수 있습니다.**

* 실제 데이터를 기준으로 동일 간격으로 나타낼 때: 기울기 정보(변화율)의 오해
* 실제 데이터와 상관없이 동일 간격으로 나타낼 때: 없는 데이터를 있다고 오해

따라서, 간격은 실제 데이터와 상관없이 일정하게 하되 각 데이터에 점을 찍어주는 것이 좋습니다. 

```python
x = [2, 3, 5, 7, 9]
y = [2, 3, 4, 7, 10]

fig, ax = plt.subplots(1, 3, figsize=(13, 4))
ax[0].plot([str(i) for i in x], y)
ax[1].plot(x, y)
ax[2].plot(x, y, marker='o') # best!

plt.show()
```

![image-20220204144608691](https://user-images.githubusercontent.com/70505378/152480656-51925c87-1a4f-4134-80e9-a9d97d9ad248.png)

#### 보간

**보간(Interpolation)**이란 데이터와 데이터 사이 데이터가 존재하지 않는 부분을 채워넣는 기법에 관한 것입니다. 

* Moving Average
* Smooth Curve with Scipy
  * `scipy.interpolate.make_interp_spline()`
  * `scipy.interpolate.interp1d()`
  * `scipy.ndimary.gaussian_filter1d()`

Presentation에는 좋은 방법일 수 있으나, 없는 데이터를 있다고 오해하게 할 수 있고 작은 차이를 없앨 수 있기 때문에 **일반적인 분석에서는 지양**할 것을 권장합니다. 

- [부드러운 곡선 그리기](https://www.delftstack.com/howto/matplotlib/matplotlib-plot-smooth-curve/)
- Noise가 많은 데이터일수록 이동평균만으로 부드러운 곡선이 그려질 수 있다!

#### 이중 축 사용

한 plot에 대해 2개의 다른 축을 **이중 축(dual axis)**이라고 하며, 2가지 방법으로 사용합니다. 

1. 같은 축을 공유하는 서로 다른 데이터를 표현

   * `.twinx()` 사용

   ```python
   fig, ax1 = plt.subplots(figsize=(12, 7), dpi=150)
   
   # First Plot
   color = 'royalblue'
   
   ax1.plot(google.index, google['close'], color=color)
   ax1.set_xlabel('date')
   ax1.set_ylabel('close price', color=color)  
   ax1.tick_params(axis='y', labelcolor=color)
   
   # # Second Plot
   ax2 = ax1.twinx()  
   color = 'tomato'
   
   ax2.plot(google.index, google['volume'], color=color)
   ax2.set_ylabel('volume', color=color)  
   ax2.tick_params(axis='y', labelcolor=color)
   
   ax1.set_title('Google Close Price & Volume', loc='left', fontsize=15)
   plt.show()
   ```

   ![image-20220204145340705](https://user-images.githubusercontent.com/70505378/152480659-6fcb04ad-2188-4d6d-8706-6bad46e5dc02.png)

2. 한 데이터에 대해 서로 다른 단위의 축을 사용(ex. radian과 degree)

   * `.secondary_xaxis()`, `.secondary_yaxis()` 사용

   ```python
   def deg2rad(x):
       return x * np.pi / 180
   
   def rad2deg(x):
       return x * 180 / np.pi
   
   fig, ax = plt.subplots()
   x = np.arange(0, 360)
   y = np.sin(2 * x * np.pi / 180)
   ax.plot(x, y)
   ax.set_xlabel('angle [degrees]')
   ax.set_ylabel('signal')
   ax.set_title('Sine wave')
   # 2번재 x축 생성
   secax = ax.secondary_xaxis('top', functions=(deg2rad, rad2deg))
   secax.set_xlabel('angle [rad]')
   
   plt.show()
   ```

   ![image-20220204145445803](https://user-images.githubusercontent.com/70505378/152480663-d75372e3-67a6-4e4a-9536-d53ce3c9103e.png)

다만, **이중 축의 사용보다는 2개의 plot을 그리는 것이 권장됩니다.**

#### ETC

* 라인 끝 단에 레이블을 추가하면 식별에 도움(범례 대신)

  ```python
  fig = plt.figure(figsize=(12, 5))
  
  x = np.linspace(0, 2*np.pi, 1000)
  y1 = np.sin(x)
  y2 = np.cos(x)
  
  ax = fig.add_subplot(111, aspect=1)
  ax.plot(x, y1,
         color='#1ABDE9',
         linewidth=2,)
  
  ax.plot(x, y2,
         color='#F36E8E',
         linewidth=2,)
  # 라인 끝 단에 label 표시
  ax.text(x[-1]+0.1, y1[-1], s='sin', fontweight='bold',
           va='center', ha='left', 
           bbox=dict(boxstyle='round,pad=0.3', fc='#1ABDE9', ec='black', alpha=0.3))
  
  ax.text(x[-1]+0.1, y2[-1], s='cos', fontweight='bold',
           va='center', ha='left', 
           bbox=dict(boxstyle='round,pad=0.3', fc='#F36E8E', ec='black', alpha=0.3))
  
  
  ax.spines['top'].set_visible(False)
  ax.spines['right'].set_visible(False)
  
  plt.show()
  ```

  ![image-20220204150324448](https://user-images.githubusercontent.com/70505378/152480665-012ebd67-0b2b-4a42-8c1c-019722838d67.png)

* Min/Max 정보(또는 원하는 포인트)는 추가해주면 도움이 될 수 있음

  ```python
  fig = plt.figure(figsize=(7, 7))
  
  np.random.seed(97)
  
  x = np.arange(20)
  y = np.random.rand(20)
  
  ax = fig.add_subplot(111)
  ax.plot(x, y,
         color='lightgray',
         linewidth=2,)
  
  ax.set_xlim(-1, 21)
  
  # max
  ax.plot([-1, x[np.argmax(y)]], [np.max(y)]*2,
          linestyle='--', color='tomato'
         )
  
  ax.scatter(x[np.argmax(y)], np.max(y), 
              c='tomato',s=50, zorder=20)
  
  # min
  ax.plot([-1, x[np.argmin(y)]], [np.min(y)]*2,
          linestyle='--', color='royalblue'
         )
  ax.scatter(x[np.argmin(y)], np.min(y), 
              c='royalblue',s=50, zorder=20)
  
  plt.show()
  ```

  ![image-20220204150354406](https://user-images.githubusercontent.com/70505378/152480668-a3257cf1-d272-43a5-ae96-8eeea6101a94.png)

* 보다 연한 색을 사용하여 uncertainty 정보 표현 가능(신뢰구간, 분산 등)

  ![image-20220204152825730](https://user-images.githubusercontent.com/70505378/152482782-bb9a4d34-0254-495e-9f3c-ea2b771d8b84.png)

기본적으로 line plot은 scatter plot이나 다른 기법들과 융합하여 많이 사용됩니다. 















<br>

<br>

## 참고 자료

* 기본 Line Plot - Line Plot의 요소: [마커의 종류](https://matplotlib.org/stable/api/markers_api.html)
* 더 정확한 Line Plot - 보간: [부드러운 곡선 그리기](https://www.delftstack.com/howto/matplotlib/matplotlib-plot-smooth-curve/)

