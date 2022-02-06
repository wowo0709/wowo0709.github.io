---

layout: single
title: "[AITech] 20220204 - Matplotlib Facet API"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 학습 내용

### Facet

**Facet**이란 **화면 상의 분할**을 의미합니다. 

화면 상에 View를 분할 및 추가하여 다양한 관점을 전달할 수 있습니다. 

* 같은 데이터셋에 **서로 다른 인코딩(bar, line, scatter, heapmap...)**을 통해 다른 인사이트를 전달
* **같은 인코딩 방법으로 동시에** 여러 feature를 볼 수 있음
* 큰 틀에서 볼 수 없는 **부분 집합을 세세**하게 보여주거나 **요약된 정보**를 보여줄 수 있음



<br>

### Figure & Ax Properties

#### dpi

DPI는 인치에 해당하는 dot 수(dots per inch)를 정하는 인자로 해상도를 의미합니다. 기본값은 100입니다.

150, 200, 300 값 등을 조정하며 원하는 해상도로 출력할 수 있습니다.

```python
fig = plt.figure(dpi=150)
# fig, ax = plt.subplots(dpi=150)
```

위처럼 처음에 figure를 생성할 때 `dpi` 인자로 설정할 수 있습니다. 

dpi는 주로 **생성한 그래프를 최종적으로 저장할 때** 따로 지정하여 사용됩니다. 







#### sharex, sharey

개별 서브플롯들이 x축 또는 y축 스케일을 공유합니다. 이를 할 수 있는 방법에는 2가지가 있습니다. 

1. `plt.subplots()`에서 지정하기

   ```python
   fig, axes = plt.subplots(1, 2, sharey=True)
   
   axes[0].plot([1, 2, 3], [1, 4, 9])
   axes[1].plot([1, 2, 3], [1, 2, 3])
   plt.show()
   ```

   ![image-20220206154537229](https://user-images.githubusercontent.com/70505378/152671340-20f355c0-75aa-4a98-9b09-a44df1880f77.png)

2. `fig.add_plots()`에서 지정하기

   ```python
   fig = plt.figure()
   ax1 = fig.add_subplot(121)
   ax1.plot([1, 2, 3], [1, 4, 9])
   ax2 = fig.add_subplot(122, sharey=ax1)
   ax2.plot([1, 2, 3], [1, 2, 3])
   plt.show()
   ```

   ![image-20220206154347915](https://user-images.githubusercontent.com/70505378/152671339-2c6adc74-0f1a-4f0c-aace-4409a494acb8.png)





#### squeeze와 flatten

`subplots()`를 사용하면 전달하는 서브플롯의 개수에 따라 다음과 같이 ax 배열이 생성됩니다. 

- 1 x 1 : 객체 1개 (`ax`)
- 1 x N 또는 N x 1 : 길이 N 배열 (`axes[i]`)
- N x M : N by M 배열 (`axes[i][j]`)

이렇게 되면, 각 경우에 따라 반복문을 수정해야 하는 문제가 발생합니다. 이럴 때, `squeeze=False` 로 전달하면 항상 2차원으로 배열을 받을 수 있어 반복문을 사용하기에 유용합니다. 

```python
n, m = 1, 3

fig, axes = plt.subplots(n, m, squeeze=False, figsize=(m*2, n*2))
idx = 0
for i in range(n):
    for j in range(m):
        axes[i][j].set_title(idx)
        axes[i][j].set_xticks([])
        axes[i][j].set_yticks([])
        idx+=1

plt.show()
```

![image-20220206155026546](https://user-images.githubusercontent.com/70505378/152671341-fe965cf8-0a3f-4959-9830-8e123248ece7.png)

만약 이와 반대로 항상 1중 반복문을 사용하고 싶다면, `flatten()` 메서드를 사용할 수 있습니다. 

```python
n, m = 2, 3

fig, axes = plt.subplots(n, m, figsize=(m*2, n*2)) # (2,3)

for i, ax in enumerate(axes.flatten()): # (6,)
    ax.set_title(i)
    ax.set_xticks([])
    ax.set_yticks([])


plt.show()
```

![image-20220206155142245](https://user-images.githubusercontent.com/70505378/152671343-2df0af12-eec9-4d84-82fc-3e65dac66ec1.png)

인덱스만 바뀌는 것이지 서브플롯의 배치 자체가 바뀌는 것은 아닙니다. 



#### aspect

`aspect`는 서브플롯의 '세로/가로' 비율입니다. 

```python
fig = plt.figure(figsize=(18,6))
ax1 = fig.add_subplot(131, aspect=1)
ax2 = fig.add_subplot(132, aspect=0.5)
ax3 = fig.add_subplot(133, aspect=0.5)

ax3.set_ylim(0,2)

plt.show()
```

![image-20220206155748307](https://user-images.githubusercontent.com/70505378/152671344-15ca9852-c9ae-4280-b20a-9154fee8e4a1.png)





<br>

### Grid Spec

#### add_gridspec

* `fig.add_gridspec(row, col)`: NxM 의 그리드에서 슬라이싱으로 서브플롯을 배치할 수 있습니다. 

```python
fig = plt.figure(figsize=(8, 5))

gs = fig.add_gridspec(3, 3) # make 3 by 3 grid (row, col)

ax = [None for _ in range(5)]

ax[0] = fig.add_subplot(gs[0, :]) 
ax[0].set_title('gs[0, :]')

ax[1] = fig.add_subplot(gs[1, :-1])
ax[1].set_title('gs[1, :-1]')

ax[2] = fig.add_subplot(gs[1:, -1])
ax[2].set_title('gs[1:, -1]')

ax[3] = fig.add_subplot(gs[-1, 0])
ax[3].set_title('gs[-1, 0]')

ax[4] = fig.add_subplot(gs[-1, -2])
ax[4].set_title('gs[-1, -2]')
# x축, y축 눈금 제거
for ix in range(5):
    ax[ix].set_xticks([])
    ax[ix].set_yticks([])

plt.tight_layout()
plt.show()
```

![image-20220206160342249](https://user-images.githubusercontent.com/70505378/152671345-0ae56043-fc66-4d17-886f-d3067a5af9b4.png)





#### subplot2grid

* `plt.subplot2grid((grid_row, grid_col), (start_row, start_col), rowspan, colspan)`: NxM 그리드에서 시작점(왼쪽 위 모서리 기준)과 가로,세로 길이로 표현할 수 있습니다. 
  * **add_gridspec**보다 사용하기 약간 까다로운 것 같습니다. 

```python
fig = plt.figure(figsize=(8, 5)) # initialize figure

ax = [None for _ in range(6)] # list to save many ax for setting parameter in each

ax[0] = plt.subplot2grid((3,4), (0,0), colspan=4)
ax[1] = plt.subplot2grid((3,4), (1,0), colspan=1)
ax[2] = plt.subplot2grid((3,4), (1,1), colspan=1)
ax[3] = plt.subplot2grid((3,4), (1,2), colspan=1)
ax[4] = plt.subplot2grid((3,4), (1,3), colspan=1,rowspan=2)
ax[5] = plt.subplot2grid((3,4), (2,0), colspan=3)


for ix in range(6): 
    ax[ix].set_title('ax[{}]'.format(ix)) # make ax title for distinguish:)
    ax[ix].set_xticks([]) # to remove x ticks
    ax[ix].set_yticks([]) # to remove y ticks
    
fig.tight_layout()
plt.show()
```

![image-20220206161046376](https://user-images.githubusercontent.com/70505378/152671346-e566bcd2-f171-4229-8661-a19306b6155b.png)





#### add_axes

* `fig.add_axes([x, y, dx, dy])`: 임의의 위치에 서브플롯을 그릴 수 있습니다.
  * 왼쪽 아래 지점 기준입니다.  

```python
fig = plt.figure(figsize=(8, 5))

ax = [None for _ in range(3)]


ax[0] = fig.add_axes([0.1,0.1,0.8,0.4]) # x, y, dx, dy
ax[1] = fig.add_axes([0.15,0.6,0.25,0.6])
ax[2] = fig.add_axes([0.5,0.6,0.4,0.3])

for ix in range(3):
    ax[ix].set_title('ax[{}]'.format(ix))
    ax[ix].set_xticks([])
    ax[ix].set_yticks([])

plt.show()
```

![image-20220206161423767](https://user-images.githubusercontent.com/70505378/152671347-7ef44ed4-db73-4c90-810a-2b6fd04a56de.png)







#### inset_axes

* `ax.inset_axes([x, y, dx, dy])`: 서브플롯 내에 또 다른 작은 서브플롯을 추가할 때 사용할 수 있습니다. 

```python
fig, ax = plt.subplots()
axin = ax.inset_axes([0.85, 0.8, 0.15, 0.2])
plt.show()
```

![image-20220206162106625](https://user-images.githubusercontent.com/70505378/152671338-ea0a4f44-9b9d-4d95-99e5-be057a2eedb7.png)

아래와 같이 간단한 요약 정보를 보여줄 대 사용할 수 있습니다. 

```python
fig, ax = plt.subplots()

color=['royalblue', 'tomato']
ax.bar(['A', 'B'], [1, 2],
       color=color
      )

ax.margins(0.2)
axin = ax.inset_axes([0.85, 0.8, 0.15, 0.2])
axin.pie([1, 2], colors=color, 
         autopct='%1.0f%%')
plt.show()
```

![image-20220206161730705](https://user-images.githubusercontent.com/70505378/152671348-3f6f802c-b569-4e15-ab2c-5d38f60270d0.png)







#### make_axes_locatable

* `make_axes_locatable(ax)`: 일반적으로 color bar에 많이 사용합니다. 

```python
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

fig, ax = plt.subplots(1, 1)

ax_divider = make_axes_locatable(ax)
ax = ax_divider.append_axes("right", size="7%", pad="2%")

plt.show()
```

![image-20220206162011166](https://user-images.githubusercontent.com/70505378/152671350-bc79e393-0ed4-4930-87ef-d946cfb66a6d.png)

```python
fig, ax = plt.subplots(1, 1)

# 이미지를 보여주는 시각화
# 2D 배열을 색으로 보여줌
im = ax.imshow(np.arange(100).reshape((10, 10)))

divider = make_axes_locatable(ax)
cax = divider.append_axes("right", size="5%", pad=0.05)

fig.colorbar(im, cax=cax)
plt.show()
```

![image-20220206162031027](https://user-images.githubusercontent.com/70505378/152671351-0ee7dac5-bdb6-4552-bc8c-9c413881ff71.png)





















<br>

<br>

## 참고 자료

* 

















<br>
