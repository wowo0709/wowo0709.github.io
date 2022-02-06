---

layout: single
title: "[AITech] 20220204 - Matplotlib More Tips"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Grid', 'Line&Span', 'Settings']
---



<br>

## 학습 내용

여기서는 앞서 살펴본 Matplotlib Text, Color, Facet 요소들 이외에 활용할 수 있는 요소들에 대해 간단히 살펴보겠습니다. 

### Grid

#### Basic grid

기본적인 Grid부터 살펴보겠습니다.

기본적인 그리드에서는 다음 파라미터를 살펴보겠습니다.

- `which` : major ticks, minor ticks
- `axis` : x, y
- `linestyle`
- `linewidth`
- `zorder` 


```python
fig, ax = plt.subplots()
ax.grid()
plt.show()
```


![output_3_0](https://user-images.githubusercontent.com/70505378/152671870-2d3bd755-78e3-405c-b013-27088209b18a.png)
    



```python
np.random.seed(970725)

x = np.random.rand(20)
y = np.random.rand(20)


fig = plt.figure(figsize=(16, 7))
ax = fig.add_subplot(1, 1, 1, aspect=1)


ax.scatter(x, y, s=150, 
           c='#1ABDE9',
           linewidth=1.5,
           edgecolor='black', zorder=10)

ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)

    
ax.grid(zorder=0, linestyle='--')    
ax.set_title(f"Default Grid", fontsize=15,va= 'center', fontweight='semibold')

plt.tight_layout()
plt.show()
```


​    
![output_4_0](https://user-images.githubusercontent.com/70505378/152671871-da1f8fa6-a86f-4eba-a750-81be627310a1.png)
​    

#### x + y = c

그리드 변경은 grid 속성을 변경하는 방법도 존재하지만 간단한 수식을 사용하면 쉽게 그릴 수 있습니다.


```python
fig = plt.figure(figsize=(16, 7))
ax = fig.add_subplot(1, 1, 1, aspect=1)


ax.scatter(x, y, s=150, 
           c=['#1ABDE9' if xx+yy < 1.0 else 'darkgray' for xx, yy in zip(x, y)],
           linewidth=1.5,
           edgecolor='black', zorder=10)

## Grid Part
x_start = np.linspace(0, 2.2, 12, endpoint=True)

for xs in x_start:
    ax.plot([xs, 0], [0, xs], linestyle='--', color='gray', alpha=0.5, linewidth=1)


ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)

ax.set_title(r"Grid ($x+y=c$)", fontsize=15,va= 'center', fontweight='semibold')

plt.tight_layout()
plt.show()
```


​    
![output_6_0](https://user-images.githubusercontent.com/70505378/152671872-953c3ce2-0cbe-4046-8824-bd606c782037.png)
​    

#### y = cx


```python
fig = plt.figure(figsize=(16, 7))
ax = fig.add_subplot(1, 1, 1, aspect=1)


ax.scatter(x, y, s=150, 
           c=['#1ABDE9' if yy/xx >= 1.0 else 'darkgray' for xx, yy in zip(x, y)],
           linewidth=1.5,
           edgecolor='black', zorder=10)

## Grid Part
radian = np.linspace(0, np.pi/2, 11, endpoint=True)

for rad in radian:
    ax.plot([0,2], [0, 2*np.tan(rad)], linestyle='--', color='gray', alpha=0.5, linewidth=1)


ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)

ax.set_title(r"Grid ($y=cx$)", fontsize=15,va= 'center', fontweight='semibold')

plt.tight_layout()
plt.show()
```


  ![output_8_0](https://user-images.githubusercontent.com/70505378/152671873-8808f1d2-16b7-4ee3-a412-011854ab83de.png)
    

#### 동심원


```python
fig = plt.figure(figsize=(16, 7))
ax = fig.add_subplot(1, 1, 1, aspect=1)


ax.scatter(x, y, s=150, 
           c=['darkgray' if i!=2 else '#1ABDE9'  for i in range(20)] ,
           linewidth=1.5,
           edgecolor='black', zorder=10)

## Grid Part
rs = np.linspace(0.1, 0.8, 8, endpoint=True)

for r in rs:
    xx = r*np.cos(np.linspace(0, 2*np.pi, 100))
    yy = r*np.sin(np.linspace(0, 2*np.pi, 100))
    ax.plot(xx+x[2], yy+y[2], linestyle='--', color='gray', alpha=0.5, linewidth=1)

    ax.text(x[2]+r*np.cos(np.pi/4), y[2]-r*np.sin(np.pi/4), f'{r:.1}', color='gray')

ax.set_xlim(0, 1.1)
ax.set_ylim(0, 1.1)

ax.set_title(r"Grid ($(x-x')^2+(y-y')^2=c$)", fontsize=15,va= 'center', fontweight='semibold')

plt.tight_layout()
plt.show()
```


​    
![output_10_0](https://user-images.githubusercontent.com/70505378/152671874-3dd5f9e4-ad60-4e2f-844b-f7655e1ba8ca.png)
​    


## 





<br>

### Line&Span

#### Line

- `axvline(x, ymin, ymax, color)`
- `axhline(y, xmin, xmax, color)`

직교좌표계에서 평행선을 원하는 부분 그릴 수도 있습니다.

선은 Plot으로 그리는게 더 편할 수 있기에 원하는 방식으로 그려주시면 됩니다.


```python
fig, ax = plt.subplots()

ax.set_aspect(1)
ax.axvline(0, color='red')
ax.axhline(0, color='green')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

plt.show()
```


​    
![output_14_0](https://user-images.githubusercontent.com/70505378/152671876-82615f93-a974-472f-8321-f0200848a5ba.png)
​    

ax의 전체 구간을 0, 1로 삼아 특정 부분에만 선을 그릴 수도 있습니다. 

다만 다음과 같이 특정 부분을 선으로 할 때는 오히려 plot이 좋습니다.


```python
fig, ax = plt.subplots()

ax.set_aspect(1)
ax.axvline(0, ymin=0.3, ymax=0.9, color='red')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

plt.show()
```


​    
![output_16_0](https://user-images.githubusercontent.com/70505378/152671878-910594e5-9bc5-4aec-b3c3-30941258def5.png)
​    

다음과 같이 활용할 수 있습니다. 

```python
fig, ax = plt.subplots(figsize=(10, 10))
ax.set_aspect(1)

math_mean = student['math score'].mean()
reading_mean = student['reading score'].mean()

ax.axvline(math_mean, color='gray', linestyle='--')
ax.axhline(reading_mean, color='gray', linestyle='--')

ax.scatter(x=student['math score'], y=student['reading score'],
           alpha=0.5,
           color=['royalblue' if m>math_mean and r>reading_mean else 'gray'  for m, r in zip(student['math score'], student['reading score'])],
           zorder=10,
          )

ax.set_xlabel('Math')
ax.set_ylabel('Reading')

ax.set_xlim(-3, 103)
ax.set_ylim(-3, 103)
plt.show()
```


​    
![output_17_0](https://user-images.githubusercontent.com/70505378/152671879-0d3f50e3-5a9d-4e38-9f62-655dc63c6f83.png)
​    

#### Span

- `axvspan(xmin, xmax, ymin, ymax, color)`
- `axhspan(ymin, ymax, xmin, xmax, color)`

선과 함께 다음과 같이 특정 부분 면적을 표시할 수 있습니다.


```python
fig, ax = plt.subplots()

ax.set_aspect(1)
ax.axvspan(0,0.5, color='red')
ax.axhspan(0,0.5, color='green')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

plt.show()
```


​    
![output_19_0](https://user-images.githubusercontent.com/70505378/152671880-8397af10-d745-415e-8eb3-f4dae4aabf7a.png)
​    



```python
fig, ax = plt.subplots()

ax.set_aspect(1)
ax.axvspan(0,0.5, ymin=0.3, ymax=0.9, color='red')

ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)

plt.show()
```


​    
![output_20_0](https://user-images.githubusercontent.com/70505378/152671881-8e370099-9c52-4cb8-9152-55c4e14f7e1f.png)
​    


특정 부분을 강조할 수도 있지만, 오히려 특정 부분의 주의를 없앨 수도 있습니다.


```python
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect(1)

math_mean = student['math score'].mean()
reading_mean = student['reading score'].mean()

ax.axvspan(-3, math_mean, color='gray', linestyle='--', zorder=0, alpha=0.3)
ax.axhspan(-3, reading_mean, color='gray', linestyle='--', zorder=0, alpha=0.3)

ax.scatter(x=student['math score'], y=student['reading score'],
           alpha=0.4, s=20,
           color=['royalblue' if m>math_mean and r>reading_mean else 'gray'  for m, r in zip(student['math score'], student['reading score'])],
           zorder=10,
          )

ax.set_xlabel('Math')
ax.set_ylabel('Reading')

ax.set_xlim(-3, 103)
ax.set_ylim(-3, 103)
plt.show()
```


​    
![output_22_0](https://user-images.githubusercontent.com/70505378/152671882-f347b99f-8ca1-4841-a5e1-2fc6d2955506.png)
​    

#### Spines

- `ax.spines` : 많은 요소가 있지만 대표적인 3가지를 살펴봅시다.
  - `set_visible`  
  - `set_linewidth`
  - `set_position`


```python
fig = plt.figure(figsize=(12, 6))

_ = fig.add_subplot(1,2,1)
ax = fig.add_subplot(1,2,2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)
plt.show()
```


​    
![output_24_0](https://user-images.githubusercontent.com/70505378/152671883-8a393330-33fb-4653-a56f-0efe82510894.png)
​    



```python
fig = plt.figure(figsize=(12, 6))

_ = fig.add_subplot(1,2,1)
ax = fig.add_subplot(1,2,2)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.spines['left'].set_position('center')
ax.spines['bottom'].set_position('center')
plt.show()
```


​    
![output_25_0](https://user-images.githubusercontent.com/70505378/152671884-6b01b4ee-3a90-4493-b4db-dad3a0fd1c17.png)
​    


축은 꼭 중심 외에도 원하는 부분으로 옮길 수 있습니다.

- `'center'` -> `('axes', 0.5)`
- `'zero'` -> `('data', 0.0)`


```python
fig = plt.figure(figsize=(12, 6))

ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

for ax in [ax1, ax2]:
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

ax1.spines['left'].set_position('center')
ax1.spines['bottom'].set_position('center')

ax2.spines['left'].set_position(('data', 0.3))
ax2.spines['bottom'].set_position(('axes', 0.2))

ax2.set_ylim(-1, 1)
plt.show()
```


![output_27_0](https://user-images.githubusercontent.com/70505378/152671885-1e75a426-1a15-4c58-be35-a584f3551b46.png)

​    



<br>

### Settings

- [Customizing Matplotlib with style sheets and rcParams](https://matplotlib.org/stable/tutorials/introductory/customizing.html)

#### mpl.rc

`plt.rcParams[]` 또는 `plt.rc()`를 사용하면 기본 설정을 바꿀 수 있습니다. 


```python
plt.rcParams['lines.linewidth'] = 2
plt.rcParams['lines.linestyle'] = ':'
```


```python
plt.rc('lines', linewidth=2, linestyle=':')
```

원래대로 되돌려 놓으려면 아래와 같이 하면 됩니다. 


```python
plt.rcParams.update(plt.rcParamsDefault)
```

#### Theme

`mpl.style.use()`를 사용하여 기본 테마를 바꿀 수 있습니다. 


```python
print(mpl.style.available)
'''
['Solarize_Light2', '_classic_test_patch', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn', 'seaborn-bright', 'seaborn-colorblind', 'seaborn-dark', 'seaborn-dark-palette', 'seaborn-darkgrid', 'seaborn-deep', 'seaborn-muted', 'seaborn-notebook', 'seaborn-paper', 'seaborn-pastel', 'seaborn-poster', 'seaborn-talk', 'seaborn-ticks', 'seaborn-white', 'seaborn-whitegrid', 'tableau-colorblind10']
'''

mpl.style.use('seaborn')
# mpl.style.use('./CUSTOM.mplstyle') # 커스텀을 사용하고 싶다면

plt.plot([1, 2, 3])
```


![output_34_2](https://user-images.githubusercontent.com/70505378/152671887-67d5a901-1114-4459-b5ff-2791f48dec68.png)
    















<br>

<br>

## 참고 자료

* **Settings**
  * [Customizing Matplotlib with style sheets and rcParams](https://matplotlib.org/stable/tutorials/introductory/customizing.html)

















<br>
