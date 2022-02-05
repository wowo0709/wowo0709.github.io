---
layout: single
title: "[AITech] 20220203 - Matplotlib Bar Plot"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 학습 내용 정리

### 기본 Bar Plot

**Bar plot**은 **카테고리에 따른 수치값을 비교**하기에 적합한 방법입니다. (개별 비교, 그룹 비교 모두 적합)

* `bar()`: x축에 카테고리, y축에 값을 표기. (default)
* `barh()`: y축에 카테고리, x축에 값을 표기. (카테고리가 많을 때 적합)

```python
# figure와 ax를 동시에 생성
fig, axes = plt.subplots(1, 2, figsize=(12, 7))

x = list('ABCDE')
y = np.array([1, 2, 3, 4, 5])

axes[0].bar(x, y)
axes[1].barh(x, y)

plt.show()
```

![image-20220204103053796](https://user-images.githubusercontent.com/70505378/152476947-e65074cd-d167-4891-acfb-08c1fe8d281d.png)





<br>

### 다양한 Bar Plot

실습용 데이터셋으로 [StudentsPerformance.csv](http://roycekimmons.com/tools/generated_data/exams) 파일을 사용합니다. 

```python
student = pd.read_csv('./StudentsPerformance.csv')
student.sample(5) # 무작위 샘플링
```

![image-20220204103352230](https://user-images.githubusercontent.com/70505378/152476950-3efcb65f-96c9-4521-b226-0468c28141de.png)

```python
group = student.groupby('gender')['race/ethnicity'].value_counts().sort_index()
display(group)
print(student['gender'].value_counts())
'''
gender  race/ethnicity
female  group A            36
        group B           104
        group C           180
        group D           129
        group E            69
male    group A            53
        group B            86
        group C           139
        group D           133
        group E            71
Name: race/ethnicity, dtype: int64
female    518
male      482
Name: gender, dtype: int64
'''
```

바 플롯을 그리는 대표적인 방법에는 크게 4가지가 있으며, 각각에 대해 하나씩 알아보겠습니다. 

<br>

#### Multiple Bar Plot

**Multiple Bar Plot**은 가장 기본으로, 여러 개의 서브 플롯 각각에 그래프를 그리는 것입니다. 

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 7))
axes[0].bar(group['male'].index, group['male'], color='royalblue')
axes[1].bar(group['female'].index, group['female'], color='tomato')
plt.show()
```

![image-20220204103800411](https://user-images.githubusercontent.com/70505378/152476953-5cd0508d-cecc-486a-b60a-2609665d8233.png)

위 사진을 보면 y축의 scale이 일치하지 않는데요, 이를 2가지 방법으로 해결할 수 있습니다. 

1. 서브플롯 생성 시 `sharey` 파라미터 사용

   ```python
   fig, axes = plt.subplots(1, 2, figsize=(15, 7), sharey=True) # sharey
   axes[0].bar(group['male'].index, group['male'], color='royalblue')
   axes[1].bar(group['female'].index, group['female'], color='tomato')
   plt.show()
   ```

2. `ax.set_ylim()`으로 서브플롯에 개별적으로 y축 조정

   ```python
   fig, axes = plt.subplots(1, 2, figsize=(15, 7))
   axes[0].bar(group['male'].index, group['male'], color='royalblue')
   axes[1].bar(group['female'].index, group['female'], color='tomato')
   
   for ax in axes:
       ax.set_ylim(0, 200)
       
   plt.show()
   ```

두 경우 모두 y축의 scale이 일치하게 됩니다. 

#### Stacked Bar Plot

**Stacked Bar Plot**은 카테고리별 크기의 경향과 각 카테고리 안에 있는 수치값 비교를 동시에 할 수 있는 방법입니다. 하지만, **가독성이 그리 높지는 않아서** 많이 사용하지는 않습니다. 

Stack Bar Plot을 그릴 때는 세로로 쌓을 때는 두 번째 bar부터 `bottom` 인자를, 가로로 쌓을 때는 두번째 bar부터 `left` 인자를 전달하면 됩니다. 

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

group_cnt = student['race/ethnicity'].value_counts().sort_index()
axes[0].bar(group_cnt.index, group_cnt, color='darkgray')

axes[1].bar(group['male'].index, group['male'], color='royalblue')
axes[1].bar(group['female'].index, group['female'], bottom=group['male'], color='tomato') # bottom

for ax in axes:
    ax.set_ylim(0, 350)
    
plt.show()
```

![image-20220204104510345](https://user-images.githubusercontent.com/70505378/152476954-22de008d-aa7b-4336-9226-9efe3f714245.png)

##### Percentage Stack Bar Plot

**Percentage Stack Bar Plot**은 Stacked bar plot을 응용하여 전체에서 비율을 나타냅니다. 

```python
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

group = group.sort_index(ascending=False) # 역순 정렬
total=group['male']+group['female'] # 각 그룹별 합


ax.barh(group['male'].index, group['male']/total, 
        color='royalblue')

ax.barh(group['female'].index, group['female']/total, 
        left=group['male']/total, 
        color='tomato')

ax.set_xlim(0, 1)
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)
# percentage 표기
for group_idx in group.index:

    gender = group_idx[0]
    group_cat = group_idx[1]
    val = (group / total)[group_idx]

    if gender == 'male':
        ax.text((val) / 2, group_cat, 
                s=f'{val * 100:.1f}%',
                ha='center',
                fontweight='bold'
                )
    else:
        ax.text(((1-val) + (val/2)), group_cat, 
                s=f'{val * 100:.1f} %',
                ha='center', 
                fontweight='bold'
                )

plt.show()
```

![image-20220204225431702](https://user-images.githubusercontent.com/70505378/152631441-c88f1ac4-a7e4-45e7-9eaf-4d19b1376049.png)

✋ `ax.bar_label(plot, label_type='center')`을 이용해 그 비율을 표시할 수도 있습니다. 

```python
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

group = group.sort_index(ascending=False) # 역순 정렬
total=group['male']+group['female'] # 각 그룹별 합


rects1 = ax.barh(group['male'].index, group['male']/total, 
        color='royalblue')
ax.bar_label(rects1, label_type='center')
rects2 = ax.barh(group['female'].index, group['female']/total, 
        left=group['male']/total, 
        color='tomato')
ax.bar_label(rects2, label_type='center')


ax.set_xlim(0, 1)
for s in ['top', 'bottom', 'left', 'right']:
    ax.spines[s].set_visible(False)

plt.show()
```

![image-20220204225658995](https://user-images.githubusercontent.com/70505378/152631443-2f73d9a8-4a2e-437d-9f00-3ec9a8e7661c.png)





#### Overlapped Bar Plot

**Overlapped Bar Plot**은 2개 이상의 bar plot을 겹쳐서 나타내는 방법입니다. 이 방법은 bar plot의 개수가 2개일 때 추천되며, 3개 이상일 때는 area plot(Seaborn 포스팅에서 다룹니다)을 추천합니다. 

Overlapped bar plot을 위해서는 각 bar plot의 `alpha`(투명도) 값을 조절해야 합니다. 

```python
group = group.sort_index() # 다시 정렬

fig, axes = plt.subplots(2, 2, figsize=(12, 12))
axes = axes.flatten()
# 여러 값의 alpha로 실험
for idx, alpha in enumerate([1, 0.7, 0.5, 0.3]):
    axes[idx].bar(group['male'].index, group['male'], 
                  color='royalblue', 
                  alpha=alpha)
    axes[idx].bar(group['female'].index, group['female'],
                  color='tomato',
                  alpha=alpha)
    axes[idx].set_title(f'Alpha = {alpha}')
    
for ax in axes:
    ax.set_ylim(0, 200)
     
plt.show()
```

![image-20220204111140006](https://user-images.githubusercontent.com/70505378/152476957-1a575948-67a7-4ee1-858a-7b3d994fb3cd.png)







#### Grouped Bar Plot

**Grouped Bar Plot**은 같은 카테고리에 속하는 bar plot을 바로 옆에 이웃하게 배치하는 방법입니다. 

Matplotlib으로는 비교적 구현이 까다로우며, 추후 Seaborn에서 이에 대해 다룹니다. Matplotlib으로 이를 구현할 수 있는 방법으로는 다음 3가지 테크닉이 필요합니다. 

1. x축 위치 조정
2. width 조정
3. xticks, xticklabels 사용

원래 x축이 0, 1, 2, 3로 시작한다면

* 한 그래프는 0-width/2, 1-width/2, 2-width/2 로 구성하면 되고
* 한 그래프는 0+width/2, 1+width/2, 2+width/2 로 구성하면 됩니다.

```python
fig, ax = plt.subplots(1, 1, figsize=(12, 7))

idx = np.arange(len(group['male'].index)) # x축 조정 목적
width=0.35 # 너비 조정 목적
# 1. x축 위치 조정 2. width 조정
ax.bar(idx-width/2, group['male'], 
       color='royalblue',
       width=width)

ax.bar(idx+width/2, group['female'], 
       color='tomato',
       width=width)
# 3. x축 라벨명 조정
ax.set_xticks(idx)
ax.set_xticklabels(group['male'].index)
    
plt.show()
```

그리고 여기서 규칙을 찾으면, 

> - 2개 : -1/2, +1/2
> - 3개 : -1, 0, +1 (-2/2, 0, +2/2)
> - 4개 : -3/2, -1/2, +1/2, +3/2
>
> 그렇다면 index i(zero-index)에 대해서는 다음과 같이 x좌표를 계산할 수 있습니다.
>
> ![image-20220204140441568](https://user-images.githubusercontent.com/70505378/152476961-0f9d794f-5e7a-40a6-aa32-f0e62dfd6c74.png)
>

```python
fig, ax = plt.subplots(1, 1, figsize=(13, 7))

x = np.arange(len(group_list))
width=0.12

for idx, g in enumerate(edu_lv):
    ax.bar(x+(-len(edu_lv)+1+2*idx)*width/2, group[g], 
       width=width, label=g)

ax.set_xticks(x)
ax.set_xticklabels(group_list)
ax.legend()    
    
plt.show()
```

![image-20220204140531932](https://user-images.githubusercontent.com/70505378/152476962-f0fa3e33-e196-4a28-8169-8ac1dbc130e1.png)







<br>

### 더 정확한 Bar Plot

#### Principle of Proportion Ink

**Principle of Proportion Ink**는 **실제값과 그래픽으로 표현되는 잉크의 양은 비례해야 한다**는 것입니다. 

이로부터 2가지 사실을 직관적으로 알 수 있습니다. 

1. x축 시작은 0부터!
2. y축 범위를 임의로 자르지 않는다!

이 원칙은 비단 bar plot 뿐 아니라, area plot, donut chart 등 다수의 시각화에서 적용되는 원칙입니다. 

```python
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

idx = np.arange(len(score.index))
width=0.3

for ax in axes:
    ax.bar(idx-width/2, score['male'], 
           color='royalblue',
           width=width)

    ax.bar(idx+width/2, score['female'], 
           color='tomato',
           width=width)

    ax.set_xticks(idx)
    ax.set_xticklabels(score.index)
# 비교를 강조할 수는 있지만, 정확한 수치 인식 불가
axes[0].set_ylim(60, 75)
    
plt.show()
```

![image-20220204140931469](https://user-images.githubusercontent.com/70505378/152476963-3bf8fcb2-6011-46aa-949d-0343f81b26f6.png)

위에서 왼쪽 그래프는 y축 범위를 임의로 조정했기 때문에 비교를 강조할 수는 있지만, 실제 수치값과의 괴리가 존재합니다. 

반면 오른쪽 그래프는 비교는 덜 될지언정, 수치값을 표현할 수 있습니다. 만약 오른쪽 그래프에서 **비교를 강조하고 싶다면 y축을 늘리는 방법을 사용**하는 것이 좋습니다. 

```python
fig, ax = plt.subplots(1, 1, figsize=(6, 10)) # figsize
# 나머지 코드는 동일
# ...
```

![image-20220204141244190](https://user-images.githubusercontent.com/70505378/152476964-9c5bc36e-265c-49e5-87ea-f07c41f4c22d.png)

#### 데이터 정렬하기

더 정확한 정보를 전달하기 위해서는 Pandas에서 `sort_values()`, `sort_index()` 등의 메서드를 사용하여 데이터를 정렬해주는 과정이 필수적입니다. 

* 데이터의 종류에 따라 다음 기준으로
  * **시계열**: 시간 순
  * **수치형**: 크기 순
  * **순서형**: 범주의 순서대로
  * **명목형**: 범주의 값 따라 정렬

#### 적절한 공간 사용

공간의 사용 측면에서 다음의 것들을 조정할 수 있습니다. 

- X/Y axis Limit (`.set_xlim()`, `.set_ylime()`)
- Margins (`.margins()`)
- Gap (`width`)
- Spines (`.spines[spine].set_visible()`)

```python
group_cnt = student['race/ethnicity'].value_counts().sort_index()

fig = plt.figure(figsize=(15, 7))

ax_basic = fig.add_subplot(1, 2, 1)
ax = fig.add_subplot(1, 2, 2)

ax_basic.bar(group_cnt.index, group_cnt)
ax.bar(group_cnt.index, group_cnt,
       width=0.7,
       edgecolor='black',
       linewidth=2,
       color='royalblue'
      )

ax.margins(0.1, 0.1) # 서브 플롯의 내부 마진 (default: 0.05)

for s in ['top', 'right']: # 위쪽, 오른쪽 변 제거
    ax.spines[s].set_visible(False)

plt.show()
```

![image-20220204141707808](https://user-images.githubusercontent.com/70505378/152476967-ca8e2b94-05c2-443a-9eda-8bcfd22e7ad4.png)

#### 복잡함과 단순함

목적과 상관없는 필요없는 복잡함은 절대 지양하는 것이 좋습니다. 

우리가 고려해 볼 만한 요소들은 다음의 것들이 있습니다. 

* Grid(`.grid()`)
* Ticklabels(`.set_ticklabels()`)
* Text(`.text()` or `.annotate()`)

```python
group_cnt = student['race/ethnicity'].value_counts().sort_index()

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

for ax in axes:
    ax.bar(group_cnt.index, group_cnt,
           width=0.7,
           edgecolor='black',
           linewidth=2,
           color='royalblue',
           zorder=10
          )

    ax.margins(0.1, 0.1)

    for s in ['top', 'right']:
        ax.spines[s].set_visible(False)

axes[1].grid(zorder=0)

for idx, value in zip(group_cnt.index, group_cnt):
    axes[1].text(idx, value+5, s=value, # 텍스트 삽입
                 ha='center', 
                 fontweight='bold'
                )
        
plt.show()
```

![image-20220204142029522](https://user-images.githubusercontent.com/70505378/152476968-95b32d53-e23b-4cd8-9242-e7b59e794f55.png)

#### ETC

그 외에 다음의 요소들이 있습니다. 

* 오차 막대를 추가하여 Uncertainty 정보 전달(`xerr`, `yerr`)
* Bar 사이 Gap이 0이라면 -> **히스토그램**
  * `.hist()`를 사용하여 구현 가능
  * 연속된 느낌을 줄 수 있음
* 다양한 Text 정보 활용
  * 제목(`.set_title()`)
  * 라벨(`.set_xlabel()`, `.set_ylabel()`)

아래는 표준편차 값을 이용하여 오차막대를 추가한 bar plot 예제입니다. 

```python
fig, ax = plt.subplots(1, 1, figsize=(10, 10))

idx = np.arange(len(score.index))
width=0.3


ax.bar(idx-width/2, score['male'], 
       color='royalblue',
       width=width,
       label='Male',
       yerr=score_var['male'], # 오차막대 그리기
       capsize=10 # 오차막대 위아래 가로선 크기
      )

ax.bar(idx+width/2, score['female'], 
       color='tomato',
       width=width,
       label='Female',
       yerr=score_var['female'],
       capsize=10
      )

ax.set_xticks(idx)
ax.set_xticklabels(score.index)
ax.set_ylim(0, 100)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
# 범례 추가
ax.legend()
# 서브플롯 제목
ax.set_title('Gender / Score', fontsize=20)
# x, y 축 라벨
ax.set_xlabel('Subject', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')

plt.show()
```

![image-20220204142338057](https://user-images.githubusercontent.com/70505378/152476943-627ed710-2df4-40b6-b367-46e9b6072569.png)

















<br>

<br>

## 참고 자료

* 
