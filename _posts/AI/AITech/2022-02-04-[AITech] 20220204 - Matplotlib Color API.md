---
layout: single
title: "[AITech] 20220204 - Matplotlib Color API"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 학습 내용

### Color에 대한 이해

**색**은 우리에게 가장 효과적인 채널 구분을 줄 수 있는 요소 중 하나입니다. 이 색을 이해하기 위해서는 기본적으로 **HSL**을 이해하는 것이 중요합니다. 

* **Hue(색조)**: 빨강, 파랑, 초록 등 색상으로 생각하는 부분
  * 빨강에서 보라색까지 있는 스펙트럼에서 0~360으로 표현
* **Saturate(채도)**: 색의 선명도
  * 원색에서 회색을 섞는 정도로 표현 가능
  * 선명하다(진하다)와 탁하다(연하다)로 구분 가능
* **Lightness(광도)**: 색의 밝기
  * 원색에서 흰색/검은색 을 섞은 정도로 표현 가능
  * 밝다와 어둡다로 구분 가능
  * Value(명도)라고도 함

여기서 저를 포함한 많은 분들이 처음에는 **채도와 광도(명도)**를 잘 구분 못하는 경우가 많습니다. 이에 아래 그림을 첨부합니다. 

![image-20220206121225941](https://user-images.githubusercontent.com/70505378/152670066-3b90219e-7161-4e0e-9f93-d55253280f8a.png)

(출처: https://toramee2vr.tistory.com/45)

색은 분명 시각적으로 강력한 요소이지만, 역시나 남용하게 된다면 오히려 보는 사람으로 하여금 혼란을 야기할 것입니다. 

따라서, 중요한 것은 **내가 전하고자 하는 내용을 독자에게 오해없이 잘 전달하는 것**입니다. 

![image-20220206121554522](https://user-images.githubusercontent.com/70505378/152670067-07905d5f-ee12-4477-83c5-47af474fcea6.png)

또한 **색**은 상징의 의미를 가지고 있죠. 예를 들면 높은 온도에는 빨강, 낮은 온도에는 파랑을 사용합니다. 

이와 같이 이미 통용되고 있는 색이라면, 기존 정보와 느낌을 잘 활용하는 것이 좋습니다. 이미 사용하는 색에는 그 이유가 있기 때문입니다. 

<br>

### Color Palette의 종류

Matplotlib에서 Color를 다루는 방법은 복잡하고 다양합니다. 따라서, 여기서는 그 중 기본적인 내용 일부를 소개합니다. 

#### 범주형 색상

기본적으로 **범주형 색상**에서는 **같은 범주끼리는 같은 색상**으로 구분하면 되겠죠? 이때에는 **색의 차이**로 구분하는 것이 권장되며, **채도, 명도 등 다른 요소를 조정하는 것은 지양**하는 것이 좋습니다. 

또, 최대 8~10개의 색상까지만 사용하는 것이 좋고 그 이상은 '기타(etc)'로 묶는 것이 좋습니다. 

`plt.cm.get_cmap(cmap).colors`에는 0~1 사이의 (r, g, b) 형태로 표현된 튜플값들이 있고, 이는 각각의 색에 해당합니다. 

```python
# color list to color map
print(plt.cm.get_cmap('tab10').colors)
'''
((0.12156862745098039, 0.4666666666666667, 0.7058823529411765), (1.0, 0.4980392156862745, 0.054901960784313725), (0.17254901960784313, 0.6274509803921569, 0.17254901960784313), (0.8392156862745098, 0.15294117647058825, 0.1568627450980392), (0.5803921568627451, 0.403921568627451, 0.7411764705882353), (0.5490196078431373, 0.33725490196078434, 0.29411764705882354), (0.8901960784313725, 0.4666666666666667, 0.7607843137254902), (0.4980392156862745, 0.4980392156862745, 0.4980392156862745), (0.7372549019607844, 0.7411764705882353, 0.13333333333333333), (0.09019607843137255, 0.7450980392156863, 0.8117647058823529))
'''
```

많은 종류의 color map들 중 범주형 색상으로 사용되는 것으로는 다음의 것들이 있습니다. 

```python
qualitative_cm_list = ['Pastel1', 'Pastel2', 'Accent', 'Dark2', 'Set1', 'Set2', 'Set3', 'tab10']
```

그러면 위 color map들을 사용하여 Scatter Plot을 그려보겠습니다. 

```python
from matplotlib.colors import ListedColormap

# Group to Number
groups = sorted(student['race/ethnicity'].unique()) # 5개
gton = dict(zip(groups , range(5)))

# Group에 따라 색 1, 2, 3, 4, 5
student['color'] = student['race/ethnicity'].map(gton)

# Scatter Plot
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
axes = axes.flatten()

student_sub = student.sample(100)

for idx, cm in enumerate(qualitative_cm_list):    
    pcm = axes[idx].scatter(student_sub['math score'], student_sub['reading score'],
                     c=student_sub['color'], # cmap에서 인덱스
                     cmap=ListedColormap(plt.cm.get_cmap(cm).colors[:5]) # 실제 색깔 값
                     )
    cbar = fig.colorbar(pcm, ax=axes[idx], ticks=range(5)) # 서브플롯 옆에 표시되는 color bar
    cbar.ax.set_yticklabels(groups) # color bar도 하나의 서브플롯이기 때문에, 축 설정 가능
    axes[idx].set_title(cm)
    
plt.show()
```

![image-20220206123537438](https://user-images.githubusercontent.com/70505378/152670069-764f50c2-a9cf-4d39-9a4b-82ded395320b.png)





#### 연속형 색상

**연속형 색상**은 **정렬된 값을 가지는 순서형, 연속형 변수**인 경우에 사용합니다. 이때 색상이 연속적이라는 것은 **단일 색조로 명도를 조절하는 것**이라고 할 수 있습니다. 

연속형 색상으로 사용되는 color map에는 다음의 것들이 있습니다. 

```python
sequential_cm_list = ['Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds',
            'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu',
            'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn']
```

그러면 위 color map들로 연속형 색상이 무엇인지 확인해볼까요?

```python
fig, axes = plt.subplots(3, 6, figsize=(25, 10))
axes = axes.flatten()

student_sub = student.sample(100)

for idx, cm in enumerate(sequential_cm_list):    
    pcm = axes[idx].scatter(student['math score'], student['reading score'],
                            c=student['reading score'], # 연속형 변수 값을 바로 전달
                            cmap=cm,
                            vmin=0, vmax=100 # 색상의 범위를 지정
                     )
    fig.colorbar(pcm, ax=axes[idx])
    axes[idx].set_title(cm)
    
plt.show()
```

범주형 색상을 사용할 때와 차이가 느껴지시나요?

연속형 색상은 말 그대로 그 값이 연속적이기 때문에 **값을 인덱스로 매핑해주는 과정이 필요없습니다.** 또한 **vmin과 vmax로 색의 범위를 지정**해주어야 하며, color bar에 축의 값을 재설정할 필요도 없습니다. 

연속형 색상을 사용하는 예로 깃허브 커밋 로그를 들 수 있으며, `ax.imshow()`를 사용하면 이를 비슷하게 만들 수 있습니다. 

```python
im =  np.random.randint(10, size=(7, 52))
fig, ax = plt.subplots(figsize=(20, 5))
ax.imshow(im, cmap='Greens')
ax.set_yticks(np.arange(7)+0.5, minor=True) # minor=True이면 축 눈금 사이에 작은 눈금을 생성
ax.set_xticks(np.arange(52)+0.5, minor=True)
ax.grid(which='minor', color="w", linestyle='-', linewidth=3) # which='minor'이면 작은 눈금을 기준으로 격자 생성
plt.show()
```

![image-20220206125644704](https://user-images.githubusercontent.com/70505378/152670070-5449798f-4a4c-486d-8a2a-62acbb3ec554.png)











#### 발산형 색상

세번째로 발산형 색상입니다. **발산형 색상**은 연속형과 유사하지만, **중앙을 기준으로 발산**한다는 차이가 있습니다. 이는 상반된 값(기온)이나, 서로 다른 2개(지지율)를 표현하는 데 적합합니다. 

발산형 색상은 보통 **양쪽에 서로 다른 색조를 사용하고, 중앙에서 명도가 가장 높고(하얀색) 양 끝으로 갈수록 명도가 낮아(검은색)집니다.**

발산형 색상 표현을 위한 color map에는 다음의 것들이 있습니다. 

```python
diverging_cm_list = ['PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']
```

```python
from matplotlib.colors import TwoSlopeNorm

fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

offset = TwoSlopeNorm(vmin=0, vcenter=student['reading score'].mean(), vmax=100)

student_sub = student.sample(100)

for idx, cm in enumerate(diverging_cm_list):    
    pcm = axes[idx].scatter(student['math score'], student['reading score'],
                            c=offset(student['math score']), 
                            cmap=cm,
                     )
    cbar = fig.colorbar(pcm, ax=axes[idx], 
                        ticks=[0, 0.5, 1], 
                        orientation='horizontal'
                       )
    cbar.ax.set_xticklabels([0, student['math score'].mean(), 100])
    axes[idx].set_title(cm)
    
plt.show()
```

![image-20220206130306637](https://user-images.githubusercontent.com/70505378/152670058-cf01ab5e-25fd-40a2-a20c-94eac43bbf1e.png)

✋ 위에서 사용한 **TwoSlopeNorm**에 대한 설명을 잠깐만 하겠습니다. 만약 TwoSlopeNorm을 사용하지 않고 `axes[idx].scatter()`에 `c=student['math score']`를 전달한다면 발산의 기준이 data들의 중간이 아닌 subplot의 중간이 됩니다. 

```python
fig, axes = plt.subplots(3, 4, figsize=(20, 15))
axes = axes.flatten()

# offset = TwoSlopeNorm(vmin=0, vcenter=student['reading score'].mean(), vmax=100)

student_sub = student.sample(100)

for idx, cm in enumerate(diverging_cm_list):    
    pcm = axes[idx].scatter(student['math score'], student['reading score'],
                            c=student['math score'], # offset(student['math score']), 
                            cmap=cm,
                     )
    cbar = fig.colorbar(pcm, ax=axes[idx], 
                        # ticks=[min(student['math score']), 50, max(student['math score'])], 
                        orientation='horizontal'
                       )
    # cbar.ax.set_xticklabels([0, student['math score'].mean(), 100])
    axes[idx].set_title(cm)
    
plt.show()
```

![image-20220206131252812](https://user-images.githubusercontent.com/70505378/152670059-a3e45c01-ba51-443e-b67b-5e8500a15454.png)

하지만 발산형 색상이란 곧 data를 기준으로 양 극단을 보여주는 것이 목적이기에, TwoSlopeNorm을 활용한 것입니다. 

<br>

### 색상 대비 더 이해하기

마지막으로 특정 부분을 강조할 때 사용할 수 있는 색상 대비들 중 **명도 대비, 채도 대비, 보색 대비**에 대해 알아보겠습니다. 

아래와 같은 기본적인 bar plot과 scatter plot에 대비를 주어서 'group A'를 강조해보겠습니다. 

```python
fig = plt.figure(figsize=(18, 15))
groups = student['race/ethnicity'].value_counts().sort_index()

ax_bar = fig.add_subplot(2, 1, 1)
ax_bar.bar(groups.index, groups, width=0.5)

ax_s1 = fig.add_subplot(2, 3, 4)
ax_s2 = fig.add_subplot(2, 3, 5)
ax_s3 = fig.add_subplot(2, 3, 6)

ax_s1.scatter(student['math score'], student['reading score'])
ax_s2.scatter(student['math score'], student['writing score'])
ax_s3.scatter(student['writing score'], student['reading score'])

for ax in [ax_s1, ax_s2, ax_s3]:
    ax.set_xlim(-2, 105)
    ax.set_ylim(-2, 105)

plt.show()
```

![image-20220206151427266](https://user-images.githubusercontent.com/70505378/152670060-17c45ef6-f398-488e-a356-c505ced04e2b.png)

#### 명도 대비

**명도 대비**란 밝은 색과 어두운 색을 배치하면 밝은 색은 더 밝게, 어두운 색은 더 어둡게 보이는 것을 말합니다. 

```python
a_color, nota_color = 'black', 'lightgray'
# group A만 강조
colors = student['race/ethnicity'].apply(lambda x : a_color if x =='group A' else nota_color)
color_bars = [a_color] + [nota_color]*4

fig = plt.figure(figsize=(18, 15))
groups = student['race/ethnicity'].value_counts().sort_index()

ax_bar = fig.add_subplot(2, 1, 1)
ax_bar.bar(groups.index, groups, color=color_bars, width=0.5)

ax_s1 = fig.add_subplot(2, 3, 4)
ax_s2 = fig.add_subplot(2, 3, 5)
ax_s3 = fig.add_subplot(2, 3, 6)

ax_s1.scatter(student['math score'], student['reading score'], color=colors, alpha=0.5)
ax_s2.scatter(student['math score'], student['writing score'], color=colors, alpha=0.5)
ax_s3.scatter(student['writing score'], student['reading score'], color=colors, alpha=0.5)

for ax in [ax_s1, ax_s2, ax_s3]:
    ax.set_xlim(-2, 105)
    ax.set_ylim(-2, 105)

plt.show()
```

![image-20220206151906746](https://user-images.githubusercontent.com/70505378/152670061-0143604a-85f8-442b-9c10-cd926211c419.png)







#### 채도 대비

**채도 대비**란 선명한 색과 흐릿한 색을 배치하면 선명한 색은 더 선명하게, 흐릿한 색은 더 흐릿하게 보이는 것을 말합니다. 

```python
a_color, nota_color = 'orange', 'lightgray'

colors = student['race/ethnicity'].apply(lambda x : a_color if x =='group A' else nota_color)
color_bars = [a_color] + [nota_color]*4

# 이하 동일
...
```

![image-20220206152024747](https://user-images.githubusercontent.com/70505378/152670062-c3e6012c-8cea-4734-9bac-b608eee0a534.png)









#### 보색 대비

**보색 대비**란 정반대 색을 사용하면 각각의 색상 모두 더 선명해보이는 것을 말합니다. 

```python
a_color, nota_color = 'tomato', 'lightgreen'

colors = student['race/ethnicity'].apply(lambda x : a_color if x =='group A' else nota_color)
color_bars = [a_color] + [nota_color]*4

# 이하 동일
...
```

![image-20220206152248640](https://user-images.githubusercontent.com/70505378/152670064-2082d185-487a-46fd-97d3-830977878060.png)





<br>

<br>

## 참고 자료

* **Color에 대한 이해**
  * [명도와 채도의 관계](https://toramee2vr.tistory.com/45)
* **Color Palette의 종류**
  * [Matplotlib.colors document](https://matplotlib.org/stable/gallery/color/named_colors.html)

















<br>
