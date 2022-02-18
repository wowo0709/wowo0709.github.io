---

layout: single
title: "[AITech][Visualization] 20220218 - Custom Matplotlib Theme"
categories: ['AI', 'AITech', 'Visualization']
toc: true
toc_sticky: true
tag: ['Color', 'Facet', 'Visualization References']
---



<br>

**_본 포스팅은 서울대 HCI Lab의 '안수빈' 강사 님의 강의를 바탕으로 제작되었습니다._** 

# 학습 내용

Matplotlib을 커스텀할 때, 기본적으로 변경할 수 있는 요소를 다크모드 시각화를 만들며 살펴봅시다.


```python
import numpy as np
import pandas as pd 
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
```

## 색의 선정

- https://developer.apple.com/design/human-interface-guidelines/ios/visual-design/color

색은 cycler를 기본으로 사용하여 전체적인 color palette를 바꿀 수 있습니다. 


```python
from cycler import cycler

raw_light_palette = [
    (0, 122, 255), # Blue
    (255, 149, 0), # Orange
    (52, 199, 89), # Green
    (255, 59, 48), # Red
    (175, 82, 222),# Purple
    (255, 45, 85), # Pink
    (88, 86, 214), # Indigo
    (90, 200, 250),# Teal
    (255, 204, 0)  # Yellow
]

raw_dark_palette = [
    (10, 132, 255), # Blue
    (255, 159, 10), # Orange
    (48, 209, 88),  # Green
    (255, 69, 58),  # Red
    (191, 90, 242), # Purple
    (94, 92, 230),  # Indigo
    (255, 55, 95),  # Pink
    (100, 210, 255),# Teal
    (255, 214, 10)  # Yellow
]

raw_gray_light_palette = [
    (142, 142, 147),# Gray
    (174, 174, 178),# Gray (2)
    (199, 199, 204),# Gray (3)
    (209, 209, 214),# Gray (4)
    (229, 229, 234),# Gray (5)
    (242, 242, 247),# Gray (6)
]

raw_gray_dark_palette = [
    (142, 142, 147),# Gray
    (99, 99, 102),  # Gray (2)
    (72, 72, 74),   # Gray (3)
    (58, 58, 60),   # Gray (4)
    (44, 44, 46),   # Gray (5)
    (28, 28, 39),   # Gray (6)
]


light_palette = np.array(raw_light_palette)/255
dark_palette = np.array(raw_dark_palette)/255
gray_light_palette = np.array(raw_gray_light_palette)/255
gray_dark_palette = np.array(raw_gray_dark_palette)/255
```


```python
print('Light mode palette')
sns.palplot(light_palette)
sns.palplot(gray_light_palette)

print('Dark mode palette')
sns.palplot(dark_palette)
sns.palplot(gray_dark_palette)
```




![output_4_1](https://user-images.githubusercontent.com/70505378/154608113-a18e7247-243b-4ea3-ad35-fa3d7c66b2fe.png)
    




![output_4_2](https://user-images.githubusercontent.com/70505378/154608117-1c657b07-7e83-4f50-9a04-77d5206add86.png)
    




![output_4_3](https://user-images.githubusercontent.com/70505378/154608120-a0d78fe2-7f3e-4dee-acd8-0f39185bdeac.png)
    




![output_4_4](https://user-images.githubusercontent.com/70505378/154608121-0587f50c-f9b3-4a43-9a86-1e2b79ba9895.png)
    


위에서 선언한 다크모드 색상을 사용하여 전체적인 colormap을 바꿉니다.


```python
# cmap 수정
mpl.rcParams['axes.prop_cycle'] = cycler('color',dark_palette)
```

다크 모드의 전체적인 배경색을 바꿔주기 위해 배경 관련 색들을 바꿔줍니다.


```python
# 전체적인 배경색 수정
mpl.rcParams['figure.facecolor']  = gray_dark_palette[-2]
mpl.rcParams['figure.edgecolor']  = gray_dark_palette[-2]
mpl.rcParams['axes.facecolor'] =  gray_dark_palette[-2]

```

기존에 검정이 었던 텍스트와 테두리는 모두 흰색으로 변경합니다.


```python
# 사용되는 텍스트 색상 흰색으로 수정
white_color = gray_light_palette[-2]
mpl.rcParams['text.color'] = white_color
mpl.rcParams['axes.labelcolor'] = white_color
mpl.rcParams['axes.edgecolor'] = white_color
mpl.rcParams['xtick.color'] = white_color
mpl.rcParams['ytick.color'] = white_color
```

꼭 색상이 아니더라도 수정하면 종종 괜찮은 해상도를 높여줍니다. 해상도가 높을수록 차트가 그려지는 데 필요한 시간이 증가하므로 너무 높일 필요는 없습니다. 또한 이미지만 저장할 때 해상도를 설정할 수 있습니다.


```python
# 해상도 조정
mpl.rcParams['figure.dpi'] = 200
```

일반적인 시각화는 좌측에서 우측으로, 상단에서 하단으로 시선이 이동하며 이에 따라 상단과 우측을 제거하면 훨씬 더 깔끔한 느낌을 줄 수 있습니다.


```python
# ax의 우측과 상단 지우기
mpl.rcParams['axes.spines.top'] = False
mpl.rcParams['axes.spines.right'] = False
```

<br>

## Facet + Dark Mode를 활용한 예시


```python
student = pd.read_csv('./StudentsPerformance.csv')
iris = pd.read_csv('./Iris.csv')
```

### Scatter Plot


```python
def score_distribution(f1, f2):
    fig = plt.figure(figsize=(12, 10),dpi=150)

    gs = fig.add_gridspec(5, 6)

    ax = fig.add_subplot(gs[:,:5])
    ax.set_aspect(1)

    for group in sorted(student['race/ethnicity'].unique()):
        student_sub = student[student['race/ethnicity']==group]
        ax.scatter(student_sub[f'{f1} score'], student_sub[f'{f2} score'], 
                   s=20, alpha=0.6, 
                   linewidth=0.5, 
                   label=group
                  )

    sub_axes = [None] * 5
    for idx, group in enumerate(sorted(student['race/ethnicity'].unique())):
        sub_axes[idx] = fig.add_subplot(gs[idx,5], aspect=1)
        sub_axes[idx].scatter(student[student['race/ethnicity']!=group][f'{f1} score'], student[student['race/ethnicity']!=group][f'{f2} score'], 
                              s=5, alpha=0.2, 
                              color= white_color,
                              linewidth=0.7, 
                              label=group,
                              zorder=5
                  )
        sub_axes[idx].scatter(student[student['race/ethnicity']==group][f'{f1} score'], student[student['race/ethnicity']==group][f'{f2} score'], 
                              s=5, alpha=0.6, 
                              color= dark_palette[idx],
                              linewidth=0.5, 
                              label=group,
                              zorder=10
                  )
        cnt = (student['race/ethnicity']==group).sum()
        sub_axes[idx].set_title(f'{group} ({cnt})', loc='left', fontsize=9)
        sub_axes[idx].set_xticks([])
        sub_axes[idx].set_yticks([])

    for axes in [ax] + sub_axes:
        axes.set_xlim(-3, 103)
        axes.set_ylim(-3, 103)

    ax.set_title(f'{f1.capitalize()} & {f2.capitalize()} Score Distribution', loc='left', fontsize=15, fontweight='bold')    
    ax.set_xlabel(f'{f1.capitalize()} Score', fontweight='medium')
    ax.set_ylabel(f'{f2.capitalize()} Score', fontweight='medium')
    ax.legend(title='Race/Ethnicity', fontsize=10)

    plt.show()
```


```python
score_distribution('math', 'reading')
```


![output_19_0](https://user-images.githubusercontent.com/70505378/154608122-85bebe9b-f585-43a1-98ee-1e2c8ed3c9b0.png)
    


### 2-2. KDE Plot


```python
def score_distribution_kde(subject):
    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(6, 5)
    ax = fig.add_subplot(gs[:5,:])

    sns.kdeplot(x=subject, hue='race/ethnicity', data=student,
                hue_order=sorted(student['race/ethnicity'].unique()),
                bw_adjust=0.4,
                fill=True,ax=ax)
    

    sub_axes = [None] * 5
    for idx, group in enumerate(sorted(student['race/ethnicity'].unique())):
        sub_axes[idx] = fig.add_subplot(gs[5,idx])
        sns.kdeplot(x=subject, data=student,
                    alpha=0.2, 
                  color= white_color,
                  linewidth=0.7, 
                  label=group, fill=True, bw_adjust=0.4,
                  zorder=5, ax=sub_axes[idx]
                  )

        sns.kdeplot(x=subject, data=student[student['race/ethnicity']==group],
                    alpha=0.6, 
                      color= dark_palette[idx],
                      linewidth=0.5, 
                      label=group, fill=True,bw_adjust=0.4,
                      zorder=10, ax=sub_axes[idx]
                  )
        cnt = (student['race/ethnicity']==group).sum()
        sub_axes[idx].set_xticks([])
        sub_axes[idx].set_yticks([])
        sub_axes[idx].set_xlabel('')
        sub_axes[idx].set_ylabel('')

    ax.set_title(subject.capitalize(), loc='left', fontweight='bold', fontsize=13)

    fig.tight_layout()
    plt.show()
```


```python
score_distribution_kde('math score')
```


![output_22_0](https://user-images.githubusercontent.com/70505378/154608123-65c5d8ef-b27d-4acc-a251-526a0c2bef1e.png)
    


### 2-3. Pairplot


```python
sns.pairplot(iris, hue='Species', corner=True)
```




![output_24_1](https://user-images.githubusercontent.com/70505378/154608108-8b32c49e-ceed-46a4-add3-cdb042c71e95.png)
    


### 







### Plotly 3D Plot

```python
import plotly.graph_objects as go

x, y, z = student['math score'], student['reading score'], student['writing score']
gtc = dict(zip(sorted(student['race/ethnicity'].unique()), raw_dark_palette[:5]))
color = student['race/ethnicity'].map(gtc)


fig = go.Figure(data=[go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(
        size=5,
        color=color,
        opacity=0.8
    )
)], layout=go.Layout(
    plot_bgcolor='rgba(255,0,0,1)',
    paper_bgcolor=f'rgb{raw_gray_dark_palette[-2]}',
    font=dict(color='white'))
)

# tight layout
fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
fig.update_layout(scene = dict(
                    xaxis_title='MATH',
                    yaxis_title='READING',
                    zaxis_title='WRITING',
    
                    xaxis = dict(
                         gridcolor="white",
                         showbackground=False,
                         zerolinecolor="white",
                        range=[0, 100]
                    ),
                    yaxis = dict(
                        gridcolor="white",
                         showbackground=False,
                        zerolinecolor="white",
                        range=[0, 100]
                    ),
                    zaxis = dict(
                        gridcolor="white",
                         showbackground=False,
                        zerolinecolor="white",
                        range=[0, 100]                    
                    )),
                    margin=dict(
                    r=10, l=10,
                    b=10, t=10)
                  )

camera = dict(
    eye=dict(x=1.4, y=1.4, z=1.4)
)


fig.update_layout(scene_camera=camera)

fig.show()
```



![image-20220218113731646](https://user-images.githubusercontent.com/70505378/154608134-be6ab2f8-d0ce-4d6b-b9a2-c6350659cb48.png)

<br>

<br>

# 참고 자료

* XAI
     - [Visual Analytics in Deep Learning: An Interrogative Survey for the Next Frontiers](https://arxiv.org/abs/1801.06889)
     - XAI using torch : https://captum.ai/
     - saliency map (heatmap visualization)
* node-link diagram (network visualization)
  * http://alexlenail.me/NN-SVG/
  * https://github.com/HarisIqbal88/PlotNeuralNet
  * http://ethereon.github.io/netscope/quickstart.html
* 딥러닝/머신러닝을 공부하는 분들에게 마지막 전달하는 AI + Visualization 자료

  - Distill.pub : https://distill.pub/ (Interactive web)
  - Poloclub : https://poloclub.github.io/ (human centric AI)
  - Google Pair : https://pair.withgoogle.com/
  - Open AI Blog : https://openai.com/blog/
* 그 외 visualization 아이디어를 얻을 수 있는 소스

  - Observable : https://observablehq.com/
  - https://textvis.lnu.se/
  - https://visimages.github.io/visimages-explorer/







<br>
