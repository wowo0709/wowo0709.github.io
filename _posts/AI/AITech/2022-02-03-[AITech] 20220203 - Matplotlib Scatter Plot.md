---
layout: single
title: "[AITech] 20220203 - Matplotlib Scatter Plot"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 학습 내용 정리

### 기본 Scatter Plot

**Scatter Plot**은 점을 사용하여 **두 feature 간의 관계를 알기 위해** 사용하는 그래프입니다. 

Scatter plot을 그리기 위해서 `.scatter()` 메서드를 사용할 수 있습니다. 

```python
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, aspect=1)

np.random.seed(970725)

x = np.random.rand(20)
y = np.random.rand(20)

ax.scatter(x, y)
ax.set_xlim(0, 1.05)
ax.set_ylim(0, 1.05)

plt.show()
```

![image-20220204151119980](https://user-images.githubusercontent.com/70505378/152482505-f8ccaaf1-65b3-45a0-a796-aecb56118213.png)

#### Scatter Plot의 요소

Scatter plot에서 변주를 줄 수 있는 요소에는 다음의 것들이 있습니다. 

* **색**: `color(c)`
* **모양**: `marker`
* **크기**: `size(s)`
* **테두리**: `linewidth`, `edgecolor`

각각은 일괄적으로 적용할 수도 있고, 리스트 형태로 각 점에 대해 개별적으로 적용할 수도 있습니다. 

```python
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, aspect=1)

np.random.seed(970725)

x = np.random.rand(20)
y = np.random.rand(20)
s = np.arange(20) * 20 # 개별적으로 size 지정

ax.scatter(x, y, 
           s= s,
           color='white',
           marker='o',
           linewidth=1,
           edgecolor='black')

plt.show()
```

![image-20220204151543470](https://user-images.githubusercontent.com/70505378/152482506-38ba3a43-f7b3-4f49-81c0-9637a665db62.png)

#### Scatter plot의 목적

Scatter plot을 통해 다음의 것들을 확인할 수 있습니다. 

* 상관 관계(양의 상관관계, 음의 상관관계, 없음)

  ![image-20220204151806030](https://user-images.githubusercontent.com/70505378/152482509-8bdddece-7818-4f12-b50c-1cf3ab090fac.png)

* 군집 관계

  ![image-20220204151826847](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220204151826847.png)







<br>

### 더 정확한 Scatter Plot

#### Overplotting

너무 많은 개수의 점을 그리게 되면 점의 분포를 파악하기 어렵습니다. 이럴 때는 다음의 방법들을 사용할 수 있습니다. 

* **투명도 조정**(`alpha`)
* **지터링(Jittering)**: 점의 위치를 약간씩 변경
* **2차원 히스토그램**: 히트맵을 사용하여 깔끔한 시각화
* **Coutour Plot**: 분포를 등고선을 사용하여 표현

![image-20220204152033843](https://user-images.githubusercontent.com/70505378/152482515-ac1580da-88c6-4f89-8f31-f22242869d1f.png)

#### 점의 요소와 인지

* **색**: 연속은 gradient, 이산은 개별 색상으로
* **마커**: 거의 구별하기 힘들다 + 크기가 고르지 않다
* **크기**: 두 feature 간의 관계보다는 각 점간의 비율에 초점을 둔다면 좋음(SWOT 분석등에 활용 가능)

![image-20220204152249833](https://user-images.githubusercontent.com/70505378/152482516-398f45f7-0814-42db-9fdf-5b2be7fc8588.png)

#### 인과관계와 상관관계

* **인과관계**와 **상관관계**는 다르다는 것을 분명히 인지해야 함
* 인과 관계는 항상 사전 정보와 함께 가정으로 제시
* 상관 관계는 추후 heatmap에서 다룸

#### 추세선

추세선을 사용하여 scatter의 패턴을 유추할 수 있습니다. 단, 추세선이 2개 이상이 되면 가독성이 떨어지므로 주의해야 합니다. 

![image-20220204152434067](https://user-images.githubusercontent.com/70505378/152482501-d2157b8f-25d2-4b87-928a-121bb32b8bdc.png)































<br>

<br>

## 참고 자료

* 
