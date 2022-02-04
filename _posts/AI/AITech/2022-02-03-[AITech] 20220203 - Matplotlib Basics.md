---
layout: single
title: "[AITech] 20220203 - Matplotlib Basics"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 학습 내용 정리

### Matplotlib 소개

왜 우리는 대표적인 시각화 라이브러리로 `Matplotlib`을 사용할까요?

* Numpy, Scipy를 베이스로 하여 다양한 라이브러리와 호환성이 높습니다. 
  * Scikit Learn, PyTorch, TensorFlow, Pandas 등
* 다양한 시각화 방법론을 제공합니다. 
  * 막대그래프, 선그래프, 산점도 등

Matplotlib 외에도 **Seaborn, Plotly, Bokeh, Altair** 등의 여러 시각화 라이브러리가 존재하며, 이 중 Seaborn과 Plotly는 추후 다른 포스팅에서 학습합니다. 

#### Import Library

`matplotlib`은 코드 상에서 `mpl`로 줄여서 사용하며, matplotlib 안에 있는 라이브러리들 중 자주 사용하는 `pyplot` 라이브러리는 `plt`로 줄여서 사용합니다. 

```python
import matplotlib as mpl
import matplotlib.pyplot as plt
```





<br>

### Plot Basic

#### Figure & Ax(es)

`matplotlib.pyplot`을 사용하기 위해서 기본적으로 **figure**와 **ax(es)**를 이해해야 합니다. 

* `figure`: 그래프/표가 그려질 큰 틀(팔레트)을 나타냄
* `ax(es)`: figure 내의 실제 서브플롯 각각을 나타냄

다음 코드를 보면 두 개념을 확실히 이해할 수 있을 것입니다. 

```python
fig = plt.figure(figsize=(12, 7)) # 큰 팔레트를 생성
fig.set_facecolor('yellow') # 팔레트의 배경색 생성
ax = fig.add_subplot() # 그래프 추가
plt.show() # 팔레트 출력
```

![image-20220203133419178](https://user-images.githubusercontent.com/70505378/152285348-15158476-9496-460b-bc88-046ac2e53911.png)

* `fig = plt.figure()`: 하나의 큰 팔레트를 생성하는 것이며, fig 변수에 저장합니다. 인자로 figsize를 전달하여 팔레트의 크기(비율)를 설정할 수 있습니다. 

* `ax = fig.add_subplot()`: 팔레트에 하나의 서브 플롯을 생성하는 것이며, ax 변수에 저장합니다. 다음과 같이 인자를 전달하여 팔레트 내에 여러 개의 서브 플롯을 생성할 수 있습니다. 

  ```python
  fig = plt.figure()
  ax1 = fig.add_subplot(221) 
  ax2 = fig.add_subplot(224) 
  # 같은 내용이지만 더 가독성을 높인다면 다음과 같이 사용 가능
  # ax1 = fig.add_subplot(2, 2, 1)
  # ax2 = fig.add_subplot(2, 2, 4)
  plt.show()
  ```

  ![image-20220203133808064](https://user-images.githubusercontent.com/70505378/152285350-7265ac2b-2106-4686-9d40-4c15d161288b.png)

* `plt.show()`: 현재 생성된 figure를 출력하기 위해서는 항상 plt.show() 함수를 호출해야 합니다. 

또는, 아래와 같이 `plt.subplots()`를 이용해 fig와 axex를 동시에 생성할 수 있습니다. 

```python
fig, axes = plt.subplots(2, 2, figsize=(12, 12))
```

이 경우, axes\[0\]\[0\], axes\[0\]\[1\], axes\[1\]\[0\], axes\[1\]\[1\] 에 ax 객체들이 저장됩니다. 

#### plt로 선 그래프 그리기

`plt.plot(x, y)`를 사용하면 선 그래프를 그릴 수 있습니다. 

plot 함수를 사용하는 방법에는 두 가지가 있습니다. 첫째는 **plt 라이브러리의 plot 함수를 호출하여 가장 최근 그래프에 plot하는 순차적 방법**이고, 둘째는 **각 ax 인스턴스의 plot 메서드를 호출하여 자신의 그래프에 plot하는 객체지향적 방식**입니다. 

주로 객체지향적 방식을 많이 사용하고, 가끔 순차적 방식이 필요할 때가 있다고 합니다. 😊

```python
fig = plt.figure()
x1 = [1, 2, 3]
x2 = [3, 2, 1]
# Pyplot API(순차적 방법)
ax1 = fig.add_subplot(211) 
plt.plot(x1) # ax1에 그리기
ax2 = fig.add_subplot(212) 
plt.plot(x2) # ax2에 그리기
# Object-oriented API(객체지향적 방법)
ax1 = fig.add_subplot(211) 
ax2 = fig.add_subplot(212) 
ax1.plot(x1) 
ax2.plot(x2)
```

(결과는 동일합니다)

![image-20220203134433388](https://user-images.githubusercontent.com/70505378/152285351-4600f232-689f-40e9-9bd6-f6a3f2535f2a.png)

✋ plt로 그리다 `plt.gcf().get_axes()`로 다시 서브 플롯 객체를 받아서 사용할 수도 있습니다. 

<br>

### Plot의 요소 알아보기

#### 한 서브플롯에서 여러 개 그리기

ax에는 동시에 다양한 그래프를 그릴 수 있습니다. 

```python
fig = plt.figure()
ax = fig.add_subplot(111) 
# 3개의 그래프 동시에 그리기(미리 지정된 색깔 순서)
ax.plot([1, 1, 1]) # 파랑
ax.plot([1, 2, 3]) # 주황
ax.plot([3, 3, 3]) # 초록

plt.show()
```

![image-20220203134649469](https://user-images.githubusercontent.com/70505378/152285352-dd741caa-e6b4-4e62-9491-30eb6fe317d2.png)

위에서 보다시피, 하나의 서브플롯에 여러 그래프를 그리게 되면 내부적으로 미리 지정된 색깔 순(파랑 -> 주황 -> 초록 -> ...)으로 그래프가 그려집니다. 

하지만 **서로 다른 종류의 그래프를 그릴 때에는 구분되지 않습니다.**

```python
fig = plt.figure()
ax = fig.add_subplot(111) 

# 선그래프와 막대그래프 동시에 그리기(서로 다른 종류의 그래프를 그릴 때는 같은 색깔)
ax.plot([1, 2, 3], [1, 2, 3]) 
ax.bar([1, 2, 3], [1, 2, 3]) 

plt.show()
```

![image-20220203134954992](https://user-images.githubusercontent.com/70505378/152285354-8d90e919-ce38-463f-b488-ffcc8cbb2234.png)



#### 색상 지정하기

그래프 색상을 지정하는 방법에는 3가지가 있습니다. 

```python
fig = plt.figure()
ax = fig.add_subplot(111) 
# 3개의 그래프 동시에 그리기
ax.plot([1, 1, 1], color='r')           # 1. 한 글자로 정하는 색상(빠르고 단순하게 그릴 때)
ax.plot([2, 2, 2], color='forestgreen') # 2. color name(색상을 잘 알고 있을 때, but 모두 알기 어려움)
ax.plot([3, 3, 3], color='#00ffff')     # 3. hex code(00~ff) (추천!!!)
plt.show()
```

위 방법들 중 3번째 방법인 **hex-code**를 이용하는 방법을 추천하며, 이에 대한 자세한 내용은 추후 **Color** 포스팅에서 다룰 예정입니다. 



#### 텍스트 사용하기

정보를 추가하기 위해 텍스트를 사용할 수도 있습니다. 

**legend**

서브플롯에 범례(legend)를 추가할 수 있습니다. 

```python
fig = plt.figure()
ax = fig.add_subplot(111) 
# 범례에 나타낼 그래프에는 label 인자를 전달한다. 
ax.plot([1, 1, 1], label='1') 
ax.plot([2, 2, 2], label='2') 
ax.plot([3, 3, 3], label='3')
# ax 객체의 legend() 메서드를 호출하여 범례를 출력한다. 
ax.legend()
plt.show()
```

![image-20220203135355174](https://user-images.githubusercontent.com/70505378/152285340-eb4c2c45-6c01-48a9-ad63-ec1b7104838f.png)

**title**

Title의 경우 figure 전체의 title을 추가(`fig.suptitle(title)`)할 수도 있고, 하나의 ax의 title을 추가(`ax.set_title(title)`)할 수도 있습니다. 

```python
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
# 각각의 subplot에 title 추가
ax1.set_title('ax1')
ax2.set_title('ax2')
# figure에 title 추가
fig.suptitle('fig')

plt.show()
```



**ticks**

ticks는 축을 나타냅니다. 

`ax.set_xticks(list)`로 x 축에 적히는 수를 직접적으로 지정할 수 있고, `ax.set_xticklabels(list)`로 x축에 적힐 텍스트를 전달할 수도 있습니다. 

```python
fig = plt.figure()
ax = fig.add_subplot(111) 
ax.plot([1, 1, 1], label='1') 
ax.plot([2, 2, 2], label='2') 
ax.plot([3, 3, 3], label='3')

ax.set_title('Basic Plot')
ax.set_xticks([0, 1, 2]) # x축에 0, 1, 2 만 나타나도록 지정
ax.set_xticklabels(['zero', 'one', 'two']) # x축의 0, 1, 2 대신 zero, one, two가 출력되도록 지정
ax.legend()

plt.show()
```

![image-20220203140023659](https://user-images.githubusercontent.com/70505378/152285344-4591756b-8378-4ce3-9fe5-0b41ff20925d.png)





**text, annotate**

일반적으로 텍스트를 추가하는 방법에는 2가지가 있습니다. 

* `ax.text(x, y, x)`: 해당 위치에 텍스트를 적는 느낌
* `ax.annotate(text, xy, [arrowprops, xytext])`: 해당 포인트를 지정하는 느낌

x, y는 텍스트의 좌측 하단을 기준으로 합니다. `ha` 파라미터(center, right, ...)로 그 기준을 변경할 수 있습니다. 

```python
fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot()

ax.plot([1,2,3])
ax.text(0, 1, 'Text using text()')
ax.annotate('Text using annotate() I', (0,2))
ax.annotate(text='Text using annotate() II', xy=(0, 3),
            xytext=(0.2, 2.8), 
            arrowprops=dict(facecolor='black'),
            )
fig.suptitle('Comparison of making text')

fig.show()
```

![image-20220203141239329](https://user-images.githubusercontent.com/70505378/152285346-2b87439d-2f36-4386-b5f8-441cfccbca04.png)





<br>

<br>

## 참고 자료

* 
