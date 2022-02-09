---

layout: single
title: "[AITech] 20220209 - Generative Model I"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Explicit/Implicit Model', 'Probability Distribution', 'Conditional Independence', 'auto-regressive model']
---



<br>

## 학습 내용

이번 'Generative Model I' 포스팅에서는 생성 모델이 무엇인지, 생성 모델에 대한 개념을 알아보도록 하겠습니다. 

### Learning a Generative Model

여러 분은 **생성 모델**이란 무엇을 하는 모델이라고 생각하시나요? 대부분의 사람들을 무언가를 만들어내는 모델이라고 생각을 할 것입니다. 

하지만, 엄밀히 말하면 generative model의 종류에는 그것만이 있는 것이 아닙니다. **생성 모델이란 입력 데이터의 확률 분포 p(x)를 알고자 하는 것**으로, 다음의 것들을 수행할 수 있습니다. 

* 입력으로 개 사진들이 주어져 있다고 해봅시다. 
  * **Generation**: x<sub>new</sub>~p(x)에서 sampling한다면, x<sub>new</sub>는 개 이미지와 같이 생겨야 한다. (**sampling**)
  * **Density estimation**: x가 개와 유사하게 생겼다면 p(x)는 높아야 하고, 그렇지 않다면 낮아야 한다. (**anomaly detection**)
    * Explicit model: 이미지 x에 대해 각각의 클래스에 속할 확률 p<sub>i</sub>(x)를 뽑아낼 수 있는 모델
    * Implicit model: 이미지 x와 유사한 이미지 x'을 생성할 수 있는 모델
  * **Unsupervised representiation learning**: 여러 장의 이미지로부터 공통된 특징을 추출할 수 있다. (**feature learning**)

생성 모델은 단순히 무언가를 생성해내는 모델(Explicit model)만은 아니며, 심지어는 무언가를 생성해내지 못하더라도 그 확률분포를 모델링할 수 있다면(Implicit model) 생성 모델이라고 할 수 있습니다. 

그렇다면, 여기서 근본적인 질문. **p(x)는 어떻게 알아낼 수 있을까요?**



<br>

### Modeling Probability Distribution

#### Basic Discrete Distributions

기본적인 이산 확률 분포에는 다음의 2가지 분포가 있습니다. 

* **베르누이 분포**: 0 또는 1이 발생하는 분포. 한 쪽의 확률이 p이면 다른 쪽의 확률은 1-p가 된다. 
  * Bernoulli distribution, X~Ber(p)
* **카테고리 분포**: m개의 사건 중 하나가 발생하는 분포. 각 사건이 발생할 확률의 합은 1이다. 
  * Categorical distribution, Y~Cat(p1, ..., pm)

예를 들어 우리가 아래와 같이 0 또는 1로 표현되는 n개의 픽셀 값에 대한 확률 분포를 예측하고자 한다고 해봅시다.

![image-20220209222234094](https://user-images.githubusercontent.com/70505378/153218021-5e689c97-6133-41d4-9fa5-d56e8ba7828d.png) 

* `p(x1, ..., xn)`
* 가능한 총 경우의 수는 **2<sup>n</sup>**가지이다. 
* **2<sup>n</sup>-1** 개의 parameter가 필요하다. 

n이 만약에 10이라고 해도 그 parameter의 수는 1,000개가 넘어가는데, n = 1024<sup>2</sup>이라면? 픽셀이 가질 수 있는 state가 0 또는 1이 아닌 (r, g, b) 값이라면? 그 parameter의 개수는 가늠할 수 없을 정도로 커지겠죠. 

그래서 얘기하고자 하는 것이 뭐냐? 우리는 **확률 분포 p(x)를 알기 위해 바로 이 parameter의 수를 줄여야 합니다.**

#### Structure Through Independence

만약 이 n개의 사건들이 서로 **독립적**이라면, 수식을 아래와 같이 변환할 수 있습니다. 

* `p(x1, ..., xn)` = `p(x1)p(x2)...p(xn)`
* 가능한 총 경우의 수는 **2<sup>n</sup>**가지이다. 
* **n** 개의 parameter가 필요하다. 

보셨나요? **각 사건이 서로 독립적이라는 가정만 주어지면, parameter의 개수는 극적으로 감소합니다.** 그 사건이 발생할 때, 다른 사건들이 0인지 1인지는 상관이 없고 오직 해당 사건의 발생 여부만이 중요해지니까요. 

하지만, 이러한 독립적이라는 가정은 우리가 실제로 활용할 만한 distribution들에 대해 적용하기에는 제한이 있습니다. 

#### Conditional Independence

그래서 우리는 **Conditional independence**, 즉 모든 사건들이 독립은 아니더라도, 조건부 독립이라는 것을 사용할 수 있습니다. 그리고 여기에는 3가지 중요한 규칙이 사용됩니다. 

1. **Chain rule**: `p(x1, …, xn) = p(x1)p(x2|x1)p(x3|x1, x2)⋯p(xn|x1, ⋯, xn-1)`
2. **Bayes' rule**: `p(x|y) = p(x, y) / p(y) = p(y | x)p(x) / p(y)`
3. **Conditional independence**: `If x ⊥ y | z, then p(x|y, z) = p(x|z)`
   * 여기서 'x ⊥ y'는 x와 y가 독립적이라는 뜻이고, '| z' 는 z가 주어진 상황이라는 것입니다. 

이 조건들을 적절히 활용하면 **X<sub>i+1</sub> ⊥ X1, …, X<sub>i-1</sub> | X<sub>i</sub> (Markov assumption)**을 이용하여, 다음과 같은 수식이 도출됩니다. 

* `p(x1, …, xn) = p(x1)p(x2|x1)p(x3|x2)⋯p(xn|xn-1)`
* **2n-1** 개의 parameter가 필요하다. 

이로써, 우리는 conditional independence를 이용하여 parameter 수를 극적으로 줄일 수 있게 되었습니다. 

**Auto-regressive model**들은 바로 이 conditional independency를 이용하는 모델들입니다. 

<br>

### Auto-regressive Model

Conditional independency를 이용한 모델들, 그러니까 Auto-regressive generative model에는 다음의 모델들이 있습니다. 

**NADE: Neural Autoregressive Density Estimator**

![image-20220209225651069](https://user-images.githubusercontent.com/70505378/153218024-30a24047-af26-4330-8ada-ea8451dabef0.png)

* NADE는 explicit model로, 주어진 입력의 density를 계산할 수 있습니다. 
* i번째 픽셀의 확률 분포는, `p(xi|x1:i-1) = σ(αihi + bi) where hi = σ(W<ix1:i-1 + c)`
  * `p(x1, …, x784) = p(x1)p(x2|x1)⋯p(x784|x1:783)`
* Continuous random variable을 모델링 할 때는, **mixture of Gaussian**이 사용될 수 있습니다. 

**Pixel RNN**

![image-20220209230157202](https://user-images.githubusercontent.com/70505378/153218026-a958b22d-ab7e-45ae-8fcc-abc5dc60ac3c.png)

앞서 우리는 RNN 또한 Auto-regressive 모델임을 배웠습니다. 따라서 이 RNN도 generative model로서 사용할 수 있겠죠. 

* Pixel RNN 역시 explicit model입니다. 

* Pixel RNN에서 픽셀들의 순서를 매기는 데는 2가지 방법이 있습니다. 

  * Row LSTM
  * Diagonal BiLSTM

  ![image-20220209230405421](https://user-images.githubusercontent.com/70505378/153218013-ac1b44e7-df75-4076-87ea-54ccc22f45ea.png)







<br>

이렇게 해서 Generative model이 무엇인지, 그 엄밀한 개념에 대해서 살펴보았고, 또 어떻게 해서 확률 분포 p(x)를 모델링할 수 있는지와 conditional independency를 이용한 auto-regressive model에는 무엇이 있는지 살펴보았습니다. 

다음 [Generative Model II] 포스팅에서는 실제로 많이 사용되는 Generative model인 VAE와 GAN에 대해 살펴보겠습니다. 

















<br>

<br>

## 참고 자료

* [NADE 논문](https://arxiv.org/abs/1605.02226)
* [Pixel RNN 논문](https://arxiv.org/abs/1601.06759)

















<br>
