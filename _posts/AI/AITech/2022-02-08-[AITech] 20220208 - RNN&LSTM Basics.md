---

layout: single
title: "[AITech] 20220208 - RNN&LSTM Basics"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 학습 내용

### RNN

#### 시퀀스 데이터

* 소리, 문자열, 주가 등의 데이터를 **시퀀스 데이터**로 분류한다. 

* 시퀀스 데이터는 **독립동등분포(i.i.d.)** 가정을 위배하기 때문에 **순서를 바꾸거나 과거 정보에 손실이 발생하면 데이터의 확률분포도 바뀐다.**

* 따라서 이전 시퀀스에 대한 정보를 가지고 앞으로 발생할 데이터의 확률 분포를 계산해야 하며, 이를 위해 조건부 확률을 이용할 수 있다. 

  ![image-20220121114516596](https://user-images.githubusercontent.com/70505378/150459169-72a12f32-2439-4e73-840a-559be2d27ff9.png)

  * 위 조건부 확률은 과거의 모든 정보를 이용하지만, 시퀀스 데이터를 분석할 때 **과거의 모든 정보들이 필요한 것은 아니다.**
    * 어떤 시점까지의 과거의 정보를 이용할 지는 데이터/모델링에 따라 달라진다. 

* 시퀀스 데이터를 다루기 위해서는 **길이가 가변적인 데이터**를 다룰 수 있는 모델이 필요하다. 

  * 이를 해결하기 위해 특정 구간 _tau_만큼의 과거 정보만을 이용하고, 그보다 더 전의 정보들은 **H<sub>t</sub>**라는 잠재변수로 인코딩해서 사용할 수 있다. 
    * 이렇게 함으로써 데이터의 길이를 고정할 수 있고, 과거의 모든 데이터를 활용하기 용이해진다. 
    * _tau_ 구간 만큼의 과거 정보를 이용하는 모델을 **Auto Regressive Model**이라 하고, 현재 시점의 입력과 인코딩된 잠재 정보를 이용하는 모델을 **Latent Autoregressive Model**이라 한다. 

  ![image-20220121115240369](https://user-images.githubusercontent.com/70505378/150459172-8ac5d9c3-3ce5-49d9-9db9-8482e2071342.png)

  * 이 잠재변수 H<sub>t</sub>를 신경망을 통해 반복해서 사용하여 **시퀀스 데이터의 패턴을 학습**하는 잠재 회귀 모델이 **RNN**이다. 

    <img src="https://user-images.githubusercontent.com/70505378/150459173-7e1479bf-afb5-454a-8336-ef95b1282c8c.png" alt="image-20220121115433935" style="zoom:67%;" />

#### RNN(Recurrent Neural Network)

* 현재 정보만을 입력으로 사용하는 완전연결신경망은 과거의 정보를 다룰 수 없다. 

* RNN은 이전 순서의 잠재변수와 현재의 입력을 활용하여 모델링한다. 

  * W: t에 따라 불변/ X, H: t에 따라 가변

  ![image-20220121115906209](https://user-images.githubusercontent.com/70505378/150459174-3020b45d-4248-4ff1-b8fa-bf36b10fa114.png)

* **RNN의 역전파**는 잠재변수의 연결그래프에 따라 순차적으로 계산한다. (맨 마지막 출력까지 계산한 후에 역전파)

  * 이를 **BPTT(Backpropagation Through Time)**라 하며 RNN의 기본적인 역전파 방식이다. 

  ![image-20220121120052256](https://user-images.githubusercontent.com/70505378/150459175-d0158bca-a493-49b7-9272-adc6d1ca8496.png)

  * BPTT를 통해 RNN의 가중치 행렬의 미분을 계산해보면 아래와 같이 **미분의 곱**으로 이루어진 항이 계산된다. 

    * 그 중 빨간색 네모 안의 항은 불안정해지기 쉽다. 
    * 이는 거듭된 값들의 곱으로 인해 값이 너무 커지거나(기울기 폭발) 너무 작아져(기울기 소실) 과거의 정보를 제대로 전달해주지 못하기 때문이다. 

    ![image-20220121120521892](https://user-images.githubusercontent.com/70505378/150459177-ae598173-a0df-431f-a4a6-538baf34ae44.png)

  * 기울기 폭발/소실 문제를 해결하기 위해 역전파 과정에서 **길이를 끊는 것**이 필요하며, 이를 **TBPTT(Truncated BPTT)**라 한다. 

    ![image-20220121120714788](https://user-images.githubusercontent.com/70505378/150459180-30e736f3-3b17-4191-a09e-85417f3d37b5.png)

* 여러가지 문제로 Vanilla RNN으로는 긴 시퀀스를 처리하는데 한계가 있고, 이를 해결하기 위해 **LSTM**이나 **GRU**와 같은 발전된 형태의 네트워크를 사용한다. 





<br>

### LSTM

LSTM(Long Short Term Memory)은 Vanilla RNN의 한계인 Long-term memory를 효과적으로 전달하기 위해 고안된 모델이다. 

![image-20220208183002864](https://user-images.githubusercontent.com/70505378/152983127-b220110a-625d-4706-96b6-dfa66117b563.png)

LSTM의 구조는 복잡해보이지만 **3개의 Gate와 1개의 Cell** 부분만 이해하면 된다. 

**Forget Gate**

Forget gate는 previous output(hidden state) `h(t-1)`과 input `x(t)`를 이용해 만든 정보 `f(t)`로 **어떤 정보를 버릴지** 결정합니다. 

![image-20220208183636794](https://user-images.githubusercontent.com/70505378/152983134-3eb40930-bec3-4e26-a2b2-f4f91aba8155.png)

**Input Gate**

Input gate는 두 가지 정보를 생성하고 이용합니다. 

* `C'(t)`: Previous output(hidden state) `h(t-1)`과 input `x(t)`를 이용해 현재 cell state에 저장할 정보 후보들을 만들어냅니다. 
* `i(t)`: Previous output(hidden state) `h(t-1)`과 input `x(t)`를 이용해 정보 후보들 중 어떤 정보를 저장할 지 선택합니다. 

최종적으로 만들어진 정보 `C'(t)`와 `i(t)`를 이용하여 **현재 cell state `C(t)`에 전달할 정보**를 만들어냅니다. 

![image-20220208185133992](https://user-images.githubusercontent.com/70505378/152983138-0da04fb5-8dfb-4f2c-98c0-cc2c1e2d0d07.png)

**Update Cell**

Update cell은 forget gate와 input gate에서 만들어진 정보들 `f(t)`, `C'(t)`, `i(t)`과 previous cell state `C(t-1)`를 이용해 **현재 cell state `C(t)`**를 만들어냅니다. 

Cell state에는 외부에는 노출되지 않는 **시간 0~t 까지의 정보들이 인코딩**되어 있습니다. 

![image-20220208185224572](https://user-images.githubusercontent.com/70505378/152983142-0bbd5d2a-faa2-40b8-9fd4-403abd974c04.png)

**Output Gate**

Output gate는 먼저 previous output(hidden state) `h(t-1)`과 input `x(t)`를 이용해 밖으로 내보낼(출력할) 정보 후보 `o(t)`를 만들어냅니다. 

그리고 만들어진 `o(t)`와 cell state `C(t)`를 이용해 **밖으로 내보낼 output(현재 hidden state) `h(t)`**를 만들어냅니다. 

![image-20220208185257047](https://user-images.githubusercontent.com/70505378/152983145-eced45e3-2a3b-47cc-939b-019be3a377f4.png)

<br>

LSTM의 구조를 요약해서 나타내면 다음과 같습니다. 

![image-20220208185800051](https://user-images.githubusercontent.com/70505378/152983146-670d9350-5fe1-4d82-a782-d6366e976e89.png)

<br>

### GRU

GRU(Gated Recurrent Unit)는 LSTM의 간소화된 형태라고 할 수 있습니다. 다만, 놀랍게도 **GRU는 더 적은 파라미터로 높은 학습 속도와 일반화 성능을 보이면서 LSTM보다 더 나은 성능을 지즈 보여줍니다.**

![image-20220208190144437](https://user-images.githubusercontent.com/70505378/152983149-67d746ac-fad8-4233-be3e-7bd8ddf4418e.png)

GRU는 2개의 gate(**reset gate** and **update gate**)를 사용하며, **cell state** 없이 **hidden state**만을 사용합니다. 





















<br>

<br>

## 참고 자료

* 

















<br>
