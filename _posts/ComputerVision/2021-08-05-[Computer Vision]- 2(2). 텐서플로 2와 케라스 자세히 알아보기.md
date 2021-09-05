---
layout: single
title: "[Computer Vision] 2(2). 텐서플로 2와 케라스 자세히 알아보기"
---



<br>

# 텐서플로 2와 케라스 자세히 알아보기

이제 텐서플로 2의 주요 핵심 개념을 자세히 살펴보자. 

<br>

### 핵심 개념

2019년 봄에 단순함과 사용 편의성에 초점을 맞춰 텐서플로의 새로운 버전이 출시됐다. 이 절에서는 텐서플로의 기본 개념을 소개하고 버전1에서 버전2로 어떻게 진화했는지 다룬다. 

##### 텐서 소개

텐서는 N차원의 배열이다. 텐서플로에서 Tensor 객체는 수치 값을 저장하기 위해 사용하며, 각 Tensor 객체는 다음 요소를 갖추고 있다. 
- **타입**: string, float32, float16, int8 등
- **형상**: 데이터 차원. 
- **순위**: 차원 개수. 
<br>
이미지와 같이 높이와 너비를 미리 알 수 없는 경우(부분적으로 형상을 알 수 없는 경우) 입력 형상을 (None, None, 3)과 같이 나타낸다. 

<br>

##### 텐서플로 그래프

연산이란 입력을 출력으로 변환하는 것이며, 텐서플로는 이 연산을 표현하기 위해 **그래프**를 사용한다. 이 그래프 개념은 텐서플로의 동작을 이해하는데 있어 매우 중요하다. 
<br>
그래프를 활용하면 다음과 같은 장점이 있다. 
- CPU에서 일부 연산을 실행하고 GPU에서 남은 연산을 실행한다. 
- 분산 모델의 경우 그래프의 다양한 부분을 여러 다른 컴퓨터에서 실행한다. 
- 불필요한 연산을 피하기 위해 그래프를 최적화해 계산 성능을 개선한다. 

<br>


##### 느긋한 실행과 조급한 실행 비교

- 텐서플로 1까지는 기본적으로 항상 **느긋한 실행(lazy execution)**을 사용했는데, 이는 프레임워크에 구체적으로 요청하기 전까지 연산이 실행되지 않기 때문이다. 
- 반면 텐서플로 2는 **조급한 실행(eager execution)**을 지원한다. 


```python
import tensorflow as tf

a = tf.constant([1,2,3])
b = tf.constant([0,0,1])
c = tf.add(a,b)

print(c)
```

    tf.Tensor([1 2 4], shape=(3,), dtype=int32)


만약 위의 코드를 텐서플로 1에서 실행했다면 결과는 다음과 같다. 
> Tensor("Add:0", shape=(3,), dtype=int32)

<br>

##### 텐서플로 2에서 그래프 생성하기

먼저 그래프 생성과 최적화 과정을 보여줄 수 있는 간단한 예제를 살펴보자. 


```python
def compute(a, b, c):
    d = a * b + c
    e = a * b * c
    return d, e
```

위 함수의 경우 d와 e를 계산할 때 동일한 연산인 a * b를 수행한다. 느긋한 실행에서는 **그래프 최적화기**가 a * b가 두번 실행되는 것을 피하기 위해 결과를 **캐시**에 저장하고 필요할 때 재사용한다. 또한 더 복잡한 연산의 경우 최적화기는 계산 속도를 높이기 위해 **병렬 처리**를 사용할 수 있다. 
<br>
하지만 조급한 실행에서는 이러한 최적화 기법이 적용될 수 없다. 다행히도 텐서플로는 이를 해결할 수 있는 모듈을 포함하고 있는데, 바로 텐서플로 **오토그래프**이다. 

<br>

##### 텐서플로 오토그래프와  tf.function

텐서플로 오토그래프 모듈은 자동 최적화를 가능하게 해 조급한 실행 코드를 그래프로 변환하기 쉽게 만든다. 
<br>
이는 단순히 함수의 맨 앞에 tf.function 데코레이터를 추가함으로써 수행된다. 


```python
@tf.function
def compute(a, b, c):
    d = a * b + c
    e = a * b * c
    return d, e
```

✋ **파이썬 데코레이터**는 함수를 감싸고 그 함수에 기능을 추가하거나 변경하는 것이 가능하게 해주는 개념이다. 

<br>
일반적으로 오토 그래프는 다음의 경우에 사용한다. 
- 모델을 다른 기기로 내보내야 할 때
- 성능이 무엇보다 중요하고 그래프 최적화를 통해 속도 개선이 필요할 때
<br>
그래프의 또 다른 장점으로 **자동 미분**을 들 수 있다. 조급한 실행모드에서 각 연산은 독립적이기 때문에 기본적으로 자동 미분이 가능하지 않지만, 텐서플로 2는 이를 가능하게 만드는 방법으로 **그래디언트 테이프(gradient tape)**를 제공한다. 

<br>

##### 그래디언트 테이프를 사용해 오차 역전파하기

여기서는 간단히 AxX = B 를 만족하는 X를 풀어야 한다고 가정하고, 그러기 위해 간단한 손실 abs(AxX - B)를 최소화할 것이다. 

그래디언트 테이프를 사용하지 않는 경우 텐서플로는 연산을 저장하는 대신 작업 결과를 계산한다. 따라서 연산과 그 연산의 입력에 대한 정보가 없으면 자동으로 손실을 미분할 수 없다. 


```python
A, B = tf.constant(3.0), tf.constant(6.0)
X = tf.Variable(20.0) # 랜덤 값
loss = tf.math.abs(A*X-B)

print(loss)
```

    tf.Tensor(54.0, shape=(), dtype=float32)


<br>
이제 tf.GradientTape를 사용해보자. tf.GradientTape의 컨텍스트에서 손실을 계산함으로써 텐서플로는 자동으로 모든 연산을 기록하고 그런 다음 역으로 이 모든 연산을 재생한다.


```python
def train_step():
    with tf.GradientTape() as tape:
        loss = tf.math.abs(A*X-B)
    dX = tape.gradient(loss, X)
    print('X = {:.2f}, dX = {:2f}'.format(X.numpy(), dX))
    X.assign(X - dX)
    
for i in range(7):
    train_step()
```

    X = 20.00, dX = 3.000000
    X = 17.00, dX = 3.000000
    X = 14.00, dX = 3.000000
    X = 11.00, dX = 3.000000
    X = 8.00, dX = 3.000000
    X = 5.00, dX = 3.000000
    X = 2.00, dX = 0.000000


위 코드는 단일 훈련 단계를 정의한다. 
<br>
train_step이 호출될 때마다 손실이 그래디언트 테이프의 컨텍스트에서 제공된다. 그 다음 이 컨텍스트는 경사를 계산하기 위해 사용된다. 그러고 나서 X 변수가 업데이트 된다. 실제로 X가 공식의 해로 수렴하는 것을 볼 수 있다. 

<br>

### 케라스 모델과 계층

---

앞에서 얻은 model 객체에는 여러 가지 유용한 메서드와 속성이 포함되어 있다. 
- inputs와 outputs: 모델 입력과 출력에 접근
- layers: 모델 계층과 형상 목록
- summary(): 모델 아키텍쳐를 출력
- save(): 훈련에서 모델, 아키텍쳐의 현 상태를 저장한다. 추후에 훈련을 재개할 때 매우 유용하다. 모델은 tf.keras.models.load_model()을 사용해 인스턴스화 할 수 있다. 
- save_weights(): 모델의 가중치만 저장한다. 

<br>

##### 순차형 API와 함수형 API

- 앞에서 사용한 **순차형 API**를 사용하는 대신 **함수형 API**를 사용할 수 있다. 


```python
model_input = tf.keras.layers.Input(shape=input_shape)
output = tf.keras.layers.Flatten()(model_input)
output = tf.keras.layers.Dense(128, activation='relu')(output)
output = tf.keras.layers.Dense(num_classes, activation='softmax')(output)
model = tf.keras.Model(model_input, output)
```

함수형 API는 순차형 API보다 훨씬 범용적으로 사용된다. 함수형 API는 모델을 분기(병렬 계층으로 구성)할 수 있고, 순차형 API는 선형 모델에만 사용 가능하다. 

<br>

##### 콜백

**케라스 콜백**은 케라스 모델의 기본 행위에 기능을 추가하기 위해 케라스 모델의 fit() 메서드에 전달할 수 있는 유틸리티 함수다. 
- CSVLogger: 훈련 정보를 CSV 파일에 로그로 남긴다. 
- EarlyStopping: 손실 혹은 메트릭이 더 이상 개선되지 않으면 훈련을 중지한다. 과적합을 피할 때 유용하다. 
- LearningRateScheduler: 스케줄에 따라 세대마다 학습률을 변경한다. 
- ReduceLROnPlateau: 손실이나 메트릭이 더이상 개선되지 않으면 학습률을 자동으로 감소시킨다. 
<br>
tf.keras.callbacks.Callback의 서브 클래스를 생성함으로써 맞춤형 콜백을 생성할 수도 있다. 
