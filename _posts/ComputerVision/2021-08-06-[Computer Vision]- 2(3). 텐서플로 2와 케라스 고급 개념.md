---
layout: single
title: "[Computer Vision] 2(3). 텐서플로 2와 케라스 고급 개념"
---



<br>

# 텐서플로 2와 케라스 고급 개념

- 지난 포스팅의 내용을 요약하면 오토그래프 모듈, tf.function 데코레이터, 그래디언트 테이프 컨텍스트는 보이지 않는 곳에서 그래프를 생성하고 관리하는 작업을 간단하게 만든다. 
- 이번 포스팅에서는 그 뒤에 숨어있는 복잡한 내부 작동 방식에 대해 알아본다. 

<br>

### tf.function 작동 방식

---

- tf.function 데코레이팅 된 함수를 호출하면 텐서플로는 연산에 맞는 그래프를 캐시에 저장하여 다음에 같은 함수가 호출될 때 재사용한다. 


```python
import tensorflow as tf
```


```python
@tf.function
def identity(x):
    print('Creating graph !')
    return x
```

<br>

위 함수의 경우 텐서플로는 그래프를 캐시에 저장하기 때문에 최소 실행 시에만 출력을 진행한다. 


```python
x1 = tf.random.uniform((10, 10))

for _ in range(10):
    identity(x1)
```

    Creating graph !

<br>

```python
x1 = tf.random.uniform((10, 10))
x2 = tf.random.uniform((10, 10))
result1 = identity(x1) # 'Creating graph !'를 출력
result2 = identity(x2) # 아무것도 출력하지 않음
```

입력 타입을 변경하는 경우 텐서플로는 다시 그래프를 생성한다. 


```python
x3 = tf.random.uniform((10, 10), dtype=tf.float16)
result3 = identity(x3) # 'Creating graph !'를 출력
```

    Creating graph !

요약하면 장식된 함수가 최초로 실행될 때마다 텐서플로는 그 **입력 타입**과 **입력 형상**에 대응하는 그래프를 캐시에 저장한다. 그 함수가 다른 타입의 입력으로 실행되면 텐서플로는 새로운 그래프를 생성해서 캐시에 저장한다. 
<br>
그럼에도 불구하고 실제 함수가 매번 실행될 때마다 정보를 로그로 남기고 싶을 경우, **tf.print**를 사용하면 된다. 
<br>
✋ tf.function이 입력 형식을 정의할 때 그 함수는 **실제 함수**가 된다. 

<br>


```python
@tf.function
def identity(x):
    tf.print("Running identity")
    return x
```


```python
x1 = tf.random.uniform((10, 10))

for _ in range(10):
    identity(x1)
```

    Running identity
    Running identity
    Running identity
    Running identity
    Running identity
    Running identity
    Running identity
    Running identity
    Running identity
    Running identity

<br>

<br>

### 텐서플로 2의 변수

---

- 모델 가중치를 저장하기 위해 텐서플로는 Variable 인스턴스를 사용한다. model.variables에 접근함으로써 모델에 포함된 모든 변수의 목록을 반환한다. 


```python
num_classes = 10
img_rows, img_cols = 28, 28
num_channels = 1
input_shape = (img_rows, img_cols, num_channels)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```


```python
# 맨 처음에는 모델 입력 형상 지정(layers 모듈의 Input 클래스)
model_input = tf.keras.layers.Input(shape=input_shape)
# 다음으로 계층들을 추가. 입력 값을 전달하고 output을 반환. 
output = tf.keras.layers.Flatten()(model_input)     
# Dense 계층의 경우 출력의 개수와 활성화 함수 지정. 
output = tf.keras.layers.Dense(128, activation='relu')(output)    
output = tf.keras.layers.Dense(10, activation='softmax')(output)
# 마지막으로 models 모듈의 Model 클래스에 input과 output을 전달하여 model 컴파일.
model = tf.keras.models.Model(model_input, output)                 
```


```python
print([variable.name for variable in model.variables])
```

    ['dense/kernel:0', 'dense/bias:0', 'dense_1/kernel:0', 'dense_1/bias:0']

<br>

각자만의 변수를 생성할 수도 있다. 


```python
a = tf.Variable(3, name = 'my_var')
print(a)
```

    <tf.Variable 'my_var:0' shape=() dtype=int32, numpy=3>

<br>

대규모 프로젝트에서는 코드를 명확하게 하고 디버깅을 쉽게 할 수 있는 변수명을 지정하는 것이 좋다. 
<br>
**Variable.assign** 메서드를 사용하면 Variable 객체의 값을 바꿀 수 있다. 


```python
a.assign(a+1)
print(a)
```

    <tf.Variable 'my_var:0' shape=() dtype=int32, numpy=4>


아래와 같은 코드는 새로운 Tensor 객체를 생성한다. 


```python
b = a + 1
print(b)
```

    tf.Tensor(5, shape=(), dtype=int32)

<br>

<br>

### 분산 전략

---

매우 큰 모델과 데이터셋을 사용할 때에는 많은 컴퓨팅 파워(또는 여러 서버)가 필요하다. **tf.distribute.Strategy API**는 모델을 효율적으로 훈련시키기 위해 여러 컴퓨터가 서로 통신하는 방법을 정의한다. 
- **MirroredStrategy**: 한 서버 내의 여러 GPU에서 훈련시키는 경우. 모델 가중치는 각 기기 사이에 싱크를 유지한다. 
- **MultiWorkerMirroredStrategy**: 여러 서버에서 훈련 시킨다는 점을 제외하면 MirroredStrategy와 동일. 
- **ParameterServerStrategy**: 여러 서버에서 훈련시킬 때 사용. 각 기기에 가중치를 동기화하는 대신 매개변수 서버에 가중치를 저장한다. 
- **TPUStrategy**: 구글 **텐서 처리 장치** 칩에서 훈련시킬 때 사용한다. 
<br>
분산 전략을 사용하려면 모델을 그 분산 전략의 범위에서 생성하고 컴파일 해야 한다. 


```python
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirroed_strategy.scope():
    model = make_model # 모델을 여기에서 생성
    model.compile([...])
```

각 기기가 각 배치의 작은 하위 집합을 받기 때문에 배치 크기를 키워야 할 수 있다. 또한 모델에 따라 학습률을 변경해야 할 수도 있다. 

<br>

<br>

### 에스티메이터 API

---

- 에스티메이터는 훈련, 평가, 예측, 서비스를 단순화한다. 
- 에스티메이터에는 텐서플로에서 제공하는 빠르게 머신러닝 아키텍쳐를 구성할 수 있는 **사전 제작된 에스티메이터**와 어떤 모델 아키텍쳐를 사용하더라도 생성될 수 있는 **맞춤형 에스티메이터** 두 가지가 있다. 
- 에스티메이터는 데이터 큐, 예외 처리, 장애 복고, 주기적 체크 포인트 등을 모두 처리한다. 
- 텐서플로 1에서는 에스티메이터를 사용하는 것이 최선이지만, 텐서플로 2에서는 케라스 API를 사용하는 것이 좋다. 

<br>

##### 사전 제작된 에스티메이터

- DNNClassifier, DNNRegressor, LinearClassifier, LinearRegressor 가 있다. 두 아키텍쳐에 기반한 **결합된 에스티메이터**인 DNNLineraComblinedClassifier와 DNNLinearCombinedRegressor 도 있다. 
- 결합된 에스티메이터(깊고 넓은 모델)는 선형 모델(기억을 위해)과 심층 모델(일반화를 위해)을 활용한다. 이 에스티메이터는 대체로 추천 혹은 순위 모델을 위해 사용된다. 
- 사전 제작된 에스티메이터는 일부 머신러닝 문제에 적합할 수 있지만, 컴퓨터 비전 문제에는 적합하지 않다. 

<br>

##### 맞춤형 에스티메이터 훈련시키기

에스티메이터를 생성하기 가장 쉬운 방식은 케라스 모델을 변환하는 것이다. 모델을 compile한 뒤, **tf.keras.estimator.model_to_estimator()**를 호출한다. 


```python
model.compile(optimizer='sgd', 
             loss='sparse_categorical_crossentropy', 
             metrics=['accuracy'])
estimator = tf.keras.estimator.model_to_estimator(model, model_dir='./estimator_dir')
```

    INFO:tensorflow:Using default config.
    INFO:tensorflow:Using the Keras model provided.
    INFO:tensorflow:Using config: {'_model_dir': './estimator_dir', '_tf_random_seed': None, '_save_summary_steps': 100, '_save_checkpoints_steps': None, '_save_checkpoints_secs': 600, '_session_config': allow_soft_placement: true
    graph_options {
      rewrite_options {
        meta_optimizer_iterations: ONE
      }
    }
    , '_keep_checkpoint_max': 5, '_keep_checkpoint_every_n_hours': 10000, '_log_step_count_steps': 100, '_train_distribute': None, '_device_fn': None, '_protocol': None, '_eval_distribute': None, '_experimental_distribute': None, '_experimental_max_worker_delay_secs': None, '_session_creation_timeout_secs': 7200, '_checkpoint_save_graph_def': True, '_service': None, '_cluster_spec': ClusterSpec({}), '_task_type': 'worker', '_task_id': 0, '_global_id_in_cluster': 0, '_master': '', '_evaluation_master': '', '_is_chief': True, '_num_ps_replicas': 0, '_num_worker_replicas': 1}


![KakaoTalk_20210806_155057969](https://user-images.githubusercontent.com/70505378/128482237-5eda3f79-2d1e-4c80-bb2d-d35bbc5af925.png)

에스티메이터를 훈련시키려면 **입력 함수**(특정 포맷의 데이터를 반환하는 함수)를 사용해야 한다. 허용되는 포맷 중 하나는 텐서플로 Dataset이다. 
<br>
여기서는 이전 포스트들에서 사용한 mnist 데이터셋을 32개 샘플의 배치로 묶은 올바른 포맷으로 반환하는 다음 함수를 정의한다. 


```python
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
```


```python
BATCH_SIZE = 32

def train_input_fn():
    # from_tensor_slices((features, labels))
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    
    train_dataset = train_dataset.shuffle(len(x_train)).batch(BATCH_SIZE).repeat()
    return train_dataset # tf.Dataset
```


```python
len(x_train), len(y_train)
```




    (60000, 60000)



위처럼 입력 함수가 정의되면 에스티메이터를 훈련시킬 수 있다. 

<br>

✋ **train_dataset의 메서드**
    - shuffle(횟수): 데이터셋을 고정된 버퍼 크기 단위로 뒤섞는다. 데이터셋을 완전히 섞기 위해서는 횟수를 데이터셋의 크기보다 크게 지정해야 한다. 
    - batch(배치 크기): 데이터를 한 번에 읽어올 개수를 지정
    - repeat(): 데이터셋을 읽다가 마지막에 도달했을 경우 다시 처음부터 조회


```python
estimator.train(train_input_fn, steps=len(x_train)//BATCH_SIZE)
```

<br>

<br>

### 텐서보드

---

텐서플로는 강력한 모니터링 도구로 **텐서보드**를 제공한다. 텐서플로를 설치하면 기본으로 설치되기 때문에 케라스의 콜백과 결합해 사용하기가 매우 쉽다. 


```python
callbacks = [tf.keras.callbacks.TensorBoard('./logs_keras')]
model.fit(x_train, y_train, epochs=10, verbose=1, validation_data=(x_test, y_test),
         callbacks = callbacks)
```

    Epoch 1/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.6724 - accuracy: 0.8316 - val_loss: 0.3583 - val_accuracy: 0.9029
    Epoch 2/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.3376 - accuracy: 0.9062 - val_loss: 0.2906 - val_accuracy: 0.9210
    Epoch 3/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.2880 - accuracy: 0.9191 - val_loss: 0.2596 - val_accuracy: 0.9291
    Epoch 4/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.2568 - accuracy: 0.9278 - val_loss: 0.2355 - val_accuracy: 0.9347
    Epoch 5/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.2331 - accuracy: 0.9352 - val_loss: 0.2170 - val_accuracy: 0.9386
    Epoch 6/10
    1875/1875 [==============================] - 3s 2ms/step - loss: 0.2142 - accuracy: 0.9403 - val_loss: 0.2029 - val_accuracy: 0.9440
    Epoch 7/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1983 - accuracy: 0.9439 - val_loss: 0.1896 - val_accuracy: 0.9475
    Epoch 8/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1850 - accuracy: 0.9479 - val_loss: 0.1788 - val_accuracy: 0.9501
    Epoch 9/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1735 - accuracy: 0.9506 - val_loss: 0.1690 - val_accuracy: 0.9534
    Epoch 10/10
    1875/1875 [==============================] - 4s 2ms/step - loss: 0.1635 - accuracy: 0.9533 - val_loss: 0.1605 - val_accuracy: 0.9551

    <tensorflow.python.keras.callbacks.History at 0x17afd585c10>

![KakaoTalk_20210806_172948168](https://user-images.githubusercontent.com/70505378/128482309-2939df65-dac5-48a5-923e-148edfe237eb.png)

<br>

이후 명령줄에서 아래 명령어를 이용해 텐서보드 인터페이스를 표기하기 위해 열 수 있는 URL을 출력한다. 보통 http://localhost:6006/ 경로를 알려준다. (현재 터미널 위치가 working directory가 아니라면 위치 이동 후 입력해야 한다)


```python
$ tensorboard --logdir ./logs_keras
```

![KakaoTalk_20210806_165904810](https://user-images.githubusercontent.com/70505378/128482272-46d6209a-e315-4ec2-b814-ab1fc22a5d9d.png)

<br>

텐서보드는 시간에 따라 모델의 성능을 모니터링하는데 배우 중요하게 사용된다. 텐서보드에서는 다음의 작업들이 가능하다. 
- 모델의 손실과 정확도 그래프 표시
- 입출력 이미지 표시
- 실행 시간 표시
- 모델의 계산 그래프 표시

<br>

* 참고 1: [텐서보드 시작하기](https://gooopy.tistory.com/98)
* 참고 2: [Tensorflow 2.0 Tensorboard 사용법](https://willbesoon.tistory.com/15)
* 참고 3: [그래프시각화 - 텐서보드](https://www.youtube.com/watch?v=hqb5Z5RDfws)

<br>

<br>

### 텐서플로 애드온과 텐서플로 확장

---

**텐서플로 애드온**

- 텐서플로 애드온은 부가적인 기능을 한 곳([Tensorflow Addons](https://github.com/tensorflow/addons))에 모아둔 것이다. 여기에는 딥러닝 분야에서 최근 고안되어 메인 텐서플로 라이브러리에 추가되지 않은 아직 안정적이지 않고 충분히 활용되지 못한 기법들을 담고 있다. 
- 텐서플로 1에서 제거된 tf.contrib를 대체한다. 

<br>

**텐서플로 확장 버전**
- 텐서플로를 위한 엔드투엔드 머신러닝 플랫폼이다. 
- **TensorFlow Data Validation**: 머신 러닝 데이터 탐색 및 검증을 위한 라이브러리. 모델을 구성하기 전에 이것을 사용할 수 있다. 
- **TensorFlow Transform**: 데이터 전처리를 위한 라이브러리. 이것을 사용하면 훈련 데이터와 평가 데이터가 동일한 방식으로 처리될 수 있다. 
- **TensorFlow Model Analysis**: 텐서플로 모델 평가를 위한 라이브러리
- **TensorFlow Serving**: 머신러닝 모델을 서비스하는 시스템. 모델을 서비스한다는 것은 일반적으로 REST API를 통해 모델의 예측을 전달하는 프로세스라는 뜻이다. 
<br>
이러한 도구는 딥러닝 모델을 구성하고 사용하는 프로세스(**데이터 검증 ➡ 데이터 변환 ➡ 모델 훈련 ➡ 모델 분석 ➡ 모델 서비스**)의 모든 단계를 다뤄 모델 생애주기의 처음부터 끝까지를 포괄하는 목표를 충족한다. 

<br>

<br>

### 텐서플로 라이트와 Tensorflow.js

- **텐서플로 라이트**는 모바일 폰과 임베디드 기기에서 모델 예측(추론)을 실행하게 설계되었다. 텐서플로 라이트는 텐서플로 모델을 tflite 포맷으로 변환하는 컨버터와 추론을 실행하기 위해 모바일 디바이스에 설치될 수 있는 인터프리터로 구성된다. 
- 좀 더 최근에는 **Tensorflow.js(tfjs)**가 거의 모든 웹 브라우저에서 딥러닝을 사용할 수 있도록 개발되었다. 이 버전은 사용자가 따로 설치할 내용은 없고 경우에 따라 시스템의 GPU 가속을 사용할 수 있다. 

<br>

<br>

### 모델 실행 장소

컴퓨터 비전 모델은 대용량 데이터를 처리하기 때문에 훈련에 긴 시간이 소요된다. 또한 효율적인 모델을 생성하려면 여러 차례 반복이 필요하다. 
위 두 사실로부터 모델을 어디에서 훈련시키고 실행할 지에 대한 결정을 할 수 있다. 

<br>

##### 로컬 컴퓨터
처음 시작할 때에는 로컬 컴퓨터가 가장 친숙하고 빠른 방법이다. 다만, 로컬 컴퓨터를 사용할 경우 GPU를 사용할 것을 추천한다. GPU에서 훈련하면 CPU를 사용할 때보다 10배~100배 까지 빨라진다. 

<br>

##### 원격 시스템

최근에는 시간단위로 GPU를 장착한 강력한 시스템을 대여할 수 있다. 
원격 시스템에 안전하게 접근할 수 있다면 다음 두 가지 옵션이 있다. 
- 원격 서버에 주피터 노트북을 실행한다. 그러면 주피터 랩이나 주피터 노트북은 브라우저를 통해 어디서든 접근 가능하다. 이는 딥러닝을 실행하는 매우 편리한 방식이다. 
- 로컬 개발 폴더와 싱크를 맞추고 코드를 원격에서 실행한다. 대부분의 IDE는 원격 서버의 싱크를 맞추는 기능을 제공한다. 이렇게 하면 자신이 선호하는 IDE에서 코딩하면서도 막강한 시스템을 누릴 수 있다. 

<br>

##### 구글 클라우드
원격 시스템을 이용하면 소프트웨어 설치, 최신 버전 업데이트 등 직접 관리해야 하는 것이 많지만, 구글 클라우드 ML을 사용해 텐서플로를 실행하면 시스템 작업이 아니라 모델에 집중할 수 있다. 
구글 클라우드 ML은 다음 작업을 하는 데 유용하다. 

- 클라우드의 탄력적으로 운용되는 자원 덕분에 빠르게 훈련이 가능하다. 
- 병렬 처리로 짧은 시간 내에 최적의 모델 매개변수를 구한다. 
- 모델이 준비되면 예측 서버를 실행하지 않고도 예측 서비스를 제공한다. 

참고: [구글 클라우드 ML 문서](https://cloud.google.com/ml-engine/docs/)

<br>

<br>

### 정리

---

- 케라스는 개발을 쉽게 하기 위해 다른 딥러닝 라이브러리를 감싼 래퍼로 설계되었다. 이제 텐서플로는 tf.keras 를 통해 케라스와 완전히 통합되었다. 이 모듈을 사용해 텐서플로 2에서 모델을 생성하는 것이 가장 좋다. 
- 텐서플로는 모델 성능과 이식성을 보장하기 위해 그래프를 사용한다. 텐서플로 2에서 수동으로 그래프를 생성하는 가장 좋은 방법은 tf.function 데코레이터를 사용하는 것이다. 
- 느긋한 실행 모드에서는 사용자가 구체적으로 겨로가를 요청하기 전까지는 계산이 수행되지 않는다. 조급한 실행 모드의 경우 모든 연산은 정의되는 시점에 실행된다. 전자는 그래프 최적화 덕분에 더 빠를 수 있지만, 후자는 사용하고 디버깅하기 더 쉽다. 텐서플로 2에서는 조급한 실행 모드를 사용한다. 
- 텐서보드에 정보를 기록하려면 tf.keras.callbacks.TensorBoard 콜백을 사용하고, 모델을 훈련시킬 때 이 콜백을 fit 메서드에 전달하면 된다. 정보를 수동으로 로깅하기 위해 tf.summary()  모듈을 사용할 수 있다. 텐서보드 URL은 터미널에서 **tensorboard --logdir 로그디렉터리** 명령어로 얻을 수 있다. 
- 텐서플로 2는 사용자 편의를 위해 그래프를 직접 관리하는 일을 제거했으며, 기본적으로 조급한 실행을 사용해 모델 디버깅이 편리해졌다. 그럼에도 불구하고 AutoGraph와 tf.function 덕분에 여전히 성능을 유지한다. 또한 케라스와 긴밀하게 통합되어 어느 때보다 쉽게 모델을 만들 수 있다. 
