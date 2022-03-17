---
layout: single
title: "[AITech][NLP] 20220310 - Part 2) RNN, LSTM, and GRU"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 KAIST '주재걸' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# RNN, LSTM, and GRU

이번 강의는 `RNN` 과 `LSTM`, `GRU`에 대한 내용입니다. 

## Basics of Recurrent Neural Networks (RNNs)

다들 RNN에 대해서는 지겹도록 많이 들으셨을 겁니다. 여기서는 장황하게 설명하지 않고, RNN을 공부함에 있어 중요하고 핵심적인 내용들을 키워드 위주로 정리해보겠습니다. 

![image-20220317140619792](https://user-images.githubusercontent.com/70505378/158777963-92e8f91f-a889-4082-99ec-c0da2de37a4c.png)

### Sequence Data

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

### RNN(Recurrent Neural Network)

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

### Types of RNNs

* **One-to-one**: Standard Neural Networks
* **One-tomany**: Image Captioning
* **Many-to-one**: Setiment Classification
* **Sequence-to-sequence**: Machine Translation
* **Many-to-Many**: Video classification on frame level

![image-20220317141100058](https://user-images.githubusercontent.com/70505378/158777966-d93824ec-f7e6-4c9c-a8e7-517d5fc0fde1.png)









<br>

## LSTM & GRU

### LSTM

LSTM(Long Short Term Memory)은 Vanilla RNN의 한계인 Long-term memory를 효과적으로 전달하기 위해 고안된 모델입니다. 

![image-20220208183002864](https://user-images.githubusercontent.com/70505378/152983127-b220110a-625d-4706-96b6-dfa66117b563.png)

LSTM의 구조는 복잡해보이지만 **3개의 Gate와 1개의 Cell** 부분만 이해하면 됩니다. 전체적인 input-output 관계부터 살펴보겠습니다. 

* `i`: input gate, cell state에 전달할 정보 생성
* `f`: forget gate,cell state에 전달하지 않고 버릴 정보 생성
* `o`: output gate, 다음 time step에 전달할 hidden state 생성
* `g`: gate gate(update cell), 현재 time step의 cell state 생성

(ifog라고 외우면 쉽습니다 😊)

![image-20220317174937832](https://user-images.githubusercontent.com/70505378/158777968-6059baf9-d99a-4784-ab07-86d67b384f36.png)

input `x`와 `h`의 길이를 모두 **h**라 한다면, LSTM의 전체 가중치 W의 파라미터 개수는 **4h \* 2h = 8h<sup>2</sup>**으로 구할 수 있습니다. `W * [h, x]`로 구한  출력 행렬의 길이는 **4h**가 되고, 이는 각각 i/f/o/g gate에 **h** 씩 분배됩니다. 각각의 gate는 가중치에 activation을 적용하여 output을 만들게 됩니다. 

이 때 주목할 것은, 당연하게도 입력 hidden state `h`와 출력 hidden state `o`의 차원이 동일하게 **h**라는 것입니다. 또한 만약 bias도 존재한다면, LSTM의 전체 parameter의 개수는 **4h \* (2h+1) = 8h<sup>2</sup> + 4h** 개가 될 것입니다 (단, 여기서 '2h'는 'x의 길이+h의 길이'인 것을 주의해주세요). 

추가적으로 activation function에 대한 얘기를 조금 더 해보겠습니다. i/f/o gate의 activation인 **sigmoid**는 다들 잘 아시다시피 값을 확률로 매핑해주는 역할을 합니다. 입력 정보에서 몇 퍼센트의 정보를 출력으로 전달할 지(또는 버릴지)에 대한 확률 값을 생성해줍니다. Gate gate(update cell)의 경우 activation으로 **tanh**를 사용했는데, tanh는 값을 -1 ~ 1 사이의 값으로 매핑해주는 함수로서 입력 정보로부터 유의미한 정보들을 뽑아내는 용도로 사용된다고 합니다. 

<br>

**Forget Gate**

Forget gate는 previous output(hidden state) `h(t-1)`과 input `x(t)`를 이용해 만든 정보 `f(t)`로 **어떤 정보를 버릴지** 결정합니다. 

![image-20220208183636794](https://user-images.githubusercontent.com/70505378/152983134-3eb40930-bec3-4e26-a2b2-f4f91aba8155.png)

**Input Gate**

Input gate는 두 가지 정보를 생성하고 이용합니다. 

* `C'(t)`: Previous output(hidden state) `h(t-1)`과 input `x(t)`를 이용해 **현재 cell state에 저장할 정보 후보**들을 만들어냅니다. 
* `i(t)`: Previous output(hidden state) `h(t-1)`과 input `x(t)`를 이용해 **정보 후보들 중 어떤 정보를 저장할 지** 선택합니다. 

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

GRU(Gated Recurrent Unit)는 LSTM의 경랸화된 형태라고 할 수 있습니다. 다만, 놀랍게도 **GRU는 더 적은 파라미터로 높은 학습 속도와 일반화 성능을 보이면서 LSTM보다 더 나은 성능을 자주 보여줍니다.**

![image-20220208190144437](https://user-images.githubusercontent.com/70505378/152983149-67d746ac-fad8-4233-be3e-7bd8ddf4418e.png)

GRU는 2개의 gate(**reset gate** and **update gate**)만을 사용하며, **cell state** 없이 **hidden state**만을 사용합니다. 

<br>

### Backpropagation in LSTM(GRU)

마지막으뢰 왜 LSTM이 gradient vanishing(exploding) 문제를 해결할 수 있는지 보겠습니다. 

input에 반복적으로 동일한 W<sub>hh</sub>를 곱하는 Vaniila RNN과 달리, LSTM(GRU)은 각 time step마다 cell state에 별개의 f(forget gate)와 elementwise multiplication을 수행하고 이후 addition 연산만이 수행되기 때문에 gradient vanishing(exploding) 문제로부터 자유롭습니다. 

따라서 더 긴 sequence에 대해서도 gradient를 잘 유지하면서 학습할 수 있게 됩니다. 

![image-20220317182052761](https://user-images.githubusercontent.com/70505378/158777959-33bd8a84-a80b-48b0-9e4b-f439ab7b7253.png)



<br>

## 실습) Word-level language modeling with RNN

이번 실습에서는 데이터 토큰화 및 전처리에 대한 내용은 다루지 않습니다. 해당 내용이 궁금하신 분들이 제 이전 포스팅인 [Part 1) Bag or Words & Word Embedding]을 참고하시길 바랍니다. 

### 모델 아키텍쳐 준비

* `RNNModel`: Embedding, RNN module, Projection 를 포함한 컨테이너 모듈. 다음과 같이 이전 hidden state와 input을 받아 다음 토큰의 log probability와 다음 hidden state를 반환합니다.

![image-20220317183305124](https://user-images.githubusercontent.com/70505378/158785190-bff94862-b358-4f95-ae7e-d4b58b79184c.png)



<br>

모델의 forward 순서는 다음과 같이 구성됩니다. 

* `input`을 embedding layer와 dropout layer에 차례로 통과시켜 embedded vector를 얻음
* `embedded vector`를 rnn layer에 통과시켜 `output`과 `next_hidden`을 얻음
* `output`을 dropout layer와 projection layer(hidden dimension -> vocab_size)를 통과시킨 후 softmax를 적용하여 `log_prob` 를 얻음

```python
class RNNModel(nn.Module):
    def __init__(self, 
        rnn_type: str,
        vocab_size: int,
        embedding_size: int=200,
        hidden_size: int=200,
        num_hidden_layers: int=2,
        dropout: float=0.5
    ):
        super().__init__()
        self.rnn_type = rnn_type
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layer = num_hidden_layers
        assert rnn_type in {'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'}

        # 정수 형태의 id를 고유 벡터 형식으로 나타내기 위하여 학습 가능한 Embedding Layer를 사용합니다.
        self.embedding = nn.Embedding(vocab_size, embedding_size)

        # Dropout은 RNN 사용시 많이 쓰입니다.
        self.dropout = nn.Dropout(dropout)

        if rnn_type.startswith('RNN'):
            # Pytorch에서 제공하는 기본 RNN을 사용해 봅시다.
            nonlinearity = rnn_type.split('_')[-1].lower()
            self.rnn = nn.RNN(
                embedding_size, 
                hidden_size, 
                num_hidden_layers,
                batch_first=True, 
                nonlinearity=nonlinearity,
                dropout=dropout
            )
        else:
            # Pytorch의 LSTM과 GRU를 사용해 봅시다.
            self.rnn = getattr(nn, rnn_type)(
                embedding_size,
                hidden_size,
                num_hidden_layers,
                batch_first=True,
                dropout=dropout
            )

        # 최종적으로 나온 hidden state를 이용해 다음 토큰을 예측하는 출력층을 구성합시다.
        self.projection = nn.Linear(hidden_size, vocab_size)

    def forward(
        self, 
        input: torch.Tensor,
        prev_hidden: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]
    ):
        """ RNN 모델의 forward 함수 구현
        위의 그림과 __init__ 함수 내 주석을 참고하여 forward 함수를 구현하세요.

        Hint 1: RNN 모델에선 Dropout을 곳곳에 적용하는 것이 성능이 좋다고 알려져 있습니다.
                예를 들어, Embedding 이후와 Projection 전에도 적용할 수 있습니다.
        Hint 2: 최종 확률값을 구하기 위해서 Projection 이후에 F.log_softmax를 사용하면 됩니다.

        Arguments:
        input -- 토큰화 및 배치화된 문장들의 텐서
                    dtype: torch.long
                    shape: [batch_size, sequence_lentgh]
        prev_hidden -- 이전의 hidden state
                    dtype: torch.float
                    shape: RNN, GRU - [num_layers, batch_size, hidden_size]
                           LSTM - ([num_layers, batch_size, hidden_size], [num_layers, batch_size, hidden_size])

        Return:
        log_prob -- 다음 토큰을 예측한 확률에 log를 취한 값
                    dtype: torch.float
                    shape: [batch_size, sequence_length, vocab_size]
        next_hidden -- 이후의 hidden state
                    dtype: torch.float
                    shape: RNN, GRU - [num_layers, batch_size, hidden_size]
                           LSTM - ([num_layers, batch_size, hidden_size], [num_layers, batch_size, hidden_size])
        """
        ### YOUR CODE HERE
        ### ANSWER HERE ###
        emb = self.dropout(self.embedding(input))
        output, next_hidden = self.rnn(emb, prev_hidden)
        log_prob = self.projection(self.dropout(output)).log_softmax(dim=-1)

        ### END YOUR CODE
        
        assert list(log_prob.shape) == list(input.shape) + [self.vocab_size]
        assert prev_hidden.shape == next_hidden if self.rnn_type != 'LSTM' \
          else prev_hidden[0].shape == next_hidden[0].shape == next_hidden[1].shape
        
        return log_prob, next_hidden
    
    def init_hidden(self, batch_size: int):
        """ 첫 hidden state를 반환하는 함수 """
        weight = self.projection.weight
        
        if self.rnn_type == 'LSTM':
            return (weight.new_zeros(self.num_hidden_layer, batch_size, self.hidden_size),
                    weight.new_zeros(self.num_hidden_layer, batch_size, self.hidden_size))
        else:
            return weight.new_zeros(self.num_hidden_layer, batch_size, self.hidden_size)
    
    @property
    def device(self):   # 현재 모델의 device를 반환하는 프로퍼티
        return self.projection.weight.device
    
    
    
rnn_type = 'LSTM'      # 'LSTM', 'GRU', 'RNN_TANH', 'RNN_RELU'
vocab_size = len(corpus.dictionary)
model = RNNModel(rnn_type, vocab_size=vocab_size)
```







<br>

### 모델 학습

**배치화**

전체 말뭉치에 대해 RNN 계산을 하여 기울기(Gradient)를 역전파하는 것은 시간도 오래걸릴 뿐만이 아니라 병렬화가 불가능합니다. 따라서 이를 배치 크기 만큼 잘라 각각을 별개의 학습 샘플로 사용합니다.

현재 데이터셋은 한 줄로 길게 구성되어 있습니다.

```
[ a b c d e <eos> f g h i j k l m n <eos> o p q r s <eos> t u v w x y z <eos> ]
```

이를 batch_size = `4`로 나누면 다음과 같습니다.

```
[[ a b c d e <eos> ], 
 [ f g h i j k ],
 [ l m n <eos> o p ],
 [ r s <eos> t u v ]]
```

개수가 부족하여 배치를 못 채운 부분은 잘라버립니다.

**Backpropagation through Time (BPTT)**

배치화를 하였음에도 불구하고 하나의 샘플이 너무 길어 RNN 역전파를 하기에는 난점이 많습니다. 이를 해결하기 위하여 학습에 **BPTT**를 사용합니다. BPTT는 **한번에 sequence_length 만큼에 대해서만 역전파를 수행해서 전체 Sequence를 학습**시키는 방법입니다.

이를 위하여 한 각 샘플을 sequence_length 나누어 줍니다. 배치화된 데이터 셋을 sequence_length = `2`로 나누면 다음과 같습니다. 아래와 같은 데이터에서 모델은 [a, b]를 통과시키고 학습 한 뒤 [c, d]를 통과시키고 학습, [e, \<eos\>]를 통과시키고 학습...의 순서로 학습을 진행합니다. 

```
[[[ a b ], [ c d ], [ e <eos> ]],
 [[ f g ],  [ h i ], [ j k ]],
 [[ l m ], [ n <eos> ], [ o p ]], 
 [[ r s ], [ <eos> t ], [ u v ]]]
```

현재 shape는 (batch_size, num_sample, sequence_length) 입니다. **BPTT는 num_sample 부분을 순회하면서 기울기를 계산**합니다. 따라서 이를 `reshape`하여 **(num_sample, batch_size, sequence_length)**로 구성하면 편리합니다.

```
[[[ a b ], [ f g ], [ l m ], [ r s ]],
 [[ c d ], [ h, i ], [ n <eos> ], [ <eos> t ]],
 [[ e <eos> ], [ j k ], [ o p ], [ u v ]]]
```

이때 첫번째 샘플인 `[[ a b ], [ f g ], [ l m ], [ r s ]]`는 각 배치의 첫번째 sequence이고, 두번째 샘플인 `[[ c d ], [ h, i ], [ n <eos> ], [ <eos> t ]]`는 각 배치의 두번째 sequence, 그리고 마지막 샘플인 `[[ e <eos> ], [ j k ], [ o p ], [ u v ]]]`는 각 배치의 마지막 부분이라는 것을 알 수 있습니다. 

위 모양으로 보면, 각 배치에 대한 학습이 위에서 아래로 가며 이루어지는 것을 알 수 있습니다. 따라서 학습 코드 구현 시 `for batch in data`와 같이 작성할 수 있게 됩니다. 

<br>

여기서는 일렬로 구성된 data를 입력으로 받아 배치화된 형태로 반환하는 함수를 정의합니다. 

```python
def bptt_batchify(
    data: torch.Tensor,
    batch_size: int,
    sequence_length: int
):
    ''' BPTT 배치화 함수
    한 줄로 길게 구성된 데이터를 받아 BPTT를 위해 배치화합니다.
    batch_size * sequence_length의 배수에 맞지 않아 뒤에 남는 부분은 잘라버립니다.
    이 후 배수에 맞게 조절된 데이터로 BPTT 배치화를 진행합니다.

    Arguments:
    data -- 학습 데이터가 담긴 텐서
            dtype: torch.long
            shape: [data_lentgh]
    batch_size -- 배치 크기
    sequence_length -- 한 샘플의 길이

    Return:
    batches -- 배치화된 텐서
               dtype: torch.long
               shape: [num_sample, batch_size, sequence_length]

    '''
    ### YOUR CODE HERE
    ### ANSWER HERE ###
    length = data.numel() // (batch_size * sequence_length) * (batch_size * sequence_length)
    batches = data[:length].reshape(batch_size, -1, sequence_length).transpose(0, 1)

    ### END YOUR CODE

    return batches
```

**모델 학습**

RNN 계열 모델의 학습은 이전까지 다뤄왔던 image model의 학습 코드와는 다릅니다. 큰 과정은 아래와 같습니다. 

* **optimizer를 사용하지 않습니다**

* RNN model에 (batch, hidden)을 전달하여 (output, hidden)을 받습니다. 

  * BPTT 학습을 위해 매 batch마다 `hidden.detach()`를 호출합니다. 

* 마지막 예측을 제외한 output과 첫번째 단어를 제외한 batch 사이의 nll_loss를 계산합니다. 

  * https://pytorch.org/docs/stable/generated/torch.nn.functional.nll_loss.html?highlight=nll_loss#torch.nn.functional.nll_loss 

    ```python
    >>> # input is of size N x C = 3 x 5
    >>> input = torch.randn(3, 5, requires_grad=True)
    >>> # each element in target has to have 0 <= value < C
    >>> target = torch.tensor([1, 0, 4])
    >>> output = F.nll_loss(F.log_softmax(input), target)
    >>> output.backward()
    ```

* `model.zero_grad()`, `loss.backward()`를 호출합니다. 

* `clip_grad_norm_` 함수를 이용해 gradient를 기울기 폭주를 방지하고 가중치를 업데이트합니다. 

  * gradient를 조정하는 과정이 필요하기 때문에 optimizer를 사용하지 않는 것입니다. 

```python
import math
from tqdm.notebook import tqdm
from torch.nn.utils import clip_grad_norm_

def train(
    model: RNNModel,
    data: torch.Tensor, # Shape: (num_sample, batch_size, sequence_length)
    lr: float
):
    model.train()
    batch_size = data.shape[1]
    total_loss = 0.

    hidden = model.init_hidden(batch_size)
    # tqdm을 이용해 진행바를 만들어 봅시다.
    progress_bar = tqdm(data, desc="Train")
    for bid, batch in enumerate(progress_bar, start=1):
        batch = batch.to(model.device) # RNN Model에 정의했던 device 프로퍼티를 사용
        
        # 특이점: optimizer를 사용하지 않음!!
		
        '''
        train 시에는 hidden을 detach해주는 과정이 필요합니다. 
        이는 BPTT를 위한 것으로, 이전 batch 시 갱신뇐 가중치는 현재 batch의 gradient에 의해 갱신되지 않고 현재의 가중치만 갱신되도록 하기 위함입니다. 
        쉽게 말해, back propagation을 끊어주는 것입니다. 
        '''
        output, hidden = model(batch, hidden)
        if model.rnn_type == 'LSTM':
            hidden = tuple(tensor.detach() for tensor in hidden)
        else:
            hidden = hidden.detach()

        # 손실 함수는 Negative log likelihood로 계산합니다.
        '''
        위에서 본 예시로부터 아래 식을 설명해보겠습니다. 아래와 같은 data가 있습니다. 
        
        [[[ a b ], [ f g ], [ l m ], [ r s ]],
 		[[ c d ], [ h, i ], [ n <eos> ], [ <eos> t ]],
 		[[ e <eos> ], [ j k ], [ o p ], [ u v ]]]
 		
 		우리는 각 batch 단위로 for문을 돌리고 있으니 처음 batch는 [[ a b ], [ f g ], [ l m ], [ r s ]] 이겠죠. 
 		여기서 [a b] -> [b c], [f g] -> [g h], [l m] -> [m n], [r s] -> [s <eos>]를 예측하도록 학습되어야 합니다. 
 		
 		이 때 
 		output: [[ b c ], [ g h ], [ m n ], [ s <eos> ]] -> size: (batch_size, sequence_length, proj_size(vocab_size)) (output의 b, c 등의 원소는 각 vocab이 다음 단어가 될 점수값, 즉 vocab_size 크기의 벡터임)
 		batch: [[ a b ], [ f g ], [ l m ], [ r s ]] -> size: (batch_size, sequence_length)
 		와 같이 됩니다. 
 		
 		따라서 output의 마지막 출력과 batch의 처음 입력은 비교할 대상이 없기 때문에, 아래와 같이 인덱싱을 해줍니다. 
        '''
        loss = F.nll_loss(output[:, :-1, :].transpose(1, 2), batch[:, 1:])
        
        model.zero_grad()
        loss.backward()
        
        # backward된 gradient를 조정하는 과정이 필요하기 때문에 optimizer.step()을 사용하지 않습니다. 
        # clip_grad_norm_을 통해 기울기 폭주 (Gradient Exploding) 문제를 방지합니다.
        clip_grad_norm_(model.parameters(), 0.25)
        for param in model.parameters():
            param.data.add_(param.grad, alpha=-lr)
        
        total_loss += loss.item()
        current_loss = total_loss / bid

        # Perplexity는 계산된 Negative log likelihood의 Exponential 입니다.
        progress_bar.set_description(f"Train - loss {current_loss:5.2f} | ppl {math.exp(current_loss):8.2f} | lr {lr:02.2f}", refresh=False)
```







<br>

### 모델 평가

Evaluation 용 코드를 작성합니다. 

```python
@torch.no_grad()
def evaluate(
    model: RNNModel,
    data: torch.Tensor
):
    ''' 모델 평가 코드
    모델을 받아 해당 데이터에 대해 평가해 평균 Loss 값을 반환합니다.
    위의 Train 코드를 참고하여 작성해보세요.

    Arguments:
    model -- 평가할 RNN 모델
    data -- 평가용 데이터
            dtype: torch.long
            shape: [num_sample, batch_size, sequence_length]

    Return:
    loss -- 계산된 평균 Loss 값
    '''
    
    model.eval()

    ### YOUR CODE HERE
    ### ANSWER HERE ###
    total_loss = 0.
    hidden = model.init_hidden(data.shape[1])
    
    for batch in data:
        batch = batch.to(model.device)
		
        # eval 시에는 가중치 갱신이 일어나지 않기 때문에 hidden.detach()가 필요하지 않습니다. 
        output, hidden = model(batch, hidden)
        total_loss += F.nll_loss(output[:, :-1, :].transpose(1, 2), batch[:, 1:]).item()
    
    loss = total_loss / len(data)
    ### END YOUR CODE

    return loss
```

<br>

### 문장 생성

```python
from tqdm.notebook import trange

num_words = 1000
temperature = 1.0

hidden = model.init_hidden(1)
input = torch.randint(vocab_size, (1, 1), dtype=torch.long).to(device)
outputs = []

for i in trange(num_words, desc="Generation"):
    with torch.no_grad():
        log_prob, hidden = model(input, hidden)

    weights = (log_prob.squeeze() / temperature).exp()
    token_id = torch.multinomial(weights, 1)
    outputs.append(token_id.item())
    input = token_id.unsqueeze(0)

outputs = [corpus.dictionary.id2token[token_id] for token_id in outputs]

with open('generate.txt', 'w') as fd:
    fd.write(' '.join(outputs).replace('<eos>', '\n'))
    
'''
de Molina Minister , shows as part of the production of universe Catherine <unk> by one of the original case staff primaries in Jacob to the site that Innis 's ceiling .
Attached of the Australian organization was dressed formed by become currants of science on the fans , China . Russian initially felt bills feet of knowledge 's .....
'''
```

**Q.** 위 코드에서 `temperature`의 역할은?

temperature는 문장 생성에서 다양성을 조정한다. 

temperature가 1보다 높아지게 되면 최종 확률 값이 점차 평탄해지고 무한대로 가면 균일 분포가 된다. 이는 원래라면 확률이 낮은 토큰이 좀 더 잘 뽑히게 된다는 뜻이고, 이는 다양한 문장을 생성하는데 도움을 준다. 그러나 이렇게 생성된 문장은 문법적으로 불안하거나 어색할 확률 역시 높아지게 된다. 

반대로 temperature가 1보다 낮으면 원래 뽑힐 확률이 높은 토큰이 더 잘 뽑히게 되며, 이는 다양성이 낮아지는 결과로 귀결된다. 그러나 이렇게 생성된 문장이 좀 더 안정적이다.

















<br>

<br>

# 참고 자료

* 
