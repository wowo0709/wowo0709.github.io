---
layout: single
title: "[AITech][NLP] 20220317 - Part 4) Transformer"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 KAIST '주재걸' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# Transformer

이번 강의는 `Transformer`에 대한 내용입니다. 

## Transformer

`Transformer`는 2017년 _Attention is all you need_ 라는 이름의 논문에 나온 자연어처리 모델입니다. 

이전까지는 RNN 구조의 모델에 attention mechanism을 활용하는 형태로 쓰였다면, Transformer 모델은 RNN 구조를 모두 걷어내고 Attention 구조만을 사용한 모델입니다. 

### Recall: RNN

지난 강의까지 열심히 공부한 RNN 구조를 상기해봅시다. Attention 메커니즘을 사용함으로써 기존 RNN의 **마지막 hidden state에만 의존하게 된다**라는 문제를 해결할 수 있었습니다(기울기 문제 등 다른 여러 문제들도 해결했었죠). 

하지만 여전히, RNN 구조를 사용할 경우 **현재 time step의 hidden state를 생성할 때는 이후 time step의 정보들을 포함하지 못하고, 이전 time step의 정보들 또한 점점 유실되는 문제**가 있습니다. 이는 RNN은 문장의 왼쪽(처음)에서 오른쪽(끝)으로 가며 hidden state를 생성하기 때문이죠. 

![image-20220318221327064](https://user-images.githubusercontent.com/70505378/159112384-8b00e453-8f06-4f47-ba9a-73c915a6235d.png)

이를 해결하기 위해 `Bi-Directional RNN` 모델이라는 것을 사용하는 시도가 있었습니다. 이 모델의 목적은 **특정 time step의 hidden state를 생성할 때 주변 단어들을 모두 고려하고 싶다**는 것입니다. 

아래 그림과 같이 'go'라는 단어의 hidden state를 생성할 경우, Forward RNN을 통해 앞 단어들의 정보를 포함한 hidden state와 Backward RNN을 통해 뒷 단어들의 정보를 포함한 hidden state를 concat하여 최종적으로 'go'의 hidden state를 생성합니다. 

![image-20220318231051143](https://user-images.githubusercontent.com/70505378/159112386-2428ecc5-c32e-4f36-81db-a25c11c09997.png)

<br>

`Transformer`는 Bi-Directional RNN과 다르게 **attention module**을 사용하여 주변 단어들에 대한 정보를 모두 포함하는 hidden state를 생성할 수 있습니다. 이제부터는 Transformer가 무엇인지, attention module의 구조와 작동 방식에 대해 알아보도록 하겠습니다. 

### What is Transformer?

간단하게 **Transformer**가 무엇인지에 대한 얘기부터 해봅시다. 

Transformer는 본래 자연어 번역을 위한 모델로, **RNN 구조 없이 Attention이라는 모듈**을 도입해서 매우 성공적인 performance를 보인 모델입니다. 

Transformer의 구조는 아래와 같습니다. 

![image-20220209144344432](https://user-images.githubusercontent.com/70505378/153200627-2499de50-375d-4fac-a5b3-5f80b429cb3b.png)

구조를 보면 아래와 같은 정보들을 발견할 수 있습니다. 

* Transformer는 기계 번역 task를 수행합니다. 
* Transformer는 Encoder 부와 Decoder 부로 나눠져 있습니다. 
* Encoder와 Decoder 부는 각각 6개의 stacked 구조로 되어 있습니다. 

그럼 이제 Transformer가 무엇을 하는 녀석이고, 어떤 구조로 되어 있는지 봤으니, 각각의 부분들을 하나씩 뜯어봅시다. 

<br>

### Encoder

Encoder의 구조 내부는 아래와 같이 생겼습니다. 

![image-20220209144737148](https://user-images.githubusercontent.com/70505378/153200631-6800a280-6cf4-40c3-b46d-dff01aba50be.png)

그러면 이제 저 **Self-Attention** 모듈이 무엇이고, 어떻게 RNN을 대체했는지에 대해 봐야겠죠?

#### **Attention**

**Attention이 무엇인가?**

우선 rough하게 말하면, Attention은 N개의 단어로 이루어진 문장을 처리할 때 **재귀적으로 N번 처리하는 것이 아니라, 한 번에 N개의 단어를 모두 이용**합니다. 다시 말하면, 1개의 단어를 처리하기 위해 다른 N-1개의 단어에 대한 정보를 동시에 활용하는 것입니다. 

이렇게 함으로써 얻을 수 있는 이점은 무엇일까요? 

첫째, **학습 시간을 줄일 수 있습니다.** N번의 과정을 기다려야 했던 RNN에 비해, 이를 한 번에 처리하는 Transformer는 훨씬 빠른 모습을 보입니다. 

둘째, **다른 단어들에 대한 정보를 더 잘 얻을 수 있습니다.** RNN 모델에서는 hidden state 생성 시 이전 time step까지의 단어들의 정보 만을 포함할 수 있었고, Bi-directional RNN 역시 time step이 진행되면서 예전 단어들의 정보는 소실되는 문제가 여전히 있었습니다. 반면 Attention 구조에서는 **동시에 다른 단어들에 대한 정보를 모두 이용**하기 때문에 그런 문제가 없습니다. 즉, Transformer는 RNN 구조의 고질적인 문제였던 **Long-Term Dependency** 문제를 깔끔히 해결했습니다. 

다만, 당연하게도 많은 정보를 한 번에 이용하는 Attention 구조는 많은 Computational resource를 요구하긴 합니다. 이것이 Transformer의 한계로 지적되기도 하죠. 

**Attention의 동작 구조**

어찌되었든, Attention은 N개의 단어를 함께 고려하며 각 단어에 대해 처리를 합니다. 아래처럼 말이죠. 

![image-20220209145736590](https://user-images.githubusercontent.com/70505378/153200632-042c5e22-9608-4103-b857-5d83a3583665.png)



위 그림을 보면 각 단어에 해당하는 x벡터가 Attention의 입력으로 들어가, 출력으로 z벡터들이 나오는데, 그 과정을 한 번 살펴봅시다. 

Attention은 아래의 과정을 수행합니다. 

> 1. _`x`의 Input feature를 embedding하여 Embedded vector로 변환_
> 2. _각각의 embedded vector에 대해 **Query/Key/Value vector** 를 생성_
>    * Query: 각 key vector의 정보를 얼마나 가져올 것인가?
>    * Key: 유사도를 구하기 위해 사용. 각 입력 벡터의 정보를 표현하는 벡터. 
>    * Value: hidden state를 구하기 위해 사용. Query와 Key로부터 구한 유사도를 곱해 hidden state를 생성하게 되는 벡터. 
>    * Query vector와 Key vector의 차원 수는 같아야 함(d<sub>k</sub>). Value vector의 차원 수는 독립적(d<sub>v</sub>). 
> 3. _각 단어에 대해 자신을 포함한 모든 단어들과 Query vector와 Key vector를 내적_ 
>    * 이 값을 Attention Score라고 함
>    * 이 Attention Score가 해당 단어와 다른 단어들과의 관계성(유사도)을 나타내는 지표라고 할 수 있음
> 4. _각각의 score를 root(d<sub>k</sub>)로 나누고 Softmax를 적용_
>    * 이 때의 d<sub>k</sub>는 dimension of key vector(or query vector)이며, 내적 값의 분산을 줄여주는 역할을 함
>    * 이 값을 Attention weight라고 함
> 5. _나눈 값에 각각에 해당하는 단어의 Value vector를 스칼라 곱하고 모두 더함 (Weighted Sum)_
>    * 이 값을 `z`라고 함

![image-20220318230227839](https://user-images.githubusercontent.com/70505378/159112385-602cf455-1da5-4103-b613-9c529c384b34.png)

✋ **잠깐! 유사도를 다른 방식으로 구할 수 없을까?**

Attention 메커니즘 초기에는 단어들 간의 유사도를 구할 때 단순히 입력 벡터들을 내적하여 구했다고 합니다. 하지만 이럴 경우, 자기자신 벡터와 내적한 경우(예를 들어 x1\*x1의 경우) 그 값이 다른 벡터와 내적한 경우보다 더 커지게 되고, 따라서 hidden state 생성 시 자기자신 단어에 편향된 정보가 포함되게 됩니다. 

✋ **잠깐! 왜 root(d<sub>k</sub>)로 나눠줘야 할까?**

Query vector와 Key vector의 내적 연산 시 차원 수가 커지면 term의 수도 많아집니다. 예를 들어, 차원 수가 2일 때 내적 연산의 결과는 `q1k1 + q2k2`와 같이 구할 수 있는데, 차원 수가 100일 경우 `q1k1 + q2k2 + ... + q100k100`과 같이 그 식이 전개됩니다. 

이때 문제는 **분산**입니다. 서로 다른 수들을 더할 경우 더한 수의 분산은 더해진 수들의 분산의 합으로 계산됩니다. 따라서 위 예시에서 qiki의 분산이 모두 1이라 가정했을 때, 차원 수가 2인 경우 그 분산은 2가 되고 차원 수가 100인 경우 분산은 100이 됩니다. 

그런데 분산이 커지게 되면 query vector와 각 key vector를 내적한 스칼라 값의 차이가 매우 크게 날 수 있고, 이는 softmax를 거치면서 큰 값에 대부분의 weight 값이 쏠리게 되는 결과로 이어집니다. 예를 들어 차원이 2인 경우 attention score가 (1.1, -0.8, -1.7)과 같이 나온다면, 차원이 100인 경우의 attention score는 (8, -11, 3)과 같이 나와서 softmax를 거치며 8에 대부분의 weight가 쏠려서 계산되어 다른 단어들에 대한 정보를 제대로 가져오지 못 하게 되는 것입니다. 

여기서 알아야 할 것이, 어떤 수를 스칼라 값 n으로 나누게 된다면 나눠진 수의 분산은 '기존 분산/n<sup>2</sup>'으로 계산된다는 것입니다. 

따라서, query vector와 key vector의 내적값(분산 d<sub>k</sub>)에 root(d<sub>k</sub>)를 나눠주면 그 **분산을 일정하게 1로 유지**할 수 있고, 그 결과로 **각 단어에 대한 정보를 골고루 가져올 수 있도록** 하는 것입니다. 

실제로 **<span style=color:red>softmax 연산을 적용할 값의 분산을 일정하게 유지시켜주는 것</span>**은 아주 중요한 일입니다. 그렇지 않다면 모델의 학습이 아예 이뤄지지 않는 상황을 겪게 될 수도 있습니다. 

<br>

아래는 2개의 단어로 이루어진 문장(Thinking Machines)에서 'Thinking'이라는 단어로부터 `z` 값을 도출하는 과정을 그림으로 표현한 것입니다. 

![image-20220209151913555](https://user-images.githubusercontent.com/70505378/153200633-7a18bc5c-1431-429f-899b-80b5a5866ac5.png)

위와 같은 과정으로 각각의 x에 대해 z를 구하는데요, 이로 인해 RNN에서 N번의 과정을 거쳐야 했던 변환은 Attention에서는 **단순한 행렬곱**으로 대체될 수 있습니다. 

바로 아래와 같이, 단어들의 sequence를 행렬로 나타내면 각각 W<sup>Q</sup>, W<sup>K</sup>, W<sup>V</sup> 행렬과 곱해서 바로 Q, K, V 행렬을 구할 수 있습니다. 

![image-20220209152225437](https://user-images.githubusercontent.com/70505378/153200634-b0859c5b-469e-4480-b7e6-d5f73316a037.png)

그리고 위에서 구한 Q, K, V vector를 이용하면 아래와 같은 간단한 수식으로 `x`에서 `z`를 구할 수 있습니다. 

![image-20220209152816894](https://user-images.githubusercontent.com/70505378/153200640-55e73f43-53e1-45b2-a78f-312898ea3124.png)

이런 과정을 통해, Attention은 자연스럽게 다른 단어들과의 관계성을 학습하게 되죠. 

![image-20220209152449829](https://user-images.githubusercontent.com/70505378/153200636-833293b0-4dbf-47cf-9c6c-ba94578d4bb9.png)

#### MHA (Multi-Headed Attention)

그런데 실제로는, Transformer는 각 Encoder(또는 Decoder)마다 8개의 Attention을 병렬적으로 함께 사용합니다. 따라서 다음과 같이 8개의 `z` 벡터가 하나의 Encoder 내에서 생성됩니다. 

![image-20220209152713155](https://user-images.githubusercontent.com/70505378/153200638-67f4c9bc-7bcb-468d-838a-5380c1964d6b.png)



이렇게 하는 이유는, 바로 아래와 같이 **여러 관점에서 다른 단어들과의 관계를 구하기 위함**입니다. 하나의 Attention만을 사용한다면, 그 Attention이 학습한 정보밖에는 활용하지 못 하는데 비해, 여러 개 Attention을 사용하면 여러 개의 관점으로 학습한 정보들을 모두 사용할 수 있다는 것이죠. 

아래 그림에서 `it_`이라는 토큰은 여러 입력 토큰으로부터 다양한 정보를 가져와 학습되는 것을 볼 수 있는데요, 입력 토큰 안에 있는 각 grid(격자 무늬)가 각 attention head 관점에서 해당 단어의 중요도(얼마나 참조했는지)를 나타냅니다. 

![image-20220209153259136](https://user-images.githubusercontent.com/70505378/153200642-526a1400-dcc5-4d5e-84f7-5aed81b08d1a.png)



#### 출력 형태 맞춰주기

위와 같은 과정으로 MHA를 통과하고 나면, 8개의 `Z` 행렬이 생성됩니다. 이제는 이 `Z` 행렬을 처음 입력 행렬이었던 `X`와 같은 형태로 맞춰줘야 합니다. 왜냐하면 Encoder 부에는 여러 개의 stacked encoder들이 있기 때문에, **이번 encoder의 출력은 다음 encoder로의 입력이 되어야 하기 때문입니다**(또는 뒤에서 살펴볼 residual connection에서 입력 X와 더해져야 하기 때문이라고 생각할 수도 있습니다).

이 과정은 단순히 행렬 `Wo`와 행렬곱함으로써 수행할 수 있습니다. 

![image-20220209154130564](https://user-images.githubusercontent.com/70505378/153200649-2b04cb46-dfa0-4722-902d-407d926315d7.png)

그래서 Self-Attention 모듈을 통과하는 과정은 아래와 같습니다.

![image-20220209154401681](https://user-images.githubusercontent.com/70505378/159019796-4fc6ab16-3b7c-413a-ae33-05ee1b60c94b.png)

최종적으로 Self-Attention의 구조는 아래와 같이 나타낼 수 있습니다. (**Encoder에서는 Self-Attention 모듈 이후에 FC-layer(FFNN)를 지나야 함을 잊지 마세요!**)

![image-20220319121559376](https://user-images.githubusercontent.com/70505378/159112387-b26f4dec-de18-4d4a-9bcc-cebc1cc0035c.png)

#### Computational amount of Self-Attention

`Self-Attention` 모듈의 계산량을 Recurrent 모듈과 비교해보겠습니다. 

* **Complexity per Layer**
  * Self-Attention: (n, d) 모양의 Q와 (d, n) 모양의 K<sup>T</sup>를 행렬곱 할 때의 계산복잡도는 O(n<sup>2</sup>d)입니다. 
  * (d, d) 모양의 W<sub>hh</sub>와 (d, 1) 모양의 h를 총 time step인 n번 반복해 계산할 때의 계산복잡도는 O(nd<sup>2</sup>)입니다. 
  * 일반적으로 연산 시 메모리 요구량은 Self-Attention이 더 많이 요구합니다. 
* **Sequential Operations**
  * Self-Attention: 행렬 연산 한 번만 필요하므로 O(1)입니다. 
  * Recurrent: n번의 연산이 순차적으로 이루어져야 하기 때문에 O(n)입니다. 
  * GPU 코어 수에 따라 병렬화가 가능한 경우, Self-Attention은 계산 작업을 병렬화하여 더 빠른 속도를 낼 수 있습니다. 
* **Maximum Path Length**
  * Self-Attention: 어떤 단어든 해당 단어의 벡터를 바로 참조할 수 있기 때문에 O(1)입니다. 
  * Recurrent: 첫 단어와 마지막 단어 사이 n번의 time step을 거쳐야 참조가 되기 때문에 O(n)입니다. 
  * 어떤 단어든 바로 직접적인 참조가 가능하다는 점에서, Self-Attention 구조는 long-term dependency 문제를 근본적으로 해결했다고 할 수 있습니다. 

![image-20220319124628302](https://user-images.githubusercontent.com/70505378/159112370-4a1dd82c-d3e5-4005-8784-44543107b6b8.png)



#### Positional encoding & Residual connection & Layer Normalization

여기까지의 과정이 이해되셨나요? 여기에 3가지만 더 추가해봅시다. 그 세 가지는 **Positional encodding**,  **Residual connection**, 그리고 **Layer normalization**입니다. 

**Positional Encodding**

Sequential data를 다루는 모든 모델에서는 **data들의 순서**가 매우 중요합니다. 그런데 위의 과정만으로는, 그 단어들의 순서를 제대로 고려해주지 못합니다. 문장을 이루는 단어들이 같아도, 순서가 다르다면 다른 출력이 나올 수 있어야 하는데, 그러지 못하는 것이죠. 

바로 이 단어들의 순서를 고려해주기 위한 것이 positional encodding입니다. 그리고 이는 단순히 Embedded vector에 Positional encodding을 위한 행렬을 더해줌으로써 수행할 수 있습니다. 

![image-20220209155752981](https://user-images.githubusercontent.com/70505378/153200653-b29bbf5c-9433-4cd7-af0b-f56f3348ee29.png)

논문에서는 이 positional encodding에 해당하는 값들을 sin, cos 값을 이용하여 생성하고, 이를 사용하는 이유는 그 값이 -1~1까지의 범위를 가지며 frequency와 offset에 따라 unique한 값을 만들어낼 수 있는 연속함수이기 때문이라고 합니다. 

![image-20220319132336154](https://user-images.githubusercontent.com/70505378/159112373-9a9451de-4499-4c6e-86da-8e46a02f80f2.png)

![image-20220319132405253](https://user-images.githubusercontent.com/70505378/159112374-8904d277-9871-4a7d-9e56-fe447d19ed50.png)

**Residual Connection**

두번째로, Residual connection입니다. Transformer의 학습 과정에서 backpropagation이 수행되다 보면, 위에서 본 positional encodding에 대한 정보가 손실되기 쉽습니다. 바로 이 정보를 견고히 유지하기 위해서 아래와 같이 Residual connection이 존재합니다. 

(진짜 진짜 마지막으로, residual connection 다음에는 layer normalization을 적용해서 학습 효과를 증진시킵니다 😊)

![image-20220209160000883](https://user-images.githubusercontent.com/70505378/153200655-28d11568-334e-4444-bbf2-313f3360312d.png)



**Layer Normalization**

학습 효과를 더욱 증진시키기 위해서, MHA와 FFNN을 각각 거친 행렬들은 layer normalization이라는 연산을 거치게 됩니다. 

![image-20220209170248715](https://user-images.githubusercontent.com/70505378/153200660-1fe0dab1-144d-477a-aee1-93b65388fd75.png)

Batch norm, Layer norm, Instance norm, Group norm 등의 normalization 연산들은, 공통적으로 우리가 원하는 데이터들의 평균:0 분산:1 로 만들고(x-mean/sigma) 우리가 원하는 평균과 분산을 주입할 수 있도록 하는 선형변환(sigma*x+mean)으로 이루어집니다. 

![image-20220319130749044](https://user-images.githubusercontent.com/70505378/159112371-2fb97dad-d193-4612-b603-44703c1a8231.png)

 Self-Attention의 layer normalization 과정은 아래와 같이 이루어집니다. 

* 각 단어의 hidden state vector 단위로 평균은 0, 분산은 1을 갖도록 합니다. 
* 각 node 단위로 동일한 affine transform을 적용하여 동일한 평균과 분산을 갖도록 합니다. 

![image-20220319131021699](https://user-images.githubusercontent.com/70505378/159112372-9cc98a63-f468-4eb1-8d2e-ea59be57d948.png)



<br>

최종 인코더의 입출력 형태는 아래와 같습니다. 

![image-20220209160018254](https://user-images.githubusercontent.com/70505378/153200658-bd799b2d-6ef2-46c2-956b-e8118b865dba.png)

<br>

### Encoder -> Decoder

다음으로 여기서는 Encoder 부를 모두 통과한 정보들이 Decoder 부에 어떻게 전달되는 지에 대해 알아보겠습니다. 여기까지 따라오셨다면 거의 다 왔습니다!!!

우리가 2개의 인코더-디코더를 사용한다고 했을 때 최종 Encoder 부의 출력은 아래 그림과 같이 Decoder 부에 전달되는데요, 과연 **어떤 정보들이 전달되는 것일까요?**

![image-20220209170756400](https://user-images.githubusercontent.com/70505378/153200662-291d94e1-a454-4647-a0cd-6b8a287f9a6d.png)

#### Encoder에서 K, V를 Decoder에 전달한다

Encoder 부에서 입력 `x`가 일련의 encoder들을 모두 지나고 나면, 위에서 본 것처럼 input x와 형상이 같은 output `z`가 출력될 것입니다. 이 최종 출력 `z`를 이용해 **K와 V matrices**를 생성하고, 이 두 행렬을 **각각의 Decoder**에게 전달합니다. 

![](http://jalammar.github.io/images/t/transformer_decoding_1.gif)

그리고 이렇게 전달된 K와 V matrix는 각각의 Decoder 내의 **Encoder-Decoder Attention** 모듈에서 사용됩니다. 이 K와 V는 **decoder가 input sequence에서 어떤 부분에 집중해야 할 지**를 알려주며, output을 생성하는 데 사용됩니다. 



<br>

### Decoder

자, 이제 Encoder에서 어떻게 sequential data를 처리하고, Decoder에게 어떤 정보를 어떻게 전달해주는 지까지 봤습니다. 정말 다 왔습니다!

여기서는 이제 마지막으로 Decoder 부에서 어떤 과정을 통해 Transformer의 output을 만들어내는 지 살펴볼 것입니다. 

#### 각 step의 output은 다음 step의 output을 만들기 위해 사용된다

Decoder 부는 **Encoder로부터 전달받은 `K`, `V`**와 **input sequence로부터 자체적으로 생성한 `Q`**를 이용하여 한 step마다 하나의 output을 만들어냅니다. 이렇게 만들어진 **이전 step까지의 output들은 다음 step의 output을 만들기 위한 정보로 사용**됩니다. (여기서 이전 step까지의 정보들도 마찬가지로 embedding과 positional encoding이 적용됩니다)

그리고 이는 다른 말로 하면, **현재 step 후의 정보들은 현재 step의 output에 영향을 주지 않는다**는 것을 말합니다. 이는 모든 정보를 함께 사용하는 Encoder 부와 다른 점입니다. 

현재 step 후의 정보들을 사용하지 않기 위해 decoder의 self-attention 모듈에서는 미래의 정보들을 masking(setting them to '-inf')합니다. 그래서 decoder의 첫번째 self-attention 모듈은 **masked-attention** 모듈이라고 불립니다. 



![](http://jalammar.github.io/images/t/transformer_decoding_2.gif)

Decoder의 Encoder-Decoder Attention 모듈은 인코더에서 살펴본 MHA(Multi-headed attention)와 동일하게 동작하며, 다른 점은 사용하는 `K`, `V`는 Encoder로부터 전달받은 값을 사용하고 `Q` 또한 직접 생성해내는 것이 아니라 아래 layer에서 생성된 값을 사용한다는 점입니다. 

#### 최종 Transformer 출력 생성

일련의 stacked decoder들을 지나 Decoder부의 최종 출력은 **vector of floats** 입니다. 이것을 어떻게 단어들로 변환할 수 있을까요?

그것이 바로 최종적으로 사용되는 **Linear layer & Softmax layer**의 역할입니다. 

**Linear Layer**는 Decoder 부의 최종 output vector에 fully connect 연산을 적용하여 **logits vector**라는 것을 생성합니다. 그리고 이 때의 **logits vector의 크기는 Transformer 모델이 알고 있는 단어의 수(output이 될 수 있는 단어 후보의 수)**와 같습니다. 

그리고 이 logits vector를 **Softmax Layer**를 거쳐 각각의 값을 확률 값으로 바꾸고, **가장 확률이 높은 값에 해당하는 단어를 이번 step의 output**으로 선택하는 것이죠. 

![image-20220209174758005](https://user-images.githubusercontent.com/70505378/153200664-43fa5d21-3825-4d82-91d6-91b0d8004964.png)



#### Label Smoothing & Warm-up Learning Rate Scheduler

이제 Transformer의 동작 방식에 대해 모두 살펴봤습니다. 여기까지 오신 분들 축하드립니다 👏👏

여기서는 최종적으로 Transformer 모델 학습 단계에서 사용되는 두 가지 기법에 대해 살펴보도록 하겠습니다. 

**Label smoothing**

Transformer는 최종 단계에 label smoothing이라는 것을 사용해 모델의 일반화 성능을 한층 더 증가시킵니다. 

이 Label smoothing에는 여러 기법들이 있는데요, Transformer에서는 Softmax layer의 출력으로 나온 probability vector에서 가장 높은 확률을 가진 인덱스의 값만 1로 만드는 원-핫 방식을 사용하는 것이 아니라, **각각의 확률 값을 직접 이용**하는 식으로 이를 사용합니다. 

이게 무슨 말이냐 하면, 예를 들어 'Thank you'라는 단어를 한국어로 번역한다고 해봅시다. 이 단어는 한국어로 '고맙습니다' 또는 '감사합니다' 모두로 번역될 수 있습니다. 그런데 **정답에 해당하는 '고맙습니다'에 해당하는 값만 1이라면, 모델이 그 값을 '감사합니다'로 예측하든 '짜증납니다'로 예측하든 모두 그냥 틀린 것이 되버린다는 것입니다.**

이 때문에 원-핫 방식을 사용하지 않고 대신에 label smoothing을 적용하여 각각의 확률값에 유사하게 예측을 하도록 유도함으로써 모델의 일반화 성능을 한층 더 높일 수 있습니다. 이러한 label smoothing 기법은 데이터가 noisy한 경우, 즉 같은 입력 값에 다른 출력 값이 나오는 데이터들이 많을수록 크게 도움이 된다고 합니다. 자세한 내용은 더 찾아보시면 좋을 것 같습니다. 

![image-20220319160320857](https://user-images.githubusercontent.com/70505378/159112376-8df6212e-6ea8-46e1-81c8-2cc3d907bd61.png)

**Warm-up learning rate scheduler**

Learning rate scheduler란 학습율을 모델이 학습하는 동안 일정하게 유지하는 것이 아니라 동적으로 변화시키는 것으로, 요즘 모델 학습 시에는 거의 필수적으로 사용되고 있습니다. 

Transformer의 경우, 어느 정도까지는 learning rate를 선형적으로 증가시키다가 이후에는 점점 낮추는 방식을 사용합니다. 이는 초반에는 큰 학습율로 minima에 빠르게 다가가면서도 local minima에 빠지지 않도록 도와주고, 이후로는 학습율을 낮추면서 세밀한 탐색을 진행하는 것으로 생각할 수 있습니다. 

이러한 warm-up scheduling 방식이 Transformer 모델 학습에 있어서 가장 좋은 성능을 내도록 도와준다는 것이 실험적으로 입증되었습니다. 하지만, 당연히 warm-up scheduling 방식이 모든 모델에 대해 가장 좋은 성능을 내도록 하는 것은 아닙니다. 

![image-20220319154801302](https://user-images.githubusercontent.com/70505378/159112375-93d37e06-0483-4855-ae2f-e6828a318ebe.png)

<br>

<br>

이제 정말 끝입니다! Transformer를 이해하는 것은 쉽지 않은 과정이지만, 워낙 많이 사용되고 떠오르고 있는 기술이기 때문에 이해해두면 아주 도움이 될 것이라고 생각합니다 🤗🤗

**Transformer**

![image-20220319162249026](https://user-images.githubusercontent.com/70505378/159112377-8ed75a52-b381-46b1-9813-74dc1493b66b.png)

Transformer 활용의 예로는 Encoder 부분만 사용해서 이미지를 분류해내는 **Vision Transformer(ViT)**, 텍스트의 내용에 해당하는 이미지를 생성해내는 **DALL-E** 등이 있습니다. 

**Vision Transformer(ViT)**

![image-20220319162418534](https://user-images.githubusercontent.com/70505378/159112380-02ca1c7f-d7f3-437b-984e-432cf11f361c.png)

**DALL-E**

![image-20220319162506278](https://user-images.githubusercontent.com/70505378/159112381-f1b013be-1fee-482f-87d4-775bb76a2622.png)









<br>

## 실습) 번역 모델 전처리

본 강의의 실습에서는 영어-한글 번역 모델을 학습하기 위해 영어-한글 번역 데이터셋을 전처리 하는 방법에 대해 다룹니다. 본 실습에서는 번역 모델의 입/출력을 만들기 위해 자주 사용되는 여러가지 자연어 전처리 기술을 익힙니다. 번역 모델은 번역하고자 하는 문장(Source)을 입력으로 받고, 번역 결과(Target)을 출력합니다. 

### Preprocess

NLP 모델에 자연어 정보를 전달하기 위해서는 적절한 형태로의 전처리가 필요합니다. 주어진 데이터셋은 Source, Target 각각 하나의 문장으로 이루어져 있고, 이 때, 한->영 번역의 경우 source는 한국어 문장, target은 영어 문장에 해당합니다. 모델에 해당 정보를 전달하기 위해서는 하나의 문장을 여러 단어로 분리하고 각각의 단어를 토큰의 id로 바꿔줄 수 있는 token2id 사전이 필요합니다.

![image-20220319220246638](https://user-images.githubusercontent.com/70505378/159122511-55fbaaa2-e338-4a33-9fbf-9ca9de9f27ab.png)

해당 과정은 가장 간단한 수준의 토큰화이며 거의 모든 자연어 전처리 과정에서 사용됩니다. 

```python
def preprocess(
    raw_src_sentence: List[str],
    raw_tgt_sentence: List[str],
    src_token2id: Dict[str, int],
    tgt_token2id: Dict[str, int],
    max_len: int
) -> Tuple[List[int], List[int]]:
    """ 번역을 위한 문장 전처리기

    전처리 규칙:
    1. 각 문장은 token2id를 활용하여 고유의 번호를 가진 토큰 id의 수열로 바뀌어야합니다.
    2. token2id에 맞는 토큰이 없을 경우 <UNK> 토큰으로 처리합니다.
    3. Source 문장은 src_token2id로, Target 문장은 tgt_token2id를 사용해야합니다.
    4. Target 문장의 처음에 <SOS> 토큰을 넣으세요.
    5. Target 문장의 끝에 <EOS> 토큰을 넣으세요.
    6. 전처리된 문장의 총 토큰 개수는 max_len을 초과하면 안됩니다. 만약 초과하면 뒤를 잘라냅니다.

    Arguments:
    raw_src_sentence -- Source 문장 단어 리스트
    raw_tgt_sentence -- Target 문장 단어 리스트 
    src_token2id -- Source 언어 토큰들을 id로 매핑하는 딕셔너리
    tgt_token2id -- Target 언어 토큰들을 id로 매핑하는 딕셔너리
    max_len -- 변환된 토큰 리스트의 최대 길이

    Return:
    src_sentence -- 전처리된 Source 문장
    tgt_sentence -- 전처리된 Target 문장

    """
    # Special tokens
    unk_token_id = Language.UNK_TOKEN_ID
    sos_token_id = Language.SOS_TOKEN_ID
    eos_token_id = Language.EOS_TOKEN_ID

    src_sentence = list(map(lambda word: src_token2id.get(word, unk_token_id), raw_src_sentence[:max_len]))
    tgt_sentence = [sos_token_id] + list(map(lambda word: tgt_token2id.get(word, unk_token_id), raw_tgt_sentence[:max(max_len-2, 0)])) + [eos_token_id]

    return src_sentence, tgt_sentence
```

<br>

### Collating

- 문장들을 빠르게 처리하기 위해서는 병렬화가 필요하나, 문장들의 길이가 다 다르기 때문에 이를 배치화 하는 것은 쉽지 않습니다.
- 다양한 길이의 문장을 배치화하기 위하여 한 배치 내의 최대 길이 문장을 기준으로 문장에 패딩을 넣는 과정이 필요합니다.
- 패드을 넣기 위하여 PAD 라는 사전에 정의한 패드 토큰을 사용합니다.

![image-20220319220447427](https://user-images.githubusercontent.com/70505378/159122512-5588bfd4-540d-4409-90fd-22c11ef6f013.png)

```python
import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(
    batched_samples: List[Tuple[List[int], List[int]]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Collate 함수
    Source/Target 문장이 주어졌을 때 적절히 PAD 토큰을 넣어 배치화하는 함수를 작성하세요

    Note 1: collate_fn을 작성하면 torch.utils.data.dataloader.DataLoade의 collate_fn 인자를 통해 사용이 가능합니다.
            만약 관심이 있다면 뒤의 테스트 코드를 확인해보세요.

    Hint: torch.nn.utils.rnn.pad_sequence 함수를 사용하면 쉽게 구현 가능합니다.

    Arguments:
    batched_samples -- 배치화할 (source 문장, target 문장) 짝 리스트

    Return:
    src_sentences -- 배치화된 Source 문장들
                        shape: (batch_size, sentence_length)
    tgt_sentences -- 배치화된 Target 문장들
                        shape: (batch_size, entence_length)

    """
    # Pad token
    PAD = Language.PAD_TOKEN_ID

    ### YOUR CODE HERE
    ### ANSWER HERE ###
    src_sentence_list, trg_sentence_list = zip(*batched_samples)
    src_sentences = pad_sequence([
        torch.Tensor(sentence).to(torch.long) for sentence in src_sentence_list
    ], batch_first=True, padding_value=PAD)
    tgt_sentences = pad_sequence([
        torch.Tensor(sentence).to(torch.long) for sentence in trg_sentence_list
    ], batch_first=True, padding_value=PAD)

    ### END YOUR CODE
    batch_size = len(batched_samples)

    assert src_sentences.shape[0] == batch_size and tgt_sentences.shape[0] == batch_size
    assert src_sentences.dtype == torch.long and tgt_sentences.dtype == torch.long
    return src_sentences, tgt_sentences
```

<br>

### Bucketing

`Bucketing`은 주어진 **문장의 길이에 따라 데이터를 그룹화하여 패딩을 적용**하는 기법입니다. 이 기법은 모델의 학습 시간을 단축하기 위해 고안되었습니다. 

위 Collating의 그림에서 보이듯이, bucketing을 적용하지 않은 경우, 배치에 패드 토큰의 개수가 늘어나 학습하는 데에 오랜 시간이 걸립니다. 그에 비하여 아래 그림과 같이 문장의 길이에 따라 미리 그룹화하여 패딩을 적용하면 학습을 효율적으로 진행할 수 있습니다.

![image-20220319220653821](https://user-images.githubusercontent.com/70505378/159122508-a483df25-472d-4eca-8aec-fef113b6d345.png)





위의 예시는 한 학습 샘플안에 하나의 문장만이 있을 때의 예시입니다. **그러나 기계 번역 문제에선 한 학습 샘플 안에 Source 문장과 Target 문장이 들어가 있으므로 한 번에 두 개의 문장을 고려해야합니다.** 

이에 대한 가장 쉬운 방법은 한 쪽을 기준으로 Bucketing 하는 것입니다. 일반적으로 짧은 문장에 대한 번역문은 짧은 문장이기 때문에 Source 문장과 Target 문장은 대략 비슷한 길이를 가질 것입니다. 따라서 Source를 기준으로 Bucketing을 하면 Target 쪽에 대해서도 괜찮은 Bucketing이 근사됩니다.

그러나 이 실습에서는 그러한 가정을 하지 않고 Source 문장과 Target 문장을 모두 고려하는 Bucketing을 작성해보겠습니다.

```python
import random
from collections import defaultdict

def bucketed_batch_indices(
    sentence_length: List[Tuple[int, int]],
    batch_size: int,
    max_pad_len: int
) -> List[List[int]]:
    """ 배치 Bucketing 함수
    문장들을 배치화하여 한 문장에 들어가는 패드의 갯수를 최대 max_pan_len개 이하로 만들어줍니다.

    Arguments:
    sentence_length -- (Source 문장의 길이, Target 문장의 길이) 짝이 담긴 리스트
    batch_size -- 배치 크기
    max_pad_len -- 최대 패딩 개수. 한 문장 내 패드 토큰 개수는 이 수를 초과하면 안됩니다.

    return:
    batch_indices_list -- 배치를 이루는 인덱스(index)의 리스트. sentence_length 리스트의 인덱스가 담겨 있습니다.

    Example:
    만약, sentence_length = [7, 4, 9, 2, 5, 10], batch_size = 3, 그리고 max_pad_len = 3 이라면
    가능한 batch_indices_list는 [[0, 2, 5], [1, 3, 4]] 입니다.
    왜냐하면 [0, 2, 5] 인덱스 위치의 문장이 비슷한 길이를 가지기 때문입니다: sentence_length[0] = 7, sentence_length[2] = 9, sentence_length[5] = 10
    이 예시는 각 샘플에 문장이 하나 밖에 없을 때의 예시지만, 각 샘플이 두 개의 문장으로 이루어진 경우도 비슷한 형식을 따릅니다.
    """
    bucket_dict = defaultdict(list)
    for index, (src_length, trt_length) in enumerate(sentence_length):
        bucket_dict[(src_length // max_pad_len, trt_length // max_pad_len)].append(index)

    batch_indices_list = [bucket[start:start+batch_size] for bucket in bucket_dict.values() for start in range(0, len(bucket), batch_size)]

    # 배치를 뒤섞는 것을 깜빡하지 마세요! 그대로 학습하면 모델이 문장 길이에 편향될 수 있습니다.
    random.shuffle(batch_indices_list)

    return batch_indices_list
```













<br>

<br>

# 참고 자료

* Illustrated transformer
  * http://jalammar.github.io/illustrated-transformer/
* The Annotated Transformer
  * http://nlp.seas.harvard.edu/2018/04/03/attention.html
* CS224n –Deep Learning for Natural Language Processing
  * http://web.stanford.edu/class/cs224n/
* Convolution Sequence to Sequence
  * https://arxiv.org/abs/1705.03122
* Fully-parallel text generation for neural machine translation
  * https://blog.einstein.ai/fully-parallel-text-generation-for-neural-machine-translation/  
