---
layout: single
title: "[AITech][NLP] 20220311 - Part 3) Seq2seq with attention, Beam Search and BLEU Score"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 KAIST '주재걸' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# Seq2seq with attention, Beam Search and BLEU Score

이번 강의는 `Seq2seq`와 `attention`에 대한 내용입니다. 

## Seq2Seq with attention

### Seq2Seq Model

`Seq2Seq` 모델은 sequence data를 입력으로 받아 다른 sequence data를 출력으로 내는 모델로, 아래와 같이 입력된 단어를 처리하는 **인코더**와 출력될 단어를 생성하는 **디코더**로 구성된 모습을 보입니다. 

![image-20220318102551661](https://user-images.githubusercontent.com/70505378/159004930-cf5fd1ff-536f-4ac1-b62d-f365be8a5ca8.png)

모든 입력 단어에 대한 처리를 마친 인코더는 최종적으로 디코더의 첫 hidden state인 **h<sub>0</sub>**를 전달합니다. 디코더의 첫 입력으로는 start token이 주어지며, 출력으로 end token이 나올 때까지 추론을 수행합니다. 

앞선 포스팅에서 살펴봤던 RNN types 중 **Sequence to sequence** type의 RNN type이라고 할 수 있습니다. 

![image-20220318102910657](https://user-images.githubusercontent.com/70505378/159004935-e0f4929e-561c-4de5-821c-6e4f3fc0db7f.png)

### Seq2Seq Model with Attention

그런데 고전적인 Seq2Seq 모델의 경우 다음의 문제들이 존재합니다. 

* 문장의 길이와 상관없이 최종적으로 fixed dimensional vector `h`에 모든 정보들을 넣어야 합니다. 
* 긴 문장의 경우 forward 과정에서 뒤에 나온 단어에 비해 앞에 나온 단어의 영향력이 많이 떨어진다. 

이 문제를 해결하기 위해, 과거에는 입력 문장의 순서를 뒤집어 앞 단어들의 정보를 보존하려는 시도도 있었습니다. 

`Attention` 메커니즘은 이 문제를 잘 해결할 수 있습니다. 고전적인 Seq2Seq 모델은 LSTM의 최종 출력 hidden state vector 하나 만을 디코더에서 사용했습니다. Attention 메커니즘에서는 각 time step에서 encoder가 생성해낸 hidden state vector(h1, h2, h3, ...)들을 디코더에 모두 전달하고, 디코더에서 이 벡터들을 선별적으로 사용하여 출력을 생성합니다. 이때 hidden state vector h<sub>t</sub>는 시각 t까지의 단어들을 인코딩하는 벡터에 해당하며, 시각 t에 들어온 입력 단어에 대한 정보가 가장 지배적으로 있습니다. 

<br>

그렇다면 디코더는 어떻게 '인코더의 정보를 선별적으로 사용'할 수 있을까요?

Attention mechanism의 구조와 함께 그 과정을 간단히 정리해보았습니다. 

1. 인코더는 매 time step마다 encoder hidden state를 생성한다. 
2. 마지막 입력에 대한 encoder hidden state는 디코더의 initial hidden state로 전달된다. 
3. Initial input과 initial hidden state를 사용해 initial decoder hidden state를 생성한다. 
4. decoder hidden state와 각 encoder state 사이 내적 계산을 통해 Attention scores를 구한다. 
5. Attention scores에 softmax 연산을 적용해 각 encoder hidden state를 사용할 비율에 해당하는 Attention distribution(Attention vector)을 구한다. 
6. Encoder hidden state들과 그에 대응하는 attention distribution의 weighted sum을 통해 Attention output(Context vector)을 생성한다. 
7. Attention output과 decoder hidden state를 사용해 initial output을 생성하고, decoder hidden state를 다음 time step의 디코더에 전달한다. 
8. 문장이 종료될 때까지 매 time step마다 3~7 과정을 반복한다. 

![image-20220318105907042](https://user-images.githubusercontent.com/70505378/159004937-b91a046b-c346-4dc8-a30b-66cb3044570d.png)

한 문장에 대한 학습 과정이 끝나면 아래와 같은 상태일 것입니다. 

![image-20220318111159706](https://user-images.githubusercontent.com/70505378/159004938-ba59f2b8-78d0-4df3-b2ef-bec6bbc019cc.png)

여기서 또 알아야 할 것이 **학습 시 주어지는 입력과 추론 시 주어지는 입력의 차이**입니다. 

학습 시에는 디코더의 output과 상관없이, 다음 time step의 디코더의 입력으로는 ground truth 단어가 들어가게 됩니다 (요즘에는 모델 일반화를 위해 학습 후반부에는 디코더의 출력을 다음 time step의 디코더의 입력으로 주는 경우도 있긴 합니다).  

그러나 추론 시에는 각 time step의 디코더의 output이 다음 time step의 디코더의 intput으로 들어가야 합니다. 그런데 이 때 디코더의 output이 틀렸다면 어떨까요? 정답에 해당하는 문장은 절대 맞힐 수가 없는 것일까요? 

이런 문제를 해결하기 위해 **Beam Search**라는 탐색 기법을 사용하는 데, 이는 아래 Beam Search 부분에서 보도록 하겠습니다. 

### Different Attention Mechanisms

앞서 Attention score를 구할 때는 encoder hidden state와 decoder hidden state 간의 내적 연산을 통해 구했었습니다. 이외에 attention score를 구하는 방법으로 다른 방법들도 있습니다. 

![image-20220318112221511](https://user-images.githubusercontent.com/70505378/159004942-095bb86a-0503-4f1d-9474-dd190aa73194.png)

**dot**

`dot` 방식의 연산은 가장 기본적인 내적 연산입니다. 

![image-20220318134240647](https://user-images.githubusercontent.com/70505378/159004945-23811168-020a-44f8-a39f-507ccd3b9f2d.png)

**general**

`general` 방식의 연산은 hidden state 사이에 learnable parameter W를 삽입해서 곱함으로써 그 표현력을 더욱 풍부하게 해줍니다. 

![image-20220318134252610](https://user-images.githubusercontent.com/70505378/159004948-faa1e5e7-b99c-4586-b8a7-fd0e2dad7c9a.png)

**concat**

`concat` 방식은 encoder와 decoder의 hidden state vector를 concat시킨 벡터를 은닉층 하나짜리 MLP를 통과시키는 방식입니다. 

![image-20220318134312461](https://user-images.githubusercontent.com/70505378/159004951-de209416-126e-4e63-913b-5ae1fc52e713.png)

**Contribution of Attention**

* NMT(기계 번역)의 성능을 크게 끌어올렸습니다. 
* 고전적 Seq2Seq의 bottleneck이었던 초기 단어에 대한 정보가 소실되는 문제를 해결하였습니다. 
* decoder output과 encoder 사이 shortcut 을 만듦으로써 gradient vanishing 문제를 완화하였습니다. 
* 출력이 왜 그렇게 나왔는지에 대한 설명가능성을 제공합니다. 





<br>

## Beam Search

이번에는 `Beam Search`가 무엇인지에 대해 알아보겠습니다. 

위에서 잠깐 언급했듯이, 문장을 생성할 때 직전 time step의 출력으로부터만 추론을 하게 되면 그 출력이 틀렸을 때 제대로 된 문장을 찾을 가능성이 없어집니다. 이를 **Exhaustive search** 또는 **Greedy search**라고 합니다. 

대신에 우리는 **Beam search**를 사용합니다. Beam search가 무조건 정답 문장을 찾는 것을 보장하는 것은 아니지만, 이전 탐색 방법들보다 훨씬 더 나은 모습을 보여줍니다. 

<br>

아래는 k=2로 설정했을 때의 beam search로 문장을 생성하는 과정입니다. 이해가 가시나요?

![image-20220318134940486](https://user-images.githubusercontent.com/70505378/159004953-365b8ed9-dcde-4284-a3c7-6117c76b9114.png)

매 time step마다 탐색을 이어갈 후보를 k개 남겨놓는 것입니다. 매 time step마다 다음 단어 후보 **k<sup>2</sup>**개를 생성하고, 그 중 점수가 가장 높은 상위 **k**개의 단어에서만 다음 탐색을 이어가는 것입니다. 

이렇게 하면 직전 time step의 출력이 정답이 아니더라도 여전히 정답을 맞힐 가능성이 있습니다. 

<br>

Greedy decoding에서는 모델이 \<end\> 토큰을 출력하면 문장 생성을 멈추게 됩니다. 

Beam search decoding에서는 서로 다른 문장들을 함께 탐색하기 때문에, \<end\> 토큰이 나타나는 타이밍이 서로 다를 수 있겠죠. 따라서 탐색 중인 문장에서 \<end\> 토큰이 출력되면 해당 문장의 탐색을 멈추고, 나머지 문장들에 대한 탐색을 계속 이어갑니다. 

보통 beam search의 종료 시점은 다음 두 가지 중 하나입니다. 

* Time step **T**에 도달할 때
* 문장 후보 **n**개가 생성될 때

탐색이 종료되고 문장 후보들이 생성되면, 아래와 같은 식으로 값이 가장 큰 후보를 최종 정답으로 출력합니다. 

![image-20220318135752596](https://user-images.githubusercontent.com/70505378/159004954-f5a3b620-9cfa-4ad9-a469-8f4da3a3b87c.png)

식의 앞에 곱해진 **1/t**은 문장의 길이가 길어질수록 값이 작아지는 문제를 보완하기 위한 normalization입니다.  

**logP<sub>LM</sub>** term은 값이 음수이고 클수록 더 높은 점수를 뜻합니다. 따라서 이 값들을 각 문장의 길이(양수)로 나눠주면 문장의 길이에 따라 값이 달라지는 문제를 보완할 수 있습니다. 



<br>

## BLEU Score

`BLEU Score`는 생성된 문장의 정확도를 측정하는 metric의 한 종류입니다. 

기존에 많이 사용되던 precision, recall, f1 score 등도 생성된 문장의 품질을 측정하는 데 사용할 수 있긴 합니다. 

![image-20220318140924568](https://user-images.githubusercontent.com/70505378/159004956-73339236-f705-4372-b3a4-88e8fc3af8a2.png)

다만, 위 수식에서 볼 수 있듯이 위 metric들은 **단어의 순서**를 고려하지 못 합니다. 문장에서 단어의 순서는 그 문장의 품질을 측정하는 데 중요한 요소인데도 말이죠. 

아래 그림을 보면 model1이 생성한 문장이 훨씬 그럴 듯해 보임에도 model2가 생성한 문장이 훨씬 나은 문장으로 평가받는 것을 볼 수 있습니다. 

![image-20220318141137898](https://user-images.githubusercontent.com/70505378/159004958-4a5a959c-6dc9-4bd7-9ba7-491ac7c4149f.png)

그래서 문장의 품질을 측정할 때는 **BLEU Score**(BiLingual Evaluation Understudy Score)를 사용합니다. BLEU Score의 정의는 다음과 같습니다. 

* **N-gram overlap** between machine translation output and reference sentence
* Compute **precision for n-grams of size one to four**
* Add **brevity penalty** (for too short translations)  
* Typically computed over the entire corpus, not on single sentences  

![image-20220318141343534](https://user-images.githubusercontent.com/70505378/159004960-b174dc3f-009e-4bdf-94aa-7ed0f8f92334.png)

BLEU Score를 사용하면 위에서 봤던 예시 문장으로부터 훨씬 더 합리적인 점수를 측정하는 것을 볼 수 있습니다. 

![image-20220318141502475](https://user-images.githubusercontent.com/70505378/159004925-eaeb831b-8efd-48fe-810d-c3f4197d5129.png)





<br>

## 실습) Subword-level Language Model

### 서브워드 토큰화의 필요성

`서브워드`는 하나의 단어를 여러 개의 단위로 분리했을 때 하나의 단위를 나타냅니다. `서브워드 토큰화`는 이런 서브워드를 단위로 토큰화를 하는 것으로, 아래와 같이 토큰화를 수행할 수 있습니다. 

**Example 1**

> "I have a meal" -> ['I', 'hav', 'e', 'a', 'me', 'al']
>
> "나는 밥을 먹는다" -> ['나', '는', '밥', '을', '먹는', '다']

**Example 2**

> "I have a meal" -> ['I', 'ha', 've', 'a', 'mea', 'l']
>
> "나는 밥을 먹는다" -> ['나', '는', '밥', '을', '먹', '는다']

위 예시처럼 단어 단위가 아닌, 더 잘게 쪼갠 서브워드를 단위로 문장을 토큰화하는 것입니다. 따라서 서브워드 토큰화 자체는 여러 방법으로 수행할 수 있습니다. 

다만, 기본적으로 아래와 같이 공백을 넘어선 서브워드를 구성하지는 않습니다. 

> "I have a meal" -> ['Iha', 've', 'am', 'ea', 'l']
>
> "나는 밥을 먹는다" -> ['나는밥', '을먹', '는다']

* 참고: [Huggingface: subword-tokenization](https://huggingface.co/transformers/tokenizer_summary.html#subword-tokenization)

그렇다면 서브워드 토큰화는 왜 필요할까요? 

첫번째 이유는 이 세상에는 **너무 많은 단어가 존재하기 때문**입니다. 단어의 개수가 많아지면 Vacabulary의 크기가 커지고, 따라서 파라미터 수(정확히 말하면 Embedding layer의 파라미터 수)도 그만큼 많아지기 때문에 모델의 크기가 커지는 결과를 불러오게 됩니다. 

그래서 단어 단위 토큰화의 문제를 극복하기 위해 '문자 단위 토큰화'도 사용을 했었는데요, 이는 또 지나치게 길어지는 sequence로 인한 성능 저하 문제를 겪으며 서브워드 토큰화가 각광을 받게 되었습니다. 

두번째 이유는 **OoV(Out of Vocabulary) 문제가 없다**는 것입니다. 학습 데이터에서 등장하지 않은 단어는 모두 Unknown 토큰 [UNK]로 처리됩니다. 이는 자주 등장하지 않는 단어를 생략해 Vocabulary의 크기를 줄이려는 시도이지만, 이로 인해 테스트 과정 중에 처음 보는 단어를 모두 [UNK]로 모델의 입력으로 넣게 되면서 모델의 성능이 저하 될 수 있습니다. 

그러나 서브워드 토큰화 단위로 자르게 된다면 최악의 경우에도 문자단위로 토큰화가 진행됩니다. 이는 서브워드 토큰화는 현재 가지고 있는 Vocabulary로 해당 단어가 토큰화 될 수 없다면 그 단어를 서브워드 단위로 쪼개서 평가하기 때문입니다. 

<br>

### Byte Pair Encoding (BPE)

대표적인 서브워드 토큰화 방법 중에 `Byte pair encoding`이 있습니다. 

**BPE Tokenizing**

BPE의 Vocab을 만드는 것은 간단합니다. 단순히 가장 많이 등장하는 연속한 짝을 찾아 추가하는 것 입니다. 다음과 같은 말뭉치가 있다고 가정해 봅시다.

```
low lower lowest newest
```

우선은 공백을 제외한 모든 문자를 Vocab에 추가하고 각 단어의 끝에 WORD_END "`_`" 붙여 단어를 구분지어 봅시다.

```
Vocab: d e i l n o r s t w _
[ l o w _ ], [ l o w e r _ ], [ l o w e s t _ ], [ w i d e s t _ ]  
```

이때 가장 많이 등장한 연속한 두 토큰을 찾아 Vocab에 추가하고 두 토큰을 붙입니다. 이 경우에는 `l o`가 세번 등장하여 가장 많았으니 `lo`로 붙여 Vocab에 추가합니다.

```
Vocab: d e i l n o r s t w _ lo
[ lo w _ ], [ lo w e r _ ], [ lo w e s t _ ], [ w i d e s t _ ] 
```

다음은 `lo w`가 세번 등장하므로 `low`를 추가합니다.

```
Vocab: d e i l n o r s t w _ lo low
[ low _ ], [ low e r _ ], [ low e s t _ ], [ w i d e s t _ ] 
```

다음은 `e s`가 두번 등장하므로 `es`를 추가합니다.

```
Vocab: d e i l n o r s t w _ lo low es
[ low _ ], [ low e r _ ], [ low es t _ ], [ w i d es t _ ] 
```

다음은 `es t`가 두번 등장하므로 `est`를 추가합니다.

```
Vocab: d e i l n o r s t w _ lo low es est
[ low _ ], [ low e r _ ], [ low est _ ], [ w i d est _ ] 
```

다음은 `est _`가 두번 등장하므로 `est_`를 추가합니다.

```
Vocab: d e i l n o r s t w _ lo low es est est_
[ low _ ], [ low e r _ ], [ low est_ ], [ w i d est_ ] 
```

`est_`는 est로 단어가 끝난다는 것을 알려주는 서브워드가 됩니다. 일반적으로 est가 나오면 단어가 끝나니 합리적입니다.

이러한 과정을 통해서 모든 단어가 추가되거나 원하는 Vocab 크기에 도달할 때까지 서브워드를 통합하여 추가하는 과정을 반복하면 됩니다.

BPE는 아래와 같이 구현할 수 있습니다. 

```python
# 단어 끝을 나타내는 문자
WORD_END = '_'

def build_bpe(
    corpus: List[str],
    max_vocab_size: int
) -> List[int]:
    """ BPE Vocab 만들기
    Byte Pair Encoding을 통한 Vocab 생성을 구현하세요.
    단어의 끝은 '_'를 사용해 주세요.
    이때 id2token을 서브워드가 긴 길이 순으로 정렬해 주세요.
    
    Note: 만약 모든 단어에 대해 BPE 알고리즘을 돌리게 되면 매우 비효율적입니다.
          왜냐하면 대부분의 단어는 중복되기 때문에 중복되는 단어에 대해서는 한번만 연산할 수 있다면 매우 효율적이기 때문입니다.
          따라서 collections 라이브러리의 Counter를 활용해 각 단어의 빈도를 구하고,
          각 단어에 빈도를 가중치로 활용하여 BPE를 돌리면 시간을 획기적으로 줄일 수 있습니다.
          물론 이는 Optional한 요소입니다.

    Arguments:
    corpus -- Vocab을 만들기 위한 단어 리스트
    max_vocab_size -- 최대 vocab 크기

    Return:
    id2token -- 서브워드 Vocab. 문자열 리스트 형태로 id로 token을 찾는 매핑으로도 활용 가능
    """
    vocab = list(set(chain.from_iterable(corpus)) | {WORD_END})
    corpus = {' '.join(word + WORD_END): count for word, count in Counter(corpus).items()}

    while len(vocab) < max_vocab_size:
        counter = Counter()
        for word, word_count in corpus.items():
            word = word.split()
            counter.update({
                pair: count * word_count 
                for pair, count in Counter(zip(word, word[1:])).items()
            })

        if not counter:
            break
        
        pair = counter.most_common(1)[0][0]
        vocab.append(''.join(pair))
        corpus = {
            word.replace(' '.join(pair), ''.join(pair)): count
            for word, count in corpus.items()
        }

    id2token = sorted(vocab, key=len, reverse=True)
    
    return id2token
```

**BPE Encoding**

만들어진 Vocab으로 텍스트 인코딩하는 방법은 몇 가지가 있습니다. 가장 쉬운 방법은 앞에서부터 토큰화하되 가장 긴 것부터 욕심쟁이 기법(Greedy Search)으로 먼저 매칭하는 방법입니다.

```
Vocab: bcde ab cd bc de a b c d e _
abcde ==> ab cd e _
```

이 방법은 최적의 인코딩을 보장하진 않지만 긴 단어를 빠르게 인코딩하는 것이 가능합니다.

두번째 방법은 가장 길게 매칭되는 것을 전체 텍스트에 대해 먼저 토큰화하는 방법입니다.

```
Vocab: bcde ab cd bc de a b c d e _
abcde ==> a bcde _
```

두번째 방법은 첫번째 방법보다 느리지만 텍스트를 좀 더 짧게 인코딩하는 것이 가능합니다.

두번째 방법을 이용한 BPE 인코딩은 아래와 같이 구현할 수 있습니다. 

```python
def encode(
    sentence: str,
    id2token: List[str]
) -> List[int]:
    """ BPE 인코더
    문장을 받아 BPE 토큰화를 통하여 고유 id의 수열로 바꿉니다.
    문장은 공백으로 단어단위 토큰화되어있다고 가정하며, Vocab은 sentence의 모든 문자를 포함한다고 가정합니다.
    찾을 수 있는 가장 긴 토큰부터 바꿉니다.
    
    Note: WORD_END를 빼먹지 마세요.

    Arguments:
    sentence -- 인코드하고자 하는 문장
    id2token -- build_bpe를 통해 만들어진 Vocab
    
    Return:
    token_ids -- 인코드된 토큰 id 수열
    """
    def recursive(word: str):
        if not word:
            return []
        for token_id, token in enumerate(id2token):
            index = word.find(token)
            if index != -1:
                break
        else:
            raise Exception("토큰을 찾을 수 없습니다!")

        return recursive(word[:index]) + [token_id] + recursive(word[index+len(token):])
    
    token_ids = list(chain.from_iterable(recursive(word + WORD_END) for word in sentence.split()))

    return token_ids
```

**BPE Decoding**

BPE로 인코딩된 것을 디코딩하는 것은 간단합니다. 그저 해당 id를 해당하는 서브워드로 만든 뒤 합치면됩나다. WORD_END는 공백으로 처리하면 쉽습니다.

```
[ 196 62 20 6 ] ==> [ I_ li ke_ it_ ] ==> "I_like_it_" ==> "I like it " -> "I like it"  
```

BPE 디코딩은 아래와 같이 구현할 수 있습니다. 

```python
def decode(
    token_ids: List[int],
    id2token: List[str]
) -> str:
    """ BPE 디코더
    BPE로 토큰화된 id 수열을 받아 문장으로 바꿉니다.
    단어단위 토큰화에서의 문장 복원은 단순히 공백을 사이에 넣는 디코딩을 사용합니다.
    문장 끝의 공백은 잘라냅니다.
    
    Arguments:
    token_ids -- 디코드하고자하는 토큰 id 수열
    id2token -- build_bpe를 통해 만들어진 Vocab

    Return:
    sentence  -- 디코드된 문장
    """
    sentence = ''.join(id2token[token_id] for token_id in token_ids).replace(WORD_END, ' ').strip()
    
    return sentence
```

<br>

### Transformers 라이브러리

위에서 작성한 BPE 구현체를 통해 서브워드 토큰화의 원리를 알 수 있지만, 위의 구현체를 실제로 사용하기에는 난점이 존재합니다. 왜냐하면 BPE Vocab을 만드는 과정은 매우 오랜 시간이 걸리기 때문입니다. 다양한 토큰화기(tokenizer)를 직접 구현하고 학습하는 것은 매우 비용이 크기 때문에, 라이브러리를 활용하여 토큰화기를 사용하는 방법을 알아봅시다.

`Transformer` 라이브러리는 다양한 Transformer 구현체를 총망라한 라이브러리입니다. Transfomer 외에도 다양한 토큰화기를 지원하는데, 이미 학습된 서브워드 토큰화기 역시 쉽게 불러올 수 있습니다.

```python
! pip install transformers

from transformers import BertTokenizerFast

# BERT 모델에서 사용하는 토큰화를 가져옵니다.
# https://huggingface.co/docs/transformers/model_doc/bert#transformers.BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained(
    "bert-base-cased",
    unk_token='<unk>',
    eos_token='<eos>'
)
```

`BertTokeniezrFast` 토큰화기는 `##`을 통하여 현 단어가 이전 단어와 연결되어 있는지를 알려줍니다. 

```python
# 서브워드 토큰화 예시
print(tokenizer.tokenize('Boostcamp AI Tech'))
token_ids = tokenizer("Boostcamp AI Tech", add_special_tokens=False).input_ids
print(token_ids)
print(tokenizer.decode(token_ids))

print(tokenizer.tokenize('qwerklhfa asdfkwej'))
token_ids = tokenizer("qwerklhfa asdfkwej", add_special_tokens=False).input_ids
print(token_ids)
print(tokenizer.decode(token_ids))
'''
['Bo', '##ost', '##cam', '##p', 'AI', 'Tech']
[9326, 15540, 24282, 1643, 19016, 7882]
Boostcamp AI Tech
['q', '##wer', '##k', '##l', '##h', '##fa', 'as', '##d', '##f', '##k', '##we', '##j']
[186, 12097, 1377, 1233, 1324, 8057, 1112, 1181, 2087, 1377, 7921, 3361]
qwerklhfa asdfkwej
'''
```

**Q.** Transformers에서 제공되는 BertTokenizerFast는 모든 조합을 만들 수 있는 서브워드 기반 토큰화기임에도 불구하고 Unknown 토큰을 받을 수 있다. 서브워드 토큰화기에서 Unknown 토큰이 발생할 수 있는 상황은 무엇이 있을까?

**A.** 다른 언어가 들어오면 Unknown 토큰이 발생할 수 있다. 영어로 학습된 BPE 토큰화기는 한글은 처리 가능한 문자가 아니기 때문에 한국어는 다룰 수 없을 것이다. 이를 해결하기 위해 GPT-2 토큰화에선 [byte-level BPE](https://huggingface.co/docs/transformers/tokenizer_summary#bytelevel-bpe)를 통해 이 세상 모든 언어를 다루고자 하는 시도 역시 있었다.

<br>

<br>

# 참고 자료

* 
