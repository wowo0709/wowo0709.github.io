---
layout: single
title: "[AITech][NLP] 20220310 - Part 1) Bag of Words & Word Embedding"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Word2Vec', 'GloVe']
---



<br>

_**본 포스팅은 KAIST '주재걸' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# Bag of Words & Word Embedding

이번 포스팅에서는 `Bag of Words` 와 `Word Embedding`에 대해 알아보겠습니다. 

## Bag of Words

**Bag of Words Representation**

`Bag of Words`는 각 단어들이 유일한 id를 갖도록 매핑해주는 과정입니다. 이 때 vector 표현을 사용하는데, 유일한 id를 갖게 하는 방법은 '원-핫 벡터' 표현을 사용하거나, 겹치지 않는 '임베딩 벡터'로 표현하는 방법이 있습니다. 

아래는 각 단어를 원-핫 벡터로 표현한 예시입니다. 

![image-20220317102254608](https://user-images.githubusercontent.com/70505378/158726062-26867c19-44bf-4068-bb27-a364ccc36111.png)

이렇듯, 자연어 처리를 위해서는 먼저 각 단어를 유일한 벡터로 표현해주는 과정이 필요합니다. 







<br>

## Word Embedding

위에서 각 단어들을 유일한 원-핫 벡터로 표현하는 방법에 대해 보았습니다. 그런데 원-핫 벡터 표현 방법은 단어가 많아질수록 벡터의 길이도 같이 늘어나기 때문에, 메모리 측면에서 아주 비효율적입니다. 

따라서 실제로는 **Word Embedding** 기법을 사용합니다. Word embedding이란 아래와 같습니다. 

* 각 단어를 유일한 임베딩된 벡터로 표현
* 의미가 비슷한 단어들은 공간 상에서도 가깝게 위치
* 대응 관계가 비슷한 단어 쌍의 벡터는 유사함

![image-20220317102917531](https://user-images.githubusercontent.com/70505378/158726064-f72267ed-6ab8-4f2a-abe4-e274364b7a85.png)

**Word2Vec**

이렇게 단어를 유일한 임베딩된 벡터로 표현하는 방법들 중 **Word2Vec**이라 합니다. Word2Vec의 아이디어는 주변 단어들의 분포로부터 특정 단어가 등장할 가능성을 매핑하는 것입니다. 

그 중 중심 단어로부터 주변 단어들을 예측하는 것을 Skip-gram 방법이라 하고, 주변 단어들로부터 중심 단어를 예측하는 것을 CBOW(Continuous Bag Of Words) 방법이라 합니다. 많은 경우에 Skip-gram 방법을 사용하는 것이 더 나은 성능을 보여서, Skip-gram 방법을 많이 사용합니다. 

![image-20220317103831478](https://user-images.githubusercontent.com/70505378/158726065-0490fcc0-de40-4f10-93d0-0bd85360b8f4.png)



그러면 Word2Vec 알고리즘이 어떻게 동작하는지 좀 더 자세히 알아보겠습니다. 아래 슬라이드는 강의 내용 중의 슬라이드입니다. 

![image-20220317104156626](https://user-images.githubusercontent.com/70505378/158726067-ccb02f91-4e88-4756-9be9-99626fffafd2.png)

먼저 주어진 문장은 다음의 과정을 거칩니다. 

* 문장 내의 단어들로 Vocabulary를 생성 (토큰화)
* Vocabulary 내의 단어들을 유일한 원-핫 벡터로 표현

Word2Vec의 학습 데이터를 준비할 때는 **'슬라이딩 윈도우'** 기법을 사용합니다. 가령 위 문장에서 window size=3이라면, 학습 데이터가 다음과 같이 만들어집니다. 

* `I`: (I, \<sos\>), (I, study)
* `study`: (study, I), (study, math)
* `math`: (math, study), (math, \<eos\>)

여기까지 했다면, 이제 학습 데이터를 위 그림에서와 같은 embedding layer에 넣어줄 수 있습니다. 위 그림을 보면 input layer와 output layer의 차원이 **V**인 것을 알 수 있는데, V는 vocabulary의 크기를 나타내고 따라서 원-핫 벡터의 차원과 같습니다. 즉, input layer와 output layer에서 각 node는 하나의 단어를 가리킵니다. 

가운데 hidden layer의 차원 N은 사용자가 결정하는 하이퍼파라미터로, 단어 임베딩 공간의 차원 수를 나타냅니다. 

<br>

이번에는 embedding layer 오른쪽에 있는 행렬 곱 그림이 무엇을 나타내는지 알아보겠습니다. (V=3, N=2로 가정)

앞서 우리는 vocabulary 내의 각 단어를 원-핫 벡터로 나타냈었습니다. 여기서는 'study'에 해당하는 원-핫 벡터 [0, 1, 0]을 입력으로 준다고 가정하겠습니다. 그러면 [0, 1, 0] 벡터에 입력 행렬 W<sub>1</sub>이 곱해져서 study라는 단어를 크기 2의 embedding vector로 나타낼 수 있습니다. 

그런데 이를 좀 더 자세히 보면, _행렬과 원-핫 벡터를 곱한다는 것은 행렬에서 특정 column(혹은 row)을 추출해내는 것과 동일_하다는 것을 알 수 있습니다. 따라서 실제 구현 시에는 입력 단어의 index를 참조해서, W1으로부터 index번째 column을 추출하는 식으로 동작합니다. 

그리고 당연하게도, W1의 각 column은 각 단어에 대한 embedding vector에 대응합니다. 이 W1이 실질적인 각 단어들의 embedding vector를 가지고 있는 것이죠. 

<br>

W<sub>1</sub>x의 연산으로 (2, 1)의 열벡터가 생성됩니다. 여기에 출력 행렬 W<sub>2</sub>를 곱하면 원-핫 벡터의 dimension과 동일한 (3, 1)의 열벡터가 출력되게 됩니다. 그리고 나서 출력 벡터에 Softmax 연산을 취해 값을 확률로 변환해줍니다. 이를 ground truth 값과 비교해서, 1에 해당하는 확률은 더욱 높아지도록, 0에 해당하는 나머지 확률들은 더욱 낮아지도록 학습됩니다. 

여기까지 보면 알 수 있듯이, 입력 행렬 W<sub>1</sub>의 각 column vector와 출력 행렬 W<sub>2</sub>의 각 row vector는 각 단어를 잘 표현하도록 학습하게 되고, 이로부터 단어들 간 관계를 학습하게 된다는 것을 알 수 있습니다. 

**GloVe**

또 다른 Word embedding 방법으로 **GloVe**(Global Vectors for Word Representation) 방법이 있습니다. 

GloVe의 Word2Vec과의 가장 큰 차이점은 문장으로부터 만들어진 학습 데이터로부터 각 단어들이 같은 window 내에서 총 몇 번 같이 등장하였는지 사전에 먼저 계산을 하고, 이를 **co-ocurrence matrix**로 생성합니다. 이후에 서로 다른 두 단어가 계산된 값에 수렴하도록 학습하는 것입니다. 

![image-20220317112859693](https://user-images.githubusercontent.com/70505378/158726069-faf9bdc7-9c39-40a6-b2e2-3587f8fe906b.png)

GloVe 방법의 장점은 다음과 같습니다. 

* 여러 번 함께 등장한 단어들의 중복된 계산을 방지하여 더 빠른 학습 속도를 보인다. 
* 비교적 적은 양의 corpus로도 좋은 성능을 보인다. 

아래 그림을 보면 GloVe 또한 단어들 간의 대응 관계를 잘 학습한 것을 확인할 수 있습니다.  아래 그림은 단어의 '원형-비교급-최상급' 간 관계를 이어서 나타낸 모습입니다. 

![image-20220317113415980](https://user-images.githubusercontent.com/70505378/158726057-08234aaf-85d2-4785-a3ca-151fc34d6ab9.png)

마지막으로, 아래 GloVe 사이트에서 그 구현 코드나 기학습된 word vector를 다운로드 받아 사용할 수 있습니다. 

* https://nlp.stanford.edu/projects/glove/  

<br>

## 실습) Data Preprocessing & Tokenization

**데이터 업로드**

corpus가 저장되어 있는 txt 파일을 불러옵니다. 

```python
with open('./corpus.txt', 'r', encoding='utf-8') as fd:
    corpus = fd.readlines() # '~~~\n' 형태로 저장
```



**영어 텍스트 토큰화 및 전처리 구현**

`토큰화`란 주어진 입력 데이터를 자연어처리 모델이 인식할 수 있는 단위로 변환해주는 과정입니다. 그 중 `단어 단위 토큰화`란 '단어'가 자연어처리 모델이 인식하는 단위가 되도록 토큰화하는 것입니다. 

간단한 **토큰화기(tokenizer)**를 직접 구현해봅니다. 

```python
from typing import List

def tokenize(
    sentence: str
) -> List[str]:
    """ 토큰화기 구현
    공백으로 토큰을 구분하되 . , ! ? 문장 부호는 별개의 토큰으로 처리되어야 합니다.
    영문에서 Apostrophe에 해당하는 ' 는 두가지 경우에 대해 처리해야합니다.
    1. not의 준말인 n't은 하나의 토큰으로 처리되어야 합니다: don't ==> do n't
    2. 다른 Apostrophe 용법은 뒤의 글자들을 붙여서 처리합니다: 's 'm 're 등등 
    그 외 다른 문장 부호는 고려하지 않으며, 작은 따옴표는 모두 Apostrophe로 처리합니다.
    모든 토큰은 소문자로 변환되어야 합나다.

    힌트: 정규표현식을 안다면 re 라이브러리를 사용해 보세요!

    예시: 'I don't like Jenifer's work.'
    ==> ['i', 'do', 'n\'t', 'like', 'jenifer', '\'s', 'work', '.']

    Arguments:
    sentence -- 토큰화할 영문 문장
    
    Return:
    tokens -- 토큰화된 토큰 리스트
    """

    ### YOUR CODE HERE 
    ### ANSWER HERE ###
    special_characters = ['.', ',', '!', '?']
    tokens: List[str] = list()
    for word in sentence.split():
      w = ""
      for idx, ch in enumerate(word):
        if ch.isalpha():
          w += ch.lower()
        elif ch == "'":
          if word[idx-1] == 'n' and idx+1 == len(word)-1 and word[idx+1] == 't':
            tokens.append(w[:-1])
            w = 'n\''
          else:
            tokens.append(w)
            w = '\''
        elif ch in special_characters:
          tokens.append(w)
          w = ch
      tokens.append(w)

    ### END YOUR CODE

    return tokens
```

토큰화가 진행되고 나면 **Vocabulary**를 만들어야 합니다. 컴퓨터는 글자를 알아볼 수 없기 대문에 각 토큰을 숫자 형식의 유일한 id에 매핑하는 과정입니다. 

이러한 매핑은 모델 학습 전에 사전 정의되어야 합니다. 이때, 모델이 다룰 수 있는 토큰들의 집합과 이 매핑을 흔히 Vocab이라고 부릅니다. 

```python
from typing import List, Tuple, Dict

# [UNK] 토큰
unk_token = "[UNK]"
unk_token_id = 0 # [UNK] 토큰의 id는 0으로 처리합니다.

def build_vocab(
    sentences: List[List[str]], # list of tokenized sentences
    min_freq: int
) -> Tuple[List[str], Dict[str, int], List[int]]:
    """ Vocabulary 만들기
    토큰화된 문장들을 받아 각 토큰을 숫자로 매핑하는 token2id와 그 역매핑인 id2token를 만듭니다.
    자주 안나오는 단어는 과적합을 일으킬 수 있기 때문에 빈도가 적은 단어는 [UNK] 토큰으로 처리합니다.
    이는 Unknown의 준말입니다.
    토큰의 id 번호 순서는 [UNK] 토큰을 제외하고는 자유입니다.

    힌트: collection 라이브러리의 Counter 객체를 활용해보세요.

    Arguments:
    sentences -- Vocabulary를 만들기 위한 토큰화된 문장들
    min_freq -- 단일 토큰으로 처리되기 위한 최소 빈도
                데이터셋(sentences)에서 최소 빈도보다 더 적게 등장하는 토큰은 [UNK] 토큰으로 처리되어야 합니다.

    Return:
    id2token -- id를 받으면 해당하는 토큰을 반환하는 리스트 
    token2id -- 토큰을 받으면 해당하는 id를 반환하는 딕셔너리
    """

    ### YOUR CODE HERE
    ### ANSWER HERE ###
    id2token: List[str] = [unk_token]
    token2id: Dict[str, int] = {unk_token: unk_token_id}

    from collections import defaultdict, Counter
    token_freq = defaultdict(int)
    for sentence in sentences:
      token_cnt = Counter(sentence).most_common()
      for token, cnt in token_cnt:
        token_freq[token] += cnt

    i = 1
    for token, freq in token_freq.items():
      if freq < min_freq:
        continue
      id2token.append(token)
      token2id[token] = i
      i += 1

    ### END YOUR CODE

    assert id2token[unk_token_id] == unk_token and token2id[unk_token] == unk_token_id, \
        "[UNK] 토큰을 적절히 삽입하세요"
    assert len(id2token) == len(token2id), \
        "id2word과 word2id의 크기는 같아야 합니다"
    return id2token, token2id
```

이제 문장을 받아 토큰화하고 이들을 적절한 id로 바꾸는 **인코딩** 함수를 정의합니다. 

인코딩 함수는 입력으로 tokenizer, setence, token2id(build_vocab의 반환값)를 받습니다. 

```python
from typing import Callable

def encode(
    tokenize: Callable[[str], List[str]],
    sentence: str,
    token2id: Dict[str, int]
) -> List[str]:
    """ 인코딩
    문장을 받아 토큰화하고 이들을 적절한 id들로 바꿉니다.
    토큰화 및 인덱싱은 인자로 들어온 tokenize 함수와 인자로 주어진 token2id를 활용합니다.
    Vocab에 없는 단어는 [UNK] 토큰으로 처리합니다.

    Arguments:
    tokenize -- 토큰화 함수: 문장을 받으면 토큰들의 리스트를 반환하는 함수
    sentence -- 토큰화할 영문 문장
    token2id -- 토큰을 받으면 해당하는 id를 반환하는 딕셔너리
    
    Return:
    token_ids -- 문장을 인코딩하여 숫자로 변환한 리스트
    """


    ### YOUR CODE HERE 
    ### ANSWER HERE ###
    token_ids: List[int] = list()
    tokens = tokenize(sentence)
    for token in tokens:
      if token in token2id: token_ids.append(token2id[token])
      else: token_ids.append(0) # token2id["[UNK]"] = 0

    ### END YOUR CODE

    return token_ids
```

거꾸로 id들이 있을 때 원문장을 복원하는 **디코딩** 함수도 필요합니다. 인코딩 과정에서 공백 및 대소문자에 대한 정보를 잃어버리고, [UNK] 토큰으로 인해 원문장을 복원할 수는 없습니다. 여기서는 간단히 공백으로 연결된 문장으로 디코딩합니다. 

```python
def decode(
    token_ids: List[int],
    id2token: List[str]
) -> str:
    """ 디코딩
    각 id를 적절한 토큰으로 바꾸고 공백으로 연결하여 문장을 반환합니다.
    """
    return ' '.join(id2token[token_id] for token_id in token_ids)
```

결과는 아래와 같습니다. 

```python
from functools import partial

id2token, token2id = build_vocab(list(map(tokenize, corpus)), min_freq=2)
input_ids = list(map(partial(encode, tokenize, token2id=token2id), corpus))

for sid, sentence, token_ids in zip(range(1, 5), corpus, input_ids):
    print(f"======{sid}=====")
    print(f"원문: {sentence}")
    print(f"인코딩 결과: {token_ids}"),
    print(f"디코딩 결과: {decode(token_ids, id2token)}\n")
    
'''
======1=====
원문: A young man participates in a career while the subject who records it smiles.

인코딩 결과: [1, 2, 3, 0, 4, 1, 0, 5, 6, 0, 7, 0, 8, 0, 9]
디코딩 결과: a young man [UNK] in a [UNK] while the [UNK] who [UNK] it [UNK] .

======2=====
원문: The man is scratching the back of his neck while looking for a book in a book store.

인코딩 결과: [6, 3, 11, 12, 6, 13, 14, 15, 0, 5, 16, 17, 1, 10, 4, 1, 10, 18, 9]
디코딩 결과: the man is scratching the back of his [UNK] while looking for a book in a book store .

======3=====
원문: A person wearing goggles and a hat is sled riding.

인코딩 결과: [1, 19, 20, 21, 22, 1, 23, 11, 0, 24, 9]
디코딩 결과: a person wearing goggles and a hat is [UNK] riding .

======4=====
원문: A girl in a pink coat and flowered goloshes sledding down a hill.

인코딩 결과: [1, 25, 4, 1, 26, 27, 22, 28, 0, 0, 29, 1, 30, 9]
디코딩 결과: a girl in a pink coat and flowered [UNK] [UNK] down a hill 
'''
```

정리해봅시다. '자연어'와 '모델이 이해할 수 있는 형태' 간의 변환을 위해서는 다음의 요소들이 필요합니다. 

* **tokenizer**: sentence를 입력으로 주면 토큰화된 단어 리스트를 반환
* **build_vocab**: 토큰화된 sentence 리스트를 입력으로 주면 word <=> id 간 매핑을 해주는 id2token, token2id를 반환
* **encoder**: tokenizer, sentence, token2id를 입력으로 주면 인코딩된 단어 리스트를 반환
* **decoder**: 인코딩된 단어 리스트와 id2token을 입력으로 주면 디코딩된(복원된) 문장을 반환





<br>

**Spacy**

다음으로 살펴볼 것은 `spacy`입니다. spacy는 **영단어 토큰화**를 지원해주는 모듈입니다. 

```python
! pip install spacy
! python -m spacy download en_core_web_sm

import spacy
spacy_tokenizer = spacy.load('en_core_web_sm')
```

spacy를 활용환 토큰화는 상대적으로 시간이 오래 걸리지만, 토큰화 외에도 해당 문장에 대해 많은 정보를 제공합니다. 자주 사용하는 정보에는 다음의 것들이 있습니다. 

* `token.text`: text를 반환
* `token.lemma_`: text의 원형을 반환
* `token.pos_`: text의 품사를 반환

```python
tokens = spacy_tokenizer("Jhon's book isn't popular, but he loves his book.")
print(*tokens, type(tokens[0]))
print ([(token.text, token.lemma_, token.pos_) for token in tokens])

'''
Jhon 's book is n't popular , but he loves his book . <class 'spacy.tokens.token.Token'>
[('Jhon', 'Jhon', 'PROPN'), ("'s", "'s", 'PART'), ('book', 'book', 'NOUN'), ('is', 'be', 'AUX'), ("n't", 'not', 'PART'), ('popular', 'popular', 'ADJ'), (',', ',', 'PUNCT'), ('but', 'but', 'CCONJ'), ('he', '-PRON-', 'PRON'), ('loves', 'love', 'VERB'), ('his', '-PRON-', 'DET'), ('book', 'book', 'NOUN'), ('.', '.', 'PUNCT')]
'''
```

이외에도 spacy는 불용어(stopword) 처리도 지원합니다. 불용어란 한 언어에서 자주 등장하지만 큰 의미가 없는 단어를 뜻합니다. 고전적인 자연어 처리에서는 이러한 단어들은 분석에 도움이 되지 않는다고 생각하였기 때문에 이를 제공합니다. 

spacy에서는 불용어 단어의 목록을 제공하고 있습니다. 

```python
stop_words = spacy.lang.en.stop_words.STOP_WORDS
print(stop_words)
query = "this"
print(query in stop_words)
'''
{'indeed', 'whom', 'can', 'myself', 'almost', 'onto', '‘m', ..., 'is', 'therefore'}
True
'''
```

spacy와 같은 모듈을 사용하면 더욱 일반적이고 편리하게 인코딩을 수행할 수 있습니다.

<br>

**Konlpy**

`Konlpy(코엔엘파이)`는 한국어 토큰화를 위한 모듈입니다. 영어와 달리, 한국어는 '교착어'로서의 특징 때문에 단순히 '공백'으로 문장을 토큰화할 수 없습니다. 

한국어에서는 단어 단위 토큰화 방법은 공백에 기반하지 않고 대신에 '형태소 분석기'를 활용합니다. 

```python
! apt-get install -y build-essential openjdk-8-jdk python3-dev curl git automake
! pip install konlpy "tweepy<4.0.0"
! /bin/bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```

아래와 같이 사용할 수 있고, 많은 기능들을 제공하니 더 찾아보시길 바랍니다. 

```python
from konlpy.tag import Mecab
tokenizer = Mecab()

text = """\
유구한 역사와 전통에 빛나는 우리 대한국민은 \
3·1운동으로 건립된 대한민국임시정부의 법통과 불의에 항거한 4·19민주이념을 계승하고, \
조국의 민주개혁과 평화적 통일의 사명에 입각하여 정의·인도와 동포애로써 민족의 단결을 공고히 하고, \
모든 사회적 폐습과 불의를 타파하며, \
자율과 조화를 바탕으로 자유민주적 기본질서를 더욱 확고히 하여 \
정치·경제·사회·문화의 모든 영역에 있어서 각인의 기회를 균등히 하고, \
능력을 최고도로 발휘하게 하며, 자유와 권리에 따르는 책임과 의무를 완수하게 하여, \
안으로는 국민생활의 균등한 향상을 기하고 밖으로는 항구적인 세계평화와 인류공영에 이바지함으로써 \
우리들과 우리들의 자손의 안전과 자유와 행복을 영원히 확보할 것을 다짐하면서 \
1948년 7월 12일에 제정되고 8차에 걸쳐 개정된 헌법을 이제 국회의 의결을 거쳐 국민투표에 의하여 개정한다.\
"""

print(tokenizer.pos(text))
print(tokenizer.morphs(text))
'''
[('유구', 'XR'), ('한', 'XSA+ETM'), ('역사', 'NNG'), ..., ('한다', 'XSV+EF'), ('.', 'SF')]
['유구', '한', '역사', ..., '한다', '.']
'''
```

















<br>

<br>

# 참고 자료

*  
