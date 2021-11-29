---
layout: single
title: "[Deep Learning] Word Vector"
categories: ['AI', 'DeepLearning']
toc: true
toc_sticky: true
tag: ['Word2Vec','Glove]
---





## Word Embedding

- express words using vectors
- vectors are trained

## Word2Vec - 한국어

- 지금까지의 텍스트 코딩 방식인 One-hot encoding, BoW(단어모음)-문서-단어 행렬 방식은 모두 단어마다 고유번호를 배정하여 사용하지만, 이 번호들에는 아무런 의미가 포함되어 있지 않고 단지 인덱스 역할만 함.
- Word Vector 에서는 
 - 단어를 고차원 공간상의 벡터로 표현함으로 단어간 거리 표현 가능. 차원이 높을수록 정교한 의미 구분 가능.
 - 보통 50~300 개 정도의 차원을 사용함
 - 단어간의 거리 (유사도) 뿐 아니라 방향성(벡터)도 찾을 수 있음.
 - 단어벡터의 각 성분이 어떤 의미를 갖는지는 알 수 없다.

- 빅카인즈 뉴스기사 데이터 활용(https://www.bigkinds.or.kr/)

### 뉴스를 이용한 단어 벡터 생성

- 단어 추출: konlpy 의 kkma() 사용
- 단어 벡터 생성: gensim 의 word2vec() 사용


```python
!pip install konlpy
```

```python
from konlpy.tag import Kkma     # 형태소 분석 및 자연어 처리 모듈 (서울대)
from gensim.models.word2vec import Word2Vec
```


```python
# IT 뉴스기사를 이용한다
!wget https://bit.ly/2X7UON2 -O news2018.xlsx# IT ë´ě¤ę¸°ěŹëĽź ě´ěŠíë¤
!wget https://bit.ly/2X7UON2 -O news2018.xlsx
```

<br>





```python
import pandas as pd
news = pd.read_excel("news2018.xlsx")
```


```python
news["본문"][:4]
```


    0    - 비핵화 수준 상응 조치 놓고\n- 양국 협상팀 막판까지 ‘밀당’\n- 1차 때와...
    1    김정은 국무위원장이 27일 시작되는 제2차 북미정상회담 성공을 위해 심혈을 기울이고...
    2    북미가 처음으로 정상 간 단독회담과 만찬을 가지며 또다시 새로운 역사 창조에 나섰다...
    3    지난해 9월 남북정상회담 당시 리선권 북한 조국평화통일위원장의 '냉면' 발언으로 정...
    Name: 본문, dtype: object

<br>


```python
kkma = Kkma()
```


```python
sentence_list = []
for sent in news["본문"]:
    
    sent_kkma_pos = kkma.nouns(sent)   # 명사만 추출
    word_list = []
    for word_pos in sent_kkma_pos:
        word_list.append(word_pos)

    sentence_list.append(word_list)
```


```python
for i in range(3):
    print(sentence_list[i])
```

    ['비핵화', '수준', '상응', '조치', '양국', '협상', '협상팀', '팀', '막판', '당', '1', '1차', '차', '때', '시간', '조율', '단계적', '접근', '동의', '예상', '종전', '종전선언', '선언', '연락', '연락사무소', '사무소', '개설', '등', '조건', '조건부', '부', '제재', '완화', '명시', '가능성', '북미', '북미회담', '회담', '빅딜', '성공', '김', '김정은', '정은', '답방', '결과', '연관', '북한', '국무', '국무위원장', '위원장', '도', '도널드', '널드', '트럼프', '미국', '대통령', '27', '27일', '일', '친교', '만찬']
    ['김', '김정은', '정은', '국무', '국무위원장', '위원장', '27', '27일', '일', '시작', '저', '2', '2차', '차', '북미', '북미정상회담', '정상', '회담', '성공', '심혈', '조선', '조선중앙통신', '중앙', '통신', '이날', '26', '26일', '하노이', '도착', '리', '호텔', '실무', '실무대표단', '대표단', '보고', '조미', '수뇌', '수뇌회담', '성공적', '보장', '나라', '현지', '파견', '사이', '접촉', '정형', '결과', '을', '구체적', '청취']
    ['북미', '처음', '정상', '간', '단독', '단독회담', '회담', '만찬', '역사', '창조', '결', '물', '도출', '북측', '영', '영변', '변', '핵', '핵시설', '시설', '폐기', '외', '추가', '추가적인', '적인', '비핵화', '조치', '미국', '금강산', '금강산관광', '관광', '등', '경제적', '체제', '체제보장', '보장', '여부', '양', '간', '톱', '담판', '김', '북한', '국무', '국무위원장', '위원장', '도', '도널드', '널드', '트럼프', '대통령', '27', '27일', '일', '오후', '6', '6시30분', '시', '30', '분', '현지', '현지시간', '시간']

```python
news.shape, len(sentence_list)
```


    ((1543, 19), 1543)


```python
[len(w) for w in sentence_list][:10]
```


    [61, 50, 63, 48, 44, 38, 38, 60, 52, 39]

<br>


```python
model = Word2Vec(sentence_list, sg=1, size=100)   #sg=1 (skip-gram), 0(CBOW)
```


```python
model["대한민국"]
```

    array([-2.4107585e-02, -7.4946046e-02,  1.5689157e-03,  1.7300507e-02,
            7.7659652e-02, -4.3071166e-02,  8.3631985e-02,  1.6745523e-01,
           -8.2903586e-02, -1.7553378e-02,  3.9016213e-02, -1.0054115e-01,
            4.1688729e-02,  1.7242630e-01, -1.8903978e-02,  1.2952442e-01,
            4.8356697e-02,  4.0910381e-01, -7.0913650e-02, -5.0823655e-02,
            1.4685905e-01, -1.2997684e-01,  2.2543812e-02, -3.7712879e-02,
            9.6920088e-02,  1.3099691e-01, -1.3746825e-01, -1.0660959e-01,
            1.1127534e-01,  1.2975276e-01, -2.8525587e-02, -1.2853998e-01,
           -8.3741836e-02, -9.9310517e-02, -2.4495709e-01, -4.1113162e-01,
            1.0418992e-02,  7.9034410e-02,  1.3711397e-01, -5.1028132e-02,
           -1.4102933e-01, -4.6473064e-02, -7.5484976e-02, -6.2391542e-02,
           -4.0519308e-02, -1.5226401e-01, -1.3334070e-01, -1.7248647e-01,
           -9.5049895e-02,  9.9440172e-02, -2.9708706e-02,  8.7483376e-02,
            8.1404611e-02,  1.3708833e-01, -1.1457676e-01, -9.5910830e-03,
           -6.4596653e-02, -2.4731688e-01,  3.0563422e-02,  1.2345860e-01,
           -3.4807574e-02,  1.6530770e-01,  1.2371200e-01, -1.2324062e-02,
            1.4210464e-01, -1.4213949e-01,  1.7249145e-01, -7.8410409e-02,
           -6.2629886e-02, -9.0875283e-02,  2.9489502e-02,  2.1956262e-01,
            3.4037119e-01,  1.0848373e-01,  3.6547065e-02, -1.5146755e-01,
            5.6681294e-02,  6.6085658e-03,  1.9274153e-02,  1.9991216e-01,
           -1.5090431e-01,  9.0067700e-02,  5.1970325e-02,  2.0268182e-01,
            4.6885550e-02, -5.2929554e-02,  6.6083498e-02, -5.8406308e-02,
           -1.1952946e-01,  5.5076398e-02,  1.2351151e-04, -3.8982730e-02,
           -1.3962780e-01,  1.2789361e-01, -1.5078008e-01, -1.4386822e-01,
           -1.3026667e-01, -1.1459819e-01, -7.1221814e-02,  1.1928054e-01],
          dtype=float32)


```python
model.wv['대한민국']
```


    array([-2.4107585e-02, -7.4946046e-02,  1.5689157e-03,  1.7300507e-02,
            7.7659652e-02, -4.3071166e-02,  8.3631985e-02,  1.6745523e-01,
           -8.2903586e-02, -1.7553378e-02,  3.9016213e-02, -1.0054115e-01,
            4.1688729e-02,  1.7242630e-01, -1.8903978e-02,  1.2952442e-01,
            4.8356697e-02,  4.0910381e-01, -7.0913650e-02, -5.0823655e-02,
            1.4685905e-01, -1.2997684e-01,  2.2543812e-02, -3.7712879e-02,
            9.6920088e-02,  1.3099691e-01, -1.3746825e-01, -1.0660959e-01,
            1.1127534e-01,  1.2975276e-01, -2.8525587e-02, -1.2853998e-01,
           -8.3741836e-02, -9.9310517e-02, -2.4495709e-01, -4.1113162e-01,
            1.0418992e-02,  7.9034410e-02,  1.3711397e-01, -5.1028132e-02,
           -1.4102933e-01, -4.6473064e-02, -7.5484976e-02, -6.2391542e-02,
           -4.0519308e-02, -1.5226401e-01, -1.3334070e-01, -1.7248647e-01,
           -9.5049895e-02,  9.9440172e-02, -2.9708706e-02,  8.7483376e-02,
            8.1404611e-02,  1.3708833e-01, -1.1457676e-01, -9.5910830e-03,
           -6.4596653e-02, -2.4731688e-01,  3.0563422e-02,  1.2345860e-01,
           -3.4807574e-02,  1.6530770e-01,  1.2371200e-01, -1.2324062e-02,
            1.4210464e-01, -1.4213949e-01,  1.7249145e-01, -7.8410409e-02,
           -6.2629886e-02, -9.0875283e-02,  2.9489502e-02,  2.1956262e-01,
            3.4037119e-01,  1.0848373e-01,  3.6547065e-02, -1.5146755e-01,
            5.6681294e-02,  6.6085658e-03,  1.9274153e-02,  1.9991216e-01,
           -1.5090431e-01,  9.0067700e-02,  5.1970325e-02,  2.0268182e-01,
            4.6885550e-02, -5.2929554e-02,  6.6083498e-02, -5.8406308e-02,
           -1.1952946e-01,  5.5076398e-02,  1.2351151e-04, -3.8982730e-02,
           -1.3962780e-01,  1.2789361e-01, -1.5078008e-01, -1.4386822e-01,
           -1.3026667e-01, -1.1459819e-01, -7.1221814e-02,  1.1928054e-01],
          dtype=float32)

<Br>


```python
for index, word in enumerate(model.wv.index2word):
    if index == 10:
        break
    print(f"word {index}/{len(model.wv.index2word)} is {word}")
```

    word 0/1956 is 회담
    word 1/1956 is 일
    word 2/1956 is 북미
    word 3/1956 is 차
    word 4/1956 is 2
    word 5/1956 is 정상
    word 6/1956 is 2차
    word 7/1956 is 김
    word 8/1956 is 북한
    word 9/1956 is 미국

<br>

```python
print(model.wv.most_similar("대한민국"))
```

    [('대한', 0.9968054294586182), ('민국', 0.9958725571632385), ('터닝포인트', 0.9953158497810364), ('근', 0.9948737621307373), ('터닝', 0.994050920009613), ('마감', 0.993889570236206), ('국내증시', 0.9935024976730347), ('정치인', 0.992567777633667), ('글로벌', 0.9920015335083008), ('외국인', 0.9918369650840759)]

```python
print(model.wv.similarity("한국","미국"))
print(model.wv.similarity("한국","일본"))
print(model.wv.similarity("미국","일본"))
```

    0.19900209
    0.45370853
    0.7131777

<br>

```python
# storing and loading the model
model.save('tmp_word2vec.model')
model = Word2Vec.load("tmp_word2vec.model")
```


```python
# a:b = c: ? 
model.wv.most_similar(positive=['한국', '미국'], negative=['서울'])
```




    [('핵', 0.6568202376365662),
     ('미', 0.6307210922241211),
     ('북핵', 0.6297447681427002),
     ('북', 0.6209843754768372),
     ('북ㆍ미', 0.6095261573791504),
     ('ㆍ', 0.6072773337364197),
     ('성명', 0.601407527923584),
     ('정상회담', 0.6000897884368896),
     ('변', 0.5984941720962524),
     ('월말', 0.5965142250061035)]

<br>

### Keras 에서 word2vec 훈련된 모델 사용하기 (그림형제 동화 예제)


```python
import requests
import re
res = requests.get('https://www.gutenberg.org/files/2591/2591-0.txt') 
grimm = res.text[2801:530661]    # 그림형제의 동화 일부만 사용
grimm = re.sub(r'[^a-zA-Z\. ]', ' ', grimm)
sentences = grimm.split('. ')  # 문장 단위로 자름
data = [s.lower().split() for s in sentences]
```


```python
len(data)  # number of sentences
```




    3468




```python
print(data[:2])
```

    [['second', 'story', 'the', 'salad', 'the', 'story', 'of', 'the', 'youth', 'who', 'went', 'forth', 'to', 'learn', 'what', 'fear', 'was', 'king', 'grisly', 'beard', 'iron', 'hans', 'cat', 'skin', 'snow', 'white', 'and', 'rose', 'red', 'the', 'brothers', 'grimm', 'fairy', 'tales', 'the', 'golden', 'bird', 'a', 'certain', 'king', 'had', 'a', 'beautiful', 'garden', 'and', 'in', 'the', 'garden', 'stood', 'a', 'tree', 'which', 'bore', 'golden', 'apples'], ['these', 'apples', 'were', 'always', 'counted', 'and', 'about', 'the', 'time', 'when', 'they', 'began', 'to', 'grow', 'ripe', 'it', 'was', 'found', 'that', 'every', 'night', 'one', 'of', 'them', 'was', 'gone']]



```python
len(data), [len(data[i]) for i in range(10)]
```




    (3468, [55, 26, 19, 26, 23, 41, 48, 21, 18, 32])

<br>


```python
model = Word2Vec(data,         # 리스트 형태의 데이터
                 sg=1,         # 0: CBOW, 1: Skip-gram
                 size=100,     # 벡터 크기
                 window=3,     # 고려할 앞뒤 폭(앞뒤 3단어)
                 min_count=3,  # 사용할 단어의 최소 빈도(3회 이하 단어 무시)
                 workers=4)    # 동시에 처리할 작업 수(코어 수와 비슷하게 설정)
```


```python
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding
```


```python
model.wv.vectors.shape
```


    (2278, 100)

<br>


```python
NUM_WORDS, EMB_DIM = model.wv.vectors.shape

emb = Embedding(input_dim=NUM_WORDS, output_dim=EMB_DIM,
                trainable=False, weights=[model.wv.vectors])   # pre-trained weights
keras_model = Sequential()
keras_model.add(emb)
keras_model.summary()
```

    Model: "sequential"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    embedding (Embedding)        (None, None, 100)         227800    
    =================================================================
    Total params: 227,800
    Trainable params: 0
    Non-trainable params: 227,800
    _________________________________________________________________

```python
i = model.wv.index2word.index('princess'); i
```


    150


```python
keras_model(i)
```


    <tf.Tensor: shape=(100,), dtype=float32, numpy=
    array([ 0.17783275, -0.08774558,  0.09323502, -0.07242519,  0.20657292,
            0.03197189,  0.16224365, -0.15539818,  0.01945902,  0.09025002,
            0.17573437, -0.06973731,  0.11954256,  0.01534615, -0.37158963,
           -0.02288678,  0.01575112,  0.05553897,  0.17502707, -0.08478002,
           -0.21321169, -0.02501886, -0.27250507,  0.11078458, -0.21503918,
            0.27691916,  0.08938914, -0.08242173, -0.11613622, -0.20222554,
            0.04449013, -0.2559901 , -0.03157396,  0.0605896 , -0.15382041,
           -0.32093048, -0.04655121, -0.11977814,  0.04055993,  0.05245483,
            0.06287044,  0.09412678, -0.08609053, -0.13557422,  0.17934753,
           -0.09852032,  0.19760892, -0.006117  , -0.18912947, -0.09823273,
            0.1347044 ,  0.09133997, -0.06159864,  0.19485788,  0.04612151,
            0.31897318, -0.05191209, -0.06640887,  0.16498116,  0.144308  ,
           -0.30118912,  0.03162405, -0.04633264,  0.0622423 , -0.43333298,
           -0.01807583,  0.01292471, -0.14541416, -0.11620581, -0.05903935,
           -0.15599987,  0.14573784,  0.2046689 , -0.04925594, -0.09984405,
            0.15318435, -0.07531588,  0.07737457,  0.3179089 , -0.15710369,
            0.10327742,  0.01241986, -0.03649237,  0.05266789, -0.12705217,
           -0.1036229 ,  0.12542848,  0.03066339, -0.08837936, -0.08336505,
           -0.19087036,  0.1684691 , -0.01485604, -0.02229792,  0.1531238 ,
           -0.08944845,  0.03167198,  0.07329231,  0.1653206 ,  0.059671  ],
          dtype=float32)>


```python
keras_model(i) == keras_model.predict([i])
```


    <tf.Tensor: shape=(1, 1, 100), dtype=bool, numpy=
    array([[[ True,  True,  True,  True,  True,  True,  True,  True,  True,
              True,  True,  True,  True,  True,  True,  True,  True,  True,
              True,  True,  True,  True,  True,  True,  True,  True,  True,
              True,  True,  True,  True,  True,  True,  True,  True,  True,
              True,  True,  True,  True,  True,  True,  True,  True,  True,
              True,  True,  True,  True,  True,  True,  True,  True,  True,
              True,  True,  True,  True,  True,  True,  True,  True,  True,
              True,  True,  True,  True,  True,  True,  True,  True,  True,
              True,  True,  True,  True,  True,  True,  True,  True,  True,
              True,  True,  True,  True,  True,  True,  True,  True,  True,
              True,  True,  True,  True,  True,  True,  True,  True,  True,
              True]]])>


```python
model['princess'] == keras_model(i)
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:1: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      """Entry point for launching an IPython kernel.
    
    <tf.Tensor: shape=(100,), dtype=bool, numpy=
    array([ True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True,  True,  True,  True,  True,  True,  True,  True,  True,
            True])>

<br>

### Keras 에서 직접 Word2Vec 훈련도 가능

- http://doc.mindscale.kr/km/unstructured/11.html

### NLTK 이용한 문장의 유사도

- NLTK (Natural Language ToolKit) ëźě´ë¸ëŹëŚŹ ěŹěŠ


```python
!pip install nltk
```

    Requirement already satisfied: nltk in /usr/local/lib/python3.7/dist-packages (3.2.5)
    Requirement already satisfied: six in /usr/local/lib/python3.7/dist-packages (from nltk) (1.15.0)



```python
# simple exercise
import nltk
nltk.download('punkt')
sentence = "At eight o'clock on Thursday morning Arthur didn't feel very good."
tokens = nltk.word_tokenize(sentence)
tokens
```

    [nltk_data] Downloading package punkt to /root/nltk_data...
    [nltk_data]   Unzipping tokenizers/punkt.zip.





    ['At',
     'eight',
     "o'clock",
     'on',
     'Thursday',
     'morning',
     'Arthur',
     'did',
     "n't",
     'feel',
     'very',
     'good',
     '.']




```python
nltk.download('averaged_perceptron_tagger')
tagged = nltk.pos_tag(tokens)
tagged[0:6]
```

    [nltk_data] Downloading package averaged_perceptron_tagger to
    [nltk_data]     /root/nltk_data...
    [nltk_data]   Unzipping taggers/averaged_perceptron_tagger.zip.





    [('At', 'IN'),
     ('eight', 'CD'),
     ("o'clock", 'NN'),
     ('on', 'IN'),
     ('Thursday', 'NNP'),
     ('morning', 'NN')]

<br>


```python
# edit_distance: 문자열이 얼마나 다른지 편집거리를 이용해 유사도 판단
sentence_list = ["우리 모두 함께 놀자", "모두 같이 놀자", "놀자", "모두 다 같이"]

for i in sentence_list:
    print("'", i, "'")
    for j in sentence_list:
        print("\t", j, " : ", end='')
        print(nltk.edit_distance(i, j), )
    print()
```

    ' 우리 모두 함께 놀자 '
    	 우리 모두 함께 놀자  : 0
    	 모두 같이 놀자  : 5
    	 놀자  : 9
    	 모두 다 같이  : 7
    
    ' 모두 같이 놀자 '
    	 우리 모두 함께 놀자  : 5
    	 모두 같이 놀자  : 0
    	 놀자  : 6
    	 모두 다 같이  : 4
    
    ' 놀자 '
    	 우리 모두 함께 놀자  : 9
    	 모두 같이 놀자  : 6
    	 놀자  : 0
    	 모두 다 같이  : 7
    
    ' 모두 다 같이 '
    	 우리 모두 함께 놀자  : 7
    	 모두 같이 놀자  : 4
    	 놀자  : 7
    	 모두 다 같이  : 0


​    <br>


```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
 
sentence_list = ['우리 모두 함께 놀자','모두 같이 놀자']
 
tfidf = TfidfVectorizer()
tfidf_vec = tfidf.fit_transform(sentence_list)

count = CountVectorizer()
count_vec = count.fit_transform(sentence_list)
```


```python
tfidf_vec[0].toarray()
```


    array([[0.        , 0.40993715, 0.40993715, 0.57615236, 0.57615236]])

<br>


```python
from sklearn.metrics.pairwise import cosine_similarity

tfidf_cosine = cosine_similarity(tfidf_vec[0].toarray(), tfidf_vec[1].toarray())[0][0]
count_cosine = cosine_similarity(count_vec[0].toarray(), count_vec[1].toarray())[0][0]

print("tfidf consine similarity : ", tfidf_cosine)
print("countvectorizer consine similarity : ", count_cosine)
```

    tfidf consine similarity :  0.4112070550676187
    countvectorizer consine similarity :  0.5773502691896258

<br>

```python
# Jaccard similarity
sentence_list = ['우리 모두 함께 놀자','모두 같이 놀자']
def get_jaccard_sim(str1, str2): 
    a = set(str1.split()) 
    b = set(str2.split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))

get_jaccard_sim(sentence_list[0], sentence_list[1])
```


    0.4

<br>

<br>

## Word2Vec - English

- https://machinelearningmastery.com/develop-word-embeddings-python-gensim/


```python
from gensim.models import Word2Vec
# define training data
sentences = [['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec'],
			['this', 'is', 'the', 'second', 'sentence'],
			['yet', 'another', 'sentence'],
			['one', 'more', 'sentence'],
			['and', 'the', 'final', 'sentence']]
# train model
model = Word2Vec(sentences, min_count=1, size=5)
# summarize the loaded model
print(model)
```

    Word2Vec(vocab=14, size=5, alpha=0.025)



```python
# summarize vocabulary (to see learned vocabulary of tokens (words) )
words = list(model.wv.vocab)
print(words)
# access vector for one word
print(model['sentence'])
# save model
model.save('model.bin')
# load model
new_model = Word2Vec.load('model.bin')
print(new_model)
```

    ['this', 'is', 'the', 'first', 'sentence', 'for', 'word2vec', 'second', 'yet', 'another', 'one', 'more', 'and', 'final']
    [-0.02676655 -0.01335588 -0.03907884 -0.0286011   0.0530019 ]
    Word2Vec(vocab=14, size=5, alpha=0.025)


    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:5: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      """



```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
%matplotlib inline

X = model[model.wv.vocab]
pca = PCA(n_components=2)
result = pca.fit_transform(X)

plt.scatter(result[:, 0], result[:, 1])
words = list(model.wv.vocab)
for i, word in enumerate(words):
    plt.annotate(word, xy=(result[i, 0], result[i, 1]))
plt.show()
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:5: DeprecationWarning: Call to deprecated `__getitem__` (Method will be removed in 4.0.0, use self.wv.__getitem__() instead).
      """




![output_49_1](https://user-images.githubusercontent.com/70505378/143829866-9f2e1a5c-a90b-49ef-a683-b26b3b8f8827.png)
    

<br>

```python
# you can continue training it later
model.train([["hello", "world"]], total_examples=1, epochs=1)
```




    (0, 2)

<br>

<br>

## Using pre-trained Word2Vec model

- GoogleNews-vectors-negative300.bin
- Korean version "ko.vec" available


```python
import gensim
from gensim.models import word2vec
from gensim.models import KeyedVectors
from sklearn.metrics.pairwise import cosine_similarity
```


```python
!wget -P ./ -c "https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz"
```

    --2021-11-20 04:39:38--  https://s3.amazonaws.com/dl4j-distribution/GoogleNews-vectors-negative300.bin.gz
    Resolving s3.amazonaws.com (s3.amazonaws.com)... 52.217.193.96
    Connecting to s3.amazonaws.com (s3.amazonaws.com)|52.217.193.96|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 1647046227 (1.5G) [application/x-gzip]
    Saving to: â./GoogleNews-vectors-negative300.bin.gzâ
    
    GoogleNews-vectors- 100%[===================>]   1.53G  69.6MB/s    in 24s     
    
    2021-11-20 04:40:02 (66.2 MB/s) - â./GoogleNews-vectors-negative300.bin.gzâ saved [1647046227/1647046227]


​    


```python
!ls -l 
```

    total 1608452
    -rw-r--r-- 1 root root 1647046227 Mar  5  2015 GoogleNews-vectors-negative300.bin.gz
    drwxr-xr-x 1 root root       4096 Nov 18 14:36 sample_data



```python
EMBEDDING_FILE = './GoogleNews-vectors-negative300.bin.gz'
word_vectors = KeyedVectors.load_word2vec_format(EMBEDDING_FILE, binary=True)
```


```python
word_vectors.vectors.shape
```




    (3000000, 300)




```python
v_apple = word_vectors["apple"] 
v_mango = word_vectors["mango"]
print(v_apple.shape)
print(v_mango.shape)
cosine_similarity([v_mango],[v_apple])
```

    (300,)
    (300,)





    array([[0.57518554]], dtype=float32)




```python
word_vectors.most_similar(["apple"]), word_vectors.most_similar("tiger")
```




    ([('apples', 0.7203598022460938),
      ('pear', 0.6450696587562561),
      ('fruit', 0.6410146355628967),
      ('berry', 0.6302294731140137),
      ('pears', 0.6133961081504822),
      ('strawberry', 0.6058261394500732),
      ('peach', 0.6025873422622681),
      ('potato', 0.596093475818634),
      ('grape', 0.5935864448547363),
      ('blueberry', 0.5866668224334717)],
     [('tigers', 0.8028031587600708),
      ('elephant', 0.6681442856788635),
      ('rhino', 0.6406095027923584),
      ('elephants', 0.6400991678237915),
      ('panther', 0.6312947273254395),
      ('Bengal_tiger', 0.6192330718040466),
      ('leopard', 0.6132040023803711),
      ('Siberian_tiger', 0.6061223745346069),
      ('leopard_cub', 0.6012793183326721),
      ('tigress', 0.5982028245925903)])



- king - man + woman


```python
word_vectors.most_similar(positive=["king","Woman"], negative=["man"])
```




    [('queen', 0.5196164846420288),
     ('princess', 0.40039342641830444),
     ('Beauty_Pageants', 0.39205846190452576),
     ('crown_prince', 0.38563376665115356),
     ('monarch', 0.3831227421760559),
     ('NYC_anglophiles_aflutter', 0.38275885581970215),
     ('queendom', 0.38235384225845337),
     ('Princess_Sirindhorn', 0.37608852982521057),
     ('kings', 0.3758448660373688),
     ('sultan', 0.37523317337036133)]




```python
words = ["soccer", "football", "baseball", "volleyball", "basketball", "tennis",
         "persimmon", "softball", "apple", "hockey", "orange", "pear", "strawberry",
         "eat", "drink", "taste", "talk", "speak", "study", "research", "have", "take"]
mat = word_vectors[words]
mat.shape
```




    (22, 300)




```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
xys = pca.fit_transform(mat)
xs = xys[:,0]
ys = xys[:,1]

plt.figure(figsize=(12,6))
plt.scatter(xs, ys)

for i, word in enumerate(words):
    plt.annotate(word, xy=(xs[i], ys[i]), rotation=30)
plt.show()
```


![output_62_0](https://user-images.githubusercontent.com/70505378/143829871-07caaf5e-2a56-404f-b03b-2f663f7f5fea.png)
    


- Pre-trained Word2Vec
  - for english: (about 3 GB, and takes long to download): https://code.google.com/archive/p/word2vec 
  - word vectors of 30+ languages: https://github.com/Kyubyong/wordvectors

<br>

<br>

## Glove - pretrained word embedder


```python
import gensim.downloader as api

glove_model = api.load('glove-twitter-25')
sample_glove_embedding=glove_model['computer']
```


```python
sample_glove_embedding
```


```python
words = ["soccer", "football", "baseball", "volleyball", "basketball", "tennis",
         "persimmon", "softball", "apple", "hockey", "orange", "pear", "strawberry",
         "eat", "drink", "taste", "talk", "speak", "study", "research", "have", "take"]
mat = glove_model[words]
mat.shape
```


```python
import matplotlib.pyplot as plt
```


```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
xys = pca.fit_transform(mat)
xs = xys[:,0]
ys = xys[:,1]

plt.figure(figsize=(12,6))
plt.scatter(xs, ys)

for i, word in enumerate(words):
    plt.annotate(word, xy=(xs[i], ys[i]), rotation=30)
plt.show()
```

<br>

### difference between Glove and Word2Vec

- Both word2vec and glove enable us to represent a word in the form of a vector (often called embedding). They are the two most popular algorithms for word embeddings that bring out the semantic similarity of words that captures different facets of the meaning of a word.
- Word2vec embeddings are based on training a shallow feedforward neural network while glove embeddings are learnt based on matrix factorization techniques.
- Glove model is based on leveraging global word to word co-occurance counts leveraging the entire corpus. Word2vec on the other hand leverages co-occurance within local context (neighbouring words).
- In practice, however, both these models give similar results for many tasks. â_Factors such as the dataset on which these models are trained, length of the vectors and so on seem to have a bigger impact than the models themselves. For instance, if I am using these models to derive  the features for a medical application, I can significantly improve performance by training on dataset from the medical domain.

<br>

<br>

## Doc2Vec

- extension of Word2Vec 
- https://lovit.github.io/nlp/representation/2018/03/26/word_doc_embedding/
- doc2vec을 만드는 과정에서 word2vec 모델이 필요하므로 필연적으로 word2vec도 생성


```python
#Import all the dependencies
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from nltk.tokenize import word_tokenize
```


```python
import nltk
nltk.download('punkt')
```


```python
data = ["I love machine learning. Its awesome.",
        "I love coding in python",
        "I love building chatbots",
        "they chat amagingly well"]

tagged_data = [TaggedDocument(words=word_tokenize(_d.lower()), tags=[str(i)]) for i, _d in enumerate(data)]
```


```python
tagged_data
```


```python
max_epochs = 100
vec_size = 5
alpha = 0.025

model = Doc2Vec(vector_size=vec_size,
                alpha=alpha, 
                min_alpha=0.00025,
                min_count=1,
                dm =1)    # dm=1:preserves word order, 0: do not preserve order
  
model.build_vocab(tagged_data)   # build a vocabulary
```


```python
model.corpus_count, model.epochs
```


```python
for epoch in range(max_epochs):
    model.train(tagged_data,
                total_examples=model.corpus_count,
                epochs=model.epochs)
    
    model.alpha -= 0.0002           # decrease the learning rate
    model.min_alpha = model.alpha   # fix the learning rate, no decay

model.save("d2v.model")
```


```python
model= Doc2Vec.load("d2v.model")
```


```python
# to infer a new document vector
test_data = word_tokenize("I love chatbots".lower())
new_v = model.infer_vector(test_data)
print(test_data)
print("new vector inferred: ", new_v)
```


```python
model.wv.similar_by_vector(new_v)
```


```python
# to find most similar doc using tags (it uses word-vectors.)
similar_doc = model.docvecs.most_similar('1')
print(similar_doc)
```


```python
# to find vector of doc in training data using tags or in other words, printing the vector of document at index 1 in training data
print(model.docvecs['1'])
```

