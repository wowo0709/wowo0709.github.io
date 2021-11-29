---
layout: single
title: "[Deep Learning] Naver Movie Sentiment Analysis"
categories: ['AI', 'DeepLearning']
toc: true
toc_sticky: true
tag: ['LSTM']
---



## Naver Movie Sentiment Analysis

- 감성분석
  - 네이버 영화평점 (Naver sentiment movie corpus v.1.0) 데이터(https://github.com/e9t/nsmc)
  - 영화 리뷰 20만건이 저장됨. 각 평가 데이터는 0(부정), 1(긍정)으로 label 됨.


- 한글 자연어 처리
  - KoNLPy(“코엔엘파이”라고 읽습니다)는 한국어 정보처리를 위한 파이썬 패키지입니다.
  - konlpy 패키지에서 제공하는 Twitter라는 문서 분석 라이브러리 사용 (트위터 분석 뿐 아니라 한글 텍스트 
    처리도 가능)
  - colab 사용 권장


```python
!pip install konlpy
```

```python
# 패키지 설치
import konlpy
import pandas as pd
import numpy as np
from konlpy.tag import Twitter # Okt
from sklearn.feature_extraction.text import TfidfVectorizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation
from sklearn import model_selection, metrics

# 토큰 파서(품사별)
def twitter_tokenizer(text):
    return Twitter().morphs(text)
```


```python
twitter_tokenizer("Welcome to data science world!...한국말도 똑 같아요...")
```

    ['Welcome',
     'to',
     'data',
     'science',
     'world',
     '!...',
     '한국말',
     '도',
     '똑',
     '같아요',
     '...']

<br>


```python
!curl -L https://bit.ly/2X9Owwr -o ratings_train.txt
!curl -L https://bit.ly/2WuLd5I -o ratings_test.txt
```

      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   152  100   152    0     0   1831      0 --:--:-- --:--:-- --:--:--  1831
    100   148    0   148    0     0    452      0 --:--:-- --:--:-- --:--:--   452
    100   318  100   318    0     0    624      0 --:--:-- --:--:-- --:--:--  1939
    100 14.0M  100 14.0M    0     0  12.5M      0  0:00:01  0:00:01 --:--:-- 12.5M
      % Total    % Received % Xferd  Average Speed   Time    Time     Time  Current
                                     Dload  Upload   Total   Spent    Left  Speed
    100   151  100   151    0     0   3081      0 --:--:-- --:--:-- --:--:--  3081
    100   147    0   147    0     0    578      0 --:--:-- --:--:-- --:--:--   907
    100   318  100   318    0     0    757      0 --:--:-- --:--:-- --:--:--   757
    100 4827k  100 4827k    0     0  5510k      0 --:--:-- --:--:-- --:--:-- 5510k

```python
# 데이터 로드
df_train = pd.read_csv('ratings_train.txt', delimiter='\t', keep_default_na=False)
df_test = pd.read_csv('ratings_test.txt', delimiter='\t', keep_default_na=False)

```


```python
df_train[:5]
```

![image-20211129164010036](https://user-images.githubusercontent.com/70505378/143828394-97242b17-a995-41a1-86c8-42ee368f9f9c.png)






```python
df_test[:5]
```



![image-20211129164031719](https://user-images.githubusercontent.com/70505378/143828400-810d2c19-e55c-4370-a3a6-988920972e83.png)

<br>


```python
df_train['document'].values == df_train['document'].to_numpy()
```


    array([ True,  True,  True, ...,  True,  True,  True])


```python
text_train, y_train = df_train['document'].to_numpy(), df_train['label'].to_numpy()
text_test, y_test = df_test['document'].to_numpy(), df_test['label'].to_numpy()
```


```python
text_train.shape, y_train.shape, text_test.shape, y_test.shape
```


    ((150000,), (150000,), (50000,), (50000,))

<br>


```python
# too much... -> let's take few of them
text_train, y_train = text_train[:2000], y_train[:2000]
text_test, y_test = text_test[:1000], y_test[:1000]
```


```python
y_train.shape, y_test.shape
```


    ((2000,), (1000,))


```python
y_train.mean(), y_test.mean()    # check distribution of classes 1 and 0
```


    (0.4945, 0.508)

<br>


```python
cv = TfidfVectorizer(tokenizer=twitter_tokenizer, max_features=3000)
X_train = cv.fit_transform(text_train)
X_test = cv.transform(text_test) # cv.fit_transform(text_test) (X)
```

```python
X_train.shape, y_train.shape, X_test.shape, y_test.shape
```


    ((2000, 3000), (2000,), (1000, 3000), (1000,))


```python
cv.get_feature_names()[-5:]
```


    ['????', '???', '???', '?????', '????']

<br>

<br>

## Linear Classification


```python
from sklearn.linear_model import SGDClassifier
clf = SGDClassifier()
result = clf.fit(X_train,y_train)
feature_names = cv.get_feature_names()
print("Training : ", result.score(X_train, y_train))
print("Testing : ", result.score(X_test, y_test))
```

    Training :  0.985
    Testing :  0.739

<br>

<br>

## Logistic Regression


```python
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
result = lr.fit(X_train,y_train)
feature_names = cv.get_feature_names()
print("Training : ", result.score(X_train, y_train))
print("Testing : ", result.score(X_test, y_test))
```

    Training :  0.916
    Testing :  0.771

<br>

<br>

## MLP


```python
# use one-hot encoded target (2 multi-class classification)
from tensorflow.keras.utils import to_categorical
y_train_ohe = to_categorical(y_train)
y_test_ohe = to_categorical(y_test)
max_words = X_train.shape[1]
batch_size = 100
nb_epoch = 5

model = Sequential()
model.add(Dense(64, input_shape=(max_words,), activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train.toarray(), y_train_ohe, epochs=nb_epoch, 
          batch_size=batch_size) 
print("Training : ", model.evaluate(X_train.toarray(), y_train_ohe))
print("Testing : ", model.evaluate(X_test.toarray(), y_test_ohe))
```


```python
# use binary target (binary classifiaction)
max_words = X_train.shape[1]
batch_size = 100
nb_epoch = 5

model = Sequential()
model.add(Dense(64, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam',
              metrics=['accuracy'])

model.fit(X_train.toarray(), y_train, epochs=nb_epoch, 
          batch_size=batch_size) 
print("Training : ", model.evaluate(X_train.toarray(), y_train))
print("Testing : ", model.evaluate(X_test.toarray(), y_test))
```

    Epoch 1/5
    20/20 [==============================] - 1s 3ms/step - loss: 0.6910 - accuracy: 0.5560
    Epoch 2/5
    20/20 [==============================] - 0s 3ms/step - loss: 0.6697 - accuracy: 0.7800
    Epoch 3/5
    20/20 [==============================] - 0s 4ms/step - loss: 0.5963 - accuracy: 0.8635
    Epoch 4/5
    20/20 [==============================] - 0s 3ms/step - loss: 0.4505 - accuracy: 0.8945
    Epoch 5/5
    20/20 [==============================] - 0s 3ms/step - loss: 0.2864 - accuracy: 0.9310
    63/63 [==============================] - 1s 3ms/step - loss: 0.2006 - accuracy: 0.9540
    Training :  [0.20061413943767548, 0.9539999961853027]
    32/32 [==============================] - 0s 3ms/step - loss: 0.4924 - accuracy: 0.7530
    Testing :  [0.4923769235610962, 0.753000020980835]

<br>

<br>

## RNN


```python
# just for checking
X_train.A[0] == X_train.toarray()[0]
```


    array([ True,  True,  True, ...,  True,  True,  True])


```python
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```


    ((2000, 3000), (1000, 3000), (2000,), (1000,))


```python
# RNN ??? ?? ??? ???
X_train_rnn = X_train.A.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test_rnn = X_test.A.reshape((X_test.shape[0], 1, X_test.shape[1]))

print(X_train_rnn.shape, X_test_rnn.shape)
```

    (2000, 1, 3000) (1000, 1, 3000)

<br>

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense

model = Sequential()
model.add(SimpleRNN(128, 
                    input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]), 
                    return_sequences=True))
# return_sequences: return the last output in the output sequence, or the full sequence
# By this, it is possible to access the hidden state output for each input time step.
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(SimpleRNN(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation="softmax"))
model.compile(loss='categorical_crossentropy', 
              optimizer='adam', metrics=['accuracy'])   
```


```python
model.summary()
```

    Model: "sequential_2"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    simple_rnn (SimpleRNN)       (None, 1, 128)            400512    
    _________________________________________________________________
    activation_1 (Activation)    (None, 1, 128)            0         
    _________________________________________________________________
    dropout_4 (Dropout)          (None, 1, 128)            0         
    _________________________________________________________________
    simple_rnn_1 (SimpleRNN)     (None, 128)               32896     
    _________________________________________________________________
    activation_2 (Activation)    (None, 128)               0         
    _________________________________________________________________
    dropout_5 (Dropout)          (None, 128)               0         
    _________________________________________________________________
    dense_6 (Dense)              (None, 2)                 258       
    =================================================================
    Total params: 433,666
    Trainable params: 433,666
    Non-trainable params: 0
    _________________________________________________________________

<br>

```python
model.fit(X_train_rnn, y_train_ohe, batch_size = 100,
          epochs=nb_epoch)
```

    Epoch 1/5
    20/20 [==============================] - 2s 24ms/step - loss: 0.6887 - accuracy: 0.5465
    Epoch 2/5
    20/20 [==============================] - 0s 6ms/step - loss: 0.6325 - accuracy: 0.7965
    Epoch 3/5
    20/20 [==============================] - 0s 6ms/step - loss: 0.4334 - accuracy: 0.8890
    Epoch 4/5
    20/20 [==============================] - 0s 6ms/step - loss: 0.2157 - accuracy: 0.9240
    Epoch 5/5
    20/20 [==============================] - 0s 7ms/step - loss: 0.1026 - accuracy: 0.9690

<br>


```python
y_pred = np.argmax(model.predict(X_test_rnn), axis=1)
print("accuracy score: ", metrics.accuracy_score(y_test, y_pred))
print("Training : ", model.evaluate(X_train_rnn, y_train_ohe))
print("Testing : ", model.evaluate(X_test_rnn, y_test_ohe))
```

    accuracy score:  0.765
    63/63 [==============================] - 1s 4ms/step - loss: 0.0617 - accuracy: 0.9870
    Training :  [0.061724983155727386, 0.9869999885559082]
    32/32 [==============================] - 0s 3ms/step - loss: 0.5956 - accuracy: 0.7650
    Testing :  [0.5956090688705444, 0.7649999856948853]

<br>

<br>

## LSTM

- https://colah.github.io/posts/2015-08-Understanding-LSTMs/


```python
from keras.layers import LSTM
model = Sequential()
model.add(LSTM(128, 
               input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]), 
               return_sequences=True))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```


```python
model.fit(X_train_rnn, y_train_ohe, batch_size = 100,
          epochs=nb_epoch)
```

    Epoch 1/5
    20/20 [==============================] - 6s 10ms/step - loss: 0.6931 - accuracy: 0.5375
    Epoch 2/5
    20/20 [==============================] - 0s 10ms/step - loss: 0.6903 - accuracy: 0.8005
    Epoch 3/5
    20/20 [==============================] - 0s 10ms/step - loss: 0.6703 - accuracy: 0.8725
    Epoch 4/5
    20/20 [==============================] - 0s 10ms/step - loss: 0.5868 - accuracy: 0.8930
    Epoch 5/5
    20/20 [==============================] - 0s 10ms/step - loss: 0.4041 - accuracy: 0.9105

<br>


```python
y_pred = np.argmax(model.predict(X_test_rnn), axis=1)
print("accuracy score: ", metrics.accuracy_score(y_test, y_pred))
print("Training : ", model.evaluate(X_train_rnn, y_train_ohe))
print("Testing : ", model.evaluate(X_test_rnn, y_test_ohe))
```

    accuracy score:  0.774
    63/63 [==============================] - 1s 5ms/step - loss: 0.2751 - accuracy: 0.9315
    Training :  [0.27514004707336426, 0.9315000176429749]
    32/32 [==============================] - 0s 5ms/step - loss: 0.4868 - accuracy: 0.7740
    Testing :  [0.4867841601371765, 0.7739999890327454]

<br>

<br>

## GRU


```python
from keras.layers import GRU
model = Sequential()
model.add(GRU(128, input_shape=(X_train_rnn.shape[1], X_train_rnn.shape[2]), return_sequences=True))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(GRU(128))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
```


```python
model.fit(X_train_rnn, y_train_ohe, batch_size = 100,
          epochs=nb_epoch)
```

    Epoch 1/5
    20/20 [==============================] - 4s 10ms/step - loss: 0.6921 - accuracy: 0.5170
    Epoch 2/5
    20/20 [==============================] - 0s 10ms/step - loss: 0.6716 - accuracy: 0.7150
    Epoch 3/5
    20/20 [==============================] - 0s 10ms/step - loss: 0.5628 - accuracy: 0.8495
    Epoch 4/5
    20/20 [==============================] - 0s 11ms/step - loss: 0.3357 - accuracy: 0.9130
    Epoch 5/5
    20/20 [==============================] - 0s 10ms/step - loss: 0.1801 - accuracy: 0.9390

<br>


```python
y_pred = np.argmax(model.predict(X_test_rnn), axis=1)
print("accuracy score: ", metrics.accuracy_score(y_test, y_pred))
print("Training : ", model.evaluate(X_train_rnn, y_train_ohe))
print("Testing : ", model.evaluate(X_test_rnn, y_test_ohe))
```

    accuracy score:  0.766
    63/63 [==============================] - 1s 5ms/step - loss: 0.1123 - accuracy: 0.9670
    Training :  [0.11229105293750763, 0.9670000076293945]
    32/32 [==============================] - 0s 5ms/step - loss: 0.5520 - accuracy: 0.7660
    Testing :  [0.552036702632904, 0.765999972820282]

<br>

<br>

## Exercise

### 한국어 불용어 처리

- 한국어  불용어 확인은 형태소 분석 라이브러리인 KonLPy 를 이용하면 됨.
- (예) 한국어 품사 중 조사를 추출하는 예
- pos (part-of-speech): 품사 (명사, 동사, ...)


```python
from konlpy.tag import Twitter
```


```python
Twitter().morphs("텍스트 데이터를 이용해서 불용어 사전을 구축하기 위한 간단 예제")
```

    ['텍스트',
     '데이터',
     '를',
     '이용',
     '해서',
     '불',
     '용어',
     '사전',
     '을',
     '구축',
     '하기',
     '위',
     '한',
     '간단',
     '예제']

<br>


```python
Twitter().pos("텍스트 데이터를 이용해서 불용어 사잔을 구축하기 위한 간단 예제")
```

    [('텍스트', 'Noun'),
     ('데이터', 'Noun'),
     ('를', 'Josa'),
     ('이용', 'Noun'),
     ('해서', 'Verb'),
     ('불', 'Noun'),
     ('용어', 'Noun'),
     ('사잔', 'Noun'),
     ('을', 'Josa'),
     ('구축', 'Noun'),
     ('하기', 'Verb'),
     ('위', 'Noun'),
     ('한', 'Josa'),
     ('간단', 'Noun'),
     ('예제', 'Noun')]

<br>


```python
Twitter().pos("텍스트 데이터를 이용해서 불용어 사전을 구축하기 위한 간단 예제", norm=True)   # norm - 오타 수정 (사잔->사전)
```

    [('텍스트', 'Noun'),
     ('데이터', 'Noun'),
     ('를', 'Josa'),
     ('이용', 'Noun'),
     ('해서', 'Verb'),
     ('불', 'Noun'),
     ('용어', 'Noun'),
     ('사전', 'Noun'),
     ('을', 'Josa'),
     ('구축', 'Noun'),
     ('하기', 'Verb'),
     ('위', 'Noun'),
     ('한', 'Josa'),
     ('간단', 'Noun'),
     ('예제', 'Noun')]

<br>


```python
Twitter().nouns("텍스트 데이터를 이용해서 불용어 사전을 구축하기 위한 간단 예제")
```

    ['텍스트', '데이터', '이용', '불', '용어', '사전', '구축', '위', '간단', '예제']

- norm: 오타수정, stem: 어근 찾기

<br>


```python
from konlpy.tag import Twitter

word_tags = Twitter().pos("텍스트 데이터를 이용해서 불용어 사전을 구축하기 위한 간단 예제", norm=True, stem=True)
print(word_tags)
stop_words = [word[0] for word in word_tags if word[1]=="Josa"]
print (stop_words)
```

    [('텍스트', 'Noun'), ('데이터', 'Noun'), ('를', 'Josa'), ('이용', 'Noun'), ('하다', 'Verb'), ('불', 'Noun'), ('용어', 'Noun'), ('사전', 'Noun'), ('을', 'Josa'), ('구축', 'Noun'), ('하다', 'Verb'), ('위', 'Noun'), ('한', 'Josa'), ('간단', 'Noun'), ('예제', 'Noun')]
    ['를', '을', '한']

<br>

### Pickling

- “Pickling” is the process whereby a Python object hierarchy is converted into a byte stream, and “unpickling” is the inverse operation, whereby a byte stream (from a binary file or bytes-like object) is converted back into an object hierarchy.
- Pickling (and unpickling) is alternatively known as “serialization”, “marshalling,” or “flattening”; however, to avoid confusion, the terms “pickling” and “unpickling” are being mostly used.
- Comparison with json
  - There are fundamental differences between the pickle protocols and JSON (JavaScript Object Notation):

  - JSON is a text serialization format (it outputs unicode text, although most of the time it is then encoded to utf-8), while pickle is a binary serialization format;
  - JSON is human-readable, while pickle is not;
  - JSON is interoperable and widely used outside of the Python ecosystem, while pickle is Python-specific;
  - JSON, by default, can only represent a subset of the Python built-in types, and no custom classes; pickle can represent an extremely large number of Python types (many of them automatically, by clever usage of Python’s introspection facilities; complex cases can be tackled by implementing specific object APIs);
  - Unlike pickle, deserializing untrusted JSON does not in itself create an arbitrary code execution vulnerability.


```python
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
cv_ = TfidfVectorizer(tokenizer=twitter_tokenizer, max_features=10)
```


```python
import os
import pickle
if not os.path.isfile("X_data.pickle"): 
    print('file does not exist')
    X_data_pre = cv_.fit_transform(text_train)
    pickle.dump(X_data_pre, open("X_data.pickle", "wb"))
```

<br>

```python
# ??? tfidf vector ??? ??
with open('X_data.pickle', 'rb') as f:
    X_data_post = pickle.load(f)
```


```python
X_data_pre.toarray() == X_data_post.toarray()
```


    array([[ True,  True,  True, ...,  True,  True,  True],
           [ True,  True,  True, ...,  True,  True,  True],
           [ True,  True,  True, ...,  True,  True,  True],
           ...,
           [ True,  True,  True, ...,  True,  True,  True],
           [ True,  True,  True, ...,  True,  True,  True],
           [ True,  True,  True, ...,  True,  True,  True]])

<br>

### Vectorizing

- BoW (Bag of Words)
  - document-term matrix
  - tfidf matrix


```python
from sklearn.feature_extraction.text import CountVectorizer
corpus = [
        'This is the first document.',
        'This document is the second document.',
        'And this is the third one.',
        'Is this the first document?',
]
vectorizer1 = CountVectorizer()
X = vectorizer1.fit_transform(corpus)
print(vectorizer1.get_feature_names())
print(X.toarray())
```

    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    [[0 1 1 1 0 0 1 0 1]
     [0 2 0 1 0 1 1 0 1]
     [1 0 0 1 1 0 1 1 1]
     [0 1 1 1 0 0 1 0 1]]

<br>

```python
vectorizer2 = TfidfVectorizer()
X = vectorizer2.fit_transform(corpus)
print(vectorizer2.get_feature_names())
print(X.toarray().round(2))
```

    ['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this']
    [[0.   0.47 0.58 0.38 0.   0.   0.38 0.   0.38]
     [0.   0.69 0.   0.28 0.   0.54 0.28 0.   0.28]
     [0.51 0.   0.   0.27 0.51 0.   0.27 0.51 0.27]
     [0.   0.47 0.58 0.38 0.   0.   0.38 0.   0.38]]

