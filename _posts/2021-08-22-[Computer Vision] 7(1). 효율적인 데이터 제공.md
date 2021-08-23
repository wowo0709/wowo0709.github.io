---
layout: single
title: "[Computer Vision] 7(1). 효율적인 데이터 제공"
---



<br>

# 효율적인 데이터 제공

훈련 데이터는 방해받지 않고 네트워크에 유입될 수 있어야 한다. 그러나 때로는 데이터셋은 종종 복잡한 구조를 갖거나 이기종 디바이스에 저장되기도 해 그 콘텐츠를 모델에 효율적으로 공급하는 프로세스를 복잡하게 만들 수 있다. 

다행히도 텐서플로는 최적화된 파이프라인을 설정하기 위한 풍부한 프레임워크로 **tf.data**를 제공한다. 

입력 파이프라인을 잘 정의해두면 모델을 훈련시키는 데 필요한 시간을 상당히 절약할 수 있을 뿐 아니라 네트워크 성능을 높일 수 있는 최적의 설정을 찾아가는 데 필요한 훈련 샘플의 전처리 과정을 수월하게 만들어준다. 

이번 포스팅에서는 이러한 최적의 파이프라인을 구성하는 방법을 알아보고 텐서플로 tf.data API를 자세히 살펴본다. 

> 이번 포스팅에서 설명하는 내용을 보여주는 주피터 노트북과 관련 소스 파일은 아래 주소에서 확인할 수 있다. 
>
> https://github.com/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2/tree/master/Chapter07

<br>

### 텐서플로 데이터 API 소개

---

#### 텐서플로 데이터 API의 이해

tf.data를 자세히 알아보기 전에 딥러닝 모델 훈련에 필요한 몇 가지 경우를 생각해보자. 

<br>

**빠르고 데이터가 많이 필요한 모델에 데이터 공급하기**

하드웨어의 성능이 좋아지면서, 신경망은 전형적인 입력 파이프라인이 훈련 배치를 '생산'하는 것보다 더 빨리 '소비'하게 되는 경향이 있다. 특히 컴퓨터 비전에서 그렇다. 

이미지 데이터셋은 보통 전체를 전처리하기에는 너무 무거우며, 그때그때 상황에 따라 이미지 파일을 읽고 디코딩하면 심각한 지연이 발생할 수 있다. 

<br>

**느긋한 구조에서 얻은 영감**

**tf.data API**는 신경망에 데이터를 공급하기 위한 명확하고 효율적인 프레임워크로, **'현재 단계를 완료하기 전에 다음 단계를 위해 데이터를 전달'**할 수 있는 입력 파이프라인을 정의하는 데 목적이 있다. 

tf.data API로 구성된 파이프라인은 함수형 언어의 '느긋한 리스트(lazy list)'와 비슷하다. 

이 파이프라인은 **필요에 따라 호출**되는 방식으로 거대하거나 무한한 데이터셋을 배치 단위로 반복할 수 있다. 이 파이프라인은 데이터를 처리하고 그 흐름을 제어하기 위한 map(), reduce(), filter(), repeat() 등의 연산을 제공하며, 계산 성능을 위해 C++을 백본으로 두고 있다. 

<br>

#### 텐서플로 데이터 파이프라인 구조

**추출 - 변환 - 적재**

tf.data API 가이드는 훈련을 위한 데이터 파이프라인과 추출, 변환, 전재의 ETL(Extract, Transform, Load) 프로세스를 병렬로 만든다. 

ETL은 컴퓨터 사이언스에서 데이터 처리를 위한 일반적인 패러다임이다. 

![ETL 파이프 라인 이해](https://ichi.pro/assets/images/max/724/1*Q25w6qRh89gKmKrPaq9osg.jpeg)

**추출** 단계에서는 데이터 소스를 선택하고 그 콘텐츠를 추출한다. 

이 소스는 문서로 명확하게 나열되거나(모든 이미지에 대한 파일명을 포함하는 csv 파일 등) 암묵적으로 나열될 수 있다(특정 이미지 폴더에 저장된 모든 이미지를 사용). 데이터 소스는 다양한 기기에 저장될 수 있으며, 이 다양한 소스를 리스트로 만들고 그 콘텐츠를 추출하는 것도 추출기가 해야 할 일이다. 

지도 학습 방식으로 신경망을 훈련시키기 위해서는 **'이미지를 따라 주석/실제 정보도 추출'**해야 한다. 

<br>

**변환** 단계에서는 가져온 데이터 샘플을 변환한다. 가장 보편적인 방법 중 하나는 추출된 데이터 샘플을 공통 포맷으로 파싱하는 것으로, 예를 들면 이미지 파일로부터 읽어들인 바이트를 행렬 표현으로 파싱하는 것이다. 

그 외에도 이미지를 동일한 차원으로 **'잘라내거나(cropping)/척도를 조정(scaling)**하는 일'이나 다양한 랜덤 연산으로 이미지를 **'보강(augmentation)'**할 수 있다. 

또한 지도 학습의 경우 **주석도 변환**돼야 한다. 주석 또한 나중에 손실 함수에 전달될 수 있도록 텐서로 파싱해야 한다. 

<br>

**적재** 단계에서 데이터는 타깃 구조로 적재된다. 이는 '모델을 실행할 기기로' 전달하는 것을 뜻한다. 처리된 데이터셋은 또한 나중에 사용/재사용할 수 있게 어딘가에 저장되거나 캐시로 저장될 수 있다. 

<br>

**API 인터페이스**

tf.data.Dataset은 tf.data API에서 제공하는 중심이 되는 클래스다. 이 클래서의 인스턴스(데이터셋)는 데이터 소스를 나타내며 방금 설명했던 '느긋한 리스트'의 패러다임을 따른다. 

```python
dataset = tf.data.Dataset.list_files("/path/to/dataset/*.png")
```

데이터셋에는 변환된 데이터셋을 제공하기 위해 적용할 수 있는 다양한 메서드도 포함되어 있다. 

예를 들어 다음 함수는 파일 콘텐츠가 균일하게 크기가 조정된 이미지 텐서로 변환된 새로운 데이터셋 인스턴스를 반환한다. 

```python
def parse_fn(filename):
    img_bytes = tf.io.read_file(filename)
    img = tf.io.decode_png(img_bytes, channels=3)
    img = tf.image.resize(img, [64,64])
    return img # 또는 이 입력에 명명하려면 '{'image': img}'
  
dataset = dataset.map(map_func=parse_fn)
```

<br>

반복할 때 `map( )`에 전달된 함수는 데이터셋의 모든 샘플에 적용된다. 

실제로 필요한 모든 변환이 적용됐으면 데이터셋은 다음처럼 느긋한 리스트/제너레이터로 사용될 수 있다. 

```python
print(tf.compat.v1.data.get_output_types(dataset))  # tf.uint8
print(tf.compat.v1.data.get_output_shapes(dataset)) # (64,64,3)

for image in dataset:
    # 이미지 관련 작업 수행
    ...
```

모든 데이터 샘플은 이미 Tensor로 반환되어 훈련한 기기에 쉽게 로딩될 수 있다. 

이를 좀 더 간단하게 하기 위해, tf.estimator.Estimator와 tf.keras.Model 인스턴스는 다음 코드에서 보듯이 훈련 시 tf.data.Dataset 인스턴스를 입력으로 직접 받을 수 있다. 

```python
keras_model.fit(dataset, ...) # 데이터에서 케라스 모델 훈련

def input_fn():
    # ... 데이터셋 구성
    return dataset
tf_estimator.train(input_fn, ...) # 에스티메이터 훈련
```

<br>

<br>

### 입력 파이프라인 구성

---

이제 ETL 절차를 염두에 두고 적어도 컴퓨터 비전 애플리케이션에 적용한 tf.data에서 제공하는 가장 보편적이면서 중요한 메서드 중 일부를 개발해본다. 

전체 목록은 [공식 문서](https://www.tensorflow.org/api_docs/python/tf/data)를 참조하세요. 

<br>

#### 추출 (소스: 텐서, 텍스트 파일, TFRecord 파일 등)

텐서플로는 데이터를 나열하고 추출하는 수많은 도구를 제공한다. 

<br>

**NumPy와 텐서플로 데이터에서 추출**

데이터 샘플이 이미 적재된 경우, 이 샘플은 `from_tensors()` 또는 `from_tensor_slices()` 정적 메서드를 사용해 tf.data에 바로 전달될 수 있다. 

두 메서드는 모두 중첩된 배열/텐서 구조를 받아들이지만, from_tensor_slices( )는 다음과 같이 데이터를 첫번째 축을 따라 샘플로 잘라낸다. 

```python
x, y = np.array([1,2,3,4]), np.array([5,6,7,8])
d = tf.data.Dataset.from_tensors((x,y))
print(tf.compat.v1.data.get_output_shapes(d)) # (TensorShape([4]), TensorShape([4]))

d_sliced = tf.data.Dataset.from_tensor_slices((x,y))
print(tf.compat.v1.data.get_output_shapes(d_sliced)) # (TensorShape([]), TensorShape([]))
```

첫번째 데이터셋 d는 1쌍의 샘플을 포함하며 각각은 4개의 값을 포함하고, 

```python
print(*d, sep='\n')

out:
  (<tf.Tensor: shape=(4,), dtype=int32, numpy=array([1, 2, 3, 4])>, <tf.Tensor: shape=(4,), dtype=int32, numpy=array([5, 6, 7, 8])>)
```



두번째 데이터셋 d_sliced는 4쌍의 샘플을 포함하며 각각은 하나의 값만 포함한다. 

```python
print(*d_sliced, sep='\n')

out:
  (<tf.Tensor: shape=(), dtype=int32, numpy=1>, <tf.Tensor: shape=(), dtype=int32, numpy=5>)
(<tf.Tensor: shape=(), dtype=int32, numpy=2>, <tf.Tensor: shape=(), dtype=int32, numpy=6>)
(<tf.Tensor: shape=(), dtype=int32, numpy=3>, <tf.Tensor: shape=(), dtype=int32, numpy=7>)
(<tf.Tensor: shape=(), dtype=int32, numpy=4>, <tf.Tensor: shape=(), dtype=int32, numpy=8>)
```

첫번째 축을 기준으로 쌍을 이루어 데이터셋이 생성된 것을 확인할 수 있다. 

<br>

**파일에서 추출**

이전 예제에서 보듯이 데이터셋은 `tf.data.Dataset.list_files( )` 정적 메서드를 사용해 파일에서 반복할 수 있다. 이 메서드는 목록의 파일 중 하나의 경로를 포함하는 문자열 텐서의 데이터셋을 생성한다. 그런 다음 `tf.io.read_file( )`을 사용해 각 파일을 열 수 있다. 

tf.data API는 이진 파일이나 텍스트 파일에서 반복하는 특수한 데이터셋도 제공한다. `tf.data.TextLineDataset( )`는 문서를 한 줄씩 읽어 들일 때 사용될 수 있으며(텍스트 파일에 이미지 파일과 레이블이 나열된 일부 공공 데이터셋에서 유용함), `tf.data.experimental.CsvDataset( )`도 CSV 파일을 파싱해 그 콘텐츠를 줄 단위로 반환할 수 있다. 

<br>

**기타 입력에서 추출 (generator, SQL 데이터베이스, range 등)**

tf.data.Dataset은 매우 다양한 입력 소스로부터 정의될 수 있다. 

단순히 숫자에서 반복하는 데이터셋은 정적 메서드인 `.range( )`로 초기화될 수 있다. 

또한 데이터셋은 `.from_generator( )`를 사용해 파이썬 제너레이터를 기반으로 구성될 수 있다. 

마지막으로 데이터가 SQL 데이터베이스에 저장된 경우에도 텐서플로는 다음을 포함해서 이를 쿼리할 수 있는 몇 가지 도구를 제공한다. 

```python
dataset = tf.data.experimental.SqlDataset(
 	"sqlite". "path/to/my_db.sqlite3", 
  "SELECT img_filename, label FROM images", (tf.string, tf.int32))
```

tf.data 문서를 보면 더 많은 데이터셋 인스턴스화 도구를 확인할 수 있다. 

<br>

<br>

#### 샘플 변환 (파싱, 보강 등)

ETL 파이프라인의 두 번째 단계는 **변환**이다. 

변환은 두 가지 유형으로 나눌 수 있다. 하나는 데이터 샘플에 개별적으로 영향을 미치고, 다른 하나는 전체 데이터셋을 편집한다. 

다음 단락부터는 전자에 해당하는 변환을 알아보고 샘플이 어떻게 전처리될 수 있는지 설명한다. 

<br>

**이미지와 레이블 파싱**

앞에서 다룬 parse_fn 메서드에서는 데이터셋에 의해 나열된 각 파일명에 대응하는 파일을 읽기 위해 `tf.io.read_file( )`을 호출한 다음 `tf.io.decode_png( )`를 사용해 바이트를 이미지 텐서로 전환했다. 

✋ **tf.io**는 decode_jpeg( ), decode_gif( ) 등도 포함하고 있다. 또한 이미지 포맷으로 무엇을 사용할 지 추론할 수 있는 보다 일반적인 decode_image( ) 도 제공한다. 

```python
dataset = tf.data.Dataset.list_files("/path/to/dataset/*.png")

def parse_fn(filename):
    img_bytes = tf.io.read_file(filename)
    img = tf.io.decode_png(img_bytes, channels=3)
    img = tf.image.resize(img, [64,64])
    return img # 또는 이 입력에 명명하려면 '{'image': img}'
  
dataset = dataset.map(map_func=parse_fn)
```

<br>

레이블을 파싱하기 위해 다양한 기법이 적용될 수 있다. 

레이블도 이미지라면(이미지 분할 혹은 편집 등), 방금 나열했던 메서드를 모두 동일하게 재사용할 수 있다. 

레이블이 텍스트 파일에 저장된 경우, `TextLineDataset( )`이나 `FixedLengthRecordDataset( )`를 사용해 텍스트 파일에서 반복할 수 있으며 **tf.strings** 같은 모듈을 사용해 텍스트 줄/레코드를 파싱할 수 있다. 

<br>

예를 들어 각 줄에 이미지 파일명과 레이블을 쉼표로 구분해 작성한 텍스트 파일로 된 훈련 데이터셋은 다음 방식으로 파싱될 수 있다. 

```python
def parse_fn(line):
    img_filename, img_lebel = tf.strings.split(line, sep=',')
    img = tf.io.decode_image(tf.io.read_file(img_filename))[0]      # 이미지 불러오기
    return {'image': img, 'label': tf.strings.to_number(img_label)} # 레이블은 문자열을 숫자로 변환
  
dataset = tf.data.TextLineDataset('path/to/file.txt').map(parse_fn) # (이미지, 레이블)
```

<br>

**TFRecord 파일 파싱**

모든 이미지 파일을 나열하여 하나씩 로딩해서 파싱하는 것은 가장 단순한 솔루션이지만 자원을 많이 소비한다. 대신, 대용량 이미지를 하나의 이진 파일에 함게 저장하면 훨씬 더 효율적일 것이다. 

따라서 텐서플로 사용자라면 **'구조화된 데이터를 직렬화하기 위해 언어 중립적이면서 플랫폼 중립적인 확장성 있는 매커니즘인 구글 프로토콜 버퍼'**(https://developers.google.com/protocol-buffers 의 문서 참조)를 기반으로 하는 **'TFRecord'** 파일 포맷을 사용하는 것이 좋을 것이다. 

'TFRecord' 파일은 데이터 샘플(이미지, 레이블, 메타데이터 같은)을 모으는 이진 파일이다. 

TFRecord 파일에는 기본적으로 샘플( {'img': image_sample1, 'label': label_sample1, ... } ) 을 구성하는 각 데이터 요소(특징)를 명명하는 딕셔너리인 직렬화된 tf.train.**Example** 인스턴스가 포함되어 있다. 샘플에 포함된 각 요소/특징은 tf.train.**Feature** 인스턴스 또는 그 서브클래스의 인스턴스다. 

이 파일 포맷은 특별히 텐서플로를 위해 개발됐기 때문에 tf.data에서 매우 잘 지원된다. 입력 파이프라인을 위한 데이터 소스로 'TFRecord' 파일을 사용하기 위해서 텐서플로 사용자는 그 파일을 `tf.data.TFRecordDataset(파일명)`으로 전달해 파일에 포함된 tf.train.Example 요소에서 반복할 수 있다. 

그 콘텐츠를 파싱하기 위해 다음 작업이 이루어진다. 

```python
dataset = tf.data.TFRcordDataset(['file1.tfrecords', 'file2.tfrecords'])
# 특징/tf.trainExample 구조를 설명하는 딕셔너리
feat_dic = {'img': tf.io.FixedLenFeature([], tf.string),   # 이미지의 바이트
            'label': tf.io.FixedLenFeature([1], tf.int64)} # 클래스 레이블

def parse_fn(example_proto): # 직렬화된 tf.train.Example을 파싱
    sample = tf.parse_single_example(example_proto, feat_dic)
    return tf.io.decode_image(sample['img'])[0], sample['label']
  
dataset = dataset.map(parse_fn)
```

<br>

**TFRecord** 파일 생성에 대한 부분은 아래 주소에서 확인하지. 

> https://ballentain.tistory.com/48

<br>

**샘플 편집**

`.map( )` 메서드는 tf.data 파이프라인의 중심이다. 이 메서드는 샘플을 파싱하는 일 외에도 추가로 샘플을 편집(이미지를 자르거나, 척도를 조정하거나, 타깃 레이블을 원-핫 인코딩하는 등)하는데 사용할 수 있다. 

<br>

<br>

#### 데이터셋 변환(뒤섞기, 압축, 병렬화 등)

**데이터셋 구조화**

tf.data API는 데이터 필터링, 샘플 뒤섞기, 샘플을 배치로 묶기 등의 연산을 위한 간단한 솔루션도 제공한다. 

아래는 데이터셋의 메서드 중 가장 자주 사용되는 것을 정리한 것이다. 

* `.batch(batch_size, ...)`: 데이터 샘플을 설정에 따라 배치로 만들어 새로운 데이터셋을 반환한다(tf.data.experimental.unbatch( )는 그 반대를 수행한다). batch( ) 다음에 .map( )이 호출되면 매핑 함수는 배치된 데이터를 입력으로 받게 된다. 
* `.repeat(count=None)`: 데이터를 count 횟수만큼 반복한다(count=None인 경우 무한 반복한다). 
* `.shuffle(buffer_size, seed, ....)`: 버퍼를 설정에 따라 채운 다음 요소를 뒤섞는다(예를 들어 buffer_size = 10이면 데이터셋을 수직으로 10개 요소씩 부분집합으로 나눈 다음 각 부분집합에서 요소의 순서를 임의로 바꾸어 이를 하나씩 반환한다). 버퍼 크기가 클수록 뒤섞는 일은 확률이 높아지지만 프로세스는 무거워진다. 
* `.filter(predicate)`: 제공된 predicate 함수의 bool 형 출력에 따라 요소를 보존/제거한다. 
* `.take(count)`: 최대로 처음 count개 요소를 포함하는 데이터셋을 반환한다. 
* `.skip(count)`: 처음 count 개수 요소를 제외한 나머지 데이터셋을 반환한다. .take( ) 메서드와 함께 이 두 메서드는 예를 들어 다음과 같이 훈련 세트와 검증 세트로 나눌 때 사용될 수 있다. 

```python
num_training_samples, num_epochs = 10000, 100
dataset_train = dataset.take(num_training_samples)
dataset_train = dataset_train.repeat(num_epochs)
dataset_val = dataset.skip(num_training_samples)
```

그 외에도 데이터를 구조화하거나 그 흐름을 제어하는 데 사용할 수 있는 메서드가 많다. (.unique( ), .reduce( ), .group_by_reducer( ) 등)

<br>

**데이터셋 병합**

일부 메서드는 데이터셋을 함께 병합하는 데 사용될 수도 있다. 

가장 간단한 메서드로는 `.concatenate(dataset)`와 정적 메서드인 `.zip(datasets)`가 있다. 전자는 제공된 데이터셋의 샘플과 현재 데이터셋의 샘플을 연결하고 후자는 다음과 같이 데이터셋의 요소를 튜플로 결합(파이썬의 zip 함수와 유사)한다. 

```python
d1 = tf.data.Dataset.range(3)
d2 = tf.data.Dataset.from_tensor_slices([[4,5], [6,7], [8,9]])
d = tf.data.Dataset.zip((d1,d2))

out:
  [0,[4,5]], [1,[6,7]], [2,[8,9]]
```

여러 소스에서 비롯된 데이터를 병합하기 위해서는 `.interleave(map_func, cycle_length, block_length, ...)`을 사용할 수 있다. 

이 메서드는 map_func 함수를 데이터셋의 요소에 적용하고 결과를 '끼워 넣는다'. 

<br>

앞의 '이미지 레이블 파싱' 부분에서 작성했던 텍스트 파일에 나열된 이미지 파일과 레이블을 사용하는 예제로 돌아가보자. 그러한 텍스트 파일이 여러 개 있고 그 안의 이미지를 단일 데이터셋에 모두 결합해야 한다면 .interleave( )는 다음처럼 적용될 수 있다. 

```python
def parse_fn(filename):
    img_bytes = tf.io.read_file(filename)
    img = tf.io.decode_png(img_bytes, channels=3)
    img = tf.image.resize(img, [64,64])
    return img 

filenames = ['/path/to/file1.txt', 'path/to/file2.txt', ...]
d = tf.data.Dataset.from_tensor_slices(filenames)
d = d.interleave(lambda f: tf.data.TextLineDataset(f).map(parse_fn), 
                 cycle_length=2, block_length=5)
```

<br>

cycle_length 파라미터는 동시에 처리되는 요소 개수를 지정한다. 

block_length 파라미터는 요소별 반환되는 연속 샘플 수를 지정한다. 

예를 들어 위의 예에서 10개의 파일이 있다고 가정하면, 2개 파일 단위로 최대 5개의 연속 행을 생성하고 그 다음을 반복한다. 

<br>

<br>

#### 적재

tf.data의 또 다른 이점은 그 모든 연산이 텐서플로 연산 그래프에 등록돼 추출되어 처리된 샘플이 Tensor 인스턴스로 반환된다는 점이다. 

따라서 ETL의 마지막 단계인 '적재'와 관련해서는 '크게 할 일이 없다.' tf.data 데이터셋을 반복하기 시작하면 생성된 샘플은 모델에 바로 전달될 수 있다. 

<br>

<br>

### 입력 파이프라인 최적화 및 모니터링

---

#### 최적화를 위한 모범 사례 따르기

tf.data API는 데이터 처리와 흐름을 최적화하기 위한 메서드와 옵션을 몇 가지 제공하는데, 이에 대해 자세히 살펴보자. 

<br>

**병렬화와 프리패치**



























<br>

<br>

### 정리

---

* 
