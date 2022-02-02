---
layout: single
title: "[AITech] 20220125 - Dataset&DataLoader"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 학습 내용 정리

### Datasets & DataLoaders

`Dataset`과 `DataLoader` 모듈은 파이토치에서 데이터를 가져와서 모델에 먹이는 일련의 과정을 담당한다. 

![image-20220125115009860](https://user-images.githubusercontent.com/70505378/150939779-edf1891b-0b6f-44a5-bc15-5e494bd0d733.png)

* **Transforms**: 데이터 전처리, 증강, 타입 변환(텐서) 등의 역할
* **Dataset**: 초기화(init) 과정, 데이터 크기, 데이터 인덱싱 방법(mapping style, 어떤 형태로 반환할 지) 등의 역할
* **DataLoader**: 배치 생성, 배치 섞기, 데이터 샘플링 등의 역할

#### Dataset 클래스

* 데이터 입력 형태를 정의하는 클래스
* Image, Text, Audio 등에 따른 다른 입력 정의

```python
import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, text, labels): # 초기 데이터 생성 방법 지정
        self.labels = labels
        self.data = text
        
    def __len__(self): # 데이터의 전체 길이
        return len(self.labels)
    
    def __getitem__(self, idx): # index 값을 주었을 때 반환 형식
        label = self.labels[idx]
        text = self.data[idx]
        sample = {"Text": text, "Class": label}
        return sample
```

**Dataset 클래스 생성 시 유의점**

* 데이터 형태에 따라 각 함수를 알맞게 정의함
* 모든 것을 데이터 생성 시점에 처리할 필요는 없음
  * Image의 Tensor 변환의 경우 학습이 필요한 시점에 변환
* 데이터 셋에 대한 표준화된 처리방법 제공 필요
* 최근에는 HuggingFace 등 표준화된 라이브러리 사용

#### DataLoader 클래스

* **Data의 Batch를 생성하고 학습 직전 Tensor로의 변환**이 메인 업무
* 병렬적인 데이터 전처리 코드의 고민 필요

```python
text = ["Happy", "Amazing", "Sad", "Unhappy", "Glum"]
labels = ["Positive", "Positive", "Negative", "Negative", "Negative"]

MyDataset = customDataset(text, labels) # Dataset 객체 생성
MyDataLoader = DataLoader(MyDataset, batch_size=2, shuffle=True) # DataLoader 객체 생성
next(iter(MyDataLoader))
# {'Text':['Glum', 'Sad'], 'Class': ['Negative', 'Negative']}

MyDataLoader = DataLoader(MyDataset, batch_size=2, shuffle=True)
for dataset in MyDataLoader:
    print(dataset)
# {'Text':['Glum', 'Sad'], 'Class': ['Negative', 'Negative']}
# {'Text':['Sad', 'Amazing'], 'Class': ['Negative', 'Positive']}
# {'Text': ['Happy'], 'Class': ['Positive']}
```

**DataLoader 파라미터**

* dataset: 데이터를 가져올 Dataset 객체로서 `map-style dataset` 또는 `iterable-style dataset`을 사용할 수 있다. 
  * map-style dataset: `__get_item()__`과 `__len()__` 프로토콜을 작성한 Dataset 객체
  * iterable-style dataset: `__iter__()` 프로토콜을 작성한 IterableDataset 객체
* batch_size, shuffle, sampler, batch_sampler, drop_last
  * sampler: 배치 생성 시 사용할 데이터의 인덱스를 generate하는 `torch.utils.data.Sampler` 객체
  * shuffle: True로 설정 시 자동으로 랜덤하게 데이터의 인덱스를 generate하는 Sampler를 사용한다. 
  * batch_sampler: 배치를 생성하는 방법이 정의된  `torch.utils.data.Sampler` 객체
  * batch_size: 인자 전달 시 batch_size 크기의 미니 배치를 생성
  * drop_last: 전체 데이터 크기와 배치의 크기가 나누어 떨어지지 않는다면 마지막 미니 배치는 버린다. 
  * 이 모든 parameter를 사용하기 위해서는 dataset 으로 map-style dataset을 전달해야 한다. 
* collate_fn: 배치를 생성하기 위한 데이터의 인덱스를 뽑은 후에 합치는(collate) 과정에서 적용할 함수
  * 서로 길이가 다른 sequential data의 길이를 맞춰주기 위한 padding을 할 때 주로 사용한다. 
* num_workers, pin_memory, timeout
  * num_workers: 멀티 프로세스 사용
  * pin_memory: 메모리를 불러올 때 pinned memory를 사용, CUDA-enabled GPU 사용 시 더 빠른 데이터 transfer 가능. 커스터마이징 시 cumstom batch 안에 pin_memory() 메서드 정의
  * timeout: 배치를 가져오는데 걸리는 시간의 timeout

```python
DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
           batch_sampler=None, num_workers=0, collate_fn=None,
           pin_memory=False, drop_last=False, timeout=0,
           worker_init_fn=None, *, prefetch_factor=2,
           persistent_workers=False)
```







<br>

## 참고 자료

* [torch.utils.data document](https://pytorch.org/docs/stable/data.html)
