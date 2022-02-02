---
layout: single
title: "[AITech] 20220126 - Custom Dataset&DataLoader 개발하기"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['파이토치', '실습', 'Dataset', 'DataLoader']
---



<br>

## 학습 내용 정리

### Custom Dataset 및 Custom DataLoader

#### Dataset

* Dataset 관련 모듈
  * torch.utils.data
    * 데이터셋의 표준을 정의하고 데이터셋을 불러오고 자르고 섞는데 쓰는 도구들이 모여있는 모듈
    * torch.utils.data.Dataset: 데이터의 입력 표준을 정의하는 클래스
    * torch.utils.data.DataLoader: 데이터의 배치를 생성하고 학습 직전 텐서로 변환하는 등 데이터를 로드해주는 클래스
  * torchvision.dataset: torch.utils.data.Dataset을 상속하는 이미지 데이터셋 모음
  * torchtext.dataset: torch.utils.data.Dataset을 상속하는 텍스트 데이터셋 모음
  * torchvision.transforms: 이미지 데이터셋에 쓸 수 있는 Tensor 변환, resizing, cropping, rotating 등의 변환 필터를 갖는 모듈
  * torchvision.utils: 이미지 데이터를 저장하고 시각화할 수 있는 도구들을 갖는 모듈
* 커스텀 Dataset 정의
  * torch.utils.data.Dataset을 상속
  
  * `__init__()`, `__len()__`, `__getitem__()` 메서드 정의(map-style dataset)
    * `__init__()`: 데이터의 위치나 파일명을 지정하고 불러오는 것과 같은 초기화 작업과 데이터의 전처리 작업을 수행. 이미지를 처리할 transforms들을 compose해서 정의
    * `__len()__`: dataset의 요소 수를 반환
    * `__getitem__()`: 데이터셋의 idx 번째 데이터의 반환 형식을 정의. 원본 데이터의 전처리, 증강 등을 수행. 
    
  * 예시 코드: Titanic Dataset
  
    ```python
    class TitanicDataset(Dataset):
        def __init__(self, path, drop_features, train=True):
            self.data = pd.read_csv(path)
            self.data['Sex'] = self.data['Sex'].map({'male':0, 'female':1})
            self.data['Embarked'] = self.data['Embarked'].map({'S':0, 'C':1, 'Q':2})
            self.train = train
            self.data = self.data.drop(drop_features, axis=1)
            
            self.X = self.data.drop('Survived', axis=1).values
            self.y = self.data['Survived']
                
            self.features = self.data.drop('Survived', axis=1).columns.tolist()
            self.classes = ['Dead', 'Survived']
    
        def __len__(self):
            return len(self.y)
    
        def __getitem__(self, idx):
            X = self.X[idx]
            if self.train:
                y = self.y[idx]
            return torch.tensor(X), torch.tensor(y)
    ```
  
    

<br>

#### DataLoader

* DataLoader 인터페이스

  > DataLoader(dataset, batch_size=1, shuffle=False, sampler=None,
  >
  >    				batch_sampler=None, num_workers=0, collate_fn=None,
  >    	
  >    				pin_memory=False, drop_last=False, timeout=0,
  >    	
  >    				worker_init_fn=None)
  * batch_size: 배치의 사이즈
  
  * shuffle: 데이터를 섞어서 사용하는 지 여부
  
  * sampler/batch_sampler: 데이터의 인덱스를 컨트롤
    * map-style에서 `__len__()`과 `__iter__()`를 구현
    * [document](https://pytorch.org/docs/stable/data.html) or [others](https://towardsdatascience.com/pytorch-basics-sampling-samplers-2a0f29f0bf2a)
      * SequentialSampler : 항상 같은 순서
      * RandomSampler : 랜덤, replacemetn 여부 선택 가능, 개수 선택 가능
      * SubsetRandomSampler : 랜덤 리스트, 위와 두 조건 불가능
      * WeigthRandomSampler : 가중치에 따른 확률
      * BatchSampler : batch단위로 sampling 가능
      * DistributedSampler : 분산처리 (torch.nn.parallel.DistributedDataParallel과 함께 사용)
    
  * num_workers: 데이터를 불러올 때 사용하는 서브 프로세스 개수
    * 개수를 크게 설정해도 CPU-GPU 사이의 병목 현상으로 인해 오히려 속도가 느려질 수 있음
    
  * collate_fn: 데이터를 배치 단위로 합칠 대 일괄적으로 적용해주는 함수
    * 보통 텍스트 처리에서 padding을 하는 등 데이터의 사이즈를 일정하게 만들어줄 때 주로 사용
    
  * pin_memory: True로 지정 시 Tensor를 CUDA 메모리에 할당, 데이터 전송이 훨씬 빠르게 이루어짐. 
  
    ![image-20220127215148720](https://user-images.githubusercontent.com/70505378/151489742-3ca86d6d-e1e0-4246-b81e-02d4ddc1cbe5.png)
  
  * drop_last: batch_size에 따라 크기가 다를 수 있는 마지막 batch를 사용할 지 여부
  
  * time_out: 양수로 지정할 경우, DataLoader가 data를 불러오는데 제한 시간
  
  * worker_init_fn: 어떤 worker(프로세스)를 불러올 것인가를 리스트로 전달
  
* **[torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)**

  * torchvision은 항상 입력 이미지로 PIL 객체를 요구합니다. 

  * transformers

    * Resize

    ```python
    torchvision.transforms.Resize(size, 
                                  interpolation=<InterpolationMode.BILINEAR: 'bilinear'>, 
                                  max_size=None, 
                                  antialias=None)
    ```

    * RandomCrop

    ```python
    torchvision.transforms.RandomCrop(size, 
                                      padding=None,
                                      pad_if_needed=False,
                                      fill=0, 
                                      padding_mode='constant')
    ```

    

    * RandomRotation

    ```python
    torchvision.transforms.RandomRotation(degrees,
                                          interpolation=<InterpolationMode.NEAREST: 'nearest'>, 
                                          expand=False,
                                          center=None,
                                          fill=0, 
                                          resample=None)
    ```

    * 이외에도 수많은 transform 클래스들이 있습니다. 

  * PIL/Tensor(Array) 변환

    * transforms.ToTensor()(image): PIL 객체를 Tensor 객체로 변환합니다. 
    * transforms.ToPILImage()(image): 텐서 또는 배열 형태의 객체를 PIL 객체로 변환합니다. 

  * Compose

    * 여러 transformer들을 하나로 묶어서 처리하는 객체

    ```python
    transforms.Compose([transforms.Resize((224,224)),
                        transforms.RandomHorizontalFlip(0.5),
                        transforms.CenterCrop(150)])(im)
    ```

  * datasets

    * torchvision.datasets 에는 CIFAR10, MNIST 등 대표적인 이미지 데이터들을 dataset 객체로 쉽게 가져올 수 있도록 하는 인터페이스가 마련되어 있습니다. 
    * `dir(torchvision.datasets)`를 참조하세요. 

  * **torchvision에서 제공하는 transform 외에도 [albumentations](https://github.com/albumentations-team/albumentations)과 같이 다양한 transformer들을 제공하는 라이브러리가 많습니다.**

  * **때로는 transformation에 의해서 input이 변하면 GT값이 변하는 경우가 있습니다. 예컨대 객체 인식(Object detection)의 경우, 물체의 위치 정보인 바운딩 박스(Bounding box)가 그렇습니다. 원본 이미지를 뒤집거나 회전시키면 그에 따라서 바운딩 박스도 좌표가 변환되어야 합니다. 이럴 경우 사용하는 입장에서는 매우 곤란한데요. 이런 골치 아픈 상황을 해결해주는 라이브러리도 있습니다. 바로 [imgaug](https://github.com/aleju/imgaug)입니다.**

<br>

### 전체적인 학습 구조(train.py)

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
from network import CustomNet
from dataset import ExampleDataset
from loss import ExampleLoss

###############################
#  첫번째 과제 Custom modeling  #
###############################

# 모델 생성
model = CustomNet()
model.train()

# 옵티마이저 정의
params = [param for param in model.parameters() if param.requires_grad]
optimizer = optim.Example(params, lr=lr)

# 손실함수 정의
loss_fn = ExampleLoss()

###########################################
#  두번째 과제 Custom Dataset & DataLoader  # 
###########################################

# 학습을 위한 데이터셋 생성
dataset_example = ExampleDataset()

# 학습을 위한 데이터로더 생성
dataloader_example = DataLoader(dataset_example)

##########################################################
#  세번째 과제 Transfer Learning & Hyper Parameter Tuning  # 
##########################################################
for e in range(epochs):
    for X,y in dataloader_example:
        output = model(X)
        loss = loss_fn(output, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```



<br>

<br>

## 참고 자료

* [Tensor padding](https://hichoe95.tistory.com/116)
* [torchvision.transforms](https://pytorch.org/vision/stable/transforms.html)



<br>
