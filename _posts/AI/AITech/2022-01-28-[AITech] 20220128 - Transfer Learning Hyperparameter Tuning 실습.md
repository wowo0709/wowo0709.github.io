---
layout: single
title: "[AITech] 20220128 - Transfer Learning&Hyperparameter Tuning 실습"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 학습 내용 정리

### Transfer Learning & Hyperparameter Tuning 실습

2주차 파이토치 심화 과제에서는 **전이 학습과 하이퍼파라미터 튜닝**에 관한 내용을 다루었는데요, 핵심 내용들만 정리해봅니다. 

#### Transfer Learning

* 대용량의 데이터(Source Task)로 학습된 이미 높은 성능을 보이는 모델을 나의 목적에 맞는 데이터(Target Task)로 재학습시켜 목적에 맞는 모델을 만드는 것
* Source task와 Target task에 정답(label) 유무에 따라 다양한 전이 학습 방법이 있는데, 그 중 두 task에서 모두 정답이 있는 경우에 **Fine-Tuning** 기법을 사용할 수 있습니다. 
  * Fine Tuning 방법에서는 가져온 모델을 전부 재학습 시킬 수도 있고, 특징 추출(Feature Extraction) 부분은 고정(frozen)시키고 분류 부분만 학습시킬 수도 있습니다.
  * 또는 epoch가 진행되면서 layer의 고정을 조금씩 풀어주는 방법도 있습니다.  

전이 학습 섹션에서는 ImageNet 데이터셋으로 학습된 ResNet18 모델을 Fashion MNIST 데이터셋으로 전이 학습시켰습니다. 

```python
imagenet_resnet18 = torchvision.models.resnet18(pretrained=True)
fashion_train = torchvision.datasets.FashionMNIST(root='./fashion', train=True, download=True)
fashion_test = torchvision.datasets.FashionMNIST(root='./fashion', train=False, download=True)
# 모델 구조 확인
print(imagenet_resnet18)
'''
ResNet(
  (conv1): Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
  (relu): ReLU(inplace=True)
  (maxpool): MaxPool2d(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
  (layer1): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
    (1): BasicBlock(
      (conv1): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(64, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer2): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(64, 128, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(64, 128, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer3): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(128, 256, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(128, 256, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(256, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (layer4): Sequential(
    (0): BasicBlock(
      (conv1): Conv2d(256, 512, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (downsample): Sequential(
        (0): Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2), bias=False)
        (1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
    )
    (1): BasicBlock(
      (conv1): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn1): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      (relu): ReLU(inplace=True)
      (conv2): Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
      (bn2): BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
    )
  )
  (avgpool): AdaptiveAvgPool2d(output_size=(1, 1))
  (fc): Linear(in_features=512, out_features=1000, bias=True)
)
'''
```

<br>

전이학습을 시키기 위해 필수적으로 해야 하는 것이 2가지 있는데요, 이는 **모델의 입력/출력 layer 수정**과 **가중치 초기화**입니다. 

**모델 입력/출력 layer 수정**

ImageNet으로 학습된 ResNet18 모델의 입력 크기는 (3, 28, 28)이고, 우리의 목적인 Fashion MNIST의 크기는 (28, 28) 입니다. 여기서 채널 개수가 다르다는 것이 중요한데요, ImageNet의 채널 개수는 3이고 Fashion MNIST의 채널 개수는 1(grayscale)입니다. 

✋ 모델의 입력 채널 개수와 데이터 셋의 입력 채널 개수는 다음과 같이 확인할 수 있습니다. 

```python
'''CNN 모델의 입력 크기 확인하기'''
imagenet_resnet18.conv1.weight.shape # torch.Size([64, 3, 7, 7]) => (batch_size, channel, height, width)
imagenet_resnet18.conv1.weight.shape[1] # 채널 개수: 3
'''Fashion MNIST 데이터셋의 입력 크기 확인하기'''
fashion_train[0] # (<PIL.Image.Image image mode=L size=28x28 at 0x7F6608B19BD0>, 9)
np.array(fashion_train[0][0]).shape # (28, 28)
```



Convolution 연산 시 kernel의 channel 수는 input의 channel 수와 동일해야 하기 때문에 첫번째 convolution layer를 수정해야 합니다. 

```python
target_model = imagenet_resnet18

FASHION_INPUT_NUM = 1
target_model.conv1 = torch.nn.Conv2d(FASHION_INPUT_NUM, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
```

> [Con2d](https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html?highlight=conv2d#torch.nn.Conv2d) 모듈의 인터페이스
>
> torch.nn.Conv2d(*in_channels*, *out_channels*, *kernel_size*, *stride=1*, *padding=0*, *dilation=1*, *groups=1*, *bias=True*, *padding_mode='zeros'*, *device=None*, *dtype=None*)

그리고 출력 layer를 우리 목적에 맞는 layer로 교체해주어야 합니다. 

Pretrained model의 출력층(Linear layer(FC layer))의 가중치의 개수는 (1000, 512)로, (out_features, in_features) 모양꼴이기 때문에 즉 output의 개수는 1000개 입니다. 이를 in_features는 동일하고 out_features는 target task, 즉 Fashion MNIST의 class의 개수와 일치하도록 교체해주어야 합니다. 

```python
FASHION_CLASS_NUM = 10
target_model.fc = torch.nn.Linear(in_features=512, out_features=FASHION_CLASS_NUM, bias=True)
```

<br>

**가중치 초기화**

이렇게 모델의 layer를 수정/교체해주었으면 초기화를 해줘야 합니다. 

보편적인 가중치 초기화 방법으로는 weight의 경우 Xavier Initialization으로, bias의 경우 in_features 크기를 n이라 했을 때 U(-1/root(n), 1/root(n))의 uniform distribution으로 해주는 방법이 있습니다. 

```python
  torch.nn.init.xavier_uniform(target_model.conv1.weight)
  torch.nn.init.xavier_uniform_(target_model.fc.weight)
  stdv = (1/target_model.fc.in_features)**(1/2)
  torch.nn.init.uniform_(target_model.fc.bias, -stdv, stdv)
```

> [torch.nn.init.xavier_uniform_()](https://pytorch.org/docs/stable/nn.init.html?highlight=uniform#torch.nn.init.xavier_uniform_)의 인터페이스
>
> torch.nn.init.xavier_uniform_(*tensor*, *gain=1.0*)
>
> [torch.nn.init.uniform_()](https://pytorch.org/docs/stable/nn.init.html?highlight=uniform#torch.nn.init.uniform_)의 인터페이스
>
> torch.nn.init.uniform_(*tensor*, *a=0.0*, *b=1.0*)

✋ 이외에도 카이밍 초기화([torch.nn.init.kaiming_uniform_()](https://pytorch.org/docs/stable/nn.init.html?highlight=uniform#torch.nn.init.kaiming_uniform_)), 정규 분포 초기화([torch.nn.init.normal_()](https://pytorch.org/docs/stable/nn.init.html?highlight=normal#torch.nn.init.normal_)), 상수 초기화([torch.nn.init.constant_()](https://pytorch.org/docs/stable/nn.init.html?highlight=torch%20init%20nn%20constant_#torch.nn.init.constant_)) 등 많은 방법이 있습니다. 

<br>

**모델 학습하기**

모델 학습에 대한 코드는 [PyTorch - Transfer Learning for Computer Vision Tutorial](https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html#convnet-as-fixed-feature-extractor)에 자세하게 나와 있습니다. 

여기서 짚고 넘어갈 것은 전체 모델의 모든 layer를 재학습 시킬 수도 있고, feature extraction 부분은 고정시키고 classification 부분만 재학습 시킬 수도 있으며, 점차 layer의 고정을 풀어주는 식으로 재학습 시킬 수도 있는 여러 방법이 있다는 것입니다. 

```python
# 모델 가중치 고정시키기
for param in target_model.parameters():
    param.requires_grad = False
```







<br>

#### Hyperparameter Tuning

하이퍼파라미터 튜닝 섹션에서는 `ray`라는 모듈을 이용하여 튜닝을 수행하는 방법을 배웠습니다. ray 모듈은 Distributed application을 만들기 위한 프레임워크로, 분산 컴퓨팅 환경에서 많이 사용되고 있습니다. 그리고 ray 모듈 안에 있는 tune이라는 모듈을 이용하여 간단하게 하이퍼파라미터 튜닝을 수행할 수 있습니다. ([Tune Document](https://docs.ray.io/en/master/tune/index.html))

ray를 이용한 튜닝 방법을 코드 레벨에서 보기 전에, 튜닝을 할 때는 다음 2가지에 대해 생각해봅시다. 

1. Tuning의 목적(종속변인)
   * 이는 우리가 튜닝을 하는 목적에 해당합니다. 즉, **어떤 값을 최대화(최소화)할 것인지**를 정하는 것입니다. 
   * 여기서는 Fashion MNIST Test dataset의 Accuracy의 최대화를 목표로 합니다. 
2. Tuning할 Hyperparameter(조작변인, 통제변인)
   * 조작변인은 값을 조정하며 최적 값을 탐색할 변수에 해당하고, 통제변인은 값을 고정시킬 변수에 해당합니다. 
   * 여기서는 조작변인으로 **Epoch, Batch size, Learning rate**를, 통제변인으로 **모델 구조 ImageNet Pretrained ResNet18, All Not-Freeze Fine Tuning**을 지정합니다. 

**ray 모듈 설치하기**

아래 커맨드를 통해 ray 모듈을 설치할 수 있습니다. 

```python
print("Install ray")
!pip uninstall -y -q pyarrow
!pip install -q -U ray[tune]
!pip install -q ray[debug]
```

**통제변인**

```python
# 통제 변인
## 1. imagenet_resnet18 모델
def get_imagenet_pretrained_model():
  imagenet_resnet18 = torchvision.models.resnet18(pretrained=True)
  target_model = imagenet_resnet18
  FASHION_INPUT_NUM = 1
  FASHION_CLASS_NUM = 10
    
  imagenet_resnet18.conv1 = torch.nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
  in_features = imagenet_resnet18.fc.in_features
  imagenet_resnet18.fc = torch.nn.Linear(in_features, FASHION_CLASS_NUM, bias=True)
  torch.nn.init.xavier_uniform_(imagenet_resnet18.fc.weight)
  stdv = (1/imagenet_resnet18.fc.in_features)**(1/2)
  torch.nn.init.uniform_(imagenet_resnet18.fc.bias, -stdv, stdv)

  return target_model
```

**조작변인**

```python
# 조작 변인
## 1. Learning Rate
def get_adam_by_learningrate(model, learning_rate:float):
  return torch.optim.Adam(model.parameters(), lr=learning_rate)
## 2. Epoch 개수
def get_epoch_by_epoch(epoch:int):
  return epoch
## 3. BatchSize 크기에 따른 데이터 로더 생성
common_transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
fashion_train_transformed = torchvision.datasets.FashionMNIST(root='./fashion', train=True, download=True, transform=common_transform)
fashion_test_transformed = torchvision.datasets.FashionMNIST(root='./fashion', train=False, download=True, transform=common_transform)

def get_dataloaders_by_batchsize(batch_size:int):
  # Mnist Dataset을 DataLoader에 붙이기
  BATCH_SIZE = batch_size
  fashion_train_dataloader = torch.utils.data.DataLoader(fashion_train_transformed, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
  fashion_test_dataloader = torch.utils.data.DataLoader(fashion_test_transformed, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

  dataloaders = {
      "train" : fashion_train_dataloader,
      "test" : fashion_test_dataloader
  }

  return dataloaders
```

**탐색 구간과 탐색기 정하기**

ray에서 사용할 수 있는 탐색기에는 여러 종류가 있습니다. 더 다양한 탐색기들에 대한 내용은 [여기](Optimizer들은 https://docs.ray.io/en/master/tune/api_docs/suggestion.html#bayesopt)에서 확인할 수 있습니다. 

```python
from ray import tune
# 탐색할 하이퍼파라미터 config 설정
config_space = {
    "NUM_EPOCH" : tune.choice([4,5,6,7,8,9]),
    "LearningRate" : tune.uniform(0.0001, 0.001),
    "BatchSize" : tune.choice([32,64,128]),
}

from ray.tune.suggest.hyperopt import HyperOptSearch
# 탐색기 Optimizer 설정
optim = HyperOptSearch(
    metric='accuracy', # hyper parameter tuning 시 최적화할 metric을 결정합니다.
    mode="max", # target objective를 maximize 하는 것을 목표로 설정합니다
)
```

**Training 함수 작성**

```python
def training(
    config # 조작 변인 learning rate, epoch, batchsize 정보
):
  # 통제 변인
  target_model = get_imagenet_pretrained_model() 

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # 학습 때 GPU 사용여부 결정. Colab에서는 "런타임"->"런타임 유형 변경"에서 "GPU"를 선택할 수 있음
  target_model.to(device)

  # 조작 변인
  NUM_EPOCH = get_epoch_by_epoch(config["NUM_EPOCH"])
  dataloaders = get_dataloaders_by_batchsize(config["BatchSize"])
  optimizer = get_adam_by_learningrate(target_model, config["LearningRate"])

  ### 학습 코드 시작
  ...
    
  # epoch 종료
  tune.report(accuracy=best_test_accuracy.item(), loss=best_test_loss)
```

**Tuning 수행**

```python
from ray.tune import CLIReporter
import ray

NUM_TRIAL = 10 # Hyper Parameter를 탐색할 때에, 실험을 최대 수행할 횟수를 지정합니다.

reporter = CLIReporter( # jupyter notebook을 사용하기 때문에 중간 수행 결과를 command line에 출력하도록 함
    parameter_columns=["NUM_EPOCH", "LearningRate", "BatchSize"],
    metric_columns=["accuracy", "loss"])

ray.shutdown() # ray 초기화 후 실행

analysis = tune.run(
    training,
    config=config_space,
    search_alg=optim,
    #verbose=1,
    progress_reporter=reporter,
    num_samples=NUM_TRIAL,
    resources_per_trial={'gpu': 1} # Colab 런타임이 GPU를 사용하지 않는다면 comment 처리로 지워주세요
)
```

**결과 확인**

```python
best_trial = analysis.get_best_trial('accuracy', 'max')
print(f"최고 성능 config : {best_trial.config}")
# 최고 성능 config : {'NUM_EPOCH': 9, 'LearningRate': 0.0009309039165529126, 'BatchSize': 32}
print(f"최고 test accuracy : {best_trial.last_result['accuracy']}")
# 최고 test accuracy : 0.9143999814987183
```

<br>

이로써 pretrained model을 가져와 transfer learning을 수행하고 hyperparameter tuning까지 수행하는 과정을 코드 레벨에서 공부했습니다. 

<br>
