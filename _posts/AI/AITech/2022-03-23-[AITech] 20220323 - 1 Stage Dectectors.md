---
layout: single
title: "[AITech][Object Detection] 20220323 - 1 Stage Detectors"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['YOLO', 'SSD', 'RetinaNet']
---



<br>

_**본 포스팅은 '송원호' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# 1 Stage Detectors

`1 stage detecor`는 2 stage dectector의 한계인 '속도' 면에서 실시간으로 동작할 수 있는 수준의 속도를 보입니다. 

1 stage detector의 특징은 아래와 같습니다. 

* Localization, Classification이 동시에 진행
* 전체 이미지에 대해 특징 추출, 객체 검출이 이루어짐 → 간단하고 쉬운 디자인
* 속도가 매우 빠름 (Real-time detection)
* 영역을 추출하지 않고 전체 이미지를 보기 때문에 객체에 대한 맥락적 이해가 높음
* Background error가 낮음  

![image-20220323103135927](https://user-images.githubusercontent.com/70505378/159626727-ea11e39e-267c-42b6-8007-f19df4e811f8.png)

## YOLOv1

`YOLO`(You Only Look Once) 모델은 현재 v5까지 나와있는 상태입니다. YOLO 모델의 발전사와 각각의 특징은 아래와 같고, 이번 포스팅에서는 YOLO v1, v2, v3에 대해 다룹니다. 

* YOLO v1 : 하나의 이미지의 Bbox와 classification 동시에 예측하는 1 stage detector 등장
* YOLO v2 : 빠르고 강력하고 더 좋게
  * 3가지 측면에서 model 향상
* YOLO v3 : multi-scale feature maps 사용
* YOLO v4 : 최신 딥러닝 기술 사용
  * BOF : Bag of Freebies, BOS: Bag of Specials
* YOLO v5: 크기별로 모델 구성
  * Small, Medium, Large, Xlarge  

YOLO v1은 아래와 같이 GoogLeNet을 변형한 구조를 backbone으로 사용하고, 마지막에는 2개의 fully connected layer를 두어 각각 box의 좌표값 및 확률을 계산합니다. 

![image-20220323104608563](https://user-images.githubusercontent.com/70505378/159626729-04c928a3-06c6-410c-b4bf-3dcd93b196b9.png)

### Pipeline

**1. (30, 7, 7) 크기의 feature map을 입력으로 받습니다. 원본 입력 이미지도 동일하게 (7, 7) grid로 나눕니다.**

![image-20220323104844796](https://user-images.githubusercontent.com/70505378/159626730-9226469c-f661-4c50-8d38-7640eea53b41.png)

**2. Feature map의 한 grid cell마다 (1, 30) 크기의 벡터를 생성합니다.**

2-1. 앞의 10개의 값은 grid cell마다 2개의 bounding box에 대해 (x, y, w, h, confidence) 를 나타냅니다. 

논문에서는 B=2로 설정했습니다. 이 값을 b라고 한다면 벡터마다 총 5xb 개의 값이 생길 것입니다. 

![image-20220323105311662](https://user-images.githubusercontent.com/70505378/159626734-d5d5b674-eeb0-40e8-a495-a4074e51bc33.png)

2-2. 뒤의 20개의 값은 각 class에 해당할 확률(probability)을 나타냅니다. 

논문에서는 C=20으로 설정했으며, RCNN과 달리 background는 클래스에 추가되지 않습니다. 여기서 C=20은 loss가 계산되는 전체 클래스 중 일부의 클래스이며, YOLOv1은 총 80개 class에 대해 classification을 수행할 수 있습니다. 

![image-20220323105559604](https://user-images.githubusercontent.com/70505378/159626736-63d70889-2026-4634-b187-af04365b1099.png)

**3. 2개의 bounding box confidence score에 대해 각각 20개의 class probability를 곱해서 하나의 bounding box 당 (20, 1) 크기의 class score vector를 생성합니다.**

![image-20220323105920511](https://user-images.githubusercontent.com/70505378/159626737-d716d726-8fb3-4443-a310-c911bff04c51.png)

**4. 각 7x7개의 grid cell에 대해 3의 과정을 반복합니다. 최종적으로 2x49개의 (20, 1) vector가 생성됩니다.**

![image-20220323110040366](https://user-images.githubusercontent.com/70505378/159626739-722b8220-80b8-4393-b349-0b5a332b0c9e.png)

**5. 각 bbox class scores에 후처리를 가합니다. ** 

아래 과정을 각 20개의 class에 대해 98개의 class scores에 가합니다. 

5-1. class score가 threshold 이하인 값은 0으로 값을 설정합니다.

5-2. class scores를 내림차순으로 정렬합니다. 

5-3. NMS를 수행합니다. 

![image-20220323110550999](https://user-images.githubusercontent.com/70505378/159626741-a6580efd-1865-4966-8270-91fae14f21e3.png)





### Loss

![image-20220323111136923](https://user-images.githubusercontent.com/70505378/159626742-9d4be1eb-d0bd-4afb-9f20-677d6b8119e3.png)

* **Localization Loss**
  * 각 grid cell의 각 bbox에 대해 **물체가 있을 때** x와 y 값을 비교
  * 각 grid cell의 각 bbox에 대해 **물체가 있을 때** w와 h 값을 비교
* **Confidence Loss**
  * 각 grid cell의 각 bbox에 대해 **물체가 있을 때** confidence 값을 비교(물체가 있는 ground truth C 값은 1)
  * 각 grid cell의 각 bbox에 대해 **물체가 없을 때** confidence 값을 비교(물체가 없는 ground truth C 값은 0)
* **Classification Loss**
  * 각 grid cell에 대해 **물체가 있을 때** class probability를 비교

### Result

* Faster R-CNN에 비해 6배 빠른 속도
* 다른 real-time detector에 비해 2배 높은 정확도
* 이미지 전체를 보기 때문에 클래스와 사진에 대한 맥락적 정보를 가지고 있음
* 물체의 일반화된 표현을 학습
* 사용된 dataset외 새로운 도메인에 대한 이미지에 대한 좋은 성능을 보임  

![image-20220323111617046](https://user-images.githubusercontent.com/70505378/159626743-d1f9f18d-89a2-4ca7-9f21-4c7b7f61a5f2.png)

<br>

## SSD

YOLO는 아래와 같은 단점을 가지고 있습니다. 

* 7x7 그리드 영역으로 나눠 bbox prediction 진행
  * 그리드보다 작은 크기의 물체는 정확하게 검출할 수 없음
* CNN의 마지막 feature map만을 사용
  * 정확도 하락

`SSD`는 아래와 같은 특징을 가지며, YOLO의 단점을 극복할 수 있습니다. 

* Extra convolution layers에 나온 feature map들 모두 detection 수행
  * 6개의 서로 다른 scale의 feature map 사용
  * 큰 feature map (early stage feature map)에서는 작은 물체 탐지
  * 작은 feature map (late stage feature map)에서는 큰 물체 탐지
* Fully connected layer 대신 convolution layer 사용하여 속도 향상
* Default box 사용 (anchor box)
  * 서로 다른 scale과 비율을 가진 미리 계산된 box 사용  

SSD는 backbone으로 VGG-16을 사용하고 Extra convolution layers를 FC layer 대신 사용합니다. 또한 입력 사이즈로는 300x300을 받습니다. 

![image-20220323112113065](https://user-images.githubusercontent.com/70505378/159626748-9e022b4c-122d-4f2c-944a-2f7e1b0e9780.png)

### Pipeline

SSD에서 중요한 것은 Extra convolution layers에서 생성된 **multi-scale feature map**들을 모두 이용한다는 것입니다.  

예를 들어 (256, 5, 5) 크기의 feature map을 어떻게 사용하는지 보여드리겠습니다. 

<br>

Feature map에 3x3 conv를 적용해 channel 차원을 N<sub>Bbox</sub> * (Offsets + Nc)로 만듭니다.

이 때 N<sub>Bbox</sub>는 각 grid cell마다 bbox의 개수, Offset은 (cx, cy, w, h), N<sub>c</sub>는 class 수 + 1(background)입니다. 

![image-20220323113523370](https://user-images.githubusercontent.com/70505378/159626750-9faf284b-2528-42d3-b09a-899ad9ba1e9b.png)

SSD에서 Bbox를 만드는 방법은 아래와 같습니다. scale과 aspect를 공식에 의해 정하고, m이 bbox의 개수에 해당합니다. 

![image-20220323114228643](https://user-images.githubusercontent.com/70505378/159626751-79e1954d-ca18-48bd-ba67-52f575ab1959.png)

예를 들면 아래와 같습니다. 

![image-20220323114246800](https://user-images.githubusercontent.com/70505378/159626752-9c57df36-5457-46f6-8ee8-2551e8be9e82.png)

N<sub>Bbox</sub> * (Offsets + Nc) 모양의 채널 방향 벡터는 아래와 같이 구성됩니다. 

m=6이라고 할 때, 5x5 feature map 단계에서 생기는 출력은 5x5x(6x(4+21)) 모양입니다. 

![image-20220323114610358](https://user-images.githubusercontent.com/70505378/159626754-dbdebbaa-b257-4b3d-9aa3-2fded2576683.png)

<br>

6 단계의 각 feature map들에 대해 위의 연산을 반복하면, 아래와 같은 과정을 통해 총 8,732 개의 bbox를 얻게 됩니다. 

![image-20220323114957539](https://user-images.githubusercontent.com/70505378/159626760-b6f79baa-55ae-4a36-b2b4-b2b473c78fdc.png)

![image-20220323115019481](https://user-images.githubusercontent.com/70505378/159626708-2a31692d-e723-4cfc-af9e-9ee5d146570e.png)

최종적으로 Hard negative mining과 NMS를 수행하여 예측합니다. 

### Loss

SSD에서 사용하는 Loss function은 아래와 같습니다. Localization loss와 Confidence loss를 함께 사용하는 모습입니다. 

Loss function을 이용해 예측한 bbox와 gound truth bbox 사이의 차이, delta에 대해 학습합니다. 

![image-20220323115423063](https://user-images.githubusercontent.com/70505378/159626711-ddbbbd86-31bd-4761-8b51-1e09fa5a20f9.png)



### Result

SSD는 YOLO보다 빠른 속도를 보여주면서 더 나은 성능을 보여줍니다. 이는 FC layer를 convolutional layer로 대체하면서 속도를 더 빠르게 하고, 마지막 feature map만 사용했던 것을 6단계의 multi-scale feature map들을 모두 사용하여 정확도를 향상시킬 수 있었던 것으로 보입니다. 

![image-20220323115557275](https://user-images.githubusercontent.com/70505378/159626714-6bdae83c-307c-4a5d-a546-1077c8ade5f9.png)

<br>

## YOLOv2

`YOLOv2`는 아래 세가지 컨셉으로 YOLOv1을 향상시켰습니다. 

* **Better**: 정확도 향상
* **Faster**: 속도 향상
* **Stronger**: 더 많은 class 예측(80 -> 9418)

**Better**

* Batch normalization

  * mAP 2% ↑

* High resolution classifier

  * YOLO v1: 224x224 이미지로 사전 학습된 VGG를 448x448 Detection 태스크에 적용
  * YOLO v2 : 448x448 이미지로 새롭게 finetuning
  * mAP 4% ↑  

* Convolution with anchor boxes

  * Fully connected layer 제거
  * YOLO v1 : grid cell의 bounding box의 좌표 값 랜덤으로 초기화 후 학습
  * YOLO v2 : anchor box 도입
  * K means clusters on COCO datasets
    * 5개의 anchor box
  * 좌표 값 대신 offset 예측하는 문제가 단순하고 학습하기 쉬움
  * mAP 5% ↑  

* Fine-grained features

  * 크기가 작은 feature map은 low level 정보가 부족
  * Early feature map은 작은 low level 정보 함축
  * Early feature map을 late feature map에 합쳐주는 passthrough layer 도입
  * 26x26 feature map을 분할 후 결합  

  ![image-20220323131720721](https://user-images.githubusercontent.com/70505378/159626715-3a0679c0-a827-4262-9450-85156d0f3869.png)

* Multi-scale training

  * 다양한 입력 이미지 사용 \{320, 352, …, 608\}
  * ≠ multi-scale feature map  

**Faster**

* Backbone model
  * GoogLeNet → Darknet-19
* Darknet-19 for detection
  * 마지막 fully conected layer 제거
  * 대신 3x3 convolution layer로 대체
  * 1x1 convolution layer 추가
    * channel 수 125 (=5 x (5+20)) -> 5개 앵커박스에 대해 (x, y, w, h, confidence + 20개 class)

![image-20220323131825766](https://user-images.githubusercontent.com/70505378/159626719-2258ed84-a0f4-4ea7-80b7-b41b2c4f8cb2.png)

**Stronger**

* Classification 데이터셋(ImageNet), detection 데이터셋(Coco) 함께 사용

  * Detection 데이터셋 : 일반적인 객체 class로 분류 ex) 개
  * Classification 데이터셋 : 세부적인 객체 class로 분류 ex) 불독, 요크셔테리어
  * “개”, “요크셔테리어”, 배타적 class로 분류하면 안된다  

* WordTree 구성 (계층적인 트리)

  * Ex. “요크셔테리어” = 물리적객체(최상위 노드) – 동물 – 포유류 – 사냥개 – 테리어(최하위 노드)
  * ImageNet 데이터셋과 CoCo 데이터셋 합쳐서 구성 : 9418 범주  

  ![image-20220323132010453](https://user-images.githubusercontent.com/70505378/159626722-3ebc60d5-7161-48bd-9d8a-0995a2539cda.png)

* ImageNet 데이터셋 : Coco 데이터셋 = 4 : 1

  * Detection 이미지 : classification loss는 특정범주(20개)에 대해서만 loss 계산
    * ex. 개 이미지 : 물리적객체 – 동물 –포유류 – 개 에 대해서 loss 계산
  * Classification 이미지 : classification loss만 역전파 수행 (IoU)  

<br>

## YOLOv3

`YOLOv3`는 YOLOv2와 비교하여 다음 두 가지를 개선하였습니다. 

* **Backbone: Darknet - 53**

  * Skip connection 적용
  * Max pooling x, convolution stride 2사용
  * ResNet-101, ResNet-152와 비슷한 성능, FPS 높음  

  ![image-20220323133115246](https://user-images.githubusercontent.com/70505378/159626723-171d78a5-ec02-461e-98c7-f9d0ad8f5cd7.png)

* **Multi-scale Feature maps**

  * 서로 다른 3개의 scale을 사용 (52x52, 26x26, 13x13)
  * Feature pyramid network 사용
    * High-level의 fine-grained 정보와 low-level의 semantic 정보를 얻음  

<br>

## RetinaNet

Single-Stage detector가 겪게 되는 필연적인 문제로 **Class imbalance problem**이 있습니다. 

많은 경우에 이미지 내에서 우리가 찾고자 하는 물체는 아주 일부에 해당하고, 나머지는 배경에 해당합니다. 이러한 배경은 모델 학습에 도움을 주지 못 함에도 불구하고 계속해서 loss를 발생시켜 모델의 올바른 학습을 방해합니다. 

* Class imbalance
  * Positive sample(객체 영역) < negative sample(배경영역)  
* Anchor Box 대부분 Negative Samples (background)
  * 2 Stage detector의 경우 region proposal에서 background sample 제거 (selective search, RPN)
  * Positive/ negative sample 수 적절하게 유지 (hard negative mining)  

`RetinaNet`은 이 문제를 개선한 1 stage detector입니다. 

![image-20220312182048676](https://user-images.githubusercontent.com/70505378/158012844-509ee4c6-1752-4eb3-9141-7799b32c1c4b.png)

### Focal Loss

Focal loss는 class imbalance problem을 완화하기 위한 식으로 구성되어 있고, 맞힐 확률을 높은 클래스에 대해서는 낮은 loss gradient를, 맞힐 확률이 낮은 클래스에 대해서는 높은 loss gradient를 발생시킵니다. 

![image-20220312182006161](https://user-images.githubusercontent.com/70505378/158012842-336938ed-13d0-4e56-86a4-a3c8b5207456.png)

이를 이용해 1 stage methods의 단점이었던 성능 면에서 큰 향상을 이루었습니다. 

![image-20220323134042542](https://user-images.githubusercontent.com/70505378/159626725-4a6f9115-2527-4e1d-ba5b-89303c661ef7.png)

<br>

<br>

정리해보겠습니다. 

* YOLOv1
  * Localization과 Classification을 동시에 수행한 1 stage detector
  * Faster RCNN에 비해 6배 빠른 속도, 다른 real-time detector에 비해 2배 높은 정확도
  * 각 grid cell마다 2개의 bounding box 예측, 처음에는 랜덤으로 좌표값 초기화
* SSD
  * FC layer를 convolutional layer로 대체
  * Backbone에서 각 단계의 feature map들을 활용하기 위해 Extra convolution layers 사용 -> Multi scale feature maps (!=FPN)
  * 정해진 scale, ratio를 갖는 default box 정의
* YOLOv2
  * Fine-grained features, Multi-scale training (!=Multi scale feature map)
  * YOLOv1에서 사용한 GoogLeNet 대신 Darknet-19를 backbone으로 사용
  * anchor box를 도입하여 좌표를 예측하는 대신 offset을 예측하는 문제로 변환
  * YOLOv1이 80개 클래스에 대해 예측했던 것을 9418개 클래스로 예측
* YOLOv3
  * Backbone으로 Dartnet-53 사용
  * FPN 사용
* RetinaNet
  * Focal loss를 사용하여 class imbalance 문제 개선
  * FPN 사용



























<br>

<br>

# 참고 자료

* Hoya012, https://hoya012.github.io/
* 갈아먹는 Object Detection, https://yeomko.tistory.com/13
* Deepsystems, https://deepsystems.ai/reviews
* https://herbwood.tistory.com
* https://arxiv.org/pdf/1506.02640.pdf (You Only Look Once: Unified, Real-Time Object Detection)
* https://arxiv.org/pdf/1512.02325.pdf (SSD: Single Shot MultiBox Detector)
* https://arxiv.org/pdf/1612.08242.pdf (YOLO9000: Better, Faster, Stronger)
* https://pjreddie.com/media/files/papers/YOLOv3.pdf (YOLOv3: An Incremental Improvemen)
* https://arxiv.org/pdf/1708.02002.pdf (Focal Loss for Dense Object Detection)
* https://github.com/aladdinpersson/Machine-Learning-Collection/tree/master/ML/Pytorch/object_detection  

