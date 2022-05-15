---
layout: single
title: "[Papers][CV][Face Detection] RetinaFace 논문 리뷰"
categories: ['AI', 'AI-Papers']
tag: []
toc: true
toc_sticky: true
---



<br>

# RetinaFace 논문 리뷰

이번에 리뷰할 논문은 2019년 발표되었고, Face Detection task에 있어 유명한 논문인 `RetinaFace` 논문입니다. 

포스팅 순서는 아래와 같습니다. 

* Introduction
* Related Work
* Multi-task Loss
* Dense Regression Branch
* Implementation Details
* Evaluation
* Conclusion

## Introduction

![image-20220419134134449](https://user-images.githubusercontent.com/70505378/163938176-bf0feba5-2eec-4a8b-a654-b24a4d6862bd.png)

논문에서는 아래 문장으로 RetinaFace를 표현합니다. 

> _A robust **single-stage face detector**, which performs **pixel-wise face localization on various scales** of faces by taking advantages of joint **extra-supervised and self-supervised multi-task learning**._

하이라이팅 되어 있는 부분이 모델의 핵심 아이디어를 나타내는데요, 이는 발표를 진행하면서 보도록 하겠습니다. 



본격적인 소개에 앞서, RetinaFace 이전에 나온 face detection 모델에는 무엇이 있고 어떤 아이디어를 사용했는지 보도록 하겠습니다. 

**MTCNN, STN**

Face detection task에서는 기본적으로 classification과 regression이 필요합니다. MTCNN과 STN은 이에 더해 5개의 facial landmark도 학습에 활용하는 방법을 사용했습니다. 

RetinaFace에서도 facial landmark 정보를 extra supervision signal로 사용하여 detection 성능을 끌어올리려 했습니다. 

**Mask R-CNN**

Mask R-CNN은 face mask 정보를 학습에 활용하였고, 성능 향상을 이루었습니다. 

하지만 모든 데이터셋에 대해 mask 정보를 labeling하는 것은 쉽지 않기 때문에, 이는 동시에 Mask R-CNN의 한계점이기도 합니다. 

RetinaFace에서는 이 대신에, 다른 unsupervised method를 사용하려는 시도를 했습니다. 

**FAN**

FAN 모델은 anchor-level attention map을 사용해 겹쳐진 얼굴에 대한 detection 성능을 향상시켰습니다. 

최근에는 여기서 발전하여 3D 정보를 face detection에 활용하려는 시도가 지배적인데요, RetinaFace 에서도 Mesh decoder라는 것을 사용해 3D 정보를 self-supervised signal로 활용하려는 시도를 했습니다. 

<br>

<br>



## Related Work

다음으로 관련된 연구들에 대해 간단히 살펴보겠습니다. 

**Image pyramid vs Feature pyramid**

![image-20220419135455797](https://user-images.githubusercontent.com/70505378/163938178-c2de9985-397e-4f71-80ed-b7888f0d3784.png)

첫번째로 image pyramid와 feature pyramid입니다. 

Image pyramid는 resizing된 각 input image에 대해 각각의 feature map들을 모두 뽑아내는 것이고, feature pyramid는 하나의 input image를 backbone network에 통과시켜 서로 다른 scale의 feature map들을 사용하는 것을 말합니다. 

최근의 추세를 따라, RetinaFace에서도 feature pyramid 방식을 채택했습니다. 



**Two-stage vs single-stage**

![image-20220419140329018](https://user-images.githubusercontent.com/70505378/163938183-b8188a0b-c1a8-44f9-ae7d-d8122f005c83.png)

다음으로는 two stage model과 one stage model에 대한 비교입니다. 

RetinaFace에서는 빠른 속도와 높은 recall을 위해 1-stage model을 채택했습니다. 

1-stage model의 고질적인 문제인 imbalanced sample로 인한 높은 False Positive rate를 해결하기 위해 OHEM이라는 별도의 sampling 기법을 사용하여 이를 완화했습니다. 



**Context Modeling**

![image-20220419140343161](https://user-images.githubusercontent.com/70505378/163938186-03daf787-35d8-4832-ae5a-a5e517676944.png)

다음으로는 context modeling인데, 이는 우리가 알고 있는 neck structure를 말합니다. RetinaFace에서는 풍부한 semantic 정보를 위해 Neck 구조를 사용했습니다. 

**Multi-task Learning**

![image-20220419140402601](https://user-images.githubusercontent.com/70505378/163938189-51d3d8f0-0658-44a1-a95e-24585fba78a3.png)

마지막으로 Multi-task Learning입니다. 

RetinaFace에서는 모델이 여러 task를 해결하도록 하는 multi-task loss를 사용했는데요, 이에 대한 자세한 설명은 아래에서 하도록 하겠습니다. 

<br>

<br>



## Multi-task Loss

![image-20220419140432450](https://user-images.githubusercontent.com/70505378/163938190-126d7fb7-e880-440f-aa8d-fe9523090519.png)

(1): Face classification loss

해당 bbox의 객체일 확률, 객체가 아닐 확률

* p<sub>i</sub>: predicted probability (obj, no_obj) (0~1)
* p<sub>i</sub>\*: gt probability (0 or 1)
* shape: 2

(2): Face box regression loss

bbox의 중심 좌표, 너비와 높이. 

* t<sub>i</sub>: \{tx, ty, tw, th\} (center point, width, height)
* t<sub>i</sub>\*: gt box
* shape: 4

(3): Facial landmark regression loss

양 눈, 코, 양 입꼬리 총 5개의 정보 이용

* l<sub>i</sub>: \{l<sub>x1</sub>, l<sub>y1</sub>, ..., l<sub>x5</sub>, l<sub>y5</sub>\}
* l<sub>i</sub>*: gt facial landmark
* shape: 10

(4): Dense regression loss

뒤에서 설명

* shape: 128 + 7 + 9 (structure&texture + camera view + illumination)

Balancing coefficients

* λ1: 0.25
* λ2: 0.1
* λ3: 0.01  

 Loss function으로부터 알 수 있듯, bbox가 실제로 face가 맞을 경우 multi-task loss를 사용하고 아닐 경우 classification loss만을 사용합니다. 

<br>

<br>



## Dense Regression branch

Dense regression branch에서는 3D Mesh Decoder를 사용하여 3D mesh 정보를 face detection에 활용합니다. 

Mesh decoder에서는 왼쪽 그림과 같이 일반적인 2D convolution 대신 graph convolution을 사용하고, 이는 3D mesh의 형태와 더불어 연산의 효율성 때문입니다. 

Graph convolution에서 kernel의 크기 k는 거쳐가는 edge의 개수를 나타내며, k가 커질수록 더 멀리 떨어진 node와도 연산을 수행하게 됩니다. 

![image-20220419142038625](https://user-images.githubusercontent.com/70505378/163938191-b087b06a-ed1d-467b-91f4-7f63aa39a135.png)

오른쪽 일련의 수식들은 graph convolution layer에서 일어나는 연산을 표현한 수식입니다. 

Graph convolution은 graph kernel g<sub>θ</sub>와 input data x가 주어졌을 때 g<sub>θ</sub>(L)x의 수식으로 나타낼 수 있고, 이 때 L = D - E 로 정의됩니다. E는 input data x가 G = \{V, E\}로 표현될 때 간선 정보를 나타내는 행렬이고, D는 E의 i번째 행을 sum한 값을 Dii 원소의 값으로 가지는 대각 행렬입니다. 

x<sub>k</sub>를 구하는 식을 보면 L과 x<sub>k-1</sub>의 곱 연산을 하는 것을 볼 수 있습니다. 이 때 L은 0이 대부분인 sparse matrix이기 때문에, 빠른 연산이 가능합니다. 

Graph convolution output인 y를 구하는 식을 정리하면, x<sub>0</sub>부터 x<sub>k-1</sub>을 concat한 후 graph kernel g<sub>θ</sub>와 곱하는 것으로 나타낼 수 있으며, 이 때 g<sub>θ</sub>는 dense matrix입니다. 

<br>

즉 정리하면, graph convolution 연산은 K번의 sparse vector-matrix multiplication 연산과 1번의 dense vector-matrix multiplication 연산이 일어나며, 이는 일반적인 2D convolution에 비해 매우 빠른 연산이 가능하고, 따라서 별도의 branch 로 사용할 수 있습니다. 

<br>



위 과정을 통해 shape과 texture를 나타내는 파라미터 P<sub>ST</sub>를 구하고 나면, camera view를 나타내는 파라미터 P<sub>cam</sub>과 광도를 나타내는 파라미터 P<sub>ill</sub>를 함께 사용하여 '랜더링된 2D face 정보'를 구합니다. 이를 실제 input image의 2D face 정보와 비교하여 dense regression loss를 계산함으로써 self-supervised learning을 할 수 있습니다. 

![image-20220419144058517](https://user-images.githubusercontent.com/70505378/163938192-10c11f32-ae40-4a44-9275-6a414ffed351.png)

<br>

<br>



## Implementation Details

| Implementation   | Details                                                      |
| ---------------- | ------------------------------------------------------------ |
| Feature Pyramid  | - **Backbone: ResNet-152 pretrained on ImageNet-11k**<br>- **P2~P5: use C2~C5(ResNet feature map) with top-down and lateral connections**<br>- **P6: 3x3 conv with stride=2 on C5** |
| Context Module   | - **Apply independent context modules on five feature pyramid levels**<br>- Increase the receptive field<br>- Enhance the rigid context modelling power<br>- **Replace 3x3 convolution with deformable convolution** |
| Loss Head        | - **Positive anchors: Multi-task loss**<br>- **Negative anchors: Only classification loss**<br>- **Mesh decoder: pretrained model** |
| Anchor Settings  | - **Scale-specific anchors on the feature pyramid levels from P2~P6**<br>- **Total 102.300 anchors, 75% from P2 to capture tiny faces (despite computational time and higher FP rate)**<br>- 0.5 < : positive samples, 0.3 > : negative samples<br>- **OHEM: alleviate significant imbalance between the positive-negative examples (99% > )**<br> ![image-20220419144952087](https://user-images.githubusercontent.com/70505378/163938196-3993100f-b36c-4e7d-914e-9a3f615d7734.png) |
| Training Details | - SGD (momentum=0.9, weight decay 5e-4, batch size 8x4)<br>- 4 NVIDIA Tesla P40 (24GB) GPUs |
| Testing Details  | - TTA (Flip, Resizing)<br />- Box voting (applied on union set of predicted face boxes using an IoU threshold at 0.4) |







<br>

<br>



## Evaluation

![image-20220419145731320](https://user-images.githubusercontent.com/70505378/163938199-8094c9e8-1cda-498c-90a1-a0d5468a24aa.png)

첫번째 evaluation 표는 위와 같습니다. 이는 기본적으로 FPN에 Context module을 사용한 구조에 추가적인 구조들을 더하는 것이 성능에 어떤 영향을 끼치는지 측정한 표입니다. 

앞서 Implementation details에서 본 것과 같이 Context module의 3x3 convolution을 deformable convolution으로 교체함으로써 성능 향상을 얻을 수 있습니다. 

또한 Extra-supervised branch인 'Facial landmark regression loss'와 self-supervised branch인 'dense regression loss'를 모두 사용할 때에 가장 높은 성능을 얻을 수 있는 것을 볼 수 있습니다. 

![image-20220419150145308](https://user-images.githubusercontent.com/70505378/163938200-2b0dc80c-f8c8-466c-b1d1-4e9de23da229.png)

두번째 evaluation 그래프는 WIDER FACE (Hard) dataset에 대한 pr curve와 AP 값을 보여줍니다. 

RetinaFace가 기존 가장 높은 성능을 보이던 ISRN의 0.903보다 높은 0.914 를 기록하였습니다. 







<br>

<br>



## Conclusion

![image-20220419150346566](https://user-images.githubusercontent.com/70505378/163938202-1c9b48d8-348a-44ec-a9a2-f0b2c721f708.png)



### Contributions

* Tiny face detection에 있어 좋은 성능을 보임
* Face landmark 정보를 이용한 extra supervised branch와 Mesh decoder를 이용한 self-supervised branch의 사용
* SOTA 성능 달성







### Codes

* [https://paperswithcode.com/paper/190500641](https://paperswithcode.com/paper/190500641)







### SOTA

현재 작성 시점 기준으로 RetinaFace는 WIDER Face dataset에 있어서 2위의 성능을 기록하고 있습니다. 

1위 성능 모델은 RetinaFace 논문 이후 2020년에 발표된 TinaFace 모델입니다. 

![image-20220419150747516](https://user-images.githubusercontent.com/70505378/163938172-5652fd72-1f86-47d5-92d1-321667d53ed5.png)

















<br>

<br>

_**논문에 대한 내용은 여기까지입니다. 아래부터는 개인적으로 새롭게 알고 느끼게 된 부분들을 정리하는 부분입니다.**_

<br>

# 새롭게 알게 된 것들

## Vocabulary

| Vocabulary   | meanings           |
| ------------ | ------------------ |
| embraced     | 포옹               |
| occluded     | 가려진             |
| coarse       | 조잡한             |
| morphable    | 변형 가능한        |
| indisputable | 논쟁의 여지가 없는 |
| impede       | 방해하다           |



<br>

## Domain-specific word



<br>

## Others

* Face Detection task에서는 classification, regression, facial landmark, 3D mesh 의 총 4개의 정보들을 이용할 수 있다는 것을 알았다. 이를 통해 face detection이 어떤 식으로 작동하는지 감을 잡을 수 있었다. 
* 2020년 발표되어 SOTA 성능을 기록 중인 TinaFace 논문도 봐야겠다. 













