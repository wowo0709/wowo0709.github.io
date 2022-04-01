---
layout: single
title: "[AITech][Object Detection] 20220330 - Advanced Object Detection 1"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Cascade RCNN', 'Deformable Convolution', 'DETR', Swin Transformer']
---



<br>

_**본 포스팅은 '송원호' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# Advanced Object Detection 1

지난 포스팅들에서는 Object Detection에서의 기본적인 지식들과 구조들에 대해 보았습니다. 

이번 포스팅과 다음 포스팅에서는 더 나아가 Object Detection에서 활용될 수 있는 진보된 구조의 모델들에 대해 살펴보도록 하겠습니다. 순서는 `Cascade RCNN`, `Deformable Convolutional Networks`, `Transformer`로 구성됩니다. 

## Cascade RCNN

### Motivation

Faster RCNN 구조를 상기해봅시다. 

모델 학습 시, Faster RCNN에서는 다음의 iou threshold 기준들로 학습에 사용할 roi들을 분류했습니다. 

* `rpn`: 0.7 이상 positive sample, 0.3 이하 negative sample
* `roi_head`: 0.5 이상 positive sample, 0.5 미만 negative sample

`Cascade RCNN`에서는 바로 이 **iou threshold**에 집중을 했습니다. '왜 0.5, 0.7로 정했을까?'에 의문을 가지며, 다른 threshold를 사용하면 학습에 어떠한 영향을 주게 될 지에 대한 실험을 진행하였습니다. 

<br>

**첫번째. Localization Performance**

아래는 iou threshold에 따른 Input IOU와 output IOU 간의 상관관계를 나타낸 그래프입니다.

* Input IOU(x축): rpn을 통과한 roi들과 gt roi 간의 IOU
* output IOU(y축): roi head를 통과한 roi들과 gt roi 간의 IOU

![image-20220330150002436](https://user-images.githubusercontent.com/70505378/161014672-f97181b1-c7a7-4939-9c6b-b9188822bfba.png)

위 결과를 보면, **input에서 낮은 IOU를 가지는 roi들은 iou threshold가 낮을수록 output에서 높은 IOU를 가진다**는 것을 알 수 있고 **input에서 높은 IOU를 가지는 roi들은 iou threshold가 높을수록 output에서 높은 IOU를 가진다**는 것을 알 수 있습니다. 

즉, Input IOU가 높을수록 높은 iou threshold에서 학습된 model이 더 좋은 결과를 낸다는 것입니다. 

<br>

**두번째. Detection Performance**

두번째로는 iou threshold에 따른 AP 점수를 비교해보았습니다. 

* IOU Threshold(x축): 최종 예측된 roi가 True/False인지 판별할 때 사용하는 값(AP의 iou threshold)으로, iou threshold가 0.5인 경우 mAP_50, 0.95인 경우 mAP_95의 성능을 나타냅니다. 
* AP(y축): AP iou threshold 값에 따른 AP 값을 나타냅니다. 

![image-20220330150836221](https://user-images.githubusercontent.com/70505378/161014683-7ca451ea-99de-4966-9584-c5462e1fa5d9.png)

전반적인 성능은 iou threshold가 0.5, 0.6일 때 가장 좋습니다. 

다만 실험 결과에서 주목할 만한 것은, **AP의 iou threshold가 높아질수록 높은 iou threshold로 학습된 model의 성능이 좋아진다**는 것입니다. 

<br>

위 실험 결과들을 정리하면 아래 두 가지 사실을 도출해 낼 수 있습니다. 

1. 낮은 iou threshold로 학습된 모델은 낮은 iou를 가지는 roi들을 예측하는 데 유리하고, 높은 iou threshold로 학습된 모델은 높은 iou를 가지는 roi들을 예측하는 데 유리하다. 
2. High quality detection(gt roi들과 높은 iou 값을 가지도록 탐지)을 수행하기 위해서는 iou threshold를 높여 학습하는 것이 유리하다. 

하지만, 앞서 봤듯이 **무작정 iou threshold를 높이게 되면 mAP 값이 떨어지게**(낮은 iou를 가지는 roi들을 예측하지 못 하기 때문에) 됩니다. 

따라서 Cascade RCNN은 이를 해결하려 했습니다. 

<br>

### Method

Cascade RCNN이 어떤 식으로 동작하는 지 보기 전에, 먼저 Faster RCNN의 동작 방식을 간단히 짚고 넘어가겠습니다. 

![image-20220330153807479](https://user-images.githubusercontent.com/70505378/161014686-90fd13de-c7a0-41b9-baf0-dee6219b1a4a.png)

rpn에서는 rpn의 roi threshold에 따라 roi 영역들을 제안하고, 이 영역들은 roi pooling이 적용된 후에 roi head에 전달되어 roi head의 roi threshold에 따라 최종 예측됩니다. 

이번에는 Cascade RCNN의 구조를 보겠습니다. 

![image-20220330154016508](https://user-images.githubusercontent.com/70505378/161014689-6e6723d8-5be3-40a0-80fa-59bb90ee20ee.png)

Cascade RCNN에서 주목할 점은 두 가지입니다. 

* roi proposal(bbox perdiction)의 연속적 전달
* 여러 개의 roi head 사용

이 때 각 roi head들은 다른 roi threshold를 가집니다(뒤 stage로 갈수록 높은 iou threshold를 가집니다(ex. 0.5, 0.6, 0.7)).  앞선 stage에서의 bbox prediction은 다음 stage의 roi proposal으로 전달되고, 위 그림의 경우 마지막 stage의 결과인 C3, B3가 최종 결과가 됩니다. 

이렇게 함으로써 다양하면서도 높은 질의 bbox 예측을 수행할 수 있다고 합니다. 

<br>

아래는 다른 모델들과의 비교로, **더 높은 AP iou threshold에서 훨씬 향상된 성능을 얻을 수 있음**을 확인할 수 있습니다. 

![image-20220330154823669](https://user-images.githubusercontent.com/70505378/161014694-e0829c84-1d20-498a-836f-9bac9e634af8.png)

참고로, Cascade RCNN의 발전사는 아래와 같으며 b, c를 모두 활용한 d 형태의 구조가 Cascade RCNN입니다. 

![image-20220330155216223](https://user-images.githubusercontent.com/70505378/161014701-d616e0dd-3c08-457a-bc1e-521913ffbb3f.png)

![image-20220330155235302](https://user-images.githubusercontent.com/70505378/161014703-0f62b45d-b819-4210-a627-8a7052a7966a.png)

정리하면, Cascade RCNN의 의의는 아래와 같습니다. 

* Bbox pooling을 반복 수행할 시 성능 향상되는 것을 증명 (Iterative)
* IOU threshold가 다른 Classifier가 반복될 때 성능 향상 증명 (Integral)
* IOU threshold가 다른 RoI head를 cascade로 쌓을 시 성능 향상 증명 (Cascade)  

<br>

<br>

## Deformable Convolutional Networks(DCN)

### Motivation

일반적인 CNN이 가지고 있는 문제점은 kernel의 모양이 직사각형이어야 한다는 것입니다. 이 때문에 같은 물체더라도 그 방향이나 관점에 따라 다르게 받아들이기 쉽습니다. 또한 다양한 모양의 물체에 능동적으로 적응하는 면이 부족하기도 하죠.  

기존 해결 방법으로는 다음 두 가지 방법이 있습니다. 

* **Geometric augmentation**
  * dataset에 rotate, flip, shift 등의 geometric augmentation을 적용
  * 사람이 데이터 셋의 분포를 보고 휴리스틱하게 augmentation을 결정해야 한다는 한계
* **Geometric invariant feature engineering**
  * dataset에서 geometric invariant한 feature들을 뽑아서 모델에 전달
  * 여전히 사람이 해야 하는 과정이 포함되기 때문에 휴리스틱하고, 발견하지 못 한 feature들에 대해서는 목표하는 학습을 하지 못 함
  * David G. Lowe, “Distinctive Image Features from Scale-Invariant Keypoints"

위 두 가지 방법의 공통점이자 한계점은 **사람이 해야 한다**는 것입니다. 

`Deformable convolution`은 이를 해결하기 위한 연산입니다.

 <br>

### Method

Deformable convolution은 weight와 더불어 **offset**을 learnable parameter로 둡니다. 학습 과정을 통해 offset은 객체를 탐지하는 데 최적화되도록 학습될 것입니다. 

![image-20220331140506471](https://user-images.githubusercontent.com/70505378/161014705-1b040632-715b-4811-acd0-6f058f937a02.png)

Deformable convolution이 어떤 식으로 동작하는 지 좀 더 자세히 알아보도록 하겠습니다. 

기본적인 convolution에서 가중치에 해당하는 weight와 더불어 offset에 해당하는 **R**가 있다고 해봅시다. 그렇다고 하면 output feature map의 위치 p0에서의 값 y(p0)는 아래와 같이 구할 수 있습니다.

![image-20220331141606505](https://user-images.githubusercontent.com/70505378/161014710-ed48e76b-18f8-4447-bc57-1bb3a5128baf.png)

R 값이 정해져 있는 기본적인 convolution과 달리, **deformable convolution**은 이 **R 값을 학습 가능한 파라미터로 두어** geometric feature 또한 학습이 가능하게 하는 것입니다. 이를 나타내면 아래와 같습니다. 

![image-20220331141957600](https://user-images.githubusercontent.com/70505378/161014712-139328d3-99a3-4f96-9e93-17f274429dd2.png)

<br>

아래와 같이, deformable convolution은 feature를 뽑아내고자 하는 물체에 맞게 커널의 offset이 조정되어 학습하는 것을 볼 수 있습니다. 

![image-20220331142511037](https://user-images.githubusercontent.com/70505378/161014715-869ea429-59f4-4ceb-8e29-1596760f734a.png)

결과적으로 deformable convolution은 object detection과 segmentation task에서 성능의 향상을 일으킬 수 있음을 실험을 통해 증명했습니다. 

![image-20220331142606323](https://user-images.githubusercontent.com/70505378/161014727-773dfb4f-961e-4b3c-94a0-5cf0e9a71e24.png)







<br>

<br>

## Transformer

NLP 분야에서 Attention 구조만을 이용한 Transformer 모델이 혁명적인 변화를 일으킨 후, Transformer는 CV 분야까지 그 영향력을 넓히고 있습니다. 

이번 섹션에서는 Transformer가 CV 분야에 어떻게 활용될 수 있는지에 대해 알아보겠습니다. 

### Transformer

`Transformer`에 대한 내용은 이전 포스팅에서 자세히 다뤘습니다. 아래 포스팅을 참고하시길 바랍니다. 

https://wowo0709.github.io/ai/aitech/AITech-20220317-Part-4)-Transformer/  

<br>

### Vision Transformer(ViT)

Transformer를 CV task에 적용하려는 초기 시도에서의 모델은 `ViT`였습니다. 

ViT의 구조는 아래와 같습니다. 

![image-20220331144517057](https://user-images.githubusercontent.com/70505378/161014729-5325693f-2a5e-4c5c-bbf8-b3b6abb58e03.png)

ViT가 동작하는 과정에 대해 살펴보도록 하겠습니다. 

**1. 이미지를 patch 단위로 나누고, 각 3D patch를 2D로 flatten**

예를 들어 (3, 16, 16) 크기의 입력 이미지가 있고 patch size를 (4, 4)로 정했다면, 입력 이미지로부터 4개의 (3, 4, 4) 크기의 patch를 얻을 수 있습니다. 각 patch는 하나의 token에 대응합니다. 

Transformer는 2D의 vector 형태를 입력으로 요구하기 때문에, 각 patch는 flatten되어 4개의 48-dimension vector가 되어야 합니다. 

![image-20220331144908910](https://user-images.githubusercontent.com/70505378/161014735-aec1d256-5e2a-4f1c-be72-2b06ce4298a8.png)

**2. Learnable한 embedding 처리**

NLP와 마찬가지로 각 patch는 E라는 matrix를 통해 embedding됩니다. 이 때 E는 learnable parameter입니다. 

![image-20220331145021369](https://user-images.githubusercontent.com/70505378/161014737-919d3d61-9897-40ed-8990-4578c571594f.png)

**3. Add class embedding, position embedding**

앞서 만들어진 embedded patch들에 class embedding patch([CLS] token)를 맨 앞에 추가합니다. 또한 이미지의 위치 정보를 포함시키기 위해 positional embedding을 수행합니다. 

![image-20220331145149305](https://user-images.githubusercontent.com/70505378/161014738-933820cb-7457-45d4-af57-df0ec974bf42.png)

**4. Transformer**

이렇게 전처리된 각 patch들은 transformer의 입력으로 들어갑니다. 

![image-20220331145322364](https://user-images.githubusercontent.com/70505378/161014741-d7c64257-38e1-4823-91e4-4dcbb8dadafd.png)

**5. Predict**

맨 앞에 있는 class embedding vector 값을 MLP Head에 입력시켜 최종 결과를 추출합니다. 

![image-20220331145415792](https://user-images.githubusercontent.com/70505378/161014743-c426c202-ad7f-4598-b94e-5baddc06b98a.png)

<br>

이렇게 간단하게 ViT의 학습 과정을 살펴보았습니다. 실제로 image classification에 있어 뛰어난 성능을 보이는 ViT이지만, 모델의 backbone으로 사용되기에는 아래와 같은 문제점들이 있습니다. 

* 굉장히 많은 양의 data 학습이 필요
* Transformer의 특성 상 computational cost가 큼
* feature map을 순차적으로 계산하는 과정 없이 class embedding vector를 MLP head에 통과시켜 최종 결과를 출력하기 때문에, 일반적인 backbone으로 사용하기 어려움







<br>

### DETR

Transformer를 detection task에 적용하려는 시도 중 `DETR` 모델이 있습니다. 

일반적인 2 stage detector의 경우 수많은 roi들 중 nms를 통해 최종 출력할 bbox prediction을 구하게 되는데, DETR에서는 그러한 post process 과정이 없다는 것이 특징 중 하나입니다. 

아래는 DETR의 구조입니다. 

![image-20220331150537816](https://user-images.githubusercontent.com/70505378/161014747-b69e0a5d-a0b0-420a-a0cf-dbd61443fb8d.png) 

DETR 또한 CNN backbone에서 추출된 feature map을 detector의 입력으로 사용하는 2 stage detector라고 할 수 있습니다. 다만, neck이나 RPN을 사용하지 않는 것이 눈에 띕니다. 

앞에서도 말했듯이 transformer는 연산량이 많은 모델이기 때문에, patch의 개수를 최대한 줄이기 위해 input으로는 backbone의 최종 output인 가장 high level의 feature map을 사용합니다. 

논문에서는 224x224의 input image를 사용하고 backbone의 최종 output인 7x7의 feature map을 detector의 입력으로 사용했습니다. 이 때 patch 단위는 각 픽셀로 하여 총 49개의 feature vector를 encoder의 입력값으로 사용했습니다. 

<br>

![image-20220331151114249](https://user-images.githubusercontent.com/70505378/161014750-fa889aa1-8bef-4551-9dfc-6b21c87f739e.png)

DETR에서는 특이하게, 최종 output을 N개로 제한합니다. 이 때 N개는 사용자가 정하는 값으로, 한 이미지 내에 존재하는 object 개수의 최댓값보다 높게 설정을 해주어야 합니다. 

이때 최종 output으로 항상 N개를 출력하게 되는데, 이미지 내의 object 개수는 N개 이하이기 때문에 항상 N:N 매핑이 불가능합니다. 이는 ground truth에서 부족한 obejct 개수만큼 no object로 padding 처리하여, ground truth와 prediction이 N:N 매핑이 되도록 합니다. 

각 예측 값이 N개로 unique하게 나타나기 때문에 post process 과정이 필요없는 것입니다. 

<br>

DETR과 다른 모델 간의 성능 비교는 아래와 같습니다. 전체적인 AP는 상승했으나, highest level feature만 사용하기 때문에 AP<sub>S</sub> 값은 하락한 것을 볼 수 있습니다. 

![image-20220331151551862](https://user-images.githubusercontent.com/70505378/161014753-d7ea9707-5ee7-419b-a425-cc85925ccc1c.png)

DETR 모델은 backbone이 아닌 detector 부분에 transformer를 사용한 모델입니다. 이는 여전히 ViT가 가지고 있던 문제점들이 남아있기 때문이죠. 

이제 마지막으로, transformer를 backbone에 사용한 Swin Transformer 모델에 대해 보도록 하겠습니다. 



<br>

### Swin Transformer

`Swin Transformer`는 CNN과 유사한 구조로 설계되어 feature map을 뽑을 수 있고, **window**라는 개념을 활용하여 computational cost를 줄였습니다. 

이것으로 transformer 구조를 backbone으로 활용할 수 있게 되었죠. 

![image-20220331152119214](https://user-images.githubusercontent.com/70505378/161014756-7b4e8b02-b108-40ca-99e0-7934fba862ac.png)

Swin Transformer의 구조는 위와 같습니다. 구조를 보면 알 수 있듯, CNN과 같이 중간에 input의 크기는 줄이고 채널은 늘려가며 계속해서 feature map을 뽑아내는 것을 알 수 있습니다. 

이러한 swin transformer의 각 부분이 어떻게 동작하는 지 보도록 하겠습니다. 

**Patch Partitioning**

전체 이미지를 patch 단위로 나누고 채널 방향으로 concat합니다. 

![image-20220331153319937](https://user-images.githubusercontent.com/70505378/161014759-5e3ed8d1-bc25-4fd4-adb3-f8f94b904f48.png)







**Linear Embedding**

Vit와 embedding 방식은 동일한데, class embedding을 제거했습니다. 

![image-20220331153454092](https://user-images.githubusercontent.com/70505378/161014762-a2f410e5-861e-48d2-acf9-bb3a105f77c0.png)







**Swin Transformer Block**

W-MSA(Window Multihead Self Attention)과 SW-MSA(Shifted Window Multihead Self Attention)을 연속적으로 사용합니다. 

![image-20220331153612883](https://user-images.githubusercontent.com/70505378/161014765-69e47c5e-6805-457a-a678-8ff5f1e604fe.png)







**Window Multi-head Attention**

![image-20220331154031525](https://user-images.githubusercontent.com/70505378/161014767-d53e418c-3aa3-46ca-861b-5cd78d28ec0c.png)

Window Multihead Self Attention은 embedding을 window 단위로 나누어 input으로 사용합니다. 기존 ViT의 경우 모든 embedding을 input으로 사용했는데, 이에 반해 W-MSA에서는 각 Window 안에서만 독립적으로 Transformer 연산을 수행합니다. 

이것으로 image size에 따라 quadratic하게 증가하던 computational cost를 linear 수준으로 대폭 줄일 수 있게 되었습니다. (아래 수식에서 M은 window size)

![image-20220331154510211](https://user-images.githubusercontent.com/70505378/161014769-90ae1253-1505-4f42-8707-c916e836a29a.png)

하지만, 이것 만으로는 window 안에서만 수행한다는 문제 때문에 receptive field가 제한된다는 문제가 있습니다. 또한 다른 window에 포함되지만 인접한 pixel들은 서로 인접해 있음에도 그 정보를 학습할 수 없게 됩니다. 

![image-20220331154039800](https://user-images.githubusercontent.com/70505378/161014768-709cba6b-4904-4bca-884b-3ad0583f49c3.png)

따라서 이어서 등장하는 SW-MSA에서 이 문제를 해결합니다. 

SW-MSA에서는 window를 cyclic shift를 통해 window size // 2 만큼 우측 하단으로 shift하고 A, B, C 구역을 masking하여 attention 연산이 되지 않도록 합니다. 연산이 진행된 후에는 다시 원래 값으로 되돌립니다. (reverse cyclic shift)

![image-20220331173704850](https://user-images.githubusercontent.com/70505378/161014774-ba6e72d4-b0ce-4359-8c68-fdcb0f5e7d10.png)

결과적으로 SW-MSA를 통해서 window들 사이의 연결성을 나타낼 수 있습니다. 



**Patch Merging**

Patch Merging에서는 input feature map의 h, w를 반으로 줄이고 채널 방향으로 concat 합니다. 이는 다시 linear layer를 통과하여 채널 차원을 맞춰준 후 다음 stage의 input이 됩니다. 

![image-20220331154723828](https://user-images.githubusercontent.com/70505378/161014772-702e7e97-e523-424d-a76d-7320b69e0eda.png)

<br>

아래는 Swin Transformer와 다른 모델들의 성능을 비교한 표입니다. ResNeXt를 backbone으로 사용했을 때보다 모든 부분에서 성능 향상이 나타났음을 볼 수 있습니다. 

![image-20220331154841870](https://user-images.githubusercontent.com/70505378/161014773-c087bc71-7e4a-4db6-b5d1-9463440328c2.png)

Swin transformer는 비교적 적은 양의 데이터로도 학습이 잘 이루어지며, window를 사용하여 computational cost를 대폭 줄였습니다. 

또한 CNN과 비슷한 구조로 설계되어 detection, segmentation 등의 backbone으로 general하게 활용될 수 있습니다. 



























<br>

<br>

# 참고 자료

* Alexey Bochkovskiy, Chien-Yao Wang, Hong-Yuan Mark Liao, “YOLOv4: Optimal Speed and Accuracy of Object
  Detection”
* Zhaowei Cai, Nuno Vasconcelos, “Cascade R-CNN: Delving into High Quality Object Detection”
* David G. Lowe, “Distinctive Image Features from Scale-Invariant Keypoints”
* Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu, Yichen Wei, “Deformable Convolutional Networks”
* Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia
  Polosukhin, “Attention Is All You Need”
* Alexey Dosovitskiy, “AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE”
* Nicolas Carion, Francisco Massa, Gabriel Synnaeve, Nicolas Usunier, Alexander Kirillov, Sergey Zagoruyko, “End-toEnd Object Detection with Transformers”
* Ze Liu, “Swin Transformer: Hierarchical Vision Transformer using Shifted Windows”  

