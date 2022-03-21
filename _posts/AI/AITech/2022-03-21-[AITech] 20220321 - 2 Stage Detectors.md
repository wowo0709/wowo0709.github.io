---
layout: single
title: "[AITech][Object Detection] 20220321 - 2 Stage Detectors"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 '송원호' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# 2 Stage Detectors

Level2의 P-stage 중 첫번째, Object detection stage가 시작되었습니다. 

이번 포스팅에서는 localization과 classification을 순차적으로 수행하는  `2 Stage Dectectors`에 대해 알아봅니다. 

## R-CNN

`R-CNN`은 객체 검출을 제안한 최초의 모델이며, 이후에 많은 모델들이 이 R-CNN에 기반하여 발전되었습니다. 따라서 R-CNN을 잘 이해하는 것이 아주 중요하다고 할 수 있습니다. 

R-CNN의 구조는 아래와 같습니다. 

![image-20220321141313168](https://user-images.githubusercontent.com/70505378/159255833-ccb1fd3a-b466-4a08-9acf-733cee7ada53.png)

### Extract Region Proposals

위에서의 두번째 단계인 `Extract Region Proposals`에서는 객체가 있을 만한 후보 영역들을 모두 추출하여 각 영역을 고정된 크기의 이미지로 변환해서 CNN에게 전달해줍니다. 

**Sliding Window**

`Sliding Window` 방식은 아래 그림처럼 고정된 크기의 window를 이미지 상에서 sliding 시키며 모든 영역을 찾는 것입니다. 

하지만 이 기법은 이미지의 대부분이 배경일 때 필요없는 연산을 하는 횟수가 너무 많아지고, 물체의 크기에 따라 다르게 예측하기도 어려워서 지금은 잘 사용되지 않습니다. 

![image-20220321141736987](https://user-images.githubusercontent.com/70505378/159255836-595366ee-c165-4c02-80b1-b69321f99124.png)

**Selective Search**

R-CNN에서 실질적으로 사용하는 영역 제안 알고리즘은 `Selective Search` 알고리즘입니다. 

아래 그림처럼 처음에는 최소 단위의 픽셀 별로 각 영역을 나누고, 점차 인접 영역들을 비슷한 것들끼리 합치면서 영역을 제안하게 되는 알고리즘입니다. 

![image-20220321142144243](https://user-images.githubusercontent.com/70505378/159255838-867d7970-a51a-4709-88db-5db26d633b8c.png)

### Inference

R-CNN은 크게 아래의 5단계에 걸쳐 학습을 진행합니다. 

1. 입력 이미지를 받는다. 
2. Selective search를 통해 2000개의 ROI(Region of Interest)를 추출한다.
3. 각 ROI를 모두 동일한 크기로 resize한다. 
   * RCNN 논문에서는 CNN 모델로 AlexNet을 사용했기 때문에 (3,227,227)로 resize합니다. 
4. ROI를 CNN에 넣어 feature를 추출한다. 
   * CNN을 통과한 각 ROI를 4096-dim의 feature vector가 된다. 
5. 분류 및 회귀를 진행한다. 
   * 분류: 각각의 4096 dimensional feature vector에 대해 Linear SVM으로 이진 분류를 수행합니다. 따라서 분류하려는 객체의 수(C+1, 물체+배경)만큼의 SVM이 필요하며, 각 SVM은 output으로 class와 sconfidence score를 반환합니다. 
   * 회귀: 각 feature vector를 입력으로 받아 regression을 수행해서 bounding box를 예측합니다. 

![image-20220321143723432](https://user-images.githubusercontent.com/70505378/159255844-bc064184-1ba7-452a-8ad5-7cd65b78db1d.png)

### Training

R-CNN은 CNN 모델로 사용되는 AlexNet, Linear SVM, Bbox regressor 모델들을 따로 학습시키기 때문에 end-to-end 모델이 아닙니다. 이 모델들을 어떻게 학습시키는지 보도록 하겠습니다. 

**AlexNet**

* Domain specific finetuning
* Dataset 구성
  * IoU > 0.5 + ground truth: positive samples
  * IoU < 0.5: negative samples
  * Positive samples 32, negative samples 96  

**Linear SVM**

* Dataset 구성
  * Ground truth: positive samples
  * IoU < 0.3: negative samples
  * Positive samples 32, negative samples 96
* Hard negative mining
  * 이미지에서 사람을 탐지하는 경우 사람은 positive sample, 배경은 negative sample에 해당한다. 모델 추론 시 '사람이 아니라고 예측했을 때 실제로 사람이 아닌 경우의 샘플'을 **'true negative sample'**, '사람이 아니라고 예측했지만 실제로는 사람인 경우의 샘플'을 **'false positive sample'**이라고 한다. 
  * 객체 탐지 시 negative sample의 수가 우세하기 때문에, 클래스 불균형으로 인하여 모델은 주로 false positive 오류를 주로 범하게 된다(배경이라고 예측할 확률이 높기 때문에). 
  * 이를 해결하기 위해 처음 linear SVM을 학습시킬 때의 false positive sample들을 epoch마다 학습 데이터에 추가하여 학습을 진행함으로써 모델이 강건해지고, false positive 오류가 줄어든다. 

**Bbox regressor**

* Dataset 구성
  * IoU > 0.6: positive samples
* Loss function
  * MSE Loss  

<br>

R-CNN을 요약하면 다음과 같습니다. 

* Selective search를 통해 2000개의 ROI 추출
* 2000개의 ROI를 각각 CNN 통과
* 강제 warping, 성능 하락 가능성
* CNN, Linear SVM, bbox regressor 따로 학습 -> end-to-end X





<br>

## SPPNet

R-CNN은 혁신적이지만, 그 한계 또한 분명합니다. 

* 각 ROI를 고정된 크기로 resize해야 하기 때문에 그 과정에서 성능이 떨어질 수 있습니다. 
* 하나의 이미지에 대해 2000개의 ROI가 모두 CNN을 통과해야 하므로 시간이 오래 걸립니다. 

`SPPNet`은 이 두가지를 해결하고자 했습니다. 아래는 R-CNN과 SPPNet을 비교한 그림입니다. 

![image-20220321150120707](https://user-images.githubusercontent.com/70505378/159255808-39e3e7ce-98ab-4315-bb3d-7b007c8e69e5.png)

SPPNet은 R-CNN과 달리 입력 이미지가 CNN을 먼저 거치고 ROI를 추출한 뒤, 각 ROI의 크기를 고정된 크기의 벡터로 변환해 FC layer에 전달해줍니다. 

### Spatial Pyramid Pooling

SPPNet을 이해하기 위해서는 **어떻게 ROI를 추출하는지**와 **어떻게 ROI를 고정된 크기로 변환하는지**에 대해 알아야 합니다. 전자는 ROI prejection으로 뒤에서 나올 Fast R-CNN에서 알아보도록 하고, 여기서는 후자인 **Spatial Pyramid Pooling** Layer에 대해 알아보겠습니다. 

결국 spatial pyramid pooling layer의 역할은 각기 다른 크기의 ROI들을 고정된 크기의 vector으로 변환하는 것입니다. 그리고 이는 아래와 같이 각 ROI를 (4, 4), (2, 2), (1,1) 크기의 영역으로 나눠서 각 영역에 대해 pooling을 진행함으로써 달성할 수 있습니다. 최종적으로는 각 ROI 마다 21개의 256-dim feature vector를 얻게 됩니다. 

![image-20220321151053943](https://user-images.githubusercontent.com/70505378/159255816-ea9f9967-1612-46a9-a780-82ba777b4263.png)

<br>

SPPNet 의 특징은 아래와 같습니다. 

* ROI projection -> 2000개의 ROI 각각이 CNN을 통과하는 과정을 삭제 -> 시간 단축
* Spatial pyramid pooling -> 강제 warping 과정을 삭제 -> 성능 향상
* CNN, Linear SVM, bbox regressor 따로 학습 -> end-to-end X







<br>

## Fast R-CNN

`Fast R-CNN`은 그 구조가 SPPNet과 상당히 유사합니다. 

![image-20220321151752195](https://user-images.githubusercontent.com/70505378/159255820-00ca0f35-95ed-41a8-b771-189fe718c6ae.png)

### ROI Projection

먼저 SPPNet에서도 사용된 ROI projection이 무엇인지 알아보겠습니다. ROI projection의 목적은 CNN을 한 번만 통과하면서 feature map 상에서 ROI를 얻어내는 것입니다. 

입력 이미지에서 ROI를 먼저 구하고, CNN을 거쳐서 나온 feature map에 이 ROI들을 projection(투영)하여 feature map 상에서의 ROI 영역을 찾아냅니다. 이 과정은 단순히 feature map의 크기에 맞춰 ROI의 크기를 키우거나 줄이는 연산만으로 가능합니다. 

![image-20220321152519117](https://user-images.githubusercontent.com/70505378/159255824-4b494854-e930-445f-ae69-6246b36f5119.png)

### ROI Pooling

SPPNet에서는 ROI projection을 이용해 얻은 ROI들을 spatial pyramid pooling layer에 통과시켜 고정된 크기의 벡터를 얻었습니다. 

Fast R-CNN은 그 대신, 단순히 ROI를 (7, 7) 크기로 나눠서 각 영역을 pooling하여 사용합니다. 

### Training

Fast R-CNN은 더 이상 분류기로 SVM을 사용하지 않습니다. 대신에 Softmax classifier를 사용해서 CNN, classifier, bbox regressor를 모두 한 번에 학습시키는 데 성공했습니다. 하지만, 입력 이미지에서 ROI를 뽑아내는 selective search 과정은 학습이 필요한 과정이 아닌 CPU 상에서 돌아가는 알고리즘이기 때문에, 아직 end-to-end라고는 할 수 없습니다. 

* multi task loss 사용
  * (classification loss + bounding box regression)
* Loss function
  * Classification : Cross entropy
  * BB regressor : Smooth L1
* Dataset 구성
  * IoU > 0.5: positive samples
  * 0.1 < IoU < 0.5: negative samples
  * Positive samples 25%, negative samples 75%  
* Hierarchical sampling
  * R-CNN의 경우 이미지에 존재하는 RoI를 전부 저장해 사용
  * 한 배치에 서로 다른 이미지의 RoI가 포함됨
  * Fast R-CNN의 경우 한 배치에 한 이미지의 RoI만을 포함
  * 한 배치 안에서 연산과 메모리를 공유할 수 있음  

<br>

Fast R-CNN을 요약하면 다음과 같습니다. 

* ROI prejection & ROI pooling -> 시간 단축, 성능 향상
* CNN, (softmax) classifier, bbox regressor를 한 번에 학습
* 여전히, end-to-end는 아님

이제 Faster R-CNN이 입력 이미지에서 ROI를 뽑아내는 과정을 학습 가능한 과정으로 대체해 완전한 end-to-end 학습이 가능하도록 할 것 같지 않나요? ㅎㅎ 

<br>

## Faster R-CNN

`Faster R-CNN`은 입력 이미지로부터 ROI를 추출하기 위해 selective search 알고리즘 대신 **RPN(Region Proposal Network)**을 사용하여, 모델을 end-to-end 구조로 만들었습니다. 

아래 그림은 Fast R-CNN과 Faster R-CNN의 모델 구조를 비교한 모습입니다. Faster R-CNN은 RPN에 Fast R-CNN module이 그대로 더해진 구조라고 할 수 있습니다. 

![image-20220321181029881](https://user-images.githubusercontent.com/70505378/159255826-e1d0402d-0ff8-460b-bf23-aeb749fa7a1f.png)

### Anchor Box

Faster R-CNN에서는 1차적으로 ROI들을 추출하기 위해 **Anchor box**라는 것을 사용합니다. Anchor box란 미리 정해진 몇 개의 box로, 이를 이용해 ROI 후보들을 제안하게 됩니다. 여기서 '후보'라고 표현한 이유는 이 단계에서 anchor box에 의해 추출된 모든 ROI들이 Fast R-CNN module에 전달되는 것은 아니기 때문입니다. 

입력 이미지는 CNN을 거쳐 작아진 크기의 feature map 형태로 출력됩니다. 이 feature map의 각 grid cell에 대해 사전에 정의한 k개의 anchor box들을 적용합니다. 따라서 이 단계에서 총 HxWxK 개의 ROI 후보들이 제안됩니다. 

예를 들어 아래와 같은 입력 이미지가 들어왔다고 가정하겠습니다. 입력 이미지의 크기는 (800, 800)이고, CNN을 거쳐 (8, 8) 크기의 feature map이 출력됩니다. 이 (8,8) 크기의 feature map의 각 grid cell(8x8=64)에 대해 anchor box를 적용해서, 총 8x8x9 개의 ROI 후보들이 제안됩니다. (여기서 anchor box의 수가 9입니다)

![image-20220321181126803](https://user-images.githubusercontent.com/70505378/159255827-ae6f48ff-8b68-4f9d-be95-44bac5266d9b.png)

### Region Proposal Network

앞서 anchor box에 의해 추출된 ROI들을 모두 사용하는 것은 비효율적입니다. **RPN이 바로 이 모든 ROI 후보들 중 객체가 있을 확률이 높은 N개의 ROI를 추출해주는 네트워크**입니다. RPN은 각 ROI 후보들에 해당하는 영역의 objectness score를 계산하고 bbox regression을 수행합니다. 

자세한 과정은 아래와 같습니다. 

1. CNN에서 나온 feature map을 input으로 받는다. 
2. **3x3 conv**를 수행하여 intermediate layer를 생성한다. 
3. **1x1 conv**를 수행하여 binary classification과 bbox regression을 수행한다. 
   * classification head: 각 grid cell마다 2k개의 채널 생성
     * 2(object or not) x k(num of anchors)
   * bbox regression head: 각 grid cell마다 4k개의 픽셀 생성
     * 4(bounding box) x k(num of anchors)
4. 3에서의 결과에 NMS를 적용하여 선택된 ROI들만을 Fast R-CNN module에 입력으로 전달한다. 

![image-20220321182717149](https://user-images.githubusercontent.com/70505378/159255829-4e726413-58b0-4d29-b0cb-35acf6014fd2.png)



논문에서는 3가지 크기와 3가지 비율을 조합하여 총 9개의 anchor box를 사용했습니다. 

![image-20220321182055391](https://user-images.githubusercontent.com/70505378/159255828-602a8348-a3eb-4b45-aca7-e1018cb3fe47.png)



### Non Maximum Supression

NMS는 RPN 과정에서 얻게 되는 여러 개의  ROI들 중 가장 적합한 ROI를 선택하는 알고리즘입니다. NMS를 거쳐 선택된 ROI들 만이 Fast R-CNN module에 전달됩니다. 

![image-20220105181432226](https://user-images.githubusercontent.com/70505378/148225736-a515aef0-42cc-4c06-86df-193457caaa2a.png)

그 과정은 아래와 같습니다. 

1. bounding box 별로 지정한 confidence score threshold 이하의 box 제거
2. 남은 bounding box를 confidence score에 따라 내림차순 정렬. 그 다음 confidence score가 높은 순의 bounding box부터 다른 box와의 IoU 값을 조사하여 IoU threshold 이상인 box를 모두 제거(포함하는 영역이 비슷하면서 confidence score가 더 낮은 bounding box는 조사할 필요가 없기 때문)
3. 2의 과정을 반복하여 남아있는 box만 선택

### Training

**RPN 학습 시**

* RPN 단계에서 classification과 regressor학습을 위해 앵커박스를 positive/negative samples 구분
* 데이터셋 구성
  * IoU > 0.7 or highest IoU with GT: positive samples
  * IoU < 0.3: negative samples
  * Otherwise : 학습데이터로 사용 X  

RPN 학습 시에는 아래와 같은 형태의 Loss funciton을 사용합니다. 

![image-20220110222120009](https://user-images.githubusercontent.com/70505378/148775735-d6f08c17-d377-46b0-9ba7-35d01ff5e05a.png)

> *i*: index of an anchor
> p<sub>i</sub>: predicted probability
> t<sub>i</sub>: parameterized coordinates
> N<sub>cls</sub>: mini-batch size
> N<sub>reg</sub>: the number of anchor locations
> *λ*: balancing parameter (default = 10)
> L<sub>cls</sub>: cross-entropy loss
> L<sub>reg</sub>: L1 smooth loss
>
> 윗첨자로 쓰인 *는 ground-truth를 의미한다.

p<sub>i</sub>* = 1 인 경우 positive sample이고, p<sub>i</sub>* = 0인 경우 negative sample입니다. 

* Positive sample: IoU가 가장 높은 anchors 또는 IoU가 0.7 이상인 경우
* Negative sample: IoU가 0.3 이하인 경우
* Ignore sample: positive도 negative도 아닌 anchors로 학습에 사용되지 않는다. (-1로 라벨링)
* IoU는 anchor와 ground truth의 간의 계산으로 얻어진다. 

RPN을 학습하기 위한 mini-batch는 하나의 이미지에서 얻은 anchors 중에서 positive 128개 + negative 128개 = 256개로 구성합니다. 

* 하나의 이미지에서 얻은 WxHxk 개의 anchor box들 중 이미지 경계를 벗어나는 anchor box들을 제거한다. (-1로 라벨링)
* 위의 positive/negative 기준에 따라 sample labeling을 수행한다. 
* positive/negative samples에서 128개씩 무작위로 뽑아 mini-batch를 생성한다. 

참고로, ignore labeling을 하는 이유는 WxHxk 개의 anchor 중에서 256개의 anchor 만을 이용해서 loss를 구하지만, RPN의 output과 ground-truth의 차원을 맞춰줘야 하기 때문입니다. 

**Fast R-CNN module 학습 시**

* Region proposal 이후 Fast RCNN 학습을 위해 positive/negative samples로 구분
* 데이터셋 구성
  * IoU > 0.5: positive samples → 32개
  * IoU < 0.5: negative samples → 96개
  * 128개의 samples로 mini-bath 구성  
* Loss 함수
  * Fast RCNN과 동일  

**전체 모델 학습 시**

Faster R-CNN은 end-to-end 모델이기 때문에 한 번에 학습이 가능합니다. Faster R-CNN에서 제안하는 학습 방식은 **4-Step Alternating Training**입니다. 

![image-20220110223246553](https://user-images.githubusercontent.com/70505378/148775736-62e7f3fb-279e-45c4-a76a-c80300627e4c.png)

위 사진에서 파란색 네트워크는 갱신이 되지 않는 네트워크입니다. 

1. RPN training 방식을 통해 학습을 한다. 이때, pretrained VGG도 같이 학습한다. 
2. (1)에서 학습된 RPN을 이용해 RoI를 추출하여 Fast RCNN 부분을 학습한다. 이때, pretrained VGG도 같이 학습한다. 
3. (1)과 (2)로 학습된 pretrained VGG를 통해 추출한 feature map을 이용해 RPN을 학습한다. 이때, pretrained VGG와 Fast RCNN 부분은 학습하지 않는다. 
4. (1)과 (2)로 학습된 pretrained VGG와 (3)을 통해 학습된 RPN을 이용해 Fast RCNN을 학습한다. 이때 pretrained VGG와 RPN은 학습하지 않는다. 

다만, 학습 방법이 매우 복잡하기 때문에 최근에는 Approximate Joint Training을 활용해 학습합니다. 

<br>

이로써, Faster R-CNN은 초기 R-CNN이 가졌던 문제점들을 모두 극복할 수 있었습니다. 하지만 **2 stage detector들은 기본적으로 localization과 classification을 순차적으로 수행하기 때문에, real-time으로 사용하기에는 무리가 있습니다.**

아래는 R-CNN, Fast R-CNN, Faster R-CNN을 비교한 표입니다. 

![image-20220321184659804](https://user-images.githubusercontent.com/70505378/159255831-ae7581b5-b44e-48f2-bf52-9c0505509aee.png)







<br>

<br>

# 참고 자료

* 
