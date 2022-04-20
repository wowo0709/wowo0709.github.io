---
layout: single
title: "[AITech][Data Annotation] 20220412 - Text Detection 1"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Regression/Segmentation', 'Character/Word', 'EAST model']
---



<br>

_**본 포스팅은 Upstage의 '이활석' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Text Detection 1

이번 포스팅은 `Text Detection`에 대한 첫번째 포스팅입니다. 

추후에 두번째 Text Detection 포스팅도 작성할 예정입니다. 

## Basics

### Text Detection의 특징

Text Detection의 경우 "Text"라는 단일 클래스를 검출하는 task이기 때문에 classification 없이 위치만 예측하는 문제입니다. 

![image-20220419225421884](https://user-images.githubusercontent.com/70505378/164139833-d7c947d6-7f64-4bfb-9acb-e34ce8b976c7.png)

텍스트 객체의 특징으로는 다음과 같은 것들이 있습니다. 

* 매우 높은 밀도

  ![image-20220419225526112](https://user-images.githubusercontent.com/70505378/164139842-1a3203f6-4db7-427d-9d2c-e27b43bca6c7.png)

* 극단적 종횡비

  ![image-20220419225535881](https://user-images.githubusercontent.com/70505378/164139851-43af7a90-343f-49e2-b9d5-b6cccdb46664.png)

* 특이 모양

  * 구겨진 영역

    ![image-20220419225610253](https://user-images.githubusercontent.com/70505378/164139865-aba1e8ba-2779-4437-9a27-43a00115e2c2.png)

  * 휘어진 영역

    ![image-20220419225557823](https://user-images.githubusercontent.com/70505378/164139857-f50fa744-5188-4ed6-a396-b68ffa345327.png)

  * 세로 쓰기 영역

    ![image-20220419225621440](https://user-images.githubusercontent.com/70505378/164139870-9628c80e-d114-4587-be32-0e6d47deb7cb.png)

* 모호한 객체 영역

  ![image-20220419225640549](https://user-images.githubusercontent.com/70505378/164139871-08609083-f781-4428-8c61-18253fadcd1f.png)

* 크기 편차

  ![image-20220419225658533](https://user-images.githubusercontent.com/70505378/164139878-07a1a5bb-85ab-457e-89e9-e2fb083d12ac.png)





<br>

### 글자 영역 표현법

이와 같이 매우 높은 다양성을 가지는 텍스트 영역은 한 가지의 방법으로 표현하기 매우 어렵습니다. 

글자 영역을 표현할 때는 크게 **사각형 표현**과 **다각형 표현**을 사용할 수 있습니다. 

**사각형 표현**

사각형 표현도 크게 세가지 종류로 나눌 수 있습니다. 

* 직사각형 (RECT, Rectangle)

  * 네 변이 이미지에 평행한 직사각형
  * (x1, y1, width, height) or (x1, y1, x2, y2)

  ![image-20220419230151559](https://user-images.githubusercontent.com/70505378/164139884-a91fe4cf-6fb7-456b-bc5b-e3ff82230994.png)

* 직사각형 + 각도 (Rotated Box)

  * 회전된 직사각형
  * (x1, y1, width, height, 𝜃) or (x1, y1, x2, y2, 𝜃)

  ![image-20220419230202288](https://user-images.githubusercontent.com/70505378/164139890-55807ebf-ed57-49b7-898b-5e6e66576d28.png)

* 사각형 (QUAD, Quadrilateral)

  * 일반 사각형
  * (x1, y1, x2, y2, x3, y3, x4, y4)
  * 첫 글자의 좌상단이 (x1, y1), 그 후 시계방향으로

  ![image-20220419230210902](https://user-images.githubusercontent.com/70505378/164139896-d223bf60-ed82-4f5c-b73c-7e2a73fa2392.png)

**다각형 표현**

또 다른 방법으로 다각형 표현을 사용할 수 있습니다. 

다각형 표현은 사각형 외에 임의의 형태의 영역을 표현할 때 유용합니다. 단 다각형 표현을 사용할 때에는 일반적으로 짝수 개의 point를 이용해야 하고, 상하 점들이 쌍을 이루도록 배치해야 합니다. 

![image-20220419235103034](https://user-images.githubusercontent.com/70505378/164139905-1bf1615d-8970-4334-b727-9f1159d9bb55.png)











<br>

<br>

## Taxonomy

이번 섹션에서는 text detection 기술을 다양한 관점에서 비교 분석 해봅니다. 

### Regression-based vs Segmentation-based

`Regression based text detection`은 흔히 object detection task에서 사용하는 앵커 박스를 이용해 bbox의 위치를 regression하는 형태입니다. 

각 grid마다 미리 정의된 scale/ratio의 anchor box를 이용해 bbox 영역을 뽑야내고, ground truth와 유사하도록 그 형태를 학습합니다. 

![image-20220420002333194](https://user-images.githubusercontent.com/70505378/164139938-20a63ab1-66fa-4a97-bf9f-12e7a8470dd5.png)

Text Detection에서 이러한 regression based method가 가지는 단점은 아래와 같습니다. 

* Arbitrary shaped text: 구겨지거나 휘어지는 등 글자가 불규칙할 경우 불필요한 영역을 포함 (Bounding box 표현 방식의 한계)
* Extreme aspect ratio: 매우 긴 텍스트 영역이 존재할 경우 bounding box의 정확도 하락 (Receptive field의 한계)

![image-20220420001559910](https://user-images.githubusercontent.com/70505378/164139914-396fe4f3-3504-42cb-9b76-d83edbef9ae1.png)

<br>

Regression 방법과 달리, `Segmentation-based text detection`에서는 이미지를 입력 받아 글자 영역 표현값들에 사용되는 픽셀 단위 정보를 뽑고, 후처리를 통해 최종 글자 영역 표현값들을 확보합니다.  

![image-20220420002259799](https://user-images.githubusercontent.com/70505378/164139928-e14d4782-a765-40e9-9488-2ff056821955.png)

좀 더 자세히 보면, 아래 예시와 같이 각 pixel에 대해 해당 pixel이 글자 영역에 속할 확률과 함께 인접한 8개 방향의 pixel이 글자 영역에 속할 확률을 계산합니다. 

![image-20220420002805494](https://user-images.githubusercontent.com/70505378/164139943-d19cc1b7-2270-41c6-b4cf-5de5d2d09588.png)

픽셀 단위 정보들을 얻은 후에는 후처리를 통해 글자 영역 표현값들을 얻어냅니다. 후처리는 아래 세 단계에 걸쳐 이루어집니다. 

* Binarization: 특정 threshold를 넘는 확률을 가지는 픽셀은 1로, 넘지 않는 확률을 가지는 픽셀은 0으로 이진화
* CCA(Connected Component Analysis): 이진화를 적용한 '글자 영역에 속할 확률 맵'과 앞서 구한 '8개 방향으로 이웃한 화소가 글자 영역에 속할 확률 맵'을 이용하여 CCA를 적용. 픽셀 단위 글자 영역 표현값 반환.  
* RBOX Fitting: 픽셀 단위 글자 영역 표현값을 RBOX 형태의 표현값으로 변환. 

![image-20220420003906852](https://user-images.githubusercontent.com/70505378/164139953-048ea037-0907-463d-b5cf-6c6c691df347.png)

이러한 segmentation-based method의 단점은 아래와 같습니다. 

* 복잡하고 시간이 오래 걸리는 후처리 과정이 필요할 수 있음
* 서로 겹치거나 인접한 글자 영역 간 구분이 어려움

![image-20220420004015587](https://user-images.githubusercontent.com/70505378/164139962-2b5576e1-a3b2-40e5-a408-770dc251be36.png)



<br>

그래서 최근에는 두 방법을 함께 사용하는 Hybrid method들도 많이 연구되고 있습니다. 

Hybrid method에서는 regression-based로 대략의 사각 영역을 추출한 뒤, segmentation-based로 해당 영역에서 화소 정보를 추출합니다. 

![image-20220420004352027](https://user-images.githubusercontent.com/70505378/164139966-32ddbf86-f5ff-4673-9195-4505f287203d.png)

대표적인 방법으로는 2018년 발표된 MaskTextSpotter라는 방법이 있습니다. 해당 방법에서는 Fast R-CNN과 Mask branch를 사용해 regression과 segmentation을 모두 활용합니다. 

![image-20220420004611050](https://user-images.githubusercontent.com/70505378/164139968-659c6448-ab2b-4c3c-9f73-0a5264414f7c.png)

<br>

### Character-based vs Word-based

다른 관점으로, Character-based text detection과 Word-based text detection으로 나눌 수 있습니다. 

`Character-based text detection`의 경우 character 단위로 글자 영역을 검출하고, 이를 조합해서 word instance를 예측해야 합니다. 따라서 character 단위의 gt label 값이 필요합니다. 

![image-20220420004852459](https://user-images.githubusercontent.com/70505378/164139981-20d202f7-12e6-4a0d-a896-52023bcfcd4a.png)

2019년 발표된 CRAFT라는 방법은 segmentation 기반의 character based method입니다. 글자 별로 그 위치를 예측하고, 추가로 글자 간 연결성에 대한 정보를 계산합니다. 그리고 두 정보를 사용하여 단어 영역을 추출합니다. 

CRAFT에서는 단어 단위 라벨링으로부터 글자 단위 라벨링을 생성해내는 weakly-supervised learning을 사용했다는 것이 또 하나의 특징적인 점입니다. 

![image-20220420005343726](https://user-images.githubusercontent.com/70505378/164139995-ae7c6824-3e38-4457-a910-364d4c40569b.png)

<br>

`Word-based text detection`의 경우 word 단위로 글자 영역을 검출하며, 이는 현재 대부분의 모델들이 사용하고 있는 방법이기도 합니다. 

![image-20220420004903252](https://user-images.githubusercontent.com/70505378/164139988-236726c9-dbac-410c-b69b-c63e561a8d2e.png)



















<br>

<br>

## EAST model

> **_EAST: An Efficient and Accurate Scene Text Detector_**

### Introduction

EAST는 2017년에 발표된 논문으로, text detection에 있어 Software 2.0 방식으로 동작하여 최초로 높은 성능을 낸 모델입니다. 

![image-20220420110614770](https://user-images.githubusercontent.com/70505378/164140006-4e3025f0-b021-4149-b14c-7b314f0604ff.png)

EAST는 segmentation 기반으로 동작하며, 픽셀 단위로 정보를 추출합니다. 픽셀 단위 정보로는 아래 2가지를 추출합니다. 

* **글자 영역 중심에 해당하는지**: Score map
* **Bounding box의 위치는 어디인지**: Geometry map (해당 픽셀이 글자 영역일 때만 추출)

![image-20220420110823522](https://user-images.githubusercontent.com/70505378/164140012-d5cd6ef2-2985-47c3-a145-b71bc65282bc.png)



<br>

### Architecture

모델 전체 architecture는 아래와 같이 segmentation에서 널리 사용되는 구조인 U-Net 구조를 보입니다. 

![image-20220420111015501](https://user-images.githubusercontent.com/70505378/164140018-3ae7f2da-54da-449b-93d3-e525eabb6807.png)

EAST는 크게 세 부분으로 구성되어 있습니다. 

1. Feature extractor stem (backbone)
   * PVANet, VGGNet, ResNet50
2. Feature merging branch
   * Unpooling&Concat
   * Adjust channel dimension by 1x1, 3x3 convolution
3. Output
   * shape: H/4 x W/4 x C



<br>

### Output

**Score map**

Score map은 크기 H/4 x W/4 x 1의 binary map(글자 영역의 중심이면 1, 아니면 0)입니다. 추론 시에는 binary map이 아닌 probability map을 반환합니다. 

이는 gt bbox의 가로 세로를 축소시켜서 생성합니다. (글자 높이의 30%만큼 end points를 안 쪽으로 이동)

![image-20220420111430737](https://user-images.githubusercontent.com/70505378/164140038-695da23d-f70e-4c0a-b3eb-774e390ec255.png)

**Geometry map**

Geometry map은 RBOX 형식 또는 QUAD 형식으로 표현할 수 있습니다. 

RBOX 형식에서는 각 픽셀(글자 영역에 해당하는 픽셀)은 5 channel(회전 각도 + bbox의 4개 경계선까지의 거리)의 값을 가집니다. 

![image-20220420111745278](https://user-images.githubusercontent.com/70505378/164140046-8d70d8f9-ca0f-426c-a694-4285deecb91d.png)

![image-20220420111948099](https://user-images.githubusercontent.com/70505378/164140051-5832785b-b1f3-426f-a802-7eca8747841e.png)

QUAD 형식에서는 글자 영역에 해당하는 각 픽셀에 대해 8 channel(bbox의 4개 꼭짓점까지의 offset(x,y))의 값을 가집니다. 

![image-20220420112201580](https://user-images.githubusercontent.com/70505378/164140057-7613ded1-f321-4171-ba10-f1448fcdfc6b.png)





<br>

### Inference

픽셀 단위 정보(Score map, Geometry map)를 추출한 뒤에는 후처리를 수행합니다. 후처리는 아래 단계로 구성됩니다. 

1. Score map 이진화
2. 사각형 좌표값 복원
3. 다수의 영역 후보들에 대해 NMS 적용

![image-20220420112748038](https://user-images.githubusercontent.com/70505378/164140063-7555a922-ac48-433f-9d3b-de243bd646e7.png)

2단계에서 영역 후보가 매우 많을 경우 기존의 NMS 방식은 연산량이 많이 요구됩니다. 따라서 EAST에서는 **Locality-Aware NMS**라는 새로운 방식을 제안합니다. 

<br>

### Locality-Aware NMS

기존의 standard NMS는 O(N^2)의 복잡도를 가집니다. 

Locality-aware NMS는 인접한 픽셀이 예측한 bbox는 동일한 text instance에 대한 bbox일 확률이 높다는 데 기반합니다. 

따라서 먼저 행 단위로 탐색하며 IOU 기반으로 비슷한 bbox들을 하나로 통합합니다. 이때 통합 시에는 score map 값으로 weighted merge를 수행합니다. 

이 연산을 모두 수행한 후에, standard NMS를 적용합니다. 

![image-20220420113258948](https://user-images.githubusercontent.com/70505378/164140071-87bd2794-41b5-4c22-82f7-d774d9eb7905.png)







<br>

### Training

EAST의 loss function에는 score map loss term과 geometric map loss term이 존재합니다. 

![image-20220420113414956](https://user-images.githubusercontent.com/70505378/164140076-88db748e-4f0e-4788-9e54-957ea3dadcdd.png)

EAST 논문 상에서는 `Ls`로 class-balanced cross-entropy를 사용하였는데, 실제 구현 시에는 segmentation task에서 주로 사용되는 dice-coefficient 등을 사용할 수 있습니다. 

![image-20220420113653793](https://user-images.githubusercontent.com/70505378/164140089-2a6f0d6f-49f5-451a-bd30-febc22ec6d44.png)

`Lg`로는 bbox 값은 IoU loss, 각도 값은 consine loss를 사용합니다. 

![image-20220420113808053](https://user-images.githubusercontent.com/70505378/164140099-3fb9f966-65b0-40d4-aa47-0555ed037eae.png)

![image-20220420113841637](https://user-images.githubusercontent.com/70505378/164140101-26db2ed2-af6d-47d0-bc50-276c0fede37e.png)









<br>

### Results

EAST는 real-time 수준까지는 아니지만 빠른 속도를 보여주며, LA-NMS이 속도 개선에 효과적임을 보여주었습니다. 

아래 표에서 T1은 네트워크 계산 시간, T2는 후처리(LA-NMS) 시간을 나타냅니다. 

 ![image-20220420114050622](https://user-images.githubusercontent.com/70505378/164140107-28690cf6-f6e5-4999-8616-d8d2b510cef4.png)

아래 이미지는 EAST의 text detection 시각화 이미지입니다. 

![image-20220420114320342](https://user-images.githubusercontent.com/70505378/164140117-82d31e3f-8f4d-442e-b30b-524701c1b806.png)

현재는 TextFuseNet, CharNet 등의 최신 모델이 ICDAR 2015 벤치마크 데이터셋에 대해 SOTA 성능을 달성하고 있습니다. 

![image-20220420114752519](https://user-images.githubusercontent.com/70505378/164139809-89347584-8970-4686-aa89-f9779d4b0bd3.png)















<br>

<br>

# 참고 자료

* 
