---
layout: single
title: "[Papers][CV][Face Detection] UperNet: Unified Perceptual Parsing for Scene Understanding 논문 리뷰"
categories: ['AI', 'AI-Papers']
tag: []
toc: true
toc_sticky: true
---



<br>

# UperNet: Unified Perceptual Parsing for Scene Understanding

이번에 리뷰할 논문은 2018년 발표된 `UperNet: Unified Perceptual Parsing for Scene Understanding`이라는 제목의 논문으로, semantic segmentation task와 관련된 논문입니다. 

목차는 아래와 같습니다. 

* Introduction
* Related Work
* Datasets&Metrics
* Overall Architecture
* Experiments
* Conclusion

<br>

<br>

## Introduction

먼저 UperNet에 대해 알기 위해선 본 논문에서 제안하는 기법은 **Unified Perceptual Parsing**이 무엇을 뜻하는지부터 알아야 할 필요가 있습니다. 

이는 주어진 이미지에서 가능한 많은 시각적 요소들을 인식하는 task로, 주어진 이미지로부터 효과적으로 여러 종류의 정보들을 추출해내는 것을 말합니다. 그리고 이는 segmentation 분야에 활용될 수 있다고 본 논문에서는 주장하고 있습니다. 

그렇다면 왜 이러한 접근 방식이 필요할까요?

이는 조금 더 human-like 한 모델 구조를 만들기 위해서라고 할 수 있습니다. 인간은 하나의 장면으로부터 여러 단계에 걸쳐 여러 정보들을 추출하여 이미지를 인식합니다. 

그러나 본 논문에서는 대부분의 모델들이 하나의 visual task에만 specific하게 동작한다는 것을 지적하며, 이 대신에 여러 visual 요소들을 계층적으로 처리하는 Unified Perceptual Parsing task를 제안했습니다. 

이는 크게 2가지 특징적 요소로 설명할 수 있습니다. 

1. 매 iteration 마다 임의의 visual task에 해당하는 데이터를 뽑고, 모델을 학습. 모델은 해당 task과 관련된 layer들에서만 학습. 
2. 하나의 네트워크 내에서 계층적인 정보들을 활용

아래 이미지를 보시면 본 논문에서 활용하는 5개의 visual task가 나와있는데요, 이는 총 Scene, Object, Part, Material, Texture 로 구성됩니다. 

![image-20220504152922629](https://user-images.githubusercontent.com/70505378/166702034-a49a044a-85d0-450a-9bbc-503b9f9df28c.png)





<br>

<br>

## Related Work

그럼 다음으로 논문에서 소개하는 관련 연구들에 대해 간단히 짚고 넘어가겠습니다. 

**Semantic segmentation**

* CRF: Segmentation task가 완료된 후 각 픽셀 간 인접한 정도와 분류 결과를 이용해 그 결과를 정교화하는 과정
* FCN: Segmentation에서 기본적인 encoder 구조로 사용되는 네트워크. Linear layer를 convolutional layer로 모두 치환하여 segmentation task를 수행. 
* Deconvolution: Convolution 연산에 의해 줄어든 resolution을 transposed convolution을 이용해 원본 크기로 복원. 
* Dilated convolution: Receptive field를 효과적으로 확장하기 위해 kernel 사이에 padding을 삽입하여 convolution 연산을 수행. 
* RefineNet: Down-sampling 과정에서 생성되는 feature map들의 정보를 모두 활용하여 예측 수행  
* PSPNet: 다양한 이미지 정보를 추출하기 위해 Spatial pyramid pooling 제안. 

**Multi task learning**

Multi-task learning은 모델이 동시에 다양한 task에 대해 학습하도록 하여 모델의 적용성과 성능을 올리려는 시도를 위해 사용됩니다. 

이미 multi-task learning과 관련하여 여러 연구들이 진행되어 왔고, UberNet이라는 모델에서는 한 번에 7가지 다른 task를 이용해 모델을 학습하는 시도를 했었습니다. 

본 논문의 UperNet 모델은 더 나아가 이질적인 데이터셋으로 multi-task learning을 수행하려는 시도를 하였고, 동시에 계층적인 정보들을 활용하여 성능을 높이려는 시도를 했습니다.  



<br>

<br>

## Datasets & Metrics

그래서 먼저 논문에서 어떤 데이터셋을 이용하여 multi-task learning을 수행했는지에 대해 보겠습니다. 

기존에 존재하던 Broden이라는 데이터셋은 아래 5개의 데이터셋을 통합하여 서로 다른 5개 task에 대한 annotation을 포함하는 데이터셋입니다. 

* ADE20K, Pascal-Context, Pascal-Part, Open Surfaces, DTD (Texture)

이 중 object, part, material에 해당하는 task는 pixel level로 annotation 되었고, texture와 scene에 해당하는 task는 image level로 annotation 되어 있다고 합니다. 

![image-20220504154308898](https://user-images.githubusercontent.com/70505378/166702039-4ede39fd-ea82-4bd2-a39c-76881e209eef.png)

논문에서는 이 데이터셋을 그대로 사용하지 않고, multi-task learning에 적합한 형태로 refine하여 새로운 Broden+라는 데이터셋을 생성하였습니다. Refine process는 아래와 같습니다. 

1. Merge similar concepts across different datasets 
2. Only include object classes which appear in at least 50000 pixels (Object parts which appear in at least 20 images considered as valid parts)
3. Manually merge under-sampled labels in Open-Surfaces
4. Map more than 400 labels in ADE20K to 365 labels

<br>

다음으로 각 task에 대한 성능을 평가하기 위해 서로 다른 metric들을 사용하였습니다. 

Segmentation의 성능을 측정하기 위해 pixel accuracy와 mean IoU를 사용하였는데, refine 과정에서 발생하는 이미지 상의 unlabeled area는 계산 시 제외하였습니다. 

![image-20220504154340002](https://user-images.githubusercontent.com/70505378/166702043-ba2a076c-fd2b-4178-bf7c-49263b1a80c7.png)







<br>

<br>

## Overall Architecture

![image-20220504154354783](https://user-images.githubusercontent.com/70505378/166702048-13e58bb0-308a-4d92-a5c4-056885012871.png)

모델 전체 구조는 위와 같습니다. 5개의 task를 동시에 학습하는 만큼 모델은 5개의 branch로 이루어져 있습니다. 

모델 backbone으로는 resnet을 사용하였으며, 풍부한 정보를 얻기 위해 Neck 구조로 FPN을 사용하고 동시에 Pyramid pooling module도 사용하였습니다. 

<br>

첫번째 scene branch에서는 장면에 대한 classification을 수행합니다. 이 때 backbone model의 highest level feature map만을 사용합니다. 

feature map에 3x3 convolution을 적용하고 global average pooling을 적용하여 classification을 수행합니다. 

<br>

두번째와 세번째 branch인 object branch와 part branch는 묶어서 보겠습니다. 

두 branch에서는 FPN의 feature map들을 fusion하여 사용합니다. 이는 low-level feature부터 high-level feature까지 다양한 정보를 이용하기 위한 것임을 알 수 있습니다. 

마찬가지로 3x3 convolution을 적용하고 classification을 수행하는데, 이 때의 classification은 pixel-wise로 수행됩니다. 

<br>

네번째 material branch에서는 FPN에서 resolution이 가장 큰 feature map만을 가져와서 사용합니다. 

Object와 part branch 학습 과정에서 object에 대한 정보가 학습되며, 이는 material branch의 예측에 도움을 줍니다. 동시에 여러 feature map들을 fusion 하는 것보다는 마지막 feature map만을 사용하는 것이 성능이 더 좋았는데, 이는 local feature를 어느정도 살리는 것이 도움이 됨을 알 수 있습니다. 

<br>

마지막 branch는 texture branch입니다. 

해당 branch에서는 특이하게 다른 branch들과 동시에 학습하지 않습니다. 대신에 다른 branch들의 학습이 완료된 후에 해당 branch만 별도로 fine-tuning 과정을 수행합니다. 

Fine-tuning을 수행할 때에도 classifier만이 학습되고, backbone이나 neck에서는 학습을 하지 않습니다. 







<br>

<br>

## Evaluation

![image-20220504155400916](https://user-images.githubusercontent.com/70505378/166702051-f61985e6-410a-4394-950b-dcdc06994fcf.png)

보시는 평가표는 기존의 모델들과 UperNet의 성능을 비교한 표입니다. 

사실 UperNet의 성능 자체는 기존 SOTA 모델인 PSPNet의 성능을 뛰어넘지 못했습니다. 다만 논문에서 강조하는 것은 dilated convolution 대신에 backbone의 convolution을 그대로 유지하고, deconvolution 대신 interpolation을 사용함으로써 속도 면에서 훨씬 좋은 성능을 기록했다는 것입니다. 속도의 향상을 기록하면서도 성능 면에서도 SOTA에 준하는 성능을 기록했다는 것이 논문이 강조하는 바입니다.  

<br>

![image-20220504160344637](https://user-images.githubusercontent.com/70505378/166702056-b2a11cc3-8e66-4c24-a918-2b6de8d4054d.png)

두번째로 제공한 평가표는 다음과 같습니다. 

Training data에서 O는 Object, P는 Part, S는 scene, M은 material, T는 texture 에 해당합니다. 

저는 사실 이 부분을 읽기 전까지 다양한 task를 동시에 학습하면서 모델의 성능을 올렸을 것이라 생각하고 논문을 읽었는데, 이 평가표를 보면 다양한 task를 동시에 학습할 수록 각 task에 대한 성능이 조금씩 하락하는 것을 알 수 있습니다. 

논문에서는 다양한 task를 동시에 학습함으로써 task 간 feature 관계를 확인할 수 있었다고 합니다. 











<br>

<br>

## Conclusion

그래서 마지막으로 논문의 결론은 아래와 같습니다. 

1. FPN, PPM 등을 활용하여 성능 향상
2. Dilated convolution, Deconvolution 대신 3x3 convolution, interpolation을 사용하여 성능을 비슷하게 유지하면서도 속도 향상
3. 이질적인 데이터셋에서의 multi-task learning을 수행하고 이것이 가능함을 실험적으로 증명
4. 여러 task를 동시에 수행할 수 있는 하나의 통합된 형태의 모델 구조 제안

종합하면 본 논문에서 의의는 segmentation task에 있어 multi-task learning을 이용할 수 있고, 계층적인 정보를 활용하는 구조를 제안했다는 것으로 정리할 수 있을 것 같습니다. 또한 dilated convolution이나 deconvolution을 사용하지 않아도 성능을 비슷하게 유지할 수 있고, 속도 면에서 큰 향상을 얻을 수 있다는 것도 실험적으로 보여주었습니다. 

마지막 첨언으로, 2021년에 나온 `knet`이라는 모델 논문이 있는데 해당 논문에서도 Unified parsing 방법을 사용하는 것 같아서 읽어보면 도움이 될 것 같습니다. 

<br>

<br>













