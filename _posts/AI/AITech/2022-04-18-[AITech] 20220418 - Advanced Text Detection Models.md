---
layout: single
title: "[AITech][Data Annotation] 20220418 - Advanced Text Detection Models"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['DBNet', 'MOST', 'TextFuseNet']
---



<br>

_**본 포스팅은 Upstage의 '이활석' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Advanced Text Detection Models

이번 포스팅에서는 EAST 모델 이후로 발표된 Text Detection model들을 소개합니다. 

## DBNet

![image-20220422140947363](https://user-images.githubusercontent.com/70505378/164906270-4369a1e8-6f62-4b2b-b270-fdb7f8243fba.png)

`DBNet`(Differentiable Binarization Net)은 2020년에 발표된 추론 속도에 중점을 둔 모델입니다. 

Segmentation 기반의 text detection 방법들은 다양한 모양의 텍스트를 유연하게 잡아낼 수 있지만, 인접한 개체 구분이 어렵다는 단점이 있습니다. 

이전 연구들에서는 이를 극복하기 위해 **Pixel Embedding** 기법을 사용하여 후처리 단계에서 embedding 된 정보를 사용하여 단어들을 구분하도록 했습니다. 

![image-20220422140319570](https://user-images.githubusercontent.com/70505378/164906258-822da23f-19d6-48f6-bef0-f183cb4e8f88.png)

DBNet에서는 이 대신에 **Adaptive thresholding** 기법을 제안합니다. 이 기법은 글자 영역이 구분되는 기준인 threshold 값을 모델이 학습하도록 하는 기법입니다. 

Threshold를 학습시키기 위해서는 threshold에 대한 ground truth 값이 필요합니다. 이러한 threshold 값은 글자 영역 내에서의 글자 간 threshold와 글자 영역 간의 threshold로 나눌 수 있습니다. 

논문에서는 글자 영역 경계 부분에서는 높은 threshold 값을 적용하여 서로 다른 글자 영역이 합쳐지지 않도록 하고, 나머지 영역에 대해서는 낮은 threshold를 적용하였습니다. 

![image-20220422141453900](https://user-images.githubusercontent.com/70505378/164906282-0b6a00f5-f1bf-43dd-b2e3-e9c839458823.png)

이렇게 threshold map을 학습을 통해 얻고자 할 때 가장 문제가 되는 것은 threshold를 통한 binarization 과정이 미분 불가능하기 때문에 end-to-end 학습이 불가하다는 것입니다. 

본 논문에서는 이를 극복하기 위해 미분이 가능하면서도 최대한 standard binarization에 가까운 differentiable binarization을 제안하였습니다. 

![image-20220422141830211](https://user-images.githubusercontent.com/70505378/164906295-a693df3f-9704-49ff-a89f-e4304594f137.png)

DBNet의 전체 training pipeline은 아래와 같이 구성됩니다. Segmentation map, Binarization map은 GT probability map과 BCE loss를 구성하고, Threshold map은 GT threshold map과 L1 loss를 구성합니다. 

![image-20220422142321505](https://user-images.githubusercontent.com/70505378/164906307-cae69d66-00a9-467d-af0a-bbad584fe5dd.png)

결과는 아래와 같습니다. 각 이미지마다 오른쪽 위는 threshold map이고 오른쪽 아래는 probability map을 나타낸 것입니다. Threshold map은 글자 영역 경계 부분에서만 높은 값을 가지고, Probability map은 각 글자 중심부에서만 높은 값을 가지는 것을 확인할 수 있습니다. 

![image-20220422142344067](https://user-images.githubusercontent.com/70505378/164906318-f78e0945-3fee-42b0-8e61-2017888b7ea8.png)

DBNet은 MLT(Nulti-lingual scene text), SROIE(Scanned Receipts OCR and Information Extraction) 데이터셋에서 1등 솔루션을 차지하고 있는 모델입니다. 

해당 결과들은 **RRC (Robust Reading Competition, OCR 전문 학회인 ICDAR에서 격년으로 개최하는 competition)** 사이트에서 확인할 수 있습니다. RRC에서는 task마다 좋은 성능을 보이는 모델들을 확인할 수 있으며, 제출 시 debugging 기능 (Method click - samples list, per sample details)도 제공하기 때문에 유용하게 사용할 수 있습니다. 



<br>

<br>

## MOST

`MOST`(Multi-Oriented Scene Text detector)는 2021년에 발표된 EAST 모델의 후속 모델로, EAST 모델의 구조를 수정하여 성능을 개선하는 것에 중점을 둡니다. 

EAST는 단순하고 빠르지만, extreme aspect ratio sample에 대한 detection 성능이 많이 떨어집니다. 이는 EAST가 가지는 **receptive field의 한계**와 **LA-NMS의 문제점**이라고 할 수 있습니다. 

![image-20220422143916433](https://user-images.githubusercontent.com/70505378/164906332-7612a4de-6ece-4ebe-97b3-4b9518e9a021.png)

MOST는 이러한 EAST의 한계를 몇 가지 모듈의 도입을 통해 극복하려 했습니다. 

* `TFAM`: Receptive field의 제약을 없애기 위해 coarse detection으로 대략의 위치를 알아낸 후 그에 맞게 receptive field를 재조정하여 NMS 전 최종 검출 결과 확보
* `PA-NMS`: NMS에 필요한 글자 영역 내 픽셀들의 상대 위치 정보를 받아서, 검출 결과에 PA-NMS 적용

### TFAM

`TFAM` 모듈은 localization branch에 존재하며 input으로 coarse detection의 출력값을 받습니다. 이때 Coarse detection의 출력값은 EAST에서의 geometry map(H/4 x W/4 x 5 (상하좌우 거리 + 각도))에 해당합니다. 

![image-20220423215017738](https://user-images.githubusercontent.com/70505378/164906342-1e56714c-92fc-46aa-8a8f-64e1e1739e57.png)

또한 TFAM에서는 deformable convolution을 사용하는데, 기존의 original deformable convolution 연산이 Feature-based sampling이라고 하면 TFAM에서 사용하는 연산은 **Localization-based sampling**입니다. 

이는 Coarse detection에서 얻은 값에 따라 box의 위치(좌표)를 추정할 수 있기 때문에, 이를 기반으로 offset 값을 설정(uniform sampling)하는 것을 말합니다. 

![image-20220423215045778](https://user-images.githubusercontent.com/70505378/164906353-a0faca93-ac44-4e2f-aa2e-0396b650df43.png)





### PA-NMS

EAST에서 사용했던 LA-NMS는 추정된 글자 영역의 score를 기반으로 일정 IoU 이상의 영역들을 먼저 합치고 NMS를 적용했습니다. 

MOST에서는 이렇게 영역들을 합칠 때 사용하는 **position-aware merge** 함수를 새롭게 제안했습니다. 

![image-20220423215400098](https://user-images.githubusercontent.com/70505378/164906365-5db6e538-1f19-40e2-b653-94829be521d5.png)

LA-NMS의 문제점은 score를 기반으로 영역들을 weighted sum하기 때문에 종횡비가 큰 영역들에 대해 성능이 좋지 않았습니다. 

![image-20220423215618763](https://user-images.githubusercontent.com/70505378/164906120-e41fb8d3-1625-4102-9e40-72d5f5ab5979.png)

PA-NMS에서는 score 값 대신 position을 고려할 수 있는 새로운 값을 사용합니다. 이를 위해 position-sensitive map이라는 것을 사용하는데, 이는 EAST에서 사용하던 geometry map과 유사합니다. 

Geometry map에서 글자 영역의 네 변까지의 '절대 거리'와 각도 값을 사용했다면, position-sensitive map에서는 각도 값을 사용하지 않고 글자 영역 네 변까지의 '상대 거리'(0~1 사이의 값)를 사용합니다. 

![image-20220423220410908](https://user-images.githubusercontent.com/70505378/164906147-d72b09da-1cce-4dfe-b32a-2494528b9d13.png)

이는 실제 경계에서 가까운 곳에서 예측한 정보일수록 더 높은 가중치를 주어 영역을 합치는 것에 해당합니다. 

![image-20220423220612154](https://user-images.githubusercontent.com/70505378/164906163-cdaae0ac-f234-4d0f-b1d3-55c649f22512.png)

<br>

다음으로 Training 부분에서 MOST는 coarse detection 시의 loss에도 변화를 주었습니다. 

기존 EAST에서는 IoU loss를 사용해 글자 영역 그대로 계산하는 반면, MOST에서는 Instance-wise IoU loss를 사용하여 예측 글자 영역의 크기에 따른 정규화를 수행했습니다. 

이는 글자 영역을 크게 예측했을 때 성능이 높게 측정되는 것을 방지하며, 검출기의 성능 평가 방식과도 연결되어 있습니다. 

![image-20220423221217085](https://user-images.githubusercontent.com/70505378/164906174-53e5183d-8278-4aa5-8100-e0f05a78e3b3.png)

아래는 loss term과 함께 MOST의 전체 training pipeline을 나타낸 것입니다. 

![image-20220423221324513](https://user-images.githubusercontent.com/70505378/164906183-edd150de-17cd-40aa-8194-6af9e93c62b4.png)

아래는 동일한 이미지에 대해 EAST(위 이미지)와 MOST(아래 이미지)의 검출 결과를 비교한 모습입니다. 

종횡비가 긴 글자 영역에 대해 MOST가 더 잘 검출해내는 것을 확인할 수 있습니다. 

![image-20220423221621519](https://user-images.githubusercontent.com/70505378/164906195-e38f6236-6c29-48f6-ac26-aee071332bee.png)



<br>

<br>

## TextFuseNet

마지막으로 살펴볼 `TextFuseNet`은 2020년에 발표된 또 다른 text detection 모델입니다. TextFuseNet의 이름에는 다양한 feature map들을 조합하여 검출 성능을 올리겠다는 의미가 담겨있습니다. 

기존 instance segmentation 기반의 방법에서는 mask rcnn 구조를 text detection에 맞게 수정한 모델들을 사용했습니다. 따라서 mask rcnn과 동일하게 RoI aligned feature를 기반으로 mask segmentation을 수행했습니다. 

하지만 이럴 경우 영역이 끊어졌을 때 이를 합치기 어렵다는 문제가 있습니다. 

 ![image-20220423222152466](https://user-images.githubusercontent.com/70505378/164906207-b3b76a06-2c4a-4171-8018-1aa05e49e588.png)

TextFuseNet에서는 해당 문제를 word 기반 특징의 한계라고 지적하며 이를 극복하기 위해 이미지 전반의 특징인 **global 특징을 함께 사용해야 한다**는 제안을 하였습니다. 

TextFuseNet의 아이디어를 요약하면 semantic segmentation branch에서는 global level feature를 얻어내고,  detection branch에서는 word/character level feature를 얻어내서 **3가지의 feature map들을 조합하여 학습**함으로써 더욱 정교화된 결과를 얻어내는 것입니다. 

모델이 다양한 feature를 학습하도록 하여 향상된 성능을 기대할 수 있고, 실제 검출 결과로는 글자 영역(word instance) 결과만을 사용합니다. 

![image-20220423223400331](https://user-images.githubusercontent.com/70505378/164906221-68307d41-db16-48fc-ba2a-b5725f205d7a.png)

아래는 word/character/global level feature들이 합쳐지는 모습을 나타낸 그림입니다. 

![image-20220423223726683](https://user-images.githubusercontent.com/70505378/164906232-818b789f-cc0e-4f4c-90a5-7f0f3d588009.png)

<br>

TextFuseNet의 학습은 weakly supervised learning이라고 표현됩니다. 

TextFuseNet에서는 detection branch 학습을 위해 character level annotation이 필요합니다. 논문에서는 직접적인 annotation을 하는 과정을 없애기 위해 모델의 학습을 3단계로 나누어 진행했습니다. 

1. 합성데이터(SynthText)로 글자 영역 supervised pretraining
2. 1단계에서 학습한 모델로 글자 영역 pseudo labeling
   * GT 단어 영역과 pseudo label 글자 영역의 IoU 값이 0.8 이상인 data만 사용
3. 2단계에서 생성한 pseudo labeled data들과 단어 영역 annotation data들을 함께 사용하여 fine-tuning

이렇게 학습된 TextFuseNet은 다양한 벤치마크 데이터셋에 대해 훌륭한 검출 성능을 보여줍니다. 

![image-20220423224429761](https://user-images.githubusercontent.com/70505378/164906242-d451a453-9f08-4056-851a-2a40700c7811.png)

TextFuseNet은 발표된 이래로 아직까지도 ICDAR 2015 Incidental Scene Text - Localization 부문에서 1위를 차지하고 있습니다. 

























<br>

<br>

# 참고 자료

* 
