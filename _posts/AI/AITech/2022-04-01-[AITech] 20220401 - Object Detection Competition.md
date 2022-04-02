---
layout: single
title: "[AITech][Object Detection] 20220401 - Object Detection Competition"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['mAP', 'Pipeline', 'Validation', 'Augmentation', 'Ensemble&TTA', '대회로 학습하는 방법']
---



<br>

_**본 포스팅은 '송원호' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# Object Detection Competition

지난 포스팅까지 object detection 전반에 대한 학습을 마쳤습니다. 이번 포스팅과 다음 포스팅에서는 실제 object detection 대회에서 어떠한 기법들이 사용되는지, 알아야 할 것은 무엇언지 등에 대해 알아보겠습니다. 

## mAP에 대한 오해

Object detection에서는 mAP가 metric으로 많이 사용됩니다. 그러나 이 mAP에 큰 맹점이 있다는 사실을 아시나요?

아래 그림을 보면 왼쪽의 탐지 성능이 훨씬 나은 것처럼 보이지만, mAP는 오른쪽 탐지 성능에서 훨씬 높습니다. 왜 그럴까요?

![image-20220402175620754](https://user-images.githubusercontent.com/70505378/161390013-a16c3832-4199-42e3-a071-4457c215d629.png)

모델이 예측한 bbox들에 대해 **confidence threshold를 낮추면 낮출수록 PR Curve의 면적은 증가**하게 되고, 따라서 mAP는 계속해서 상승하게 됩니다. 

이를 이해하기 위해 강의에서 제공한 예시 상황을 보도록 하겠습니다. 

### 예시 상황: 8개의 얼굴이 존재하는 어떤 데이터셋에서 총 5개의 얼굴이 검출된 상황

![image-20220402183626677](https://user-images.githubusercontent.com/70505378/161390014-a6fe923c-9cb7-40fa-b84b-376ba6b14e4a.png)

그리고 위 bbox를 confidence로 내림차순 정렬하고, PR curve를 그리기 위해 precision과 recall을 아래와 같이 구할 수 있습니다. 

![image-20220402184326067](https://user-images.githubusercontent.com/70505378/161390017-4c0debe3-5800-41fa-adbb-957bdd1c5d70.png)

우리는 여기서 confidence threshold에 따라 최종 검출된 bbox를 모두 사용할 수도, 일부만 사용할 수도 있습니다. 여기서는 threshold=0.7로 설정한 경우(박스가 적지만 질이 높은 경우)와 threshold=0.1로 설정한 경우(박스가 많지만 질이 낮은 경우)로 case를 나눠보겠습니다. 

**Case 1. Confidence Threshold = 0.7**

Confidence threshold를 높게 설정한다는 것은 질이 높은 bbox들만을 적게 사용한다는 것을 뜻합니다. 

![image-20220402184502267](https://user-images.githubusercontent.com/70505378/161390018-8fb3828c-e829-42c2-a4ce-3cdb593f4fd3.png)

이 경우 PR curve와 AP는 아래와 같이 구해집니다. 

![image-20220402184555372](https://user-images.githubusercontent.com/70505378/161390020-26200789-bbdd-42b9-95dd-e1033f14d068.png)



**Case 2. Confidence Threshold = 0.1**

Confidence threshold를 낮게 설정한다는 것은 질이 낮은 bbox들도 많이 사용한다는 것을 뜻합니다. 

![image-20220402184154957](https://user-images.githubusercontent.com/70505378/161390015-b0bb4b79-68cd-4ec2-93e0-fcf8b15b699c.png)

이 경우에는 PR curve와 AP가 아래와 같이 구해집니다. 

![image-20220402184808798](https://user-images.githubusercontent.com/70505378/161390021-941ce875-039e-46a9-a9ab-8f2e949c69ff.png)

<br>

즉, 우리가 설정한 confidence threshold 미만의 bbox들 중 TP(True Positive)가 하나라도 있다면, AP 점수는 상승할 수 밖에 없습니다. 

따라서 이로부터 모델의 성능과 mAP 점수 사이의 괴리가 발생하는 것입니다. 실제 연구에서는 대부분 0.05를 threshold로 설정하여 mAP를 평가하며, 적용하는 분야의 특징에 따라 threshold나 metric을 조정하는 것이 좋을 것입니다. 

<br>

<br>

## Pipeline 구축

Object detection 뿐 아닌, 모든 모델을 사용하는 competition의 경우 pipeline을 잘 설계하고 구축하는 것이 매우 중요합니다. 본 강의에서 제안하는 pipeline 순서는 아래와 같습니다. 

* EDA
  * task에 적합한 EDA 수행
* 파이프라인 구축
  * 어떤 라이브러리를 사용할 것인가?
  * 직접 구현할 것인가?
  * 공유된 코드를 활용할 것인가?
  * "Dataset -> Submission file"까지의 모든 과정을 구축하는 것을 **파이프라인 구축**이라고 함
* **Validation set 찾기**
* 성능을 올리기 위한 시도 반복

제가 이번 p stage 과정에서 수행한 eda와 구축한 pipeline 은 추후에 따로 포스팅하도록 하겠습니다. 

여기서는 강의에서 강조하는 'validation set 찾기'와 '성능을 올리기 위한 시도 반복'에 대해 보겠습니다. 

<br>

<br>

## Validation set 찾기

Competition에서는 리더보드가 있습니다. 테스트 데이터셋(혹은 별도의 private 데이터셋)으로 모델의 성능을 평가하는 것이죠. 

**좋은 Validation set**이란, validation set의 점수와 리더보드의 점수의 변화 경향이 비슷한 데이터셋을 말합니다. 즉 validation set 점수가 올라가면 리더보드 점수도 올라가고, validation set 점수가 떨어지면 리더보드 점수도 떨어지는 것이 좋습니다. 또한, 절대적인 수치 자체도 비슷한 것이 좋습니다. 

<br>

이러한 validation set을 만드는 방법에는 크게 아래의 세 가지 방법이 있습니다. 

* Random split
* K-fold
* Stratified k-fold
* Group k-fold

**Random split**

* 전체 데이터를 랜덤하게 Train/Valid로 분리
* Train 데이터로는 학습을, Valid 데이터로는 검증을 진행

![image-20220402230148029](https://user-images.githubusercontent.com/70505378/161390022-20da1779-414a-4281-bc0c-77b6603ef6b8.png)

**K-fold**

* 전체 데이터를 일정 비율로 Train/Valid로 분리
* Split 수 만큼의 독립적인 모델을 학습하고 검증

![image-20220402230343829](https://user-images.githubusercontent.com/70505378/161390023-c42fb8a5-7aa6-4035-bf97-baa16135a119.png)

**Stratified k-fold**

* 데이터 분포를 고려하지 않는 K-fold 방식과 달리, fold마다 유사한 데이터 분포를 갖도록 분리
* 데이터 분포가 imbalance한 상황에서 더욱 좋음

![image-20220402230442319](https://user-images.githubusercontent.com/70505378/161390025-d59de5ba-a0f6-4db8-a386-c3fd9e7e0793.png)





**Group k-fold**

* Group(Class)을 기준으로 validation set을 분리
* 하나의 클래스는 전부 validation set, 나머지 클래스들은 전부 train set이 되는 K개(여기서 K는 클래스 개수) fold를 생성

![image-20220402230930104](https://user-images.githubusercontent.com/70505378/161390027-bf3b57df-180f-42e4-9207-367e66505223.png)



<br>

<br>

## 성능을 올리기 위한 시도 반복

### **Data Augmentation**

**Albumentations**

모델의 성능을 올리는 방법 중 하나는 다양한 augmentation을 적용해보고, 최적의 augmentation 조합을 찾는 것입니다. 

유용한 augmentation library로 그 유명한 `Albumentations`이 있습니다. Detection task에서는 이미지에 변형이 가해지면서 ground truth bbox 값 또한 변경 될 수 있다는 것이 문제인데요, albumentations 에서는 image와 ground truth bbox 를 전달해주면 이를 자동으로 처리해줍니다. 

![image-20220402231651826](https://user-images.githubusercontent.com/70505378/161390031-e7c2a682-3582-4c48-b71f-5756fbd13bd2.png)

**Cutmix**

각광받는 augmentation 기법으로 `Mixup`, `Cutout`, `Cutmix`와 같은 방법들이 있습니다. Label smoothing을 적용하여 모델의 일반화 성능을 한층 더 끌어올릴 수 있습니다. 

![image-20220402231827667](https://user-images.githubusercontent.com/70505378/161390034-5037d486-05fd-4c04-8622-de65dbb5be03.png)

그런데 detection에서 cutmix와 같은 방법은 Cropping에 따라 객체가 잘리거나, 객체가 없는 부분만 가져올 수도 있습니다. 

**Mosaic**

이에 대한 대안으로 `Mosaic` 기법이 대두되었습니다. Mosaic augmentation은 4개의 이미지를 그대로 붙여서 1개의 이미지처럼 사용하는 기법입니다. 

![image-20220402232331391](https://user-images.githubusercontent.com/70505378/161390004-44e7dd6f-daa8-4a27-b5ef-191ae4d59d4a.png)

Cutmix 기법과 달리 객체가 잘리거나 사라질 문제도 없고, 4개의 이미지를 하나의 이미지 처럼 사용하기 때문에 batch size를 크게 설정하는 것과 같은 효과를 얻을 수 있습니다. Detection task에서는 GPU memory의 한계로 batch size를 매우 작게 설정해야 할 때가 있는데, Mosaic와 같은 기법을 사용하면 이 문제를 완화할 수 있을 것 같습니다.  

### **Ensemble & TTA**

Ensemble(또는 TTA)은 inference 과정에서 성능을 굉장히 많이 끌어올릴 수 있는 강력한 기법입니다. 여러 모델들을 함께 사용하는 ensemble은 특히나 detection task에서 그 효과가 매우 높습니다. 

Detection에서 사용되는 ensemble 기법으로 크게 세 가지가 있습니다. 

* NMS(Non Maximum Suppresion)
* Soft NMS
* WBF(Weight Box Fusion)

**NMS**

`NMS`에 대해서는 앞선 포스팅들에서 모델들에 대해 설명할 때 함께 설명했었습니다. Detection model에서 roi들을 걸러낼 때 사용했던 방법이죠. 

NMS는 아래와 같이 bbox들을 confidence 순으로 내림차순 정렬하고, 상위 bbox와 같은 클래스이면서 iou가 threshold 이상인 하위 bbox들은 모두 제거하는 방식입니다. 

![image-20220402233738415](https://user-images.githubusercontent.com/70505378/161390007-4a904aa6-30b4-4771-ba19-e00e5e022096.png)

그런데 NMS에서는 다른 bbox들을 제거하기 때문에, iou가 threshold 이상이지만 다른 객체를 나타내고 있는 경우에 문제가 발생할 수 있습니다. 예를 들면 아래와 같습니다. 

![image-20220402233831397](https://user-images.githubusercontent.com/70505378/161390008-1eb77c87-d3c3-46a4-b903-7655691f5e50.png)

**Soft NMS**

따라서 이러한 문제를 해결하기 위해 `Soft NMS`에서는 하위 bbox들을 제거하지 않습니다. 대신에, 하위 bbox들의 confidence score를 낮춰서 추후 최종 bbox들을 선택하는 confidence threshold에서 걸러질 수 있도록 합니다. 

![image-20220402233950574](https://user-images.githubusercontent.com/70505378/161390009-bfba8deb-7b25-4851-bb1b-3b25f5c935ec.png)

**WBF**

현 시점에서 가장 강력한 detection ensemble 방법은 `WBF`입니다. WBF에서는 bbox들을 개별적으로 고려하는 것이 아니라, 이를 적절히 합쳐서 사용합니다. 

WBF는 같은 클래스로 예측한 이웃한 bbox들을 confidence score에 기반하여 그 좌표를 가중평균(weighted summation)해서 하나의 합쳐진 bbox를 만들어냅니다. 

![image-20220402234327289](https://user-images.githubusercontent.com/70505378/161390010-01de9b47-7bfe-4e04-a8bb-b3d5fa692e42.png)

이 때 하이퍼파라미터를 통해 일부 bbox들은 fusion에 사용하지 않는 등의 처리를 할 수 있습니다. 

<br>

위에서 어떤 앙상블 기법이 있는지에 대해 알아봤다면, 이번에는 어떤 모델들을 앙상블에 활용할 수 있는지 알아보겠습니다. 

크게 아래와 같은 방법들이 있습니다. 

* Seed Ensemble
* Framework Ensemble
* Snapshot Ensemble
* Fold Ensemble
* Stochastic Weight Averaging (SWA)

**Seed Ensemble**

`Seed Ensemble`은 randomness를 결정짓는 seed를 바꿔가며 여러 모델을 학습 시킨 후 앙상블하는 방법입니다. 

![image-20220402234922937](https://user-images.githubusercontent.com/70505378/161390011-605953d2-7e5f-48b3-b5cb-becca0f8b67a.png)

**Framework Ensemble**

`Framework Ensemble`은 서로 다른 여러 라이브러리에서의 모델을 앙상블하는 방법입니다. 다양성이 추가될수록 좋습니다. 

**Snapshot Ensemble**

`Snapshot Ensemble`은 동일한 아키텍쳐이지만 서로 다른 local minima에 빠진 신경망을 앙상블하는 방법입니다. 

**Fold Ensemble**

`Fold Ensemble`은 같은 모델을 서로 다른 fold로 학습시킨 결과를 앙상블하는 방법입니다. 

**Stochastic Weight Averaging (SWA)**

`Stochastic Weighted Averaging`은 앞서 본 방법들이 학습이 완료된 모델을 앙상블하는 것과 달리, 학습 과정에서 사용하는 방법입니다. 

이는 서로 다른 모델들의 가중치를 평균을 내서 더 general한 모델을 만들어내는 방법입니다. 

![image-20220402235241579](https://user-images.githubusercontent.com/70505378/161390012-13ed30a9-3abf-4e4c-a2fb-8ee8a81c0fc0.png)

#### github - WBF

다양한 앙상블 기법을 코드 레벨에서 이해하고 싶으신 분들은 아래 사이트를 참고하시기 바랍니다. 

* [https://github.com/ZFTurbo/Weighted-Boxes-Fusion](https://github.com/ZFTurbo/Weighted-Boxes-Fusion)

<br>

<br>

## 컴피티션으로 학습하는 방법

모델의 최고 성능을 이끌어내야 하는 대회에서는 여러 토론과 실험이 매우 중요합니다. 이 과정에서, 모든 이론들을 from scratch로 직접 다 구현하는 것은 불가능에 가깝습니다. 

대회에서는 사용하는 모델의 전체적인 구조와 중요한 하이퍼파라미터에 대한 이해를 빠르게 하고, 실험을 진행하는 것이 좋습니다. 하지만 이렇게 반복적인 실험 만으로는, 해당 모델을 내가 체화했다고는 말 하기 힘들 것입니다. 

대회 중에는 필요한 부분에 대한 이해와 많은 실험을 하더라도, 대회 종료 후에는 대회 중에서 사용했던 모델이나 기법들에 대한 논문을 읽어보고, 이를 이해하여 내 것으로 만드는 과정이 훨~~씬 중요합니다. 

대회에서 사용했던 지식들을 구체화하고 체화하는 과정이 자신의 성장에 큰 도움이 될 것입니다. 









<br>

<br>

# 참고 자료

* 
