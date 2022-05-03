---
layout: single
title: "[AITech][Semantic Segmentation] 20220501 - Semantic Segmentation Competition"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 KAIST의 '김현우' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Semantic Segmentation Competition

이번 포스팅에서는 강의에서 소개된 Semantic Segmentation 대회에서 사용할 수 있는 기법들을 정리해봅니다. 

## Basics

### 주의해야 할 사항들

* 디버깅 모드

  1. 샘플링을 통해서 데이터 셋의 일부분만 추출
  2. Epoch를 1~2 정도 설정하여 Loss가 감소하는지 확인
  3. Step에 따라 loss가 제대로 감소한다면 전체 실험 진행

* 시드 고정

  * 실험의 재현을 원한다면 시드 관련 요소들을 모두 고정하는 것이 필요

  ![image-20220503140745389](https://user-images.githubusercontent.com/70505378/166433924-f8c60698-06a2-4d47-9a08-b03729e8225b.png)

* 실험 기록

  * Network 종류, Augmentation 방법, Hyperparameter 등 성능에 영향을 주는 조건을 바꿔가며 실험을 진행한 후, 그 결과를 기록
  * Notion, google 스프레드 시트, github issue 등을 잘 활용하면 편리

* 실험은 한 번에 하나씩

* 팀원마다의 역할 분배

### Validation

* 제출 전 모델의 성능을 평가하고, validation score와 leader board(제출 시) score가 잘 align되는 validation set을 찾는 것이 중요

* 방법

  * Hold out

    ![image-20220503172444085](https://user-images.githubusercontent.com/70505378/166433926-689a264a-0283-4974-868a-65b78252ccdb.png)

    * 장점: 빠른 속도
    * 단점: val set은 학습에 사용하지 못 함

  * K-Fold

    ![image-20220503172332461](https://user-images.githubusercontent.com/70505378/166433925-86f8bfc1-fcbb-4100-8891-f4db3acfc838.png)

    * 장점: 모든 데이터셋을 학습에 활용, Ensemble 효과로 인해 대부분의 경우 모델 성능이 향상
    * 단점: Hold out 방법에 비해 K배의 시간 소요

  * Stratified K-Fold: Fold마다 class distribution을 동일하게 split

    ![image-20220503173044193](https://user-images.githubusercontent.com/70505378/166433929-f9df5329-0011-4b52-98d9-82a415ff639f.png)

  * Group K-Fold

    * Train-Validation 사이 data leakage를 방지하는 방법
    * 각 인물에 대한 이미지가 3개씩 있을 때, train과 validation set에 동일한 인물의 이미지가 들어있으면 cheating이 발생한다. 이를 방지하기 위해 group=인물로 지정할 수 있다. 

    ![image-20220503173100745](https://user-images.githubusercontent.com/70505378/166433930-487a5cb1-8535-4eb6-a9bb-66fc8548a549.png)

### Augmentation

* 기본이자 가장 일반적인 성능 향상을 보이는 기법들: Rotation, Flip, Transpose
* Torchvision.transforms: Pytorch에서 제공하는 transformation
  * Center crop, Color Litter, Grayscale, RandomCrop, RandomHorizontalFlip, RandomResizedCrop, ...
* Albumentations: Classification, Semantic segmentation, Object detection, Pose estimation 등 다양한 task에서 사용 가능 (이미지와 함께 mask, bbox 등을 함께 전달)
  * Resize, RandomRotate90, Cutout, VerticalFlip, ShiftScaleRotate, RandomResizedCrop, Normalize, ToTensor
* 도메인에 맞는 augmentation을 적절히 사용하는 것이 가장 중요
* 이외의 augmentation 기법들
  * Cutout: 이미지의 일부를 가림
  * Gridmask: Cutout의 경우 객체의 중요 부분 혹은 context information을 삭제할 수 있다는 단점을 해결하기 위해, 규칙성 있는 박스를 통해 cutout을 진행
  * Mixup: 두 이미지의 투명도(alpha)를 조절하여 하나의 이미지로 결합 (gt label도 alpha 값 비율에 맞게 변화)
  * Cutmix: Mixup과 다르게 alpha를 조절하는 방식이 아닌, 이미지 일부를 잘라낸 공간에 다른 이미지를 결합하는 것
  * SnapMix: CAM(Class Activation Map)을 이용해 이미지 및 라벨을 mixing하는 방법
  * CropNonEmptyMaskIfExists: Object가 존재하는 부분을 중심으로 crop할 수 있다면 model의 학습을 효율적으로 할 수 있음
  * ObjectAug/CopyPasteAug: Segmentation task에서 object mask를 다른 이미지에 붙여넣어 개수가 적은 클래스의 샘플의 개수를 늘리는 방법

### LR Scheduler

* CosineAnnealingLR

  * Learning rate의 최대값과 최소값을 정해, 그 범위의 학습율을 Cosine 함수를 이용해 스케쥴링하는 방법
  * 최대값과 최소값 사이에서 learning rate를 급격히 증가시켰다가, 감소시키기 때문에 saddle point, 정체 구간을 벗어날 수 있음

  ![image-20220503175542618](https://user-images.githubusercontent.com/70505378/166433934-fabdbc43-e61f-4a0d-8701-788596369dc5.png)

* ReduceLROnPlateau

  * Metric의 성능이 향상되지 않을 때 learning rate를 조절하는 방법
  * Local minima에 빠졌을 때 learning rate를 조절하여, 효과적으로 빠져나옴

  ![image-20220503175822701](https://user-images.githubusercontent.com/70505378/166433935-db53fb33-8c66-438f-83de-83e54243f87e.png)

* Gradual Warmup

  * 학습을 시작할 때 매우 작은 learning rate로 출발해서 특정 값에 도달할 때까지 learning rate를 서서히 증가시키는 방법
  * 이 방식을 사용하면 weight가 불안정한 초반에도 비교적 안정적으로 학습 가능
  * Backbone 네트워크의 weight가 망가지는 것을 방지

  ![image-20220503180140557](https://user-images.githubusercontent.com/70505378/166433936-e1200512-a675-4d0c-af0e-cee19eefb62b.png)

### Optimizer/Loss

* Adam, AdamW, AdamP, Radam

* Lookahead optimizer

  * Adam이나 SGD를 통해 k번 업데이트 후, 처음 시작했던 point와의 interpolation을 통해 최종 위치를 결정
  * Adam이나 SGD로는 빠져나오기 힘든 Local minima를 빠져나올 수 있게 한다는 장점

  ![image-20220503180400075](https://user-images.githubusercontent.com/70505378/166433938-bb4ae832-013c-44d1-b4f3-e940d5e658cc.png)

* 다양항 loss 사용
  * Compound Loss 계열은 imbalanced segmentation task에 강인한 모습
  * MICCAI 2020 HECKTOR Challenge 1등 DiceFocal loss, 2등 DiceTopK loss  

![image-20220503180415848](https://user-images.githubusercontent.com/70505378/166433939-12c66cec-ae14-4b11-b786-6ce497004c97.png)







<br>

<br>

## Advances

### Ensemble

* 5-fold Ensemble
  * 5-Fold Cross validation을 통해 만들어진 5개의 모델을 ensemble
* Epoch ensemble
  * 중간중간 서로 다른 epoch에서 최고 성능을 달성한 weight를 이용해 예측한 후 결과를 ensemble
* SWA (Stochastic Weight Averaging)
  * 각 step마다 weight를 업데이트 시키는 SGD와 달리 일정 주기마다 weight를 평균 내는 방법
* Seed Ensemble
  * Random한 요소를 결정짓는 seed만 바꿔가며 여러 모델을 학습시킨 후 Ensemble하는 방법
* Resize Ensemble
  * Input 이미지의 size를 다르게 학습해 ensemble하는 방법
* TTA (Test time augmentation)
  * Test set으로 모델의 성능을 테스트할 때, augmentation을 수행하는 방법
  * 원본 이미지와 함께 augmentation을 거친 N장의 이미지를 모델의 입력하고, 각각의 결과를 평균
  * ttach 라이브러리

### Pseudo Labeling

1. 모델 학습을 진행
2. 성능이 가장 좋은 모델에 대해 Test 데이터셋에 대한 예측을 진행
   * 이 때 softmax를 취한 확률값이나 softmax를 취하기 전의 값, torch.max를 취하기 전의 값을 예측
   * Test 데이터셋은 모델의 예측값이 threshold(ex. 0.9) 보다 높은 결과물을 이용
3. 2단계에서 예측한 Test 데이터셋과 Train 데이터셋을 결합해 새롭게 학습을 진행
4. 3단계에서 학습한 모델로 Test 데이터셋을 예측

![image-20220503181547981](https://user-images.githubusercontent.com/70505378/166433942-10df3052-aef9-4c43-ba24-2e5793717e64.png)





<br>

<br>

## Competitions

**학습 이미지가 많고 큰 경우 네트워크를 한 번 학습하는 데 시간이 오래 걸려서 충분한 실험을 하지 못 함**

* 학습 속도를 높여줄 필요가 있음

* fp16 (부동 소수점 변경)

* 실험 간소화

  * 일부 데이터 사용
  * 단일 fold로만 검증

* 가벼운 모델 사용

  * Parameter 수가 적은 모델들로 실험하고 최종은 성능이 잘 나오는 모델로 확인

* Input 이미지의 size를 줄여서 실험

  * SANZCR CLIP 대회에서는 downconv라는 커스텀 모듈을 만들어 이미지를 패치 단위로 잘라 concat 함으로써 시간을 단축

    * resize로부터 오는 성능 하락을 어느정도 보완

    ![image-20220503183856760](https://user-images.githubusercontent.com/70505378/166433918-39a0b09a-7d7e-4206-b2c9-14a61e5352bb.png)

  * Window 단위로 잘라서 각각을 input으로 사용

    * Window size > Stride size: overlapping 기법
      * 장점: stride가 작아 그 만큼 많은 input image를 얻어서 다양한 정보를 얻을 수 있음. 이로부터 ensemble 등의 기법 사용 가능. 
      * 단점: 중복되는 정보가 많아 학습 데이터가 늘어나는 양에 비해 성능의 차이는 적고 학습 속도가 오래 걸림
    * Window size = Stride size: Non overlapping 기법

  * Sliding window 적용 시 과도한 배경 영역 추출

    * Background 영역은 조금만 샘플링

  * Edge에 잘린 object들은 유의미하지 않고, 오히려 학습에 방해가 될 수 있음

    * Window의 center 부분만 crop하여 사용
    * Object detection 이후 segmentation 사용

**Label Noise가 있는 경우**

* Segmentation에서는 데이터에 noise가 존재하고 annotation 실수가 존재하는 경우가 다수 존재
* 따라서 data cleansing 과정이 필수적
* Label Noise를 해결하기 위한 연구들도 존재
  * 2020-ECCV - Learning with Noisy Class Labels for Instance Segmentation.
  * 2020-NIPS - Disentangling Human Error from the Ground Truth in Segmentation of Medical Images.
  * 2020-MICCAI - Characterizing Label Errors: Confident Learning for Noisy-labeled Image Segmentation. 
* 이외에도 label smoothing, pseudo labeling을 통한 label preprocessing 등 





























<br>

<br>

# 참고 자료

* 
