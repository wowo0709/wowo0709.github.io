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

이번에 리뷰할 논문은 2020년 발표된 `Anomaly Detection` 분야의 논문인 `DROCC: Deep Robust One-Class Classification` 이라는 제목의 논문입니다. 

목차는 아래와 같습니다. 

* Terminology
* Introduction
* Related Work
* Anomaly Detection
* One-class Classification with Limited Negatives (OCLN)
* Evaluation
* Conclusion

## Terminology

본격적인 논문 리뷰에 앞서 Anomaly detection에서 등장하는 용어들에 대해 보도록 하겠습니다. 

### Anomaly Detection

![image-20220424183508276](https://user-images.githubusercontent.com/70505378/165060401-f996dbde-86a8-47d6-8ddf-b3a2d8adac00.png)

`Anomaly detection`은 다수의 normal sample과 소수의 abnormal sample을 구별해내는 문제입니다. 그리고 이는 세 가지 기준에 따라 나눠 볼 수 있습니다. 

**Training data에 따른 분류**

먼저 training data에 따른 분류입니다. 

**Supervised anomaly detection**은 정상 sample과 비정상 sample의 label이 모두 존재하는 경우를 말합니다. 정확도가 비교적 높다는 장점이 있지만, labeling 시간이나 비정상 sample 취득의 어려움, class-imbalance 문제 등 해결해야 하는 문제들이 존재합니다. 

**Semi-supervised anomaly detection**은 One-class classification이라고도 하며, 정상 sample로만 학습을 진행하는 경우를 말합니다. Semi-supervised learning에서는 학습한 정상 boundary 밖에 있는 sample들을 모두 비정상 sample로 간주합니다. 이 경우에도 비정상 sample 없이 정상 sample들만을 최대한 다양하게 모아야 한다는 문제점이 있을 수 있습니다. 

**Unsupervised anomaly detection**은 labeling 없이 다량의 sample로 학습시키는 방법입니다. 이는 주로 GAN이나 auto encoder 등 데이터의 주성분을 학습할 수 있는 모델을 사용합니다. 

Unsupervised anomaly detection은 다량의 sample에서 비정상적인 sample은 매우 소량이기 때문에 이에 대한 특징을 학습시키는 것 만으로 정상 sample들에 대한 특징을 추출할 수 있다는 주장에 기반합니다. 따라서 unlabeled data로 학습을 진행한 후, input과 복원된 output 간의 차이로 비정상 sample을 구별합니다. 

해당 방법은 data labeling 과정이 불필요하기 때문에 편리하지만, 다른 두 방법에 비해 정확도가 비교적 낮고 하이퍼파라미터에 따라 성능이 크게 좌우되는 방법이기도 합니다. 

<br>

**Abnormal sample의 정의에 따른 분류**

다음으로 Abnormal sample의 정의에 따라 novelty detection과 outlier detection으로 나눌 수 있습니다. 

**Novelty detection**은 이전에 없던 형태의 sample을 찾아내는 방법이고, 해당 sample은 novel sample 또는 unseen sample이라고 합니다. 

**Outlier detection**은 학습한 데이터와 전혀 관련 없는 비정상적인 sample을 찾아내는 방법이고, 해당 sample은 outlier sample 또는 abnormal sample이라고 합니다. 

예를 들어 골든 리트리버, 닥스훈트, 도베르만, 말티즈의 총 4가지 종류의 강아지 sample으로 모델을 학습시켰을 때 test 시 불독 이미지는 novel sample에 해당하고 토끼 이미지는 outlier sample에 해당합니다. 

하지만 위 정의는 자주 혼재되어 사용되기 때문에 그렇게 잘 정의된 용어는 아니라고 합니다. 

<br>

**Normal sample의 class 개수에 따른 분류**

마지막으로 Normal sample의 class 개수에 따라 분류할 수도 있습니다. 

Normal sample이 하나의 클래스로 구성되어 있는 경우 **one-class normal sample detection**에 해당하고, 여러 개 클래스로 구성되어 있는 경우 **multi-class normal sample detection**에 해당합니다. 

출처: [https://hoya012.github.io/blog/anomaly-detection-overview-1/](https://hoya012.github.io/blog/anomaly-detection-overview-1/)

<br>

### OCLN (One-class Classification with Limited Negatives)

![image-20220424183738449](https://user-images.githubusercontent.com/70505378/165060408-c6c8d3c2-7119-4b12-8972-f5940b04e9fc.png)

`OCLN`은 anomaly detection에서 조금 더 진보된 real world와 유사한 형태의 문제라고 할 수 있습니다. 

실제로 모든 경우에 해당하는 sample들을 다 모으기는 불가능합니다. OCLN은 이렇게 제한적인 negative sample 하에서 positive sample을 잘 구별해내도록 하는 목적이 추가된 문제라고 할 수 있습니다. 이때 positive sample은 내가 찾고자 하는 단 하나의 클래스에 속하는 sample을 말하고, negative sample은 이외의 클래스에 속하는 sample들을 말합니다. 

OCLN의 주요 목적은 낮은 FPR을 기록하면서, 동시에 높은 recall을 기록하는 것입니다. 

실생활 속의 OCLN task로 아이폰, 갤럭시에서 시리, 빅스비를 부를 때 반응하도록 하는 문제를 들 수 있습니다. 

낮은 FPR을 기록한다는 것은 '시리'와 유사한 '수리'나 '시린' 등의 입력이 들어왔을 때 이를 '시리'로 인식하지 않아야 한다는 것을 말합니다.  





<br>

<br>

## Introduction

이제 본격적으로 본 논문에 대한 설명을 시작하겠습니다. 

먼저 본문에서 제안하는 DROCC는 모델 아키텍쳐가 아닌, anomaly detection을 위한 하나의 '방법론'임을 먼저 밝힙니다. 실제로 본문에서는 모델 아키텍쳐로 다양한 backbone 모델을 사용할 수 있다고 이야기합니다. 

<br>

그런데 여기서 한 가지 의문이 들 수 있습니다. Normal sample과 abnormal sample을 구별하는 anomaly detection과 binary classification의 차이점은 무엇일까요?

이는 크게 2가지 차이점을 들 수 있습니다. 

1. \# of abnormal samples
2. Multi-class normal sample

첫번째 차이점으로 abnormal sample의 수를 들 수 있습니다. Anomaly detection에서는 비정상 데이터의 수가 정상 데이터의 수에 비해 현저히 낮습니다. 이렇게 불균형이 심각한 데이터로 단순한 binary classification을 수행한다면 모델은 성능을 높이기 위해 모든 데이터를 정상 데이터로 분류하도록 학습하게 될 수 있습니다. 

두번째 차이점으로 normal sample이 하나가 아닌 여러 개 클래스로 구성되어 있을 수 있다는 점입니다. A, B, C, D 클래스가 존재한다고 할 때 A~C 클래스를 하나의 클래스로 묶고 D 클래스와 binary classification을 수행한다면 좋은 성능을 보이지 못 할 수 있습니다. 

<br>

본문에서는 기존에 있던 anomaly detection 방법들에 비해 DROCC는 다양한 domain에 적용이 가능하며, 다른 side-information을 요구하지 않는다고 말하고 있습니다. 



<br>

<br>

## Related Work

그럼 이제 기존에 사용되던 anomaly detection 방법에는 무엇이 있는지 보도록 하겠습니다. 

첫번째로 **Generative Modeling** 방법이 있습니다. 

이는 GAN 또는 AutoEncoder 모델을 사용하여 수행할 수 있습니다. 하지만 이러한 방법은 latent space로부터 이미지를 복원해내는 decoding step이 추가되기 때문에 더 어려운 문제를 풀어야 한다는 문제점이 있습니다. 

두번째로 **SVM**을 사용하는 방법이 있습니다. 

실제로 SVM을 활용한 DeepSVDD라는 논문은 one-class classification을 목적으로 한 첫번째 모델이기도 합니다. 

하지만 linear model인 SVM을 사용할 경우 representation collapse 문제가 발생하기 쉽습니다. Representation collapse는 간단히 학습한 데이터의 feature와 테스트 시 데이터의 feature가 다를 때 이에 제대로 대응하지 못 하는 문제입니다. 

세번째로 **Transformation** 기반의 방법이 있습니다. 여기서 말하는 transformation은 데이터에 가하는 변환을 의미합니다. 

이는 self-supervised method로, sample에 다양한 변환을 가해서 모델이 sample에 가해진 변환이 무엇인지 맞히도록 학습하는 방법입니다. 이 방법에서 sample은 모델이 가해진 변환을 제대로 예측했을 경우에만 정상 sample로 간주됩니다. 

이러한 transformation 기반의 방법은 상당히 domain에 의존적이라는 문제점이 있습니다. 

<br>

그럼 이제 본 논문에서 제안한 DROCC 방법에 대해 알아보도록 하겠습니다.  





<br>

<br>

## Anomaly Detection

DROCC에서 anomaly detection은 특정 feature를 -1에서 1 사이의 값으로 대응시키는 task로 간소화시킬 수 있습니다. 이때 -1은 비정상 sample, 1은 정상 sample에 대응합니다. 

![image-20220424185633782](https://user-images.githubusercontent.com/70505378/165060411-fac7a96e-db59-4135-bc02-0838af18c1b7.png)

DROCC는 기본적으로 다음 가설에 기반합니다. 

> *The set of typical points S lies on a low dimensional locally linear manifold that is well-sampled.*
>
> _In other words, outside a small radius around a training (typical) point, most points are anomalous._

즉, 학습 데이터로 사용되는 typical data는 더 작은 feature 수를 가지는 매니폴드 상에 투영할 수 있고, 이 매니폴드 위에 있지 않은 sample들은 anomaly data에 해당한다는 것입니다. 

이로부터 DROCC 방법은 기본적으로 학습 데이터로 normal data만을 사용하는 semi-supervised method라는 것을 알 수 있습니다(논문에서는 이를 unsupervised method라고 표현하고 있습니다). 

<br>

DROCC에서 loss term과 negative sample 집합은 아래 두 식과 같이 표현됩니다. 

![image-20220424185830576](https://user-images.githubusercontent.com/70505378/165060412-92e461af-8b2c-4409-bd3b-7e8f0dc7a860.png)

Loss term은 매우 간단하게 모델의 가중치에 대한 L2 규제 항과 정상 sample을 1로, 비정상 sample을 -1로 매핑하는 것에 대한 손실값을 포함합니다. 

이때 Negative sample은 보는 것과 같이 모든 정상 sample로부터 `r` 이상, `γ · r` 이하로 떨어진 sample로 정의됩니다. 

그런데 앞서 DROCC는 학습 데이터에 비정상 sample들이 포함되어 있지 않다고 가정했었습니다. 그렇다면 학습 단계에서 어떻게 모델에 비정상 sample들을 줄 수 있을까요?

<br>

본문에서는 이러한 negative sample들을 주어진 positive sample로부터 생성하는 방법을 제안했고, 이것이 DROCC의 핵심 아이디어입니다. 

Negative sample 집합에 해당하는 N<sub>i</sub>(r)이 있다고 했을 때, 생성된 negative sample들은 해당 영역에 존재해야 할 것입니다. 앞서 negative sample은 모든 positive sample로부터 거리가 r 이상 떨어져야 한다고 했는데, 실제로 negative sample 생성 시에는 모든 positive sample에 대해 거리를 계산하는 것은 매우 오랜 시간이 소요되기 때문에 이 대신에 **gradient ascent** 방법을 사용했습니다. 

Gradient ascent에 대한 설명은 뒤에서 하도록 하겠습니다. 

어쨌든, DROCC는 positive sample로부터 negative sample을 생성해내는 방법을 제안했고, 이에 대한 수식은 아래와 같습니다. 

![image-20220424191049129](https://user-images.githubusercontent.com/70505378/165060415-8c62ca93-05c3-4ec3-900d-aa7951f7b9ed.png)

<br>

그럼 이제 DROCC의 전체 training process와 함께 gradient ascent를 통해 negative sample을 생성하는 방법에 대해 설명하겠습니다. 

Initial steps는 초기 setup 단계이며, 모델은 주어진 positive sample data들로 1차 학습을 진행합니다. 

![image-20220424191329201](https://user-images.githubusercontent.com/70505378/165060418-4b2ea26c-9c04-4527-86df-202753732269.png)

1차 학습이 완료되면 모델은 생성된 negative sample과 주어진 positive sample들을 사용하여 앞서 본 loss term으로 학습을 한 번 더 진행합니다. 

![image-20220424191510426](https://user-images.githubusercontent.com/70505378/165060394-54fc3dbc-a9e2-46b2-93f0-ce57e2f59d4c.png)

DROCC steps에서 **Adversarial search** 부분이 negative sample을 생성하는 부분에 해당하는데요, 총 3단계로 구성됩니다. 

1. 배치 데이터 분포 하에서 랜덤하게 feature h를 추출합니다. 그리고 positivie sample인 x에 h를 더해서 x+h 샘플이 negative sample이라고 가정하여 loss term을 구합니다. 
2. 해당 loss의 미분을 통하여 gradient ascent를 수행합니다. 이를 통하여 loss 값이 커지는 feature h를 찾습니다. 
3. 정규화를 통해 negative sample 영역에 속하는 h를 생성합니다. 

3단계를 거친 뒤 생성된 negative sample과 기존에 존재하던 positive sample을 전체 loss term에 넣어 모델을 학습시킵니다. 

<br>

아래 그림은 모델의 1-d 매니폴드가 존재한다고 했을 때, DROCC 과정을 거쳐 생성된 positive sample과 negative sample의 분포를 나타낸 모습입니다. 

(c)에 해당하는 분포가 방금 말씀드린 DROCC에 의해 생성된 분포인데요, 보시는 것과 같이 조금 부정확한 분포를 보완하기 위해 본문에서는 DROCC-LF 기법을 소개했고, 바로 뒤이어서 설명해보도록 하겠습니다. 



<br>

<br>

## One-class Clssification with Limited Negatives (OCLN)

실제 세계에서 가능한 모든 경우의 normal sample들을 모으는 것은 불가능한 일입니다. 이에 따라 분포 상에서 빈 부분이 생성되면 정상 sample을 비정상 sample로 잘못 예측할 수 있습니다. 

OCLN은 이러한 문제를 해결하기 위한 진보된 방법입니다. 여기서는 찾고자 하는 하나의 sample을 positive sample, 나머지 sample들을 negative sample이라고 칭하겠습니다. 예를 들어 아이폰에서 '시리'라는 음성을 구분할 때 '시리'가 positive sample에 해당하고 나머지 음성들은 negative sample에 해당합니다. 

논문에서는 outlier exposure 방법에 기반한 **DROCC-OE** 방법을 먼저 언급하는데요, 이는 본문이 주로 다루고 있는 내용이 아니라 생략하도록 하겠습니다. 

본문에서는 더 진보된 학습법으로 **DROCC-LF**를 제안합니다. 앞서 DROCC 기법에서는 negative sample을 판단하는 척도로 positive sample과의 euclidean distance를 사용했습니다. 

DROCC-LF에서는 euclidean distance 대신에 mahalanobis distance를 사용하여 노이즈에 더 강건한 모델을 학습할 수 있었다고 합니다. 

![image-20220425180603077](https://user-images.githubusercontent.com/70505378/165060676-67e33bc4-1ab1-4c26-b3b6-42ed862906d9.png)

마할라노비스 거리는 하나의 point와 distribution 사이의 거리를 측정할 때 distribution의 co-variance, 즉 공분산 정보를 활용하는 거리로, 자세한 설명은 아래 사이트를 참고해주세요. 

* [https://gaussian37.github.io/ml-concept-mahalanobis_distance/](https://gaussian37.github.io/ml-concept-mahalanobis_distance/)

이외에는 DROCC와 마찬가지로 gradient descent phase와 gradient ascent phase로 학습을 진행합니다. 

![image-20220425180656473](https://user-images.githubusercontent.com/70505378/165060683-4450684d-ed58-462e-84fb-8ebc49083584.png)



<br>

<br>

## Evaluation

본 논문에서는 두 task에 대한 evaluation을 수행하였습니다. 

먼저 Anomaly detection task에 대한 evaluation 결과는 아래와 같습니다. 

본문에서는 DROCC로 학습한 모델을 다양한 domain에 적용시켜 봄으로써 거의 모든 경우에 기존의 모델들보다 더 나은 성능을 보임을 확인했습니다. 

_Image data_

![image-20220425180923582](https://user-images.githubusercontent.com/70505378/165060685-bea3f895-9a31-4658-83df-fc11cc496f9c.png)

_Tabular data_

![image-20220425180939296](https://user-images.githubusercontent.com/70505378/165060689-8a3425b6-0dcb-46e4-b616-57f4fd1af3cb.png)

_Time series data_

![image-20220425180955024](https://user-images.githubusercontent.com/70505378/165060691-55d1d344-28b6-4f8a-9b36-ad8f1a8b83fe.png)

<br>

또한 OCLN task에 대해서도 evaluation을 진행하였습니다. 

보시는 그래프는 Marvin과 Seven을 wake-word로 설정하고 FPR을 각각 3%, 5%로 제한했을 때의 recall 값을 나타낸 것입니다. 

DROCC-LF가 가장 높은 점수를 보이는 것을 확인할 수 있습니다. 

![image-20220425181201689](https://user-images.githubusercontent.com/70505378/165060694-397bf575-fba3-41e3-b18f-75e5be761308.png)

<br>

<br>

## Conclusion

이상 논문에 대한 설명이었습니다. 

본 논문이 제안한 DROCC는 gradient ascent를 이용해 negative sample을 생성함으로써 anomaly detection 성능을 높이고, 이를 OCLN task까지 넓힐 수 있음을 보여줬습니다. 

또한 기존의 방법들과 다르게 이미지, 음성, 시계열 데이터에 대해 다양하게 적용이 가능함을 보여주었습니다. 

<br>

<br>

_**논문에 대한 내용은 여기까지입니다. 아래부터는 개인적으로 새롭게 알고 느끼게 된 부분들을 정리하는 부분입니다.**_

<br>

# 새롭게 알게 된 것들

## Vocabulary

| Vocabulary   | meanings                       |
| ------------ | ------------------------------ |
| salient      | 현저한                         |
| reminiscent  | 연상시키는                     |
| utter        | 발언; 전적인, 무조건의         |
| perturbation | 섭동, 마음의 동요; 불안의 원인 |
| tweaking     | 조정                           |



<br>

## Domain-specific word

**Mahalanabis distance**

![image-20220425182612258](https://user-images.githubusercontent.com/70505378/165061511-995713b8-e836-4337-b82f-c4bf1b4e5df0.png)

> - 기존에 알고 있던 유클리디안 거리에 공분산 계산이 더해진 것으로도 이해할 수 있습니다. 만약 Σ=σ2IΣ=σ2I인 형태라면 즉, 각 클래스 간의 공분산이 모두 0인 상태라면 마할라노비스 거리는 유클리디안 거리와 동일합니다.
> - 따라서 **마할라노비스 거리에서는 공분산이 중요한 역할**을 합니다.
>
> 참조: [https://gaussian37.github.io/ml-concept-mahalanobis_distance/](https://gaussian37.github.io/ml-concept-mahalanobis_distance/)

<br>

## Others

* 조금 낯선 분야라서 이해하는 게 쉽지는 않았다. 하지만 anomaly detection, OCLN에 대한 기본적인 이해와 왜 사용되는지, 어떤 알고리즘을 주로 사용하는지 등에 대해 전반적으로 이해할 수 있는 계기가 되었다. 













