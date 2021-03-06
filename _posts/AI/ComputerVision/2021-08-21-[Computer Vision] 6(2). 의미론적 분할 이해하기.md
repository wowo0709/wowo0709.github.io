---
layout: single
title: "[Computer Vision] 6(2). 의미론적 분할 이해하기"
categories: ['AI', 'ComputerVision']
---



<br>

# 의미론적 분할 이해하기

**의미론적 분할(semantic segmentation)**은 이미지를 의미있는 부분으로 분할하는 작업을 지칭하는 좀 더 포괄적인 용어다. 의미론적 분할은 객체 분할과 인스턴스 분할을 모두 아우른다. 

앞선 포스팅에서 다뤘던 이미지 분류와 객체 탐지와는 달리, 분할 작업에는 밀도 높은 픽셀 단위의 예측을 반환하는, 즉 **입력 이미지의 각 픽셀마다 레이블을 할당**하는 기법이 필요하다. 

인코더-디코더가 객체 분할에 탁월한 이유와 그 결과를 더 개선할 수 있는 방법을 자세히 설명한 다음, 더 복잡한 작업인 인스턴스 분할을 위한 솔루션들을 알아볼 것이다. 

<br>

### 인코더-디코더를 사용한 객체 분할

---

![KakaoTalk_20210821_185805440](https://user-images.githubusercontent.com/70505378/130324265-655bee95-2fa7-4d13-b820-4a0548d2e281.png)

인코딩-디코딩 네트워크는 한 도메인의 데이터 샘플을 다른 도메인으로 매핑하기 위해 훈련된다. 객체 분할을 컬러 도메인의 이미지를 클래스 도메인으로 매핑하는 것으로 보면 마찬가지이다. 

사진의 각 픽셀 값과 컨텍스트가 주어졌을 때 각 픽셀에 타깃 클래스 중 하나를 할당해 동일한 높이와 너비를 갖는 **레이블 맵(label map)**을 반환해야 한다. 

이미지를 취해 레이블 맵을 반환하도록 인코더-디코더를 가르칠 때는 몇 가지 고려할 사항이 있는데, 이에 대해 지금부터 알아본다. 

<br>

#### 개요

---

U-Net같은 네트워크가 객체 분할에 어떻게 사용되는 지와 어떻게 그 출력을 더 처리해서 정제된 레이블 맵을 생성하는 지 보겠습니다. 

<br>

**레이블 맵으로 디코딩하기**

인코더-디코더 네트워크가 레이블 맵을 바로 출력하도록 구성하면 그 결과의 품질이 현저히 떨어지기 때문에, 분류기를 사용할 때와 마찬가지로 범주형 값을 출력하도록 해야 한다. 

이미지 분류와 마찬가지의 기법을 의미론적 분할에서도 사용하는 데, 대신 이미지 단위가 아니라 **픽셀 단위로 적용**한다는 것이 차이점이다. 이 네트워크는 픽셀 단위의 점수를 담고 있는 HxWxN 텐서를 반환한다. 

이러한 출력 텐서를 얻으려면 모델을 구성할 때 Do = N으로, 즉 출력 채널의 개수를 클래스 개수와 동일하게 설정하기만 하면 된다. 그런 다음 분류기처럼 훈련시킨다. 

<br>

소프트맥스 값과 실제 값을 원-핫 인코딩으로 나타낸 레이블 맵과 비교하기 위해 손실로는 **교차-엔트로피 손실(cross-entropy loss)**을 사용한다. 또한 이와 유사하게 HxWxN 예측은 채널 축을 따라 가장 높은 값을 갖는 인덱스를 선택함으로써(채널 축에서 argmax를 적용) 픽셀별 레이블로 변환될 수 있다. 

아래 코드는 앞선 포스팅에서 본 FCN-8s 코드를 객체 분할을 위해 모델을 훈련시키도록 조정한 코드이다. 

```python
inputs = Input(shape=(224,224,3))
out_ch = num_classes = 19 # Cityscapes 데이터셋 객체 분할

# [...] FCN-8s 아키텍처 구성(앞선 포스팅 참고)

outputs = Conv2DTranspose(filters=out_ch, kernel_size=16, strides=8, 
                          padding='same', activation=None)(m2)
seg_fcn = Model(inputs, outputs)
seg_fcn.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# [...] 네트워크 훈련.(앞선 포스팅 참고)

# 레이블 맵 예측
lebel_map = np.argmax(seg_fcn.predict(image), axis=-1)
```

전체 코드: https://github.com/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2/blob/master/Chapter06/ch6_nb5_build_and_train_a_fcn8s_semantic_segmentation_model_for_smart_cars.ipynb

<br>

**분할 손실과 지표를 사용해 훈련하기**

FCN-8s와 U-Net 같은 최신 아키텍처를 사용하는 것이 의미론적 분할을 위해 우수한 시스템을 구성하는 핵심이다. 그렇지만 여기서에서 최적의 수렴을 위한 **적절한 손실**이 필요하다. 교차-엔트로피가 기본 손실로 사용되기는 하지만, 조밀한 분류의 경우 몇 가지 예방책이 필요하다. 

이미지 또는 픽셀 단위의 분류는 보통 데이터셋의 **클래스 불균형** 문제가 발생한다. 이미지 분류에서는 이를 모든 클래스가 동일한 비율로 등장하도록 데이터를 추가 또는 삭제할 수 있지만, 픽셀 단위 분류에서 이 문제는 해결하기 어렵다. 

<br>

따라서 분할 모델의 편향을 방지하기 위해 손실 함수를 조정한다. 

하나의 해법으로, 각 클래스의 교차 엔트로피 손실에 대한 기여도를 측정하는 것이 일반적이다. 훈련 이미지에서 등장하는 클래스가 작을수록 손실에 대한 가중치를 높이는 것이다. 

가중치 맵은 일반적으로 실제 레이블 맵으로부터 계산된다. 각 픽셀에 적용된 가중치는 클래스에 따라 설정될 뿐만 아니라, 다른 요소와 상대적인 픽셀 위치 등에 따라 설정된다. 

또 다른 해법으로는 교차 엔트로피 자체를 클래스 비율에 영향받지 않는 다른 비용 함수로 교체하는 것이다. 여기에서 **IoU(Intersection over Union)** 와 **(쇠렌센 -) 다이스 계수**가 등장한다. 두 지표는 모두 두 집합이 얼마나 잘 겹치는지를 측정한다. 

![image-20210821212617149](https://user-images.githubusercontent.com/70505378/130324272-1ca0a0e5-29c9-424c-8647-c60e4def636a.png)

의미론적 분할에서 'Dice'  계수는 각 클래스에 대해 예측된 마스크가 실제 마스크와 얼마나 잘 겹치는 지 측정하기 위해 사용된다. 

한 클래스에 대해 분자는 정확하게 분류된 픽셀 개수를 나타내고 분모는 예측된 마스크와 실제 마스크 모두에서 이 클래스에 속한 전체 픽셀 개수를 나타낸다. 따라서 지표로서 'Dice' 계수는 **한 클래스가 이미지에서 취하는 상대적 픽셀 수에 따라 달라지지 않는다.**

다중 클래스 분할 작업에서 일반적으로 각 클래스에 대한 'Dice' 계수를 계산한 다음 그 결과의 평균을 낸다. 

<br>

'Dice' 계수는 0과 1 사이의 값으로 정의되는데, 예측된 마스크와 실제 마스크가 겹치지 않을수록 0에 가깝다. 따라서 이를 손실 함수로 사용하기 위해 이 점수를 뒤집는다. 대체로 K개의 클래스에 적용되는 의미론적 분할의 경우 'Dice' 손실은 일반적으로 다음과 같이 정의된다. 

![image-20210821213747013](https://user-images.githubusercontent.com/70505378/130324288-b6cdb753-1878-4acf-981e-5e3678492823.png)

분모가 0이 되는 경우를 방지하기 위해 위 공식의 분자와 분모에 _입실론_ (1e-6보다 작은 값)을 더해주기도 한다. 

<br>

텐서플로에서 이 손실은 다음처럼 구현될 수 있다. 

```python
def dice_loss(labels, logits, num_classes, eps=1e-6, spatial_axes=[1,2]):
  # 로짓을 확률로, 실제 값을 원-핫 인코딩된 텐서로 변환:
  pred_proba = tf.nn.softmax(logits, axis=-1)
  gt_onehot = tf.one_hot(labels, num_classes, dtype=tf.float32)
  # 다이스 계수의 분자와 분모 계산
  num_perclass = 2 * tf.reduce_sum(pred_proba * gt_onehot, axis = spatial_axes)
  den_perclass = tf.reduce_sum(pred_proba + gt_onehot, axis = spatial_axes)
  # 배치와 클래스에 대한 평균과 다이스 계수 계산:
  dice = tf.reduce_mean((num_class + eps) / (den_class + eps))
  return 1 - dice
```

 <br>

**조건부 랜덤 필드로 후처리**

모든 픽셀에 레이블을 올바르게 지정하는 일은 불가능에 가까운 일이지만, 결과를 후처리하여 명백한 단점들을 바로잡는 기법들이 있다. 

이 기법들 중 **조건부 랜덤 필드(conditional random fields, CRFs)** 기법이 전체 효율성 측면에서 가장 유명하다. 

CRF는 원본 이미지로 돌아가 각 픽셀의 컨텍스트를 고려함으로써 픽셀 단위의 예측을 개선한다. 예를 들면 두 픽셀 사이의 색 변화가 작으면 동일한 클래스에 속할 가능성이 높다고 예측하는 식이다. 공간과 색 기반의 모델과 예측기가 제공하는 확률 맵(여기서는 CNN에서 출력한 소프트맥스 텐서)을 고려해 CRF 기법은 시각적 윤곽선 측면에서 더 나은 정교화된 레이블 맵을 반환한다. 

<br>

#### 고급 예제 - 자율 주행 자동차를 위한 이미지 분할

---

[깃허브 저장소](https://github.com/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2/tree/master/Chapter06)

위 깃허브 저장소의 5번, 6번 주피터 노트북에서는 자율 주행 자동차를 위한 이미지 분할 실습을 진행한다. 이 작업을 해결하기 위해 FCN과 U-Net 모델을 이 절에서 보여주는 몇 가지 비결을 사용해 훈련시킨다. 손실을 계산할 때 각 클래스를 올바르게 평가하는 방법과 레이블 맵을 사후 처리하는 방법 등을 보여준다. 

<br>

<br>

### 더 까다로운 인스턴스 분할

---

객체 분할을 위해 훈련된 모델을 사용하면 소프트맥스 출력은 각 픽셀에 대해 N개의 클래스 중 하나에 속할 확률을 나타낸다. 하지만 이 출력이 **한 클래스의 동일한 인스턴스에 속하는지 여부를 나타내지는 않는다**. 

이번 포스팅의 마지막 부분에서는 **객체 분할**과 **객체 탐지**를 위한 솔루션을 확장함으로써 **인스턴스 분할**을 달성하는 2가지 방식을 설명한다. 

<br>

#### 객체 분할에서 인스턴스 분할까지

---

**경계를 고려하기**

의미론적 마스크를 이용해 각 인스턴스를 식별하는 데 있어 가장 중요한 것은 **정확한 의미론적 마스크**를 구하는 것이다. 여기서 정확하다는 것은, 최소한 **겹치지 않는 요소에 대해 정밀한 윤곽선으로 마스크를 생성**할 수 있어야 한다. 그리고 이를 위한 방법은 훈련 손실을 그에 맞게 조정하는 것이다. 

두 인스턴스에 모두 근접한 픽셀을 올바르게 분리하도록 네트워크를 학습시키기 위해 여러 인스턴스의 경계에 위치한 잘못 분류된 픽셀에 더 큰 패널티를 부과하도록 손실 함수에 가중치를 부과한다. 그리고 이 가중치는 픽셀별로 계산된다. 각 픽셀에 대해, 그리고 각 클래스에 대해 두 개의 가장 가까운 클래스 인스턴스까지의 픽셀 거리를 계산에 넣는다. (두 거리가 작을수록 가중치가 커진다)

이 가중치 맵은 사전에 계산되고 실제 마스크와 함게 저장되어 훈련하는 동안 함께 사용될 수 있다. 

<br>

캐글 2018 데이터 사이언스 볼 우승자을이 만든 맞춤형 U-Net은 각 클래스에 대해 두 개의 마스크를 출력하는데, 하나는 **픽셀별 클래스 확룔을 나타내는 마스크**이며 나머지는 **클래스 경계를 나타내는 마스크**이다. 적절한 훈련을 거친 다음 두 에측 마스크에서 얻은 정보는 각 클래스에 대해 잘 구분된 요소를 얻기 위해 사용될 수 있다. 

<br>

**사후 처리를 통해 인스턴스 마스크로 변환**

앞에서 설명했듯이, 정확한 마스크를 구하면 적절한 알고리즘을 적용해 그로부터 겹치지 않는 인스턴스를 식별할 수 있다. 이 사후처리는 일반적으로 **마스크 침식(mask erosion)**과 **팽창(dilation)** 같은 **모폴로지 함수(morphological functions)**를 사용해 이루어진다. 

**워터셰드 변환(Watershed transforms)**은 클래스 마스크를 인스턴스로 더 분할하는 또 다른 일반적인 알고리즘이다. 이 알고리즘은 채널이 하나인 텐서를 취해 이를 각 값이 고도를 나타내는 지형면으로 간주한다. 다양한 기법을 사용해 인스턴스 경계를 나타내는 능선의 꼭대기를 추출한다. 

예를 들어 토론토 대학의 '민 바이'(Min Bai)와 '라쿠엘 우터슨'(Raquel Urtasun)이 [논문](https://openaccess.thecvf.com/content_cvpr_2017/papers/Bai_Deep_Watershed_Transform_CVPR_2017_paper.pdf)에서 제안한  네트워크는 예측된 의미론적 마스크와 원본 RGB 이미지를 모두 입력으로 가져와 능선을 식별하는 데 사용될 수 있는 에너지 맵을 반환한다. RGB 정보 덕분에 이 솔루션은 겹치는 인스턴스도 정확하게 분할할 수 있다. 

<br>

#### 객체 탐지부터 인스턴스 분할까지 - Mask R-CNN

---

객체 탐지 모델은 객체 인스턴스에 대한 경계 상자를 반환한다. 다음 단락부터는 어떻게 이 결과들이 더 정교화된 인스턴스 마스크로 바뀌는지 알아본다. 

<br>

**의미론적 분할을 경계 상자에 적용하기**

의미론적 분할에 있어 객체 탐지는 추가 분석을 위한 사전 준비 단계로 사용되어 단일 인스턴스를 포함한 이미지 패치를 제공한다. 따라서 인스턴스 분할은 다음 두 단계의 문제가 된다. 

1. 객체 탐지 모델을 사용해 타깃 클래스의 각 인스턴스에 대한 경계 상자를 반환한다. 
2. 각 패치를 의미론적 분할 모델에 제공해 인스턴스 마스크를 얻는다. 

예측된 경계 상자가 정확하다면, 분할 네트워크는 해당 패치에서 어느 픽셀이 캡처된 클래스에 속하는지, 어느 픽셀이 배경의 일부이고 어느 픽셀이 다른 클래스에 속하는지 분류하면 된다. 

<br>

**Faster R-CNN으로부터 인스턴스 분할 모델 구성하기**

간단하게 사전 훈련된 탐지 네트워크와 그 뒤에 사전 훈련된 분할 네트워크를 배치해 사용할 수 있지만, 두 네트워크를 하나로 이어서 엔트-투-엔트 방식으로 훈련하면 전체 파이프라인의 성능이 확실히 나아진다. 공통 계층을 통해 분할 손실을 역전파하면 추출된 특징이 탐지와 분할 작업에서 모두 의미 있게 된다. 이것이 '카이밍 히'(Kaiming He) 팀이 [논문](https://openaccess.thecvf.com/content_ICCV_2017/papers/He_Mask_R-CNN_ICCV_2017_paper.pdf)에서 제안한 **Mask R-CNN**의 아이디어이다. 

<br>

Mask R-CNN은 주로 Faster R-CNN에 기반하여 하나의 영역 제안 네트워크로 구성되며 제안된 각 영역에 대한 클래스와 상자 오프셋을 예측하는 두 개의 분기가 뒤로 이어진다. 

저자는 여기에 세번째 병렬 분기를 추가해 확장함으로써 각 영역의 요소에 대한 이진 마스크를 출력한다. 이 추가된 분기는 표준 합성곱과 전치 합성곱 한 쌍으로만 구성된다. 이 병렬 처리는 Faster R-CNN의 정신을 따르며, 이것이 일반적으로 순차형인 다른 인스턴스 분할 기법과 대조되는 점이다. 

![image-20210821224638341](https://user-images.githubusercontent.com/70505378/130324296-0f788d63-5d7e-47d8-86de-b214881d4363.png)

이 병렬화 덕분에 분류와 분할을 분리할 수 있게 되었다. 

분할 분기는 N개의 이진 마스크를 출력하도록 정의되지만, 다른 분기에 의해 예측된 클래스에 대응하는 마스크만 최종 예측과 훈련 손실을 계산할 때 사용된다. 즉 인스턴스 클래스의 마스크만 분할 분기에 적용되는 교차 엔트로피 손실에 영향을 준다. 이렇게 함으로써 분할 분기는 클래스 간에 경쟁 없이 레이블 맵을 예측함으로써 그 작업을 단순화할 수 있다. 

<br>

<br>

### 정리

---

* 의미론적 분할은 객체 분할부터 인스턴스 분할을 모두 아우르는 포괄적인 개념이다. 
* 객체 분할을 위해서는 픽셀 단위로 클래스 예측을 수행해야 한다. 
    * 손실로는 다이스 손실을 사용하며, 이는 객체 수가 적은 클래스에 더 큰 패널티를 부과함으로써 클래스의 총 객체 수와 상관없이 손실을 계산하기 위함이다. 
    * 조건부 랜덤 필드 기법을 이용해 레이블 맵을 사후 처리하여 더 정교한 레이블 맵을 얻는다. 
* 인스턴스 분할을 달성하기 위한 방법에는 2가지가 있다. 
    * 객체 분할에서 출력된 객체 분할 마스크를 인스턴스 분할을 위해 사용한다. 이를 위해서는 각 객체가 겹치지 않도록 마스크를 생성하는 것이 중요하다(이를 위해 손실을 변형한다). 워터셰드 변환 등을 통해 클래스 마스크의 각 값을 고도로 변환하는 사후 처리를 통해 인스턴스 분할을 수행한다. 
    * 객체 탐지에서 출력된 박스를 인스턴스 분할을 위해 사용한다. 이것이 Mask R-CNN의 아이디어이다. 
