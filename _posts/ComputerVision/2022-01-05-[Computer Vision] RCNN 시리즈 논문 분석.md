---
layout: single
title: "[Computer Vision] RCNN 시리즈 논문 분석"
categories: ['AI', 'ComputerVision']
toc: true
toc_sticky: true
---

<br>

## RCNN

### RCNN의 구조

![image-20220105173456273](https://user-images.githubusercontent.com/70505378/148225750-2cf83cf1-19d3-4575-90a9-0c9b06fea844.png)

1. 이미지 입력
2. 약 2000개의 bottom-up region proposals를 추출
3. CNN을 이용해 각각 region proposal의 특징 추출
4. linear SVM을 이용해 각각의 label 분류

**wraped region**은 CNN의 입력으로 넣어주기 위해 크기를 맞춰주는 과정. 

<br>

### RCNN에서의 세 가지 모델

#### 1. Region Proposal

![image-20220105173809873](https://user-images.githubusercontent.com/70505378/148225752-d27e0bf5-c08f-447a-ba6d-80d4e0f1578a.png)

* CNN의 입력으로 사용할 후보 영역 추출. 

* 'Selective search algorithm'을 사용

**Selective search algorithm**

![image-20220105173950401](https://user-images.githubusercontent.com/70505378/148225754-e6b0fa27-aa6b-493a-8a46-acf47003cf5f.png)

1. 색상, 질감, 영역크기 등을 이용해 (non-objective) segmentation을 수행. 이 작업을 통해 많은 small segmented areas를 얻는다. 
2. Bottom-up(상향식) 방식으로 small segmented areas들을 합쳐 더 큰 segmented areas를 얻는다. 
3. (2) 작업을 반복하여 최종적으로 약 2000개의 region proposal을 생성한다. 

* 2000장의 region proposal을 얻고 나면 wrapping을 통해 이미지를 227x227의 고정된 크기로 변환한다. (CNN 입력)

#### 2. CNN

![image-20220105174332917](https://user-images.githubusercontent.com/70505378/148225757-5682280b-c9b9-41a8-9164-fa5166fea0a9.png)

* 추출된 2000개 region proposal에 대해 각각 ConvNet 연산을 수행
*  각각의 region proposal에 대해 4096-dimensional feature vector를 추출
* 모델 훈련 시 pre-train된 AlexNet을 Domain-specific fine-tuning 수행

**Domain-specific fine-tuning**

![image-20220105174746664](https://user-images.githubusercontent.com/70505378/148225719-cb9af3a2-46d4-4f31-8237-4745b38321f1.png)

* 2000장의 region proposals와 ground-truth box의 IoU 값이 0.5 이상이면 positive sample로, 0.5 미만이면 negative sample로 분류한다. 
  * 이렇게 함으로써 ground truth만 positive sample로 정의할 때보다 30배 많은 학습 데이터 확보 가능
  * positive sample은 객체가 포함된 sample, negative sample은 객체가 포함되지 않은 배경 sample을 의미
* positive sample 32개 + negative sample 96개, 총 128개의 sample을 하나의 미니 배치로 생성하여 fine-tuning 수행
* fine-tuning 수행 시 pre-trained Alexnet 모델의 마지막 softmax layer를 제거하고 domain specific한 N+1 way classification을 수행
  * 모델 추론 시에는 소프트맥스 layer는 사용하지 않음
  * CNN 층의 최종 목표는 4096-dimensional feature vector를 추출하는 것이기 때문

**IoU(Intersection over Union)**

![image-20220105175324262](https://user-images.githubusercontent.com/70505378/148225721-23782feb-c096-4468-85dc-035c930736e1.png)

![image-20220105175337402](https://user-images.githubusercontent.com/70505378/148225724-9a7d2824-40b2-4ca4-9ac0-4b3ebf2fb82f.png)

<br>

#### 3-1. Linear SVMs

![image-20220105175406797](https://user-images.githubusercontent.com/70505378/148225729-40b10730-fc34-45f3-a161-9ba646ed4d94.png)

* 각각의 4096-dimensional feature vector에 대해 SVM으로 이진 분류를 수행. 따라서 분류하려는 객체의 수만큼의 SVM이 필요. 
* 학습이 한차례 끝나면 hard negative mining 기법을 적용하여 재학습을 수행
* Linear SVM에서는 output으로 class와 confidence score를 반환. 

**hard negative mining**

![image-20220105175924113](https://user-images.githubusercontent.com/70505378/148225731-3b450deb-5314-4c19-b4bc-10e947b12aa7.png)

* 이미지에서 사람을 탐지하는 경우 사람은 positive sample, 배경은 negative sample에 해당한다. 모델 추론 시 '사람이 아니라고 예측했을 때 실제로 사람이 아닌 경우의 샘플'을 **'true negative sample'**, '사람이 아니라고 예측했지만 실제로는 사람인 경우의 샘플'을 **'false positive sample'**이라고 한다. 
* 객체 탐지 시 negative sample의 수가 우세하기 때문에, 클래스 불균형으로 인하여 모델은 주로 false positive 오류를 주로 범하게 된다(배경이라고 예측할 확률이 높기 때문에). 
* 이를 해결하기 위해 처음 linear SVM을 학습시킬 때의 false positive sample들을 epoch마다 학습 데이터에 추가하여 학습을 진행함으로써 모델이 강건해지고, false positive 오류가 줄어든다. 

#### 3-2. Bounding box regressor

Selective search를 통해 얻은 bounding box의 크기를 조정해주는 bounding box regressor를 모델의 최종단에서 사용한다. 

* N개의 training pair인 {(P<sup>i</sup>,G<sup>i</sup>)}*i*=1,...,*N* 에 대해 P<sup>i<sup> = (*P**x**i*,*P**y**i*,*P**w**i*,*P**h**i*)는 해당 region에 대한 추정값으로 각각은 region 중심의 x,y좌표와 width와 height를 나타내고, 이에 대응되게 G<sup>i</sup> = (*G**x**i*,*G**y**i*,*G**w**i*,*G**h**i*)은 해당 region에 대한 ground truth이다.

![image-20220105180646988](https://user-images.githubusercontent.com/70505378/148225733-e5f3040d-2dc3-43a8-b1f8-ac3ee1c0f4af.png)

* 위의 식을 만족하는 최적의 w를 찾는 것이며, 각 기호의 의미는 다음과 같다. 

  * ϕ<sub>5</sub>(P<sup>i</sup>): P<sup>i</sup>에 해당하는 feature vector를 의미하며, 여기서는 fine-tuning된 CNN의 output이다. 

  * λ: ridge regression에서의 하이퍼 파라미터

  * t

    ![image-20220105181244132](https://user-images.githubusercontent.com/70505378/148225734-b51cb08b-b29f-4f1e-959c-9813660628c7.png)

<br>

### Non maximum supression

최종적으로 얻게 되는 여러 개의 bounding box 중 가장 적합한 bounding box를 선택하는 알고리즘

![image-20220105181432226](https://user-images.githubusercontent.com/70505378/148225736-a515aef0-42cc-4c06-86df-193457caaa2a.png)

1. bounding box 별로 지정한 confidence score threshold 이하의 box 제거
2. 남은 bounding box를 confidence score에 따라 내림차순 정렬. 그 다음 confidence score가 높은 순의 bounding box부터 다른 box와의 IoU 값을 조사하여 IoU threshold 이상인 box를 모두 제거(포함하는 영역이 비슷하면서 confidence score가 더 낮은 bounding box는 조사할 필요가 없기 때문)
3. 2의 과정을 반복하여 남아있는 box만 선택

<br>

### 단점

* 이미지 한 장당 2000개의 region proposal을 추출하므로 학습 및 추론의 속도가 느리다. 
* 2000개의 region proposal 각각에 대해 ConvNet 연산을 수행하기 때문에 연산 시간이 오래 걸린다. 
* 3가지 모델을 사용하다보니 구조와 학습 과정이 복잡하다. 또한 end-to-end 학습이 불가하다. 

<br>

<br>

## Fast RCNN

### 개요

이전 모델인 RCNN과 SPPNet 모델의 성능을 향상시키기 위해 고안된 모델. 

* 학습이 multi-stage pipeline으로 이뤄진다. 따라서 복잡하고 시간이 오래 걸리며 end-to-end 학습이 불가하다. 
* 학습에서 메모리와 시간이 많이 필요하다. 

이를 다음과 같이 개선. 

* 기존 모델들보다 더 높은 정확도를 기록. 
* multi-task loss와 single stage 학습을 이용해 시간을 단축. 
* 네트워크의 모든 layer들을 갱신 가능. end-to-end 학습 가능. 
* feature caching을 위한 메모리 불필요. 

<br>

### Fast RCNN의 구조 

![image-20220105182352809](https://user-images.githubusercontent.com/70505378/148225738-8b00bf55-8383-4013-85d9-2627c07dd1ee.png)

1. input image와 RoI(region of Interest)들이 입력으로 사용. 
2. 각각의 RoI는 ConvNet 연산을 통해 고정된 크기의 feature map으로 pooling되고, FC(Fully Connected) layer를 통해 feature vector로 매핑됨. 
3. RoI 별 2개의 output을 갖는다. 하나는 softmax probabilities이고, 다른 하나는 per-class bounding-box regression offset이다. 

<br>

### RoI pooling layer

![image-20220105182800467](https://user-images.githubusercontent.com/70505378/148225741-1dfc6ae8-069f-4f4f-97f3-4f160574e0cc.png)

ConvNet 연산 후 얻어지는 feature vector는 FC 층의 입력으로 사용되기 때문에 고정된 크기의 feature vector가 필요하다. 

RCNN에서는 각각의 region proposal에 대해 ConvNet 연산을 수행했으며, SPPNet에서는 원본 이미지에 대해 ConvNet 연산 적용 후 얻은 feature map에 대해 'RoI projection'과 'Spatial pyramid pooling'을 적용해 고정된 크기의 feature vector를 얻어내었다. 

**RoI projection**

RoI projection은 input image보다 크기가 작아진 feature map에서의 RoI 영역을 알기 위해 수행하는 연산이다. 

RoI의 위치(좌표 및 높이, 너비), 입력 image와 feature map의 크기 비율을 이용해 feature map에서 RoI 영역을 지정하는 과정이다. 

**Spatial pyramid pooling**

![image-20220105183135806](https://user-images.githubusercontent.com/70505378/148225743-3e74affb-dd4e-48e1-969d-78a5f7887ddc.png)

* 위 사진과 같이 하나의 feature map에 대해 max pooling을 수행하여 고정된 크기의 벡터를 얻는다. 
* 1단계에서 1분할된 이미지(feature map 원본)에 대해, 2단계에서 4분할된 이미지에 대해, 3단계에서 16분할된 이미지에 대해 max pooling을 수행하여 최종적으로 1x256, 4x256, 16x256 크기의 벡터를 얻는다. 
  * 여기서 256은 feature map의 개수. 
* 최종적으로 모든 벡터들을 이어 붙여 21x256 크기의 feature vector를 얻는다. 

<br>

RoI pooling layer에서는 RoI projection은 동일하게 수행하고, Spatial pyramid pooling 연산 대신 feature map을 7x7로 분할하여 한번의 pooling 연산만을 수행한다. 

<br>

### Hierarchical sampling

Hierarchical sampling을 통해 학습에 사용될 Mini batch를 구성한다. 

먼저 N개의 image를 sampling한 후, 각각의 image에서 R/N 개의 RoI를 추출한다. 같은 image에서 추출된 RoI는 순전파와 역전파 시 computation과 memory를 공유할 수 있다. N을 작게 하면 mini-batch의 연산을 줄인다. 

논문에서는 N=2, R=128로 설정하여 64개의 RoI를 각 이미지에서 추출하였다. 이 때, 16개는 ground truth와의 IoU 값이 0.5 이상인 sample(positive sample, 객체)에서 추출하고, 나머지 48개는 IoU 값이 0.1~0.5 사이의 sample(negative sample, 배경)에서 추출한다. 

<br>

### Multi-task loss

Multi-task loss를 이용함으로써 end-to-end 학습이 가능해진다. 

![image-20220105184756748](https://user-images.githubusercontent.com/70505378/148225748-384ad2db-34a4-44e0-8ab1-3e314eb0bbde.png)

* p = (p0, ..., pk): K+1 개의 class score
* u: class ground truth
* t<sup>u</sup>: u 클래스의 bounding box 좌표를 조정하는 값
* v: bounding box 좌표 값의 ground truth
* L<sub>cls</sub>는 cross-entropy error로 계산된다.
* L<sub>loc</sub>는 다음과 같이 정의된다.
  * L<sub>loc</sub>(t<sup>u</sup>,v) = Σ*i*∈{*x*,*y*,*w*,*h*}*smooth*<sub>L1</sub>(t<sub>i</sub>u−v<sub>i</sub>)
    * *smooth*<sub>L1</sub> = 0.5x<sup>2</sup> if ∣*x*∣<1
    * *smooth*<sub>L1</sub> = ∣*x*∣−0.5 otherwise

* [u >= 1]는 indicator function으로 해당 클래스에 속할 때만 loss를 계산한다. 

<br>

<br>

## Faster RCNN











































