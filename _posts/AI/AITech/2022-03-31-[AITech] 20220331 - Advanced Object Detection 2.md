---
layout: single
title: "[AITech][Object Detection] 20220331 - Advanced Object Detection 2"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['YOLOv4', 'M2Det', 'CornerNet']
---



<br>

_**본 포스팅은 '송원호' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# Advanced Object Detection 2

지난 포스팅에 이어 Advanced Object Detection이라는 주제로 포스팅을 이어갑니다. 

이번 포스팅에서 살펴볼 모델은 `YOLOv4`, `M2Det`, `CornerNet`입니다. 

## YOLO v4

### Overview

`YOLO v4` 모델은 실시간으로 사용할 수 있는 detection 모델의 요구에 따라 다른 detector들보다 빠르면서 정확도가 높은 모델입니다. Object detection에서 사용하는 최신 방법들을 소개하고 직접 실험해본 논문이기도 합니다. 

또한 YOLO v4는 BOF, BOS 방법들을 실험을 통해서 증명하고, 최상의 성능을 내는 조합들을 찾았으며 이를 GPU 학습에 더 효율적이고 적합한 방법들로 변형했습니다. BOF와 BOS는 아래와 같습니다. 

* `BOF(Bag of Freebies)`: Inference 비용을 늘리지 않고 정확도를 향상시키는 방법
* `BOS(Bag of Specials)`: Inference 비용을 조금 높이지만 정확도를 크게 향상시키는 방법(Cascade, Deformable convolution 등)

**Object Detection Model**

YOLO v4에서 실험을 위해 정리한 객체 탐지 모델 파이프라인은 아래와 같습니다. 

![image-20220331185509712](https://user-images.githubusercontent.com/70505378/161206655-450a651c-42e8-4424-a0f7-d9813a3fe3eb.png)

* Input
  * Image, Patches, Image Pyramid, ...
* Backbone
  * GPU platform: VGG, ResNet, ResNeXt, DenseNet, ...
  * CPU platform: SqueezeNet, MobileNet, ShuffleNet, ...
* Neck
  * Additional blocks: SPP, ASPP, ...
  * Path-aggregation blocks: FPN, PAN, NAS-FPN, BiFPN, ...
* Head
  * Dense prediction(one-stage): RPN, YOLO, SSD, RetinaNet, CornerNet, FCOS, ...
  * Sparse prediction(two-stage): Faster R-CNN, R-FCN, Mask R-CNN, ...

**Bag of Freebies**

Bag of Freebies는 아래와 같이 크게 3개의 그룹으로 정리했습니다. 

![image-20220331214951559](https://user-images.githubusercontent.com/70505378/161206657-55b44471-41ae-4448-97e9-bef32e9d64cb.png)

* Data Augmentation
  * 입력 이미지를 변화시켜 과대적합을 방지하고, 다양한 환경에 강건하게 학습하는 방법
* Semantic Distribution Bias
  * 데이터 셋에 특정 라벨(배경)이 많은 경우 불균형을 해결하기 위한 방법
* Bounding Box Regression
  * Bounding box 좌표 값을 예측(회귀)할 때 사용하는 손실 함수

**Bag of Specials**

Bag of Specials는 아래와 같이 크게 5개의 그룹으로 정리했습니다. 

![image-20220331222806395](https://user-images.githubusercontent.com/70505378/161206659-2f31f1b6-c658-42b3-918a-e4d7327d9c98.png)

* Enhance of Receptive field
  * Feature map의 receptive field를 키워서 검출 성능을 높이는 방법
* Attention module
  * Feature map의 어느 부분에 더 집중할 지 가중치를 두어 fusion하는 방법
* Feature Integration
  * Feature map을 통합하기 위한 방법(Neck)
* Activation Function
  * 좋은 activation 함수는 gradient를 더 효율적으로 전파하고 더 좋은 feature들을 전달
* Post-processing method
  * NMS와 같이 불필요한 bbox를 제거하는 등의 방법

<br>

최종적으로 YOLO v4의 backbone에서 실험을 진행한 BOF 및 BOS는 아래와 같습니다. 

* **Activations** : ReLU, leaky-ReLU, parametric-ReLU, ReLU6, SELU, Swish, or Mish
* **Bounding box regression loss** : MSE, IoU, GIoU, CIoU, DIoU
* **Data augmentation** : CutOut, MixUp, CutMix
* **Regularization method** : DropOut, DropPath, Spatial DropOut, DropBlock
* **Normalization** : Batch Normalization (BN), Cross-GPU Batch Normalization (CGBN or SyncBN), Filter Response Normalization (FRN), Cross-Iteration Batch Normalization (CBN)
* **Skip-connections** : Residual connections, Weighted residual connections, Multi-input weighted
  residual connections, Cross stage partial connections (CSP)
* **Others** : label smoothing  

<br>

### Architecture

YOLO v4의 detector의 디자인 고려사항은 아래와 같았습니다. 

* 작은 물체 검출하기 위해서 큰 네트워크 입력 사이즈(resolution) 필요
* 네트워크 입력 사이즈가 증가함으로써 큰 receptive field 필요 → 많은 layer 필요
* 하나의 이미지로 다양한 사이즈의 물체 검출하기 위해 모델의 용량이 더 커야 함 → 많은 parameter 필요  

![image-20220331223915159](https://user-images.githubusercontent.com/70505378/161206665-b91a2ff2-0c49-46a2-8b9a-86252369e39e.png)

**Cross Stage Partial Network(CSPNet)**

YOLO v4에서는 YOLOv3까지 사용하던 DarkNet에 **CSP 알고리즘**을 적용시킨 **CSPNet** 구조를 backbone으로 사용했습니다. 

YOLO v4는 더 많은 수의 파라미터를 필요로 했기 때문에 정확도를 유지하면서도 어느 정도의 경량화가 필요했습니다. 이를 위해 CSP 방법을 사용했습니다. 

CSP를 설명하기 위해 DenseNet 구조를 보겠습니다. DenseNet은 아래와 같이 이전 단계의 feature map과 이 feature map에 convolution 연산을 적용한 feature map을 계속해서 channel 방향으로 concat하는 방법을 사용합니다. 

![image-20220331224737124](https://user-images.githubusercontent.com/70505378/161206668-f0cb1133-c907-4a3a-b2ef-1d133f9592e9.png)

이러한 방법의 문제점은 back propagation 시 동일한 feature map의 gradient 정보가 계속해서 재사용 된다는 것입니다. 이는 메모리 상 비효율적입니다. 

![image-20220331224825842](https://user-images.githubusercontent.com/70505378/161206670-6c99a369-c91c-4cb1-8e4b-d34dfe4503e9.png)

CSP 방법은 아래 그림과 같이 feature map의 일부는 convolution 연산을 거치지 않고 최종 출력에 그냥 concat해주는 방식입니다. 이렇게 하면 재사용되는 gradient information을 줄일 수 있어 메모리 효율적이면서도 성능을 유지할 수 있습니다. 

![image-20220331225227269](https://user-images.githubusercontent.com/70505378/161206675-2fcc6e74-4799-4bce-bca3-ec941af1993b.png)

![image-20220331225259803](https://user-images.githubusercontent.com/70505378/161206678-f370ce0a-ead9-4598-9424-9ae5e6e7025d.png)

CSP 방법은 정확도를 유지하면서도 메모리 cost를 감소시켜 모델 경량화에 도움을 주면서 연산 bottleneck을 제거합니다. 또한 일반적인 적용이 가능하기 때문에 다양한 backbone에서 사용할 수 있는 방법입니다. 

**Data Augmentation**

YOLO v4에서 사용한 새로운 data augmentation 기법으로 **Mosaic**와 **Self-Adversarial Training** 이 있습니다. 

Mosaic는 4개의 이미지를 붙여서 하나의 이미지 형태로 만드는 것으로, 하나의 입력 이미지에 4장의 이미지에 해당하는 정보들이 들어있기 때문에 배치 사이즈를 효과적으로 키울 수 있고, 그만큼 적은 양의 데이터로 빠른 학습이 가능하다고 합니다. 

![image-20220331231223110](https://user-images.githubusercontent.com/70505378/161206680-021951f9-3ecf-49ef-8ae4-302e16f96a1e.png)

또 다른 방법인 Self-Adversarial Training은 기존 이미지에서 '객체를 지운' 동일한 이미지를 한 번 더 학습 이미지로 사용하는 방법입니다.  즉 동일한 배경을 가지는 이미지 내에서 객체가 있는 이미지와 객체가 없는 이미지를 모두 학습에 사용함으로써, 모델을 더욱 강건하게 학습시킬 수 있습니다. 

**Modifications**

YOLO v4에서는 기존의 SAM(Spatial Attention Module)과 PAN(Path Aggregation Network)를 수정하여 사용했습니다. 

![image-20220331231638705](https://user-images.githubusercontent.com/70505378/161206682-7b837fcf-715c-4bfb-9c58-08e23515159d.png)

또한 Cross mini-batch Normalization이라는 개선된 형태의 normalization 기법을 사용했는데, 이는 accumulated batch norm을 사용하여 배치 사이즈가 작은 경우에도 큰 것과 같은 효과를 낼 수 있다고 합니다. 

![image-20220331231742907](https://user-images.githubusercontent.com/70505378/161206684-6d2919b0-e1b3-480d-bf91-12490b918f65.png)

<br>

이와 같이 YOLO v4 논문에서는 정말 다양한 실험들을 진행해 가장 좋은 성능을 보이는 방법들을 채택했습니다. 

여기서 중요한 것은 YOLO v4 논문에서 선택한 기법들은 해당 데이터셋과 모델 구조 하에서 가장 좋은 성능을 보이는 기법들이라는 것입니다.

따라서 우리는 다양한 데이터셋과 모델 하에서 다양한 실험을 통해 좋은 방법을 찾아내고, 그에 대한 인사이트를 기르는 것이 중요하다는 교훈을 얻을 수 있습니다. 



<br>

<br>

## M2Det

이번에는 1-stage detector에 multi-scale, multi-level을 적용한 모델인 `M2Det` 에 대해 알아보도록 하겠습니다. 

### Overview

**Feature Pyramid의 한계점**

물체의 scale 변화는 detection에서 중요한 요소입니다. 일반적으로 backbone의 low level feature는 크기가 작고 사소한 물체를 잡아내는 데 유리하고, high level feature는 크기가 크고 중요한 물체를 잡아내는 데 유리하다고 알려져있고, 이 정보들을 함께 사용하기 위해 **FPN**과 같은 구조를 사용하는 것입니다. 

하지만 논문의 연구진들은 이러한 backbone의 multi-scale 정보만을 이용하는 것은 object detection task를 수행하기에 충분하지 않다고 주장합니다. 이에 대한 근거로 classification을 위한 backbone이 **single-level layer**라는 점을 들고, 따라서 **FPN으로는 single-level 정보만을 나타낼 수 있다**고 합니다. 

논문에서는 이러한 single-level layer의 정보들은 물체의 scale을 잡아내기에는 용이해도, **물체의 shape을 잡아내기에는 부족**하다고 말합니다. 그리고 이러한 shape 정보를 잘 잡아낼 수 있는 것이 기존의 multi-scale 정보 만을 위한 FPN 대신에, **multi-scale + multi-level 정보를 모두 이용하는 새로운 fpn 구조**를 사용하는 것이라고 주장합니다. 

일반적으로 low-level layer의 feature는 간단한 외형을, high-level layer의 feature는 복잡한 외형을 나타내는 데 적합하다고 합니다. 이때 말하는 multi-level(fpn) 관점에서의 low-level/high-level과 하나의 backbone 내에서 layer의 깊이를 말하는 low-level/high-level은 다른 것이니 혼동하지 않도록 주의하시길 바랍니다. 

![image-20220401112220435](https://user-images.githubusercontent.com/70505378/161206687-44d97609-70c4-4f36-b79b-fdba9a07fb1d.png)

위 사진은 multi-scale, multi-level에 따라 모델이 집중하는 feature를 시각화한 것입니다. 

낮은 scale에서는 작고 사소한 물체들을, 높은 scale에서는 크고 중요한 물체들을 잡아내는 것을 볼 수 있습니다. 

여기서 주목할 것은 level에 따라 집중하는 feature입니다. 낮은 level에서는 비교적 단순한 shape을 가지는 신호등을, 높은 level에서는 복잡한 shape을 가지는 사람을 잡아내는 것을 볼 수 있습니다. 

이와 같이, 논문에서는 물체의 다양한 shape을 잘 잡아내기 위해서는 multi-level이 필요하다고 주장합니다. 

<br>

### Architecture

논문에서는 multi-level, multi-scale의 feature pyramid인 `MLFPN`을 제안했습니다. 그리고 이를 SSD에 합쳐서 M2Det이라는 one stage detector 모델을 만들어냈습니다. 

![image-20220401114300231](https://user-images.githubusercontent.com/70505378/161206690-11dc7238-7abc-464e-83c7-5989f9e1c826.png)

각 부분이 어떤 역할을 하는지 먼저 간단하게 적어보도록 하겠습니다. 

* `FFMv1`: Backbone network에서 다른 level의 2개의 feature map을 뽑아 합쳐서 base feature 생성.
* `TUM`: UNET처럼 Encoder, Decoder 구조를 사용하여 각 level의 multi-scale feature map(Decoder의 feature maps)을 생성.
* `FFMv2`: 이전 level의 TUM의 출력(Decoder의 가장 마지막 출력 feature map)과 base feature를 합쳐서 다음 level의 TUM의 입력으로 전달. 
* `SFAM`: multi-level의 multi-scale feature map들을 concat하고, attention 연산을 한 뒤 SSD Head의 입력으로 전달.  

그러면 각 모듈에 대해 좀 더 자세히 알아보겠습니다. 

**FFMv1(Feature Fusion Module v1)**

Backbone network에서 서로 다른 scale의 두 feature map을 합쳐서 semantic 정보가 풍부한 base feature를 만들어내는 역할을 합니다. 

논문에서는 VGG의 4번째와 5번째 feature map을 사용했으며, 이 경우 아래와 같이 fusion이 수행됩니다. 

![image-20220401115242501](https://user-images.githubusercontent.com/70505378/161206691-1d305250-4bea-447d-9f6d-a4f0419830e3.png)

**TUM(Thinned U-shape Module)**

TUM에서는 Encoder-Decoder 구조를 사용하여 각 level의 multi-scale feature map을 생성하고, 마지막 출력 feature map은 다음 level에 전달됩니다. 

![image-20220401115702546](https://user-images.githubusercontent.com/70505378/161206692-76095f01-e03e-4104-93a7-2a205b06883a.png)

**FFMv2(Feature Fusion Module v2)**

FFMv2에서는 base feature와 이전 level의 TUM에게서 전달 받은 feature map을 fusion하여 다음 level의 TUM의 입력으로 전달합니다. 

이때의 fusion은 아래와 같이 수행됩니다. 

![image-20220401115917633](https://user-images.githubusercontent.com/70505378/161206694-41654904-184c-49af-9058-1b7c7b23df92.png)



**SFAM(Scale-wise Feature Aggregation Module)**

SFAM은 multi level의 TUM들에서 생성된 multi-scale feature map들을 합치는 과정을 수행합니다. 이때 합치는 과정은 다음과 같습니다. 

각 level의 multi-scale feature map들에서 scale이 같은 것들끼리 concat(scale-wise concatenation) 시킵니다. 이렇게 함으로써 concat된 각 scale의 feature map들은 multi-level 정보를 포함하게 됩니다. 

![image-20220401133607256](https://user-images.githubusercontent.com/70505378/161206695-5a02cb19-a64b-436d-b284-7b5486eb3fdc.png)

Concatenation 과정을 후에는 **Attention** 연산이 이어집니다. 이때 사용하는 방법은 SENet에서 제안된 Squeeze-Excitation으로, feature map의 각 channel에 가중치를 부여하여 곱하는 것으로 수행됩니다. 

![image-20220401133839292](https://user-images.githubusercontent.com/70505378/161206698-a9409ec7-bc4e-415a-a44e-b7d87d4db102.png)

**SSD Head(Single Shot Detector Head)**

이렇게 생성된 multi-scale의 multi-level feature map들은 SSD Head에 전달됩니다. 이때 **SSD Head**라고 한 이유는 SSD 모델의 입력으로 들어가는 것이 아니기 때문입니다. 

Multi-scale의 feature map들은 SSD의 입력이 아니라, **각 scale에 해당하는 feature map 역할**을 대신하여 detection을 수행합니다.

![image-20220401135049054](https://user-images.githubusercontent.com/70505378/161206702-b55ac0b6-c0b6-4f71-8a5b-9e140973acc5.png)

<br>

M2Det에서는 8개의 TUM을 사용했으며, 각 level마다 6개의 scale features를 사용했습니다. 

Detection stage에서는 6개의 feature마다 2개의 convolution layer를 추가하여 regression과 classification을 수행했으며, 6개의 anchor box와 soft-nms 기법을 사용했다고 합니다. 

<br>

M2Det은 multi-scale에서 더 나아가, 기존에 잘 잡아내지 못 했던 물체들의 외형 정보를 multi-level을 이용하여 잡아낼 수 있었습니다. 그 결과 아래와 같은 성능 향상을 보여주기도 했습니다. 

![image-20220401135633554](https://user-images.githubusercontent.com/70505378/161206705-030c0d5e-dd9a-4cfe-a227-26a50ec14074.png)



<br>

<br>

## CornerNet

마지막으로 anchor box를 사용하지 않은 anchor-free model인 `CornerNet`에 대해 알아보겠습니다. 

### Overview

CornetNet은 anchor box를 사용함으로써 발생하는 단점들을 해결하려는 시도에서 나온 모델입니다. 

Anchor box는 분명 편리하게 roi들을 추출해주기는 하지만, roi의 숫자가 매우 많아진다는 단점이 있습니다. 또한 이 경우에 positive sample(객체)은 적고, 대부분이 negative sample(배경)이 됩니다. 즉, class imbalance 문제가 발생한다는 것이죠. 

Anchor box의 또 하나의 단점은 사람이 scale, ratio, stride 등을 휴리스틱하게 결정해야 한다는 것입니다. 

<br>

**CornerNet**은 anchor box를 사용하지 않는 1 stage detector로, anchor box 대신 좌측 상단(top-left)과 우측 하단(bottom-right) 점을 이용하여 객체를 검출합니다. 

Center(중심점)가 아니라 Corner(모서리)를 사용하는 이유는 center의 경우 4개의 면을 모두 고려해야 하는 반면, corner를 사용하면 2개만 고려하면 되기 때문이라고 합니다. 

<br>

### Architecture

![image-20220401140444724](https://user-images.githubusercontent.com/70505378/161206710-2e30e314-5ed3-4055-a4fa-e4d592526ab9.png)

CornerNet의 전체 아키텍쳐는 위와 같습니다. 

* `Hourglass Network`: Segmentation, pose estimation에서 사용하는 모델로, encoder-decoder 구조로 구성된 backbone
* `Prediction Module`: 각 객체에 대해 top-left와 bottom-right corner를 예측합니다. 이 때 top-left corner에서는 corner pooling이라는 추가적인 연산이 진행됩니다. 
  * Heapmaps: 채널의 개수가 클래스의 개수와 같은 heat map 형태의 feature map 생성. 각 채널 feature map의 pixel들은 해당 클래스 객체의 top-left corner인지를 0, 1 binary mask로 표현. 
  * Embeddings: Top-left corner를 그에 대응하는 Bottom-right corner에 매핑
  * Offsets: Hourglass 과정에서 발생할 수 있는 quantization error에 대응하여 미세한 박스 조정을 수행

그럼 좀 더 자세히 각 부분에 대해 보도록 하겠습니다. 

**Hourglass**

![image-20220401142116546](https://user-images.githubusercontent.com/70505378/161206713-734e6acf-7918-4114-b5bf-a0dff3a7c731.png)

Hourglass network는 pose estimation에서 주로 사용하는 모델입니다. 

[Encoder-Decoder]로 구성된 하나의 hourglass module을 여러 개 stack한 것이 hourglass network이며, Encoder-Decoder 구조를 반복적으로 사용하여 global과 local 정보를 모두 잘 추출할 수 있다는 장점을 가지고 있습니다. 

Encoding part와 Decoding part는 각각 다음의 역할을 수행합니다. 

* Encoding Part
  * Feature 추출 : convolution layer + maxpooling layer
  * 별도의 branch로 convolution 진행해서 스케일마다 feature 추출 (upsampling 과정에서 조합)
  * CornerNet에서는 maxpooling 대신 stride 2 사용, 스케일 5번 감소  
* Decoding Part
  * Encoder 과정에서 스케일별로 추출한 feature 조합
  * Upsampling 과정에서는 Nearest Neighbor Sampling, feature조합에서는 element-wise addition 사용  

**Perdiction Module**

Hourglass network를 거쳐 추출된 feature map은 prediction module에 전달됩니다. 

아래서 부터는 Prediction Module을 구성하는 detecting corner, grouping corner, corner pooling 과정에 대해 살펴보겠습니다. 

![image-20220401142916292](https://user-images.githubusercontent.com/70505378/161206715-fc32a3c4-98e4-4cb6-b2a5-b1b21bee64f1.png)



**Detecting Corner**

Detecting Corner 과정은 크게 heapmap과 offset 생성 과정으로 나뉩니다. 

<br>

먼저 heapmap 생성 과정에 대해 알아보겠습니다. 

**2개의 heapmap을 통해서 각각 top-left와 bottom-right corner를 예측**합니다. Heapmap은 (C, H, W)의 shape을 가지며, 이때 C는 클래스 개수입니다. 각 채널은 클래스에 해당하는 corner의 위치를 나타내는 binary mask(0 or 1)입니다. 

그렇다면 이렇게 corner를 예측할 때는 어떻게 학습을 시켜야 할까요?

Ground truth corner와 predicted corner간 거리가 가까울수록 낮은 loss를 부여하고, 멀수록 높은 loss를 부여하는 것이 자연스럽습니다. 

따라서 이 과정에서는 모든 negative location(정확히 맞히지 못 한 corner)에 동일한 loss를 부여하는 것이 아닌, **원 모양의 positive location** 구역을 정해 그 원 안에 들어오는 negative location들은 loss 값을 줄이는 방법을 사용했습니다. 이때 positive location의 반지름은 물체의 크기에 따라 결정됩니다. 

![image-20220401144259746](https://user-images.githubusercontent.com/70505378/161206716-65f03714-7e52-4b0c-90bb-e7fe61bffd78.png)

Detection heapmap을 구하는 과정에서 실제로 사용하는 loss 함수로는 Focal loss를 변형한 형태의 loss를 사용합니다. 정답에 근접한 예측값에는 낮은 loss를 부여할 수 있습니다. 

![image-20220401144440975](https://user-images.githubusercontent.com/70505378/161206717-b8fffc18-c798-4d8b-aa27-c8cf9b8377a6.png)

<br>

다음으로 offset 생성 과정에 대해 간단히 보겠습니다. 

Input image는 **backbone을 통과하면서 floating point loss가 발생**하고, heapmap에서 image로 위치를 매핑시킬 때 그로 인한 차이가 발생할 수 있습니다. 이는 크기가 작은 물체일 경우 큰 차이를 불러 일으킬 수 있습니다. 

Offset은 이렇게 손실된 정보들을 보완하기 위해 **heapmap에서 예측한 위치를 약간 조정**하는 역할을 합니다. 이때 손실 함수로는 Smooth L1 loss를 사용합니다. 

![image-20220401145045794](https://user-images.githubusercontent.com/70505378/161206719-371a9995-4e6c-45f3-8eaf-12b3f4885d53.png)







**Grouping Corner**

Grouping Corner 과정에서는 top-left corner와 bottom-right corner를 올바르게 짝지어 주기 위한 embedding을 생성합니다. 

Top-left corner와 Bottom-right corner는 생성된 embedding 값의 차이에 의해 그룹이 지어지며, embedding 값 사이의 거리가 작으면 같은 물체의 bbox에 속하는 것입니다. 

![image-20220401145615039](https://user-images.githubusercontent.com/70505378/161206723-4e49d86d-799f-499a-9597-c0dfe8d1b67e.png)







**Corner Pooling**

마지막으로 Corner pooling 과정이 무엇인지 알아보겠습니다. 

사실, 앞선 detecting corner와 grouping corner 과정을 하기 전에 **corner pooling** 과정이 선행되어 이루어집니다. 이 Corner pooling은 **corner에 해당하는 픽셀 위치를 특정**시켜주기 위한 과정인데요, 코너는 하나의 픽셀에 대응하고 이 픽셀이 '어떤 값을 가져야 corner다!'라는 등의 특징이 없기 때문에, 위치를 특정시켜 주는 과정이 필요합니다. 

![image-20220401150555587](https://user-images.githubusercontent.com/70505378/161206630-fe9d58ec-ada4-4b5b-8a49-01b6776849c6.png)

Top-left corner를 구하기 위한 corner pooling 연산은 아래와 같이 이루어집니다. Backbone에서 출력된 feature map을 한 번은 right->left로 가며 더 큰 값으로 이어지는 값들을 채우고, 한 번은 bottom->top으로 가며 더 큰 값으로 이어지는 값들을 채웁니다. 

이후 두 feature map을 element-wise summation하여 최종 corner pooling된 feature map을 생성합니다. 이렇게 생성된 feature map에서 주변 값들보다 큰 값을 가지는 픽셀들이 corner에 해당하는 픽셀이라고 특정 할 수 있는 것입니다. 

이렇게 corner pooling을 통과한 feature map으로부터 heapmap, embedding, offset들을 계산하게 됩니다. 

![image-20220401151236285](https://user-images.githubusercontent.com/70505378/161206636-5198e7ff-329e-48bb-9f2a-59668b116a1e.png)

<br>

CenterNet이 나왔을 당시의 성능은 아래 표처럼 anchor box를 사용하는 다른 모델들에 비해 더 좋은 모습을 보여주기도 했습니다. 

![image-20220401151401497](https://user-images.githubusercontent.com/70505378/161206642-be33dca6-7a5e-4553-9c41-b8ebbe8fdec9.png)



<br>

### Follow-up

CornerNet 이후로 anchor-free 모델들이 꾸준히 발표되고 있습니다. 

**CenterNet**

* Keypoint heatmap을 통해 중심점(center)예측
* Center사용하여 단 하나의 anchor box생성
  * Keypoints grouping 과정이 필요 없어 시간 단축
  * NMS 과정 x  
* https://arxiv.org/pdf/1904.08189.pdf (CenterNet: Keypoint Triplets for Object Detection)

![image-20220401151819960](https://user-images.githubusercontent.com/70505378/161206645-3f71ab79-c11f-4be7-aebd-2e20b5d2dd34.png)

**FCOS**

* 중심점으로부터 바운딩 박스의 경계까지의 거리 예측
* FPN을 통해 multi-level 예측  
* https://arxiv.org/pdf/1904.01355.pdf (FCOS: Fully Convolutional One-Stage Object Detection)

![image-20220401151911147](https://user-images.githubusercontent.com/70505378/161206650-1919186f-bd67-4be1-8e32-c5b43ef9f48c.png)





<br>

<br>

# 참고 자료

* Hoya012, https://hoya012.github.io/
* https://herbwood.tistory.com
* https://arxiv.org/pdf/2004.10934.pdf (YOLOv4: Optimal Speed and Accuracy of Object Detection)
* https://arxiv.org/pdf/1911.11929.pdf (CSPNet: A New Backbone that can Enhance Learning Capability of CNN)
* https://arxiv.org/pdf/1905.04899.pdf (CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features)
* https://arxiv.org/pdf/1810.12890.pdf (DropBlock: A regularization method for convolutional networks)
* https://arxiv.org/pdf/1811.04533.pdf (M2Det: A Single-Shot Object Detector based on Multi-Level Feature Pyramid Network)
* https://arxiv.org/pdf/1808.01244.pdf (CornerNet: Detecting Objects as Paired Keypoints)
* https://arxiv.org/pdf/1904.08189.pdf (CenterNet: Keypoint Triplets for Object Detection)
* https://arxiv.org/pdf/1904.01355.pdf (FCOS: Fully Convolutional One-Stage Object Detection)
* https://giou.stanford.edu/GIoU.pdf (Generalized intersection over union: A metric and a loss for bounding box regression)
* https://arxiv.org/pdf/1406.4729.pdf (Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition)
* https://arxiv.org/pdf/1807.06521.pdf (Cbam: Convolutional block attention module)
* https://arxiv.org/pdf/1612.03144.pdf (Feature Pyramid Networks for Object Detection)
* https://arxiv.org/vc/arxiv/papers/1908/1908.08681v2.pdf (Mish: A Self Regularized Non-Monotonic Neural Activation Function)  

