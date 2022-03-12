---
layout: single
title: "[AITech][CV] 20220307 - Part 2) Semantic Segmentation"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['FCN', 'U-Net', 'DeepLab']
---

_**본 포스팅은 POSTECH '오태현' 강사 님의 강의를 바탕으로 작성되었습니다. **_

<br>

# Semantic Segmentation

이번 포스팅에서는 기존 CNN 구조에서 발전된 형태의 네트워크로 수행할 수 있는 **Semantic Segmentation**이라는 CV 분야의 새로운 task에 대해 알아보겠습니다. 

## Semantic segmentation

앞선 포스팅에서는 CNN 구조를 이용해 이미지를 분류하는 Image classification task에 대해 살펴보았었습니다. 이번 포스팅에서 다룰 semantic segmentation은 간단히 **픽셀 단위 분류**라고 생각하면 쉽습니다. Segmentation에는 여러 종류가 있는데(semantic, instance, panoptic), 그 중 semantic segmentation은 클래스 단위로 픽셀을 분류합니다. 즉, 다른 사람이어도 하나의 클래스로 분류합니다.

의료나 자율 주행 등의 분야에 활용 가능성이 높은 기술입니다. 

![image-20220311230748191](https://user-images.githubusercontent.com/70505378/158005896-171d398b-884e-4c53-9262-b18e113946b2.png)

<br>

## Semantic segmentation architectures

그렇다면 이제 이 semantic segmentation은 어떤 아키텍처로 어떻게 구현할 수 있는지 살펴봅시다. 

### Fully Convolutional Networks (FCN)

Semantic segmentation을 위한 네트워크들의 조상격인 **FCN** 아키텍처에 대해 살펴보겠습니다. 

![image-20220311231417921](https://user-images.githubusercontent.com/70505378/158005898-66945897-baf3-4e3e-b617-f683357c03e8.png)

**어떻게?**

* 기존 CNN 아키첵쳐의 마지막 fc layer를 1x1 convolution layer로 대체

  * Fully connected layer: Output a fixed dimensional vector and discard spatial coordinates
  * Fully convolutional layer: Output a classification map which has spatial coordinates  

  ![image-20220311231534090](https://user-images.githubusercontent.com/70505378/158005900-596ffed5-742d-4035-b9c0-ca8a9ad73482.png)

FC layer를 1x1 convolution layer로 교체할 때 얻을 수 있는 효과는 **공간적인 정보를 유지**할 수 있다는 것입니다. 

FC layer를 거친 최종 output vector는 각 클래스에 속할 확률을 표현했었습니다. 1x1 convolution에서는 연산을 거친 output feature map의 각 channel이 각 클래스에 속할 확률을 표현합니다. 가령, 총 클래스 개수가 C개이면 C개의 1x1 filter를 사용해 최종 feature map은 C 크기의 channel dimension을 갖게 되고, 각각의 channel의 픽셀 값은 그 픽셀이 해당 클래스에 속할 값(확률은 아님)을 나타냅니다. 그 값이 클수록 해당 픽셀이 해당 클래스에 속할 확률이 높아지는 것입니다. 

이렇게 공간 정보가 담긴 C개의 feature map을 겹쳐놓으면, 이미지에 대한 heat map을 얻을 수 있습니다. 

![image-20220311232354444](https://user-images.githubusercontent.com/70505378/158005901-35ff2c9f-1d4c-4ab8-852c-721471ff8a33.png)

**해상도 문제**

여기서 문제!

CNN을 거친 이미지는 (h, w) 크기가 매우 작아져 있을 것입니다. 따라서 이를 최종적으로 원본 이미지의 크기만큼 Upsampling 해주어야 합니다. 

![image-20220311232536968](https://user-images.githubusercontent.com/70505378/158005903-1beb7d2d-ae40-4b06-b9c1-aa88a682973f.png)

그리고 Upsampling을 수행할 수 있는 방법으로는 대표적으로 다음 세 가지 방법이 있습니다. 

* Unpooling
* Transposed convolution
* Upsample and convolution

각각의 연산에 대한 자세한 설명은 여기서는 생략하겠습니다. 더 궁금하신 분들은 [이 포스팅](https://wowo0709.github.io/ai/computervision/Computer-Vision-6(1).-%EC%9D%B8%EC%BD%94%EB%8D%94-%EB%94%94%EC%BD%94%EB%8D%94%EB%A1%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%B3%80%ED%99%98/)을 참고하거나 검색해보시길 추천합니다. 

Transposed convolution은 그 연산 과정에서 일부 픽셀의 uneven overlap이 발생해서, 이를 방지하도록 padding과 stride parameter를 잘 조정해주는 것이 중요하다고 합니다. 이렇게 불규칙한 픽셀의 overlap이 발생하면 checkerboard artifact는 생성하는 문제를 야기해 출력 이미지에 좋지 않은 영향을 준다고 합니다. 

![image-20220312000706897](https://user-images.githubusercontent.com/70505378/158005907-3fa5cca9-7b3c-401c-907a-5dc2a5b9f241.png)

이에 반해  upsampling과 convolution을 함께 사용하면 모든 픽셀이 골고루 값에 영향을 주어서 일부 픽셀만 overlap되는 문제를 해결할 수 있다고 합니다. (**\{Nearest-neighbor(NN), Bilinear\}** interpolation followed by **convolution**)

![image-20220311234537973](https://user-images.githubusercontent.com/70505378/158005904-33ea85b4-b8e5-4dbe-b955-b90c725b686e.png)

처음에는 각 연산의 작동 과정을 완벽히 이해하는 것이 어렵기 때문에, 이러한 기법들이 upsampling에 사용된다는 것만 기억해두면 좋습니다.  

<br>

**공간정보 유실 문제**

하지만 이렇게 Upsampling을 하더라도, 이미 해상도가 낮아진 상황에서 잃어버린 정보를 모두 복원하는 것은 불가능합니다. 

따라서 맨 마지막 출력 feature map만 사용하는 것이 아니라, 중간 단계의 activation map들을 가져와 upsampling하고 그 정보를 함께 사용함으로써 더 정확한 결과를 얻을 수 있습니다. 

![image-20220311235452669](https://user-images.githubusercontent.com/70505378/158005905-684445f7-e6ba-4714-a399-76262c7ce9f9.png)

FCN-32s, FCN-16s, FCN-8s로 갈수록 더 낮은 단계의 activation map까지 사용하게 되고, 따라서 더 자세한 결과를 얻을 수 있습니다. 

![image-20220311235723644](https://user-images.githubusercontent.com/70505378/158005906-9445201c-5946-492a-aa0e-0fdd004a311a.png)



### Hypercolumns for object segmentation

`Hypercolumn`에 관한 논문은 FCN이 발표된 논문과 비슷한 시기에 발표된 또 다른 object segmentation을 task로 한 연구로, 낮은 layer와 높은 layer 간 feature들의 융합을 강조한 연구였습니다. 

![image-20220312114216114](https://user-images.githubusercontent.com/70505378/158005909-5e448715-15b7-4b6d-88a9-8a9f2e984222.png)

다만, FCN과 다른 점은 hypercolumn은 end-to-end 학습이 불가능하다는 것입니다. hypercolumn은 별도의 알고리즘으로 서로 다른 해상도의 이미지에서 object detection을 먼저 수행하고 segmentation을 해서 융합하는 식으로 구현되었습니다. 

![image-20220312114150800](https://user-images.githubusercontent.com/70505378/158005908-735b1c48-dd78-45dd-b4e9-32bd8fdc6331.png)



### U-Net

`U-Net`은 segmentation task의 부흥기를 시작한 모델로, 전체 이미지보다 이미지의 세세한 부분에 집중하는 대부분 모델들의 시초인 모델입니다. 

U-Net도 fully convolutional network이며, end-to-end로 학습이 가능합니다. 아래는 U-Net의 구조이고, Downsampling과 Upsampling이 연속적으로 일어나는 것이 특징입니다. 

![image-20220312115431737](https://user-images.githubusercontent.com/70505378/158005910-fd556867-a001-411a-a1d7-e4acd27c9f21.png)

각각의 convolution은 k=3, s=1, p=0으로 수행됩니다. U-Net은 이름에서 처럼 네트워크 구조가 U자를 그리며 양쪽이 대칭적인 것이 특징입니다. 

**Contracting path**에서는 CNN과 동일하게 이미지에 convolution 연산을 적용해 feature map을 뽑아내고, **Expanding path**에서는 뽑아낸 최종 feature map의 Upsampling과 함께 contracting path에서 만들어졌던 대칭 부분의 feature map을 concatenate하여 사용합니다. 

**feature map 사이즈 문제**

위 구조를 보면 각 feature map의 크기가 모두 2의 배수인 것을 볼 수 있는데, 이는 contracting path에서의 모든 feature map의 (h, w)가 짝수여야 expanding path에서 제대로 복원이 가능하기 때문입니다. 

따라서, U-Net을 사용할 때는 **feature map의 (h, w) 크기가 홀수가 되지 않도록** 유념해야 합니다. 

**U-Net PyTorch code**

U-Net의 구조는 아래와 같은 pytorch layer로 표현될 수 있습니다. 

![image-20220312120547641](https://user-images.githubusercontent.com/70505378/158005912-62891dd2-7bca-449c-8ae1-b585c8a5ef95.png)



### DeepLab

`DeepLab`은 segmantation task에서 중요한 한 획을 그은 모델입니다. 2015년 DeepLab v1에서 시작해 v2, v3를 거쳐 현재는 2018년 발표된 DeepLab v3+가 가장 최신 모델입니다. 

DeepLab에서 사용하는 주요 기법에 대해 알아보겠습니다. 

 **Conditional Random Fields (CRFs)**

CRF는 모델에 최초 output에 몇 차례의 후처리를 더하여 **output을 정교화시키는 과정**에 해당합니다. 조금 더 자세히는 이미지의 각 픽셀을 그래프의 노드로 여겨서 각 그래프를 적절하게 확장시켜나가는 최적화 기법이라고 할 수 있습니다. 

![image-20220312143017421](https://user-images.githubusercontent.com/70505378/158005914-dc241266-5020-4724-974a-4b616ffdbb9e.png)





**Dilated convolution (Atrous convoltion)**

Dilated convolution은 커널 사이에 padding을 넣어서 동일한 파라미터를 가지는 커널로 확장된 형태의 convolution을 수행하는 것을 말합니다. 이 연산으로 **receptive field를 exponentially expand** 할 수 있습니다. 

![image-20220312143226106](https://user-images.githubusercontent.com/70505378/158005915-c23ed1aa-b0ba-4649-81e0-4c6c947da210.png)



**Depthwise separable convolution**

Depthwise separable convolution은 기존 (C<sub>in</sub>, H, W) 크기의 커널을 C<sub>out</sub>개 사용해서 한 번에 처리했던 convolution 연산을 **Depthwise convolution**((H, W) 크기의 커널 C<sub>in</sub>개)과 **pointwise(1x1) convolution**((C<sub>in</sub>, 1, 1) 크기의 커널 C<sub>out</sub>개) 연산으로 쪼개서 같은 효과를 내면서도 그 **파라미터 수를 줄이고 학습 속도를 촉진**시킬 수 있습니다. 

![image-20220312144443347](https://user-images.githubusercontent.com/70505378/158005916-d4b4ebee-d05c-454f-a72c-70f43d9c9dca.png)

D<sub>K</sub>를 kernel size, M을 input channels, N을 output channels라고 하면, 

* Standard conv: D<sub>K</sub><sup>2</sup>MN
* Depthwise separable conv: D<sub>K</sub><sup>2</sup>M + MN
  * depthwise convolution: N = 1
  * pointwise convolution: D<sub>K</sub> = 1

으로 파라미터 수를 줄일 수 있습니다. 

마지막으로 DeepLab v3+의 네트워크 아키텍쳐를 보며 포스팅을 마치겠습니다. 

![image-20220312145231434](https://user-images.githubusercontent.com/70505378/158005917-6d7658fa-26dc-42b1-9c58-146008898ad3.png)

























<br>

<br>

# 참고 자료



* Semantic segmentation
  * Chen et al., Rethinking Atrous Convolution for Semantic Image Segmentation, arXiv 2017 
  * Novikov et al., Fully Convolutional Architectures for Multi-Class Segmentation in Chest Radiographs, T-MI 2016 
  * Aksoy et al., Semantic Soft Segmentation, SIGGRAPH 2018

* Semantic segmentation architectures
  * Long et al., Fully Convolutional Networks for Semantic Segmentation, CVPR 2015 
  * Hariharan et al., Hypercolumns for Object Segmentation and Fine-Grained localization, CVPR 2015 
  * Ronneberger et al., U-Net: Convolutional Networks for Biomedical Image Segmentation, MICCAI 2015 
  * Chen et al., Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs, ICLR 2015 
  * Howard et al., MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications, arXiv 2017 
  * Chen et al., Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation, ECCV 2018  

<br>

