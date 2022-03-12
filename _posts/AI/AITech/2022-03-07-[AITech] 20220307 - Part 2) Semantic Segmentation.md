---
layout: single
title: "[AITech][CV] 20220307 - Part 2) Semantic Segmentation"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['FCN', 'U-Net', 'DeepLab']
---



<br>

_**본 포스팅은 POSTECH '오태현' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# Semantic Segmentation

이번 포스팅에서는 기존 CNN 구조에서 발전된 형태의 네트워크로 수행할 수 있는 **Semantic Segmentation**이라는 CV 분야의 새로운 task에 대해 알아보겠습니다. 

## Semantic segmentation

앞선 포스팅에서는 CNN 구조를 이용해 이미지를 분류하는 Image classification task에 대해 살펴보았었습니다. 이번 포스팅에서 다룰 semantic segmentation은 간단히 **픽셀 단위 분류**라고 생각하면 쉽습니다. Segmentation에는 여러 종류가 있는데(semantic, instance, panoptic), 그 중 semantic segmentation은 클래스 단위로 픽셀을 분류합니다. 즉, 다른 사람이어도 하나의 '사람' 클래스로 분류합니다.

![image-20220312150739539](https://user-images.githubusercontent.com/70505378/158006303-97ce7dbb-36b7-42bb-a9fd-30699db48d47.png)

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

## 실습) Classification to Segmentation

이번 포스팅에서의 실습은 Classification model에서 Segmentation model로의 발전 방법에 대한 실습입니다. 

앞서 이론에서 semantic segmentation model은 기존의 classification model의 마지막 FC layer를 1x1 convolution layer로 교체함으로써 생성할 수 있다고 했습니다. 

이 때 우리가 알아갈 것은 **Classifier의 FC layer parameter를 Segmentation 모델의 1x1 convolution layer parameter로 그대로 사용할 수 있다는 것**입니다. FC layer는 **in_channels * out_classes** 개의 paramter를 갖고, 1x1 convolution layer는 **(1x1xin_channels) 개의 kernel을 out_classes 개** 갖게 되어 두 layer에서 parameter의 수가 같습니다. 

따라서 FC layer의 paramter를 적절히 reshaping하여 1x1 convolution layer의 parameter로 사용할 수 있습니다. 

여기서는 다음 3개의 코드를 보도록 하겠습니다. 

*  **VGG11Backbone**: VGG-11의 backbone에 해당하는 코드
* **VGG11Classification**: VGG11Backbone의 출력값을 받아 glabal average pooling을 거친 다음 FC layer를 통과시켜 최종 classification을 수행하는 코드
* **VGG11Segmentation**: VGG11Backbone의 출력값을 받아 1x1 convolution layer를 통과시켜 semantic segmentation을 수행하는 코드. 이때, VGG11Classification의 FC layer의 weights를 가져와 1x1 convolution layer의 weights 값으로 사용하기 위해 **copy_last_layer** 메서드를 구현합니다. 

 **VGG11Backbone**

```python
import torch
import torch.nn as nn

class VGG11BackBone(nn.Module):
  '''
  VGG-11의 backbone에 해당하는 부분입니다.
  총 8개의 convolution layer로 구성되어 있습니다.
  '''
  def __init__(self):
    super(VGG11BackBone, self).__init__()

    self.relu = nn.ReLU(inplace=True)
    
    # Convolution Feature Extraction Part
    self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
    self.bn1   = nn.BatchNorm2d(64)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
    self.bn2   = nn.BatchNorm2d(128)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
    self.bn3_1   = nn.BatchNorm2d(256)
    self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    self.bn3_2   = nn.BatchNorm2d(256)
    self.pool3   = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
    self.bn4_1   = nn.BatchNorm2d(512)
    self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.bn4_2   = nn.BatchNorm2d(512)
    self.pool4   = nn.MaxPool2d(kernel_size=2, stride=2)

    self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.bn5_1   = nn.BatchNorm2d(512)
    self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
    self.bn5_2   = nn.BatchNorm2d(512)
  
  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.relu(x)
    x = self.pool1(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.relu(x)
    x = self.pool2(x)

    x = self.conv3_1(x)
    x = self.bn3_1(x)
    x = self.relu(x)
    x = self.conv3_2(x)
    x = self.bn3_2(x)
    x = self.relu(x)
    x = self.pool3(x)

    x = self.conv4_1(x)
    x = self.bn4_1(x)
    x = self.relu(x)
    x = self.conv4_2(x)
    x = self.bn4_2(x)
    x = self.relu(x)
    x = self.pool4(x)

    x = self.conv5_1(x)
    x = self.bn5_1(x)
    x = self.relu(x)
    x = self.conv5_2(x)
    x = self.bn5_2(x)
    x = self.relu(x)

    return x
```





**VGG11Classification**

```python
class VGG11Classification(nn.Module):
  def __init__(self, num_classes = 7):
    '''
    VGG-11의 classifier에 해당하는 부분입니다.
    VGG11BackBone의 출력값을 받아 max pooling - global average pooling - fully connected를 통과하여 최종 prediction logits를 출력합니다.
    '''
    super(VGG11Classification, self).__init__()

    self.backbone = VGG11BackBone()
    self.pool5   = nn.MaxPool2d(kernel_size=2, stride=2)
    self.gap      = nn.AdaptiveAvgPool2d(1)
    self.fc_out   = nn.Linear(512, num_classes)

  def forward(self, x):
    x = self.backbone(x)
    x = self.pool5(x)
    x = self.gap(x)
    x = torch.flatten(x, 1)
    x = self.fc_out(x)

    return x
```





**VGG11Segmentation**

```python
class VGG11Segmentation(nn.Module):
  def __init__(self, num_classes = 7):
    '''
    VGG-11를 재구성하여 semantic segmentation을 해결하기 위한 모델에 해당하는 부분입니다.
    VGG11BackBone의 출력값을 받아 1x1 convolution을 통과하여 픽셀별 classification을 수행한 다음,
    max pooling으로 인하여 줄어든 resolution을 bilinear upsampling을 통해 입력 이미지의 크기로 확장합니다.
    '''
    super(VGG11Segmentation, self).__init__()

    self.backbone = VGG11BackBone()
    
    '''==========================================================='''
    '''======================== TO DO (1) ========================'''
    ### 모델의 마지막 layer의 in_features 값을 어떻게 가져올까??

    in_features=512

    with torch.no_grad():
      self.conv_out = nn.Conv2d(in_features, num_classes, 1)

    self.fc_out = VGG11Classification().fc_out
    self.copy_last_layer(self.fc_out)
    '''======================== TO DO (1) ========================'''
    '''==========================================================='''
  
    self.upsample = torch.nn.Upsample(scale_factor=16, mode='bilinear', align_corners=False)


  def forward(self, x):
    x = self.backbone(x)
    x = self.conv_out(x)
    x = self.upsample(x)
    assert x.shape == (1, 7, 224, 224)

    return x


  def copy_last_layer(self, fc_out):
    """
    VGG-11 classifier의 마지막 fully-connected layer인 'fc_out'을 입력으로 받아,
    해당 'fc_out'의 weights를 __init__에서 구현한 1x1 convolution filter의 weights로 copy하는 method입니다.
    """

    '''==========================================================='''
    '''======================== TO DO (2) ========================'''
    
    reshaped_fc_out = fc_out.weight.detach()
    reshaped_fc_out = torch.reshape(reshaped_fc_out, self.conv_out.weight.size())
    self.conv_out.weight = nn.Parameter(reshaped_fc_out)

    '''======================== TO DO (2) ========================'''
    '''==========================================================='''
    assert self.conv_out.weight[0][0] == fc_out.weight[0][0]
    
    return 
```

**Check output size**

```python
test_input = torch.randn((1, 3, 224, 224))

modelC = VGG11Classification()
out = modelC(test_input)
print('The output shape of the classification network:', out.shape)

modelS = VGG11Segmentation()
out = modelS(test_input)
print('The output shape of the segmentation network:', out.shape)

'''
The output shape of the classification network: torch.Size([1, 7])
The output shape of the segmentation network: torch.Size([1, 7, 224, 224])
'''
```

<br>

Segmentation 모델을 추가적으로 학습시키지 않고 Pre-trained Classification model의 FC layer parameter들을 copy해오는 것 만으로 아래와 같은 결과를 얻을 수 있습니다. Classification model은 사람이 마스크를 어떤 식으로 착용하고 있는지에 대한 데이터셋으로 학습된 상태입니다. 따라서 segmentation 결과가 마스크 주위에 집중되어 있는 것을 볼 수 있습니다. 

아래 결과가 정확하지는 않은데, 이는 마스크 영역에 해당하는 픽셀별 ground truth가 주어지지 않았기 때문이며 또한 입력 이미지에 비해 16분의 1 사이즈의 feature map에서 픽셀별 예측을 진행하고 단순히 bilinear interpolation을 진행했기 때문입니다. 

더 정확한 결과를 위해서는 segmentation model을 ground truth와 함께 학습시키고, 최종 단에서 upsampling 시 단순한 bilinear interpolation이 아니라 더 나은 기법을 사용해 볼 수 있습니다. 

![image-20220312175452680](https://user-images.githubusercontent.com/70505378/158011514-df6bce16-a1b1-438d-a582-51529a1b90e0.png)





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

