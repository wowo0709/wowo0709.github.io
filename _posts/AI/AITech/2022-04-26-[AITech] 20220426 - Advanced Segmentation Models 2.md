---
layout: single
title: "[AITech][Semantic Segmentation] 20220426 - Advanced Segmentation Models 2"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['DeepLabv2', 'PSPNet', 'DeepLabv3', 'DeepLabv3+']
---



<br>

_**본 포스팅은 KAIST의 '김현우' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Advanced Segmentation Models 2

이번 포스팅에서는 지난 포스팅에 이어 receptive field를 확장시킴으로써 segmentation 성능을 끌어올린 모델들에 대해 보도록 하겠습니다. 

Deeplab v2부터 Deeplabv3, Deeplabv3+ 까지 다루며 PSPNet에 대해서도 살펴보겠습니다. 

## DeepLab v2

이전 포스팅에서 다뤘던 DeepLab v1은 dilated convolution을 사용해서 receptive field를 효과적으로 확장시킨 모델이었습니다. 

![image-20220426150613811](https://user-images.githubusercontent.com/70505378/165428741-2b9663c1-94f0-4559-8f4c-cc9717ad407c.png)

`Deeplab v2`에서 달라진 점은 두가지입니다. 

첫번째로 **backbone으로 ResNet-101**을 사용했습니다. 단, 기존 ResNet-101의 conv4, conv5에서는 1x1 convolution(stride=2)을 통해 downsampling을 진행하고 convolution 연산을 수행한데 반해, Deeplab v2에서는 downsampling을 수행하지 않고 convolution 대신 dilated convolution을 사용하였습니다. 

![image-20220426151918820](https://user-images.githubusercontent.com/70505378/165428747-39fb5cc1-22ad-497a-97d9-1b0c34262256.png)

두번째로는 **ASPP(Atrous Spatial Pyramid Pooling)** 모듈을 사용한다는 점입니다. 기존 FC6, FC7, FC8(Score) block에 ASPP 모듈을 사용하여 더 다양한 레벨의 feature를 함께 사용할 수 있도록 하였습니다. 

![image-20220426152713823](https://user-images.githubusercontent.com/70505378/165428749-798b4502-7a3b-487a-90b0-1ea2ee1f036a.png)

이상의 두가지 방법으로 모델 성능이 개선되었음을 evaluation을 통해 확인할 수 있습니다. 

![image-20220426152923958](https://user-images.githubusercontent.com/70505378/165428750-cdf91e91-4e0b-4c26-b745-92f0fdb1c315.png)

![image-20220426153051261](https://user-images.githubusercontent.com/70505378/165428753-a35c4439-9779-4d02-b1e7-ce66651e8660.png)



<br>

<br>

## PSPNet

이번에는 조금 다른 방식으로 receptive field를 넓히려고 시도한 `PSPNet`에 대해 보겠습니다. 

PSPNet은 세가지 문제점을 제시합니다. 

* **Mismatched Relationship**
  * 호수 주변에 boat가 있는데 기존 모델(FCN)은 car로 예측
  * 원인: boat의 외관이 car와 비슷하기 때문
  * idea: 주변의 특징을 고려 (e.g. water 위의 boat)
* **Confusion Categories**
  * FCN은 skyscraper를 building과 혼동
  * 원인: ADE20K dataset의 특성상 비슷한 범주인 building과 skyscraper 존재
  * idea: Category 간의 관계를 사용하여 해결 (global contextual information)
* **Inconspicuous Classes**
  * FCN은 pillow를 bed sheet로 예측
  * 원인: pillow의 객체 사이즈가 작을 뿐 아니라 bed sheet의 커버와 같은 무늬 예측에 한계
  * idea: 작은 객체들도 global contextual information을 사용

![image-20220426153809166](https://user-images.githubusercontent.com/70505378/165428754-347338e0-9041-4f71-88f9-6fd1fce5737e.png)

또한 논문에서는 FCN에서 pooling을 진행하는 것 대비 실제 receptive field는 훨씬 좁다고 합니다. 즉, receptive field의 이론적인 크기와 실제 크기 간의 차이를 보여주기도 했습니다. 

![image-20220426154056090](https://user-images.githubusercontent.com/70505378/165428755-8ebb2498-fb87-4784-b11a-757bf408e0aa.png)

<br>

PSPNet에서는 충분히 큰 receptive field를 가지면서 global context를 고려할 수 있도록 하기 위해 **Pyramid Pooling module**을 사용합니다. 

이는 feature map에 각기 다른 size의 **average pooling**을 가하여 합치는 것으로, 다양한 context를 파악해서 예측하는 데 도움을 준다고 합니다. 

다양한 pooling으로 생성된 feature map들은 sub-region 각각에 conv를 진행하여 channel이 1인 feature map이 생성(1x1x1, 2x2x1, 3x3x1, 6x6x1)되고, 이는 upsampling 된 뒤에 pooling 전 feature map과 concat되어 최종 score map을 예측합니다. 

 ![image-20220426155037033](https://user-images.githubusercontent.com/70505378/165428758-7988ce5f-ad9b-46dc-bb9a-8b7bad4f31dc.png)

결과를 보면 아래와 같이 더 자연스러운 예측이 가능하고, 하나의 물체를 여러 개로 라벨링하는 경우가 줄어든 것을 볼 수 있습니다. 

![image-20220427103913087](https://user-images.githubusercontent.com/70505378/165428760-129cf805-37a7-4a61-8890-a45b84c37d67.png)











<br>
<br>

## DeepLab v3

`DeepLabv3`는 간단히 이야기하고 넘어가겠습니다. 

DeepLab v3는 DeepLab v2에서 사용한 ASPP 모듈과 PSPNet에서 사용한 Global Average Pooling 연산을 함께 사용합니다. Atrous convolution을 적용할 때는 zero padding을 적절히 추가하여 feature map의 크기가 변화하지 않도록 하고, global average pooling 후 크기를 upsampling 할 때는 bilinear interpolation을 사용합니다. 

![image-20220427104341303](https://user-images.githubusercontent.com/70505378/165428764-848cf02a-140e-41cd-a873-92bc0710064b.png)

















<br>

<br>

## DeepLab v3+

마지막으로 `DeepLab v3+`에 대해 살펴보겠습니다. 

DeepLab v3+ 의 전체 구조는 아래 그림의 (c)와 같습니다. 

![image-20220427104910308](https://user-images.githubusercontent.com/70505378/165428767-d5b993a3-8891-49ba-8e1d-6444aa0739fa.png)

기존 Deeplab에서 사용하던 ASPP와 다른 모델들의 Decoder 부에서 사용하던 점진적인 복원과 skip connection을 활용하는 것을 볼 수 있습니다. 

**Encoder**

* 수정된 Xception을 backbone으로 사용(DCNN)
* ASPP 모듈 사용
* Backbone 내 low-level feature와 ASPP 모듈 출력을 모두 decoder에 전달

![image-20220427105138644](https://user-images.githubusercontent.com/70505378/165428768-0d8a0d00-eaa6-4bce-98e9-d3e65ed6483a.png)

**Decoder**

* ASPP 모듈의 출력을 (bilinear) upsampling하여 low-level feature와 concat
* 결합된 정보는 convolution 연산 및 upsampling되어 최종 결과 도출
* 기존의 단순한 upsampling 연산을 개선시켜 detail을 유지하도록 함

![image-20220427105305119](https://user-images.githubusercontent.com/70505378/165428770-1bbb43a2-b8c1-4093-9a06-2d73fbd5fb79.png)

<br>

DeepLab v3+에서 사용한 Xception 구조는 효율적인 convolution 연산을 위해 **Depthwise Separable Convolution**을 사용합니다. 

Depthwise Separable Convolution은 depthwise convolution과 pointwise convolution을 차례로 적용한 것입니다. 

먼저 **depthwise convolution**은 각 채널마다 다른 filter를 사용하여 채널 별로 convolution 연산 후 결합하는 연산입니다. 

![image-20220427110018922](https://user-images.githubusercontent.com/70505378/165428772-96b83ed7-f44e-4891-a127-0260377768f5.png)

다음으로 **pointwise convolution**은 1x1 convolution을 나타냅니다. 

![image-20220427110258208](https://user-images.githubusercontent.com/70505378/165428773-632fa308-fbd6-427a-afaa-ddcba3f89ebf.png)

두 연속된 연산으로 convolution과 동일한 결과를 내면서도 훨씬 적은 수의 parameter로 효율적인 연산을 할 수 있습니다. 

코드상에서는 아래와 같이 구현할 수 있습니다. 

![image-20220427110427891](https://user-images.githubusercontent.com/70505378/165428774-0babb6ba-0303-4586-abe2-cc23100c44f8.png)



DeepLab v3+에서는 기존의 Xception 구조를 조금 수정하여 사용했습니다. 

![image-20220427111055055](https://user-images.githubusercontent.com/70505378/165428777-074d7938-e21f-4c14-96be-c9481d61242c.png)

* Entry Flow
  * Maxpooling 연산을 [Depthwise Separable Convolution + BatchNorm + ReLU] 로 변경
* Middle Flow
  * 8번의 block 반복을 16번의 block 반복으로 더 깊은 구조 사용
* Exit Flow
  * Maxpooling 연산을 [Depthwise Separable Convolution + BatchNorm + ReLU] 로 변경
  * Depthwise Separable Convolution 연산 추가

<br>

DeepLab v3+의 전체구조는 아래와 같습니다. 

![image-20220427111755397](https://user-images.githubusercontent.com/70505378/165428778-628a6d5f-767b-47a4-aec5-16924f7fbb93.png)































<br>

<br>

## 실습) DeepLab v3+

DeepLab v3+의 구조와 코드를 비교해보세요!

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_block(in_ch, out_ch, k_size, stride, padding, dilation=1, relu=True):
    block = []
    block.append(nn.Conv2d(in_ch, out_ch, k_size, stride, padding, dilation, bias=False))
    block.append(nn.BatchNorm2d(out_ch))
    if relu:
        block.append(nn.ReLU())
    return nn.Sequential(*block)


class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, dilation=1):
        super().__init__()
        if dilation > kernel_size//2:
            padding = dilation
        else:
            padding = kernel_size//2
            
        self.depthwise_conv = nn.Conv2d(
            in_ch, in_ch, kernel_size, stride, padding,
            dilation=dilation, groups=in_ch, bias=False
        )
        self.pointwise_conv = nn.Conv2d(
            in_ch, out_ch, 1, 1, bias=False
        )
        self.bn = nn.BatchNorm2d(in_ch)
        
    def forward(self, x):
        out = self.depthwise_conv(x)
        out = self.bn(out)
        out = self.pointwise_conv(out)
        return out


class XceptionBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, dilation=1, exit_flow=False, use_1st_relu=True):
        super().__init__()
        if in_ch != out_ch or stride !=1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else: 
            self.skip = None
        
        if exit_flow:
            block = [
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, in_ch, 3, 1, dilation),
                nn.BatchNorm2d(in_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, stride, dilation),
                nn.BatchNorm2d(out_ch) 
            ]
        else:
            block = [
                nn.ReLU(),
                DepthwiseSeparableConv2d(in_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),            
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, 1, dilation),
                nn.BatchNorm2d(out_ch),            
                nn.ReLU(),
                DepthwiseSeparableConv2d(out_ch, out_ch, 3, stride, dilation),
                nn.BatchNorm2d(out_ch)                
            ]
   
        if not use_1st_relu: 
            block = block[1:]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        output = self.block(x)
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x

        x = output + skip
        return x
    
    
class Xception(nn.Module):
    def __init__(self, in_channels):
        super(Xception, self).__init__()        
        self.entry_block_1 = nn.Sequential(
            conv_block(in_channels, 32, 3, 2, 1),
            conv_block(32, 64, 3, 1, 1, relu=False),
            XceptionBlock(64, 128, 2, 1, use_1st_relu=False)
        )
        self.relu = nn.ReLU()
        self.entry_block_2 = nn.Sequential(
            XceptionBlock(128, 256, 2, 1),
            XceptionBlock(256, 728, 2, 1)
        )
        
        middle_block = [XceptionBlock(728, 728, 1, 1) for _ in range(16)]
        self.middle_block = nn.Sequential(*middle_block)
        
        self.exit_block = nn.Sequential(
            XceptionBlock(728, 1024, 1, 1, exit_flow=True),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1024, 1536, 3, 1, 2),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1536, 1536, 3, 1, 2),
            nn.BatchNorm2d(1536),
            nn.ReLU(),
            DepthwiseSeparableConv2d(1536, 2048, 3, 1, 2),
            nn.BatchNorm2d(2048),
            nn.ReLU(),
        )
            
    def forward(self, x):
        out = self.entry_block_1(x)
        features = out
        out = self.entry_block_2(out)
        out = self.middle_block(out)
        out = self.exit_block(out)
        return out, features
    
    
class AtrousSpatialPyramidPooling(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.block1 = conv_block(in_ch, 256, 1, 1, 0, 1)
        self.block2 = conv_block(in_ch, 256, 3, 1, 6, 6)
        self.block3 = conv_block(in_ch, 256, 3, 1, 12, 12)
        self.block4 = conv_block(in_ch, 256, 3, 1, 18, 18)
        self.block5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_ch, 256, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv = conv_block(256*5, 256, 1, 1, 0)
         
    def forward(self, x):
        upsample_size = (x.shape[-1], x.shape[-2])
        
        out1 = self.block1(x)
        out2 = self.block2(x)
        out3 = self.block3(x)
        out4 = self.block4(x)
        out5 = self.block5(x)
        out5 = F.interpolate(
            out5, size=upsample_size, mode="bilinear", align_corners=True
        )
        
        out = torch.cat([out1, out2, out3, out4, out5], dim=1)
        out = self.conv(out)
        return out


class Decoder(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.block1 = conv_block(128, 48, 1, 1, 0)
        self.block2 = nn.Sequential(
            conv_block(48+256, 256, 3, 1, 1),
            conv_block(256, 256, 3, 1, 1),
            nn.Conv2d(256, num_classes, 1)
        )
    
    def forward(self, x, features):
        features = self.block1(features)
        feature_size = (features.shape[-1], features.shape[-2])
        
        out = F.interpolate(x, size=feature_size, mode="bilinear", align_corners=True)
        out = torch.cat((features, out), dim=1)
        out = self.block2(out)
        return out

    
class DeepLabV3p(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()
        self.backbone = Xception(in_channels)
        self.aspp = AtrousSpatialPyramidPooling(2048)
        self.decoder = Decoder(num_classes)
        
    def forward(self, x):
        upsample_size = (x.shape[-1], x.shape[-2])

        backbone_out, features = self.backbone(x)
        aspp_out = self.aspp(backbone_out)
        
        out = self.decoder(aspp_out, features)
        out = F.interpolate(
            out, size=upsample_size, mode="bilinear", align_corners=True
        )
        return out
```











<br>

<br>

# 참고 자료

* 
