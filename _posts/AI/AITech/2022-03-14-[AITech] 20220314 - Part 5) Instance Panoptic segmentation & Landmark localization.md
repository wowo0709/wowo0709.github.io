---
layout: single
title: "[AITech][CV] 20220314 - Part 5) Instance/Panoptic segmentation & Landmark localization"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 POSTECH '오태현' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# Instance/Panoptic segmentation & Landmark localization

이번 포스팅에서는 앞서 살펴본 CV task들 외에 Instance/Panoptic segmentation과 Landmark localization에 대해 알아보겠습니다. 

## Instance segmentation

아래 그림에서 보다시피, Instance segmentation은 semantic segmentation + **distinguishing instances** task가 합쳐진 복합적인 분야라고 할 수 있습니다. 

![image-20220314203720576](https://user-images.githubusercontent.com/70505378/158292489-50de27ec-9449-4997-ae5b-e2d8f05946a5.png)

이러한 instance segmentation을 수행하는 대표적인 모델들에 대해 알아보도록 하겠습니다. 

**Mask R-CNN**

`Mask R-CNN`은 이름에서 보는 것처럼 Faster R-CNN과 상당히 유사한 구조를 가지고 있습니다. 다만 다음의 부분들이 개선되었다고 할 수 있습니다. 

* 기존의 Faster R-CNN의 RoI Pooling layer는 정수 단위의 픽셀에서의 분류만 했지만, Mask R-CNN의 **RoI Align layer는 소수점 단위의 픽셀까지 지원**합니다. 따라서 더 정교한 분류가 가능합니다. 
* Faster R-CNN에 있었던 classification과 box regression branch에 더해 **Mask branch**가 더해집니다. 아래 그림은 최종 80개 클래스로 분류하는 모델의 구조이고, mask branch의 마지막 결과물을 보면 80개의 feature map이 출력되는 것을 볼 수 있습니다. Classification branch의 결과에 따라 mask branch의 출력 결과를 가져와 instance segmentation을 수행합니다. 

![image-20220314204254000](https://user-images.githubusercontent.com/70505378/158292493-7e21462a-0a81-4967-9045-733c770c26ba.png)

이를 보면 성능 향상을 위한 RoI Align layer를 제외하면, Faster R-CNN과 달라진 부분은 mask branch 하나라는 것을 알 수 있습니다. 이는 다른 적절한 branch를 사용할 경우, 다양한 task로의 적용이 가능하다는 말인데요, 실제로 같은 논문에서 **key point branch**를 추가하여 skeleton extraction을 수행하는 모델로의 전환도 쉽게 할 수 있음을 보여주었습니다. 



**YOLACT**

위 Mask R-CNN은 two-stage segmenter인데요, 당연히 single-stage segmenter 또한 존재합니다. 바로 `YOLACT`(You Only Look At CoefficienTs)입니다. YOLACT는 실시간 segmentation이 가능합니다. 

YOLACT의 핵심 구조는 다음과 같습니다. 

* Feature Pyramid: 빠르면서도 높은 성능을 보이는 pyramid 구조의 feature extraction network 사용
* Protonet: 매 detection box마다 N개의 segmentation mask를 생성했던 Mask R-CNN과 달리, 전체 이미지에 대해 여러 개의 mask를 생성합니다. 이 때의 출력 결과는 완전한 mask는 아니고, 추후에 완벽한 mask를 만들어내기 위한 여러 사전 재료들 같은 느낌입니다. 
* Prediction head: Protonet에서 생성한 prototypes를 적절히 합성(weighted sum)하기 위한 mask coefficient를 생성합니다. 

![image-20220314210018118](https://user-images.githubusercontent.com/70505378/158292496-964f8611-58db-4052-91cc-95016fbe20f1.png)

최종적으로 Crop과 Threshold를 지나 segmentation이 완료되게 됩니다. 

**YolactEdge**

YOLACT는 실시간으로 사용이 가능한 정도의 빠른 추론 속도를 보여주기는 하지만, edge device와 같은 곳에 올릴 수 있을 정도로 소형화되어 있지는 않습니다. 

`YolactEdge`는 이전 프레임의 key frame에 해당하는 feature를 다음 프레임 추론 시에 재활용하면서, 성능을 비슷하게 유지하면서도 훨씬 낮은 메모리 사용률을 달성하게 되었습니다. 

![image-20220314210809727](https://user-images.githubusercontent.com/70505378/158292499-b28ca738-ba13-4e5c-bf7e-2f1a3c157517.png)

다만, 아직까지 video에 대한 segmentation은 완벽한 수준은 아니라고 합니다. 



<br>

## Panoptic segmentation

앞서 살펴본 Instance segmentation은 배경에 대한 masking을 수행하지는 않습니다. 오히려 semantic segmentation이 배경에 대한 masking을 수행하죠. 

`Panoptic segmentation`은 이 두 기술을 융합하여 instance segmentation + background masking을 모두 수행하는 task입니다. 

![image-20220314211150350](https://user-images.githubusercontent.com/70505378/158292501-c61e0f95-2078-44db-a4b8-23f0db7bb076.png)

**UPSNet**

`UPSNet`은 기존의 Mask R-CNN 구조에 **semantic head**와 **panoptic head**를 더하여 만들어진 모델입니다. 

![image-20220314211730576](https://user-images.githubusercontent.com/70505378/158292504-9bfc2256-ae5a-4ec5-88f8-ac5a21587eae.png)

* Instance head: 각 인스턴스에 대한 masking 생성(Instance segmentation의 결과와 동일)
* Semantic head: semantic masking 생성(Sementic segmentation의 결과와 동일)
* Panoptic head: 두 결과를 융합하여 최종 panoptic segmentation output 생성







**VPSNet**

`VPSNet`은 video에서 panoptic segmentation을 수행하기 위한 네트워크입니다. 

영상에서의 segmentation은 object tracking이 중요한 이슈인데요, 즉 같은 instance는 이전 프레임과 이후 프레임에서 같은 색으로 구분되어야 한다는 것입니다. 

* Fuse: VPSNet은 이전 프레임에서의 feature map 정보를 Align(feature mapping)을 통해 픽셀이 어디에서 어디로 갔을 것인지 예측하고, 이를 현재 프레임의 실제 feature map과 합쳐서 사용합니다. 
  * 이를 통해 시간의 변화에 더 자연스러운 segmentation 결과를 생성할 수 있습니다. 
* Track head: Track head에서는 이전 프레임의 feature map 정보와, 앞서 생성한 현재 프레임의 feature map 정보를 비교해 instance가 어디에서 어디로 이동했는 지 tracking하고 동일한 instance끼리 match합니다. 
* UPSNet: 이후의 모델 구조는 UPSNet과 동일합니다. 

![image-20220314212620404](https://user-images.githubusercontent.com/70505378/158292508-8f49126c-7435-4cdf-bed6-24fdfc137122.png)





<br>

## Landmard localization

`Landmark localization(Keypoint estimation)` 또한 픽셀 분류를 이용하는 또 하나의 task입니다. 미리 정의해놓은 landmark(key point)를 찾아내고 추적하는 것을 말합니다. 

![image-20220315101833388](https://user-images.githubusercontent.com/70505378/158292512-4e23d167-e3f0-4858-85e2-e5d6edbf7abd.png)

Landmark localization을 수행하는 모델에는 무엇이 있는 지 살펴봅시다. 

**Coordinate regression VS Heapmap classification**

먼저 모델을 보기 전에, landmard localization을 수행하는 방법에는 무엇이 있는지부터 봅시다. 

* Coordinate regression: 각 landmark에 대해 (x,y)를 regression하여 2*N 개의 최종 출력을 생성
  * 해당 방법은 다소 부정확하고 일반화에 문제가 있음
* Heapmap classification: 하나의 채널이 각 landmark를 예측하는 heapmap이 되어 최종적으로 N개의 feature map을 생성
  * 더 좋은 성능을 보이지만 높은 계산 비용이 듦

![image-20220315102418053](https://user-images.githubusercontent.com/70505378/158292515-673a08ce-d01d-4a8d-988d-587a447dde5d.png)

(x, y) landmark label이 주어졌을 때 heapmap label은 아래와 같은 수식을 통해 구할 수 있습니다. 

![image-20220315102607745](https://user-images.githubusercontent.com/70505378/158292516-7daa34aa-ed6f-4b4e-9f2f-11d25af41839.png) 

**Hourglass network**

`Hourglass network`는 2016년 발표된 landmark localization에 최적화된 구조를 보이는 네트워크입니다. 

Hourglass network는 아래와 같이 **'Convolution-Upsampling'을 반복하는 여러 개의 블록을 이어붙인 구조**로 되어 있고, 이를 stacked hourglass modules라고 합니다. 이를 통해 한 번에 추론을 완료하지 않고, 점점 더 정제하고 정교화 해나가는 과정이 가능하게 됩니다. 

![image-20220315103027033](https://user-images.githubusercontent.com/70505378/158292456-bf33826a-e68a-49a8-b4c7-d2a3916e1187.png)

하나의 hourglass module은 아래와 같은 구조를 가지고 있습니다. U-Net의 형태와 흡사한데, 다음의 두 가지가 다릅니다. 

* U-Net은 축소 단계의 map을 그대로 확장 단계에 전달하는 데 비해, hourglass module은 한 번의 convolution layer를 거쳐 전달합니다. 
* U-Net은 전달된 map을 concat하는 데 비해, houglass module은 sum합니다. 

![image-20220315103438095](https://user-images.githubusercontent.com/70505378/158292460-69c7eff7-230f-429d-856d-ea714554a89d.png)

**DensePose**

신체 일부 keypoint를 찾는 hourglass network와 달리, `DensePose`는 신체의 모든 부분을 keypoint로 찾아냅니다. 이는 곧 3D Map 형태로 keypoint를 만들어내는 것이고, 이러한 형태를 UVMap 형태라고 합니다. 

* UVMap은 3D 형태를 2D 형태로 펼쳐놓은 map을 말합니다. Motion이 변해도 UVMap과 3D 위치 간 관계는 변하지 않아서, UV map을 얻는다는 것은 곧 3D mesh 형태를 얻는 것과 동일하다고 할 수 있습니다. 

![image-20220315103839344](https://user-images.githubusercontent.com/70505378/158292463-28e7dd6f-40f9-4db4-bd73-8aa5984aaeed.png)

(UV Map)

![image-20220315104319681](https://user-images.githubusercontent.com/70505378/158292465-63f49ddc-ff01-439d-9cd9-61faac511ab5.png)

DensePose의 구조는 Mask R-CNN과 유사하게, Faster R-CNN에 **3D surface regression branch**를 추가한 구조입니다. 아래 그림에서 Patch는 신체 각 부위를 segmentation한 output에 해당합니다. 

![image-20220315104626798](https://user-images.githubusercontent.com/70505378/158292468-77c4fb81-a5e0-4ae5-ac87-2704bd84123c.png)

**RetinaFace**

`RetinaFace`는 FPN 구조에 앞서 얘기한 모든 task들을 각각 수행하는 branch들을 모두 추가하여 만들어진 모델입니다. 이를 FPN에 Multi-task branches가 더해졌다고 표현합니다. 

![image-20220315105029386](https://user-images.githubusercontent.com/70505378/158292470-0b3952e8-27cc-4426-a288-a9482afb565a.png)

모델이 **Multi-task**를 수행할 경우 아래와 같은 장점이 있습니다. 

* **적은 데이터 양으로도 좋은 성능을 보일 가능성이 높아집니다.** 하나의 데이터에 대해 N개의 task를 수행한다면 N개의 gradient를 얻기 때문에, 더욱 빠르고 강건한 학습이 가능해집니다. 

이렇게, FPN에 **Target-task branches**를 추가하여 모델 구조를 만드는 것이 현재 CV에서의 모델 디자인 패턴 중 하나의 큰 부분입니다. 





<br>

## Detecting object as keypoints

앞선 object detection 포스팅에서 최근에는 bounding box로 detection을 하는 방법 외에 여러 연구들이 활발하게 진행되고 있다는 얘기를 했었습니다. 여기서는 그러한 모델들을 몇 개 살펴보겠습니다. 

**CornerNet**

`CornerNet` 모델 구조는 매우 직관적입니다. 아래 그림을 보면 하나의 브랜치에서는 Top-left corners를 예측하고, 다른 하나의 브랜치에서는 Bottom-right corners를 예측하여 두 결과를 취합하여 최종 결과를 냅니다. 

![image-20220315110414149](https://user-images.githubusercontent.com/70505378/158292473-f9479ead-e3a3-4dc0-8145-21b9325247aa.png)



CornerNet은 아주 빠른 추론 속도를 보이기는 하지만, 정확도가 떨어지는 모습을 보입니다. 

**CenterNet(1)**

초기의 CenterNet(1)은 object의 중심점이 중요하다는 데 착안하여, Top-left와 Bottom-right 좌표에 더해 Center 좌표를 추가적으로 예측했습니다. 

![image-20220315110548439](https://user-images.githubusercontent.com/70505378/158292477-474c9f25-2c93-47c2-868a-89bf6f09ca68.png)

하지만 이는 꼭 필요치 않은 좌표를 추가적으로 구한다는 면에서 비효율적입니다. 

**CenterNet(2)**

그래서 CenterNet(2)는 Center 좌표에 더하여 꼭 필요한 width과 height로 위치를 특정했습니다. 

![image-20220315110823409](https://user-images.githubusercontent.com/70505378/158292482-adbb0430-d615-434f-bdca-b52f6e67b854.png)

2019년 발표된 CenterNet(2)는 아래와 같이 개선된 다른 FasterRCNN이나 RetinaNet, YOLOv3 보다 더 좋은 성능을 보입니다. 

![image-20220315111009275](https://user-images.githubusercontent.com/70505378/158292485-6c0fd6c1-10e6-46b7-bcff-61e65b856355.png)

<br>

마치면서, 강의의 끝에서 강사님께서 강조하신 두 가지를 적어보겠습니다. 

* 새로운 모델을 만들 때, 밑바닥부터 새로 만들기보다는 기존에 있는 모델들을 활용하는 디자인 패턴을 따르는 것이 더 쉽고, 성능을 보장해준다. 
* 데이터의 표현, 출력 표현을 바꾸는 것이 모델 성능의 큰 향상을 일으킬 수 있다. 

<br>

## 실습) Pose Estimation

이번 강의의 실습은 Hourglass network로 Pose estimation을 수행하는 모델을 구현해보는 것입니다. 

**Hourglass module 커스텀 구현**

먼저 Hourglass module 하나를 직접 구현해보면서 그 흐름을 파악 해보겠습니다. 

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
  def __init__(self, num_channels=256):
    super(ResidualBlock, self).__init__()
    # 입출력 간 (c, h, w)가 변하지 않음
    self.bn1 = nn.BatchNorm2d(num_channels)
    self.conv1 = nn.Conv2d(num_channels, num_channels//2, kernel_size=1, bias=True)

    self.bn2 = nn.BatchNorm2d(num_channels//2)
    self.conv2 = nn.Conv2d(num_channels//2, num_channels//2, kernel_size=3, stride=1,
                              padding=1, bias=True)

    self.bn3 = nn.BatchNorm2d(num_channels//2)
    self.conv3 = nn.Conv2d(num_channels//2, num_channels, kernel_size=1, bias=True)

    self.relu = nn.ReLU(inplace=True)

  def forward(self, x):
    residual = x

    out = self.bn1(x)
    out = self.relu(out)
    out = self.conv1(out)

    out = self.bn2(out)
    out = self.relu(out)
    out = self.conv2(out)

    out = self.bn3(out)
    out = self.relu(out)
    out = self.conv3(out)

    out += residual

    return out


class Hourglass(nn.Module):
  def __init__(self, block, num_channels=256):
    super(Hourglass, self).__init__()

    self.downconv_1 = block(num_channels)
    self.pool_1 = nn.MaxPool2d(kernel_size=2)
    self.downconv_2 = block(num_channels)
    self.pool_2 = nn.MaxPool2d(kernel_size=2)
    self.downconv_3 = block(num_channels)
    self.pool_3 = nn.MaxPool2d(kernel_size=2)
    self.downconv_4 = block(num_channels)
    self.pool_4 = nn.MaxPool2d(kernel_size=2)

    self.midconv_1 = block(num_channels)
    self.midconv_2 = block(num_channels)
    self.midconv_3 = block(num_channels)
    
    self.skipconv_1 = block(num_channels)
    self.skipconv_2 = block(num_channels)
    self.skipconv_3 = block(num_channels)
    self.skipconv_4 = block(num_channels)

    self.upconv_1 = block(num_channels)
    self.upconv_2 = block(num_channels)
    self.upconv_3 = block(num_channels)
    self.upconv_4 = block(num_channels)

  def forward(self, x):
    x1 = self.downconv_1(x)
    x  = self.pool_1(x1)

    '''======================================================='''
    '''======================== TO DO ========================'''
    x2 = self.downconv_2(x)
    x = self.pool_2(x2)

    x3 = self.downconv_3(x)
    x = self.pool_3(x3)

    x4 = self.downconv_4(x)
    x = self.pool_4(x4)

    x = self.midconv_1(x)
    x = self.midconv_2(x)
    x = self.midconv_3(x)

    x4 = self.skipconv_1(x4)
    x = F.upsample(x, scale_factor=2)
    x = x + x4
    x = self.upconv_1(x)

    x3 = self.skipconv_1(x3)
    x = F.upsample(x, scale_factor=2)
    x = x + x3
    x = self.upconv_1(x)

    x2 = self.skipconv_1(x2)
    x = F.upsample(x, scale_factor=2)
    x = x + x2
    x = self.upconv_1(x)

    x1 = self.skipconv_1(x1)
    x = F.upsample(x, scale_factor=2)
    x = x + x1
    x = self.upconv_1(x)
    '''======================== TO DO ========================'''
    '''======================================================='''

    return x
```

<br>

**Stacked Hourglass Network 공식 깃허브 구현**

`Stacked Hourglass Network`의 공식 깃허브 구현 코드입니다. HourglassNet의 최종 반환값은 각 stack(hourglass module)의 출력 heapmap입니다. 

```python
'''
Hourglass network inserted in the pre-activated Resnet
Use lr=0.01 for current version
(c) YANG, Wei
'''
import torch.nn as nn
import torch.nn.functional as F

# from .preresnet import BasicBlock, Bottleneck


__all__ = ['HourglassNet', 'hg']

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()

        self.bn1 = nn.BatchNorm2d(inplanes)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=True)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 2, kernel_size=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        return out


class Hourglass(nn.Module):
    def __init__(self, block, num_blocks, planes, depth):
        super(Hourglass, self).__init__()
        self.depth = depth
        self.block = block
        self.hg = self._make_hour_glass(block, num_blocks, planes, depth)

    def _make_residual(self, block, num_blocks, planes):
        layers = []
        for i in range(0, num_blocks):
            layers.append(block(planes*block.expansion, planes))
        return nn.Sequential(*layers)

    def _make_hour_glass(self, block, num_blocks, planes, depth):
        hg = []
        for i in range(depth):
            res = []
            for j in range(3):
                res.append(self._make_residual(block, num_blocks, planes))
            if i == 0:
                res.append(self._make_residual(block, num_blocks, planes))
            hg.append(nn.ModuleList(res))
        return nn.ModuleList(hg)

    def _hour_glass_forward(self, n, x):
        up1 = self.hg[n-1][0](x)
        low1 = F.max_pool2d(x, 2, stride=2)
        low1 = self.hg[n-1][1](low1)

        if n > 1:
            low2 = self._hour_glass_forward(n-1, low1)
        else:
            low2 = self.hg[n-1][3](low1)
        low3 = self.hg[n-1][2](low2)
        up2 = F.interpolate(low3, scale_factor=2)
        out = up1 + up2
        return out

    def forward(self, x):
        return self._hour_glass_forward(self.depth, x)


class HourglassNet(nn.Module):
    '''Hourglass model from Newell et al ECCV 2016'''
    def __init__(self, block, num_stacks=2, num_blocks=4, num_classes=16):
        super(HourglassNet, self).__init__()

        self.inplanes = 64
        self.num_feats = 128
        self.num_stacks = num_stacks
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=True)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_residual(block, self.inplanes, 1)
        self.layer2 = self._make_residual(block, self.inplanes, 1)
        self.layer3 = self._make_residual(block, self.num_feats, 1)
        self.maxpool = nn.MaxPool2d(2, stride=2)

        # build hourglass modules
        ch = self.num_feats*block.expansion
        hg, res, fc, score, fc_, score_ = [], [], [], [], [], []
        for i in range(num_stacks):
            hg.append(Hourglass(block, num_blocks, self.num_feats, 4))
            res.append(self._make_residual(block, self.num_feats, num_blocks))
            fc.append(self._make_fc(ch, ch))
            score.append(nn.Conv2d(ch, num_classes, kernel_size=1, bias=True))
            if i < num_stacks-1:
                fc_.append(nn.Conv2d(ch, ch, kernel_size=1, bias=True))
                score_.append(nn.Conv2d(num_classes, ch, kernel_size=1, bias=True))
        self.hg = nn.ModuleList(hg)
        self.res = nn.ModuleList(res)
        self.fc = nn.ModuleList(fc)
        self.score = nn.ModuleList(score)
        self.fc_ = nn.ModuleList(fc_)
        self.score_ = nn.ModuleList(score_)

    def _make_residual(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=True),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_fc(self, inplanes, outplanes):
        bn = nn.BatchNorm2d(inplanes)
        conv = nn.Conv2d(inplanes, outplanes, kernel_size=1, bias=True)
        return nn.Sequential(
                conv,
                bn,
                self.relu,
            )

    def forward(self, x):
        out = [] # num_stacks만큼의 출력값을 담아서 out으로 반환
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)

        for i in range(self.num_stacks):
            y = self.hg[i](x)
            y = self.res[i](y)
            y = self.fc[i](y)
            score = self.score[i](y)
            out.append(score)
            if i < self.num_stacks-1:
                fc_ = self.fc_[i](y)
                score_ = self.score_[i](score)
                x = x + fc_ + score_

        return out
```

<br>

**(x, y) keypoint를 heatmap 형태로 변환**

(x, y) keypoint 형태로 되어있는 label을 모델을 학습시키기 위해 heapmap 형태로 변환해주는 과정이 필요합니다. 

```python
class BodyLandmarkDataset(Dataset):
  def __init__(self, data_root, is_Train=True, input_size=224, transform=None):
    super(BodyLandmarkDataset, self).__init__()

    # ...

  def __getitem__(self, index):
    # ...

  def __len__(self):
    # ...
  
  def _load_img_list(self, data_root, is_Train):
    # ...

  def _load_img_ID(self, path):
    # ...

  def _get_heatmaps_from_json(self, anno_path, org_size):
    # Parse point annotation
    with open(anno_path, 'r') as json_file:
      pts = json.load(json_file)
    pts = np.array([(pt['pt_x'], pt['pt_y']) for pt in pts['DataList'][0]['coordinates']])

    pts[:,0] = pts[:,0] / org_size[1] * self.hm_size
    pts[:,1] = pts[:,1] / org_size[0] * self.hm_size

    heatmap = np.zeros((self.n_landmarks, self.hm_size, self.hm_size), dtype=np.float32)
    for i, pt in enumerate(pts):
      heatmap[i] = self._draw_labelmap(heatmap[i], org_size, pt, self.sigma)
    
    return heatmap

  def _draw_labelmap(self, heatmap, org_size, pt, sigma):
    # Draw a 2D gaussian
    # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
    H, W = heatmap.shape[:2]

    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
    br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
    if (ul[0] >= heatmap.shape[1] or ul[1] >= heatmap.shape[0] or
            br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return heatmap, 0

    # Generate gaussian
    size = 6 * sigma + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1

    '''======================================================='''
    '''======================== TO DO ========================'''
    g = np.exp(- ((x-x0) ** 2 + (y-y0) ** 2) / (2 * sigma ** 2))
    '''======================== TO DO ========================'''
    '''======================================================='''

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], heatmap.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], heatmap.shape[0]) - ul[1]
    # Image range
    heatmap_x = max(0, ul[0]), min(br[0], heatmap.shape[1])
    heatmap_y = max(0, ul[1]), min(br[1], heatmap.shape[0])

    heatmap[heatmap_y[0]:heatmap_y[1], heatmap_x[0]:heatmap_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return heatmap
    
    return anno_path
```

**Visualization**

학습시킨 모델로 최종 출력을 시각화합니다. Hourglass network의 최종 출력은 heapmap 형태이기 때문에, 이를 다시 (x, y) keypoint 형태로 변환해주는 과정이 필요합니다. 

Hourglass Network의 최종 출력 `model(imgs)[-1]` 히트맵에서 각 keypoint에 대한 (x, y)를 뽑아내 시각화 해보겠습니다. 

```python
import matplotlib.pyplot as plt

n_vis = 5

# Visualize the result of validation dataset
for iter, (imgs, hm_gt) in enumerate(train_loader):
  '''============================================================'''
  '''======================== TO DO Main ========================'''
  # GPU 연산을 위해 이미지 tensor를 GPU로 보내기 (필요한 경우, 변수의 type도 수정해주세요)
  imgs = imgs.float().to(device)
  
  # 모델에 이미지 forward (gradient 계산 X)
  with torch.no_grad():
    preds = model(imgs)[-1].cpu().numpy() # 마지막 결과(네트워크의 최종 출력) 가져오기
  '''======================== TO DO Main ========================'''
  '''============================================================'''


  # for each sample in a batch
  # print(imgs.size()) # (batch_size=8,channels=3,height=320,width=320)
  # print(preds.size()) # (batch_size=8, channel(=num_classes)=22, height=80, width=80)
  imgs = imgs.cpu().numpy()
  for img, pred_hm in zip(imgs, preds):
    # Re-convert pre-processed input image to original format
    img = np.moveaxis(img, 0, -1)
    img = (img * STD) + MEAN
    img = (img*255).astype(np.uint8).copy()

    for hm in pred_hm:
      '''======================================================='''
      '''==================== TO DO Decoding ==================='''
      # 최댓값을 갖는 좌표점 추출
      y, x = np.where(hm == hm.max())
      '''==================== TO DO Decoding ==================='''
      '''======================================================='''
      # image, center(80*4=320), radius, color, thickness
      cv2.circle(img, (x[0]*4, y[0]*4), 3, (255,0,0), -1)
    
    plt.imshow(img)
    plt.show()
  

  if iter == (n_vis-1): # batch_size * n_vis 만큼 시각화 출력
    break
```

![image-20220317151815378](https://user-images.githubusercontent.com/70505378/158749606-c173c06b-6db8-4f69-add1-49ffdae722de.png)













<br>

<br>

# 참고 자료


* Instance segmentation

  * Kirillov et al., Panoptic segmentation, CVPR 2019 
  * He et al., Mask R-CNN, ICCV 2017 
  * Bolya et al., YOLACT Real-time Instance Segmentation, ICCV 2019
  *  Liu et al., YolactEdge: Real-time Instance Segmentation on the Edge (Jetson AGX Xavier: 30 FPS, RTX 2080 Ti: 
    170 FPS), arXiv 2020
* Panoptic segmentation

  * Xiong et al., UPSNet: A Unified Panoptic Segmentation Network, CVPR 2019 
  * Kim et al., Video Panoptic Segmentation, CVPR 2020  
* Landmark localization

  * Cao et al., OpenPose: Realtime Multi-Person 2D Pose Estimation using Part Affinity Fields, IEEE TPAMI 2019 
  * Jin et al., Pixel-in-Pixel Net: Towards Efficient Facial Landmark Detection in the Wild, arXiv 2020 
  * Wang et al., Adaptive Wing Loss for Robust Face Alignment via Heatmap Regression, ICCV 2019 
  * Newell et al., Stacked Hourglass Networks for Human Pose Estimation, ECCV 2016 
  * Guler et al., DensePose: Dense Human Pose Estimation in the Wild, CVPR 2018  
* Detecting objects as keypoints

  * Law et al., CornerNet: Detecting Objects as Paired Keypoints, ECCV 2018 
  * Duan et al., CenterNet: Keypoint Triplets for Object Detection, ICCV 2019 
  * Zhou et al., Objects as Points, arXiv 2019  



<br>

