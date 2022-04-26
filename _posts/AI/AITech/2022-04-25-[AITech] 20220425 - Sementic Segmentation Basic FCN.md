---
layout: single
title: "[AITech][Semantic Segmentation] 20220425 - Semantic Segmentation Basic: FCN"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 KAIST의 '김현우' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Semantic Segmentation Basic: FCN

이번 포스팅에서는 2014년 발표된 대표적인 딥러닝을 이용한 segmentation model인 FCN을 살펴보며 semantic segmentation의 기본에 대한 이해를 해보겠습니다. 

## FCN (Fully Convolutional Networks)

![image-20220425104814115](https://user-images.githubusercontent.com/70505378/165025621-a12eafe3-c6c9-42e7-9e6a-0cb36d93466a.png)

FCN의 아이디어에 대한 설명은 앞서 segmentation 포스팅들에서 많이 했습니다. 이를 3가지로 정리하면 아래와 같습니다. 

1. VGG-16 네트워크 백본을 사용 (Backbone: feature extracting network)

   * AlexNet, VGG-16, GooLeNet으로 실험 시 VGG의 성능이 가장 좋았다고 함

   ![image-20220425111220527](https://user-images.githubusercontent.com/70505378/165025639-0147e92a-29b2-4e47-8c7a-1ea819d5e6ac.png)

2. VGG 네트워크의 FC layer를 Convolution layer로 대체

   * 각 픽셀 예측의 위치 정보를 보존
   * Input의 height, width 값과 상관없이 모델 동작 가능 (다양한 크기의 입력 이미지 사용 가능)

   ![image-20220425111203141](https://user-images.githubusercontent.com/70505378/165025637-7296442c-afe0-45f8-8c13-ef5bccafa3f5.png)

3. Transposed Convolution을 이용해 원본 이미지와 동일한 크기의 출력맵에서 pixel-wise prediction을 수행

   * Downsampling(Convolution)으로 작아진 크기의 feature map을 다시 Upsampling(Transposed convolution)을 통해 원본 입력 이미지와 동일한 크기로 변환

   * 예시: 3x3 transpose convolution with stride=2

     ![image-20220425110057295](https://user-images.githubusercontent.com/70505378/165025626-75ca4662-e9b9-44c9-9246-de0ee5cb5094.png)

   * 예시: 3x3 transpose convolution with stride=2, padding=2

     ![image-20220425110137151](https://user-images.githubusercontent.com/70505378/165025630-f6d574fc-eb59-4920-bfee-183e3b01f51f.png)

**Transposed Convolution**

`Transposed Convolution`이 왜 그렇게 불리는지에 대해 간단히 알아보고 넘어가겠습니다. 

4x4의 input과 3x3 kernel의 convolution 연산은 아래와 같이 나타낼 수 있습니다. 

![image-20220425110811800](https://user-images.githubusercontent.com/70505378/165025634-11a1737d-1510-4de6-8c25-74decb228eb7.png)

이때 이에 대한 전치 행렬곱 연산은 아래와 같이 나타낼 수 있습니다. 

![image-20220425110903833](https://user-images.githubusercontent.com/70505378/165025635-ae40a37d-b32e-44b0-b514-101a1d38d410.png)

여기서 알 수 있듯, transposed convolution 연산은 convolution 전 input을 그대로 복원하는 연산이 아닙니다. 다만 input의 크기를 복원해주는 연산입니다. 

따라서 transposed convolution을 deconvolution(convolution의 역연산)이라고 부르는 것은 엄밀히 말하면 틀렸습니다. 

정리하면 아래와 같습니다. 

* Convolution과 마찬가지로 학습이 가능한 파라미터를 통해 이미지의 크기를 다시 복원하는 convolution 연산
* 엄밀한 명칭은 Transposed convolution이라고 부르는게 정확하지만 많은 경우에 deconvolution이라는 용어와 혼재되어 사용되니 참고

<br>

이제부터는 FCN이 동작하는 과정에 대해 알아보겠습니다. 

 가장 기본 모델인 `FCN-32s` 모델은 아래 구조를 가지고 있습니다. 

![image-20220425112801708](https://user-images.githubusercontent.com/70505378/165025641-975f4314-55df-4b92-a967-6e267d4e295c.png)

입력 이미지는 5번의 conv block을 거치며 width와 height가 x(1/32) 됩니다. 

이후 세 번의 1x1 conv 연산을 통해 feature map의 각 pixel 별로 각 클래스에 해당할 score 값을 계산합니다. 이때의 feature map의 channel 수는 num_classes와 동일합니다. 

Score를 계산하고 나면, Deconv 연산을 통해 크기를 x(32) 해서 입력 이미지와 동일한 크기로 만들어줍니다. 

그런데 여기서 발생하는 문제는  Score map을 바로 32배 키워버리기 때문에 세밀한 예측이 어렵다는 것입니다. 

<br>

이런 문제에 착안하여 score map을 중간 단계의 feature map과 합쳐서 정보를 보강하고 더 세밀한 예측이 가능하도록 `FCN-16s`, `FCN-8s`가 등장했습니다. 

FCN-16s 의 구조는 아래와 같습니다. 

![image-20220425113308377](https://user-images.githubusercontent.com/70505378/165025644-d33e4fc3-483f-4ca2-9a69-4590363bacea.png)

Score map은 먼저 x(2) 크기만 키워져서 입력 이미지의 1/16 크기의 score map이 됩니다. 

이 score map에 4번의 conv block을 거쳐 계산된 x(1/16) 크기의 feature map을 element-wise로 더해줍니다. 이때 channel 수가 맞지 않기 때문에 feature map은 먼저 1x1 conv 연산을 거쳐 num_classes 만큼의 channel 수를 가지도록 합니다. 

두 map을 더한 다음에는 x(16) 해서 입력 이미지와 동일한 크기로 만들어줍니다. 

그리고 이와 동일하게, FCN-8s는 3번의 conv block을 거쳐 크기가 x(1/8) 인 feature map까지 사용하는 구조입니다. 

![image-20220425113728083](https://user-images.githubusercontent.com/70505378/165025645-0f335a0b-b843-44f7-933f-4a5f16e8bebd.png)

<br>

아래 결과를 보면 FCN-8s가 가장 좋은 성능을 보이는 것을 알 수 있습니다. 

![image-20220425113821849](https://user-images.githubusercontent.com/70505378/165025647-9a22dfb2-090a-4136-8fdd-8b8c72a25041.png)

![image-20220425113849157](https://user-images.githubusercontent.com/70505378/165025649-46218e28-078a-48de-8ad8-7c6dbe8d510b.png)



<br>

**Further Reading**

사실 FCN 논문 본문에서는 VGG-16의 구조를 그대로 가져왔기 때문에, FC6 block에서 1x1 convolution이 아닌 7x7 convolution을 사용했습니다. 

그런데 이 경우, output size가 input size와 동일하지 않은 문제가 생기기 때문에 이를 아래 과정으로 보완합니다. 

* 첫번째 conv block에서 input image에 zero padding 100 적용
* Padding에 의해 output size는 intput size보다 커지게 되는데, 이를 input size에 맞춰주기 위해 crop(또는 resize)

![image-20220425115214106](https://user-images.githubusercontent.com/70505378/165025652-e27eb382-00fd-4485-909d-d0cd026c7a22.png)

논문을 읽을 때 해당 사실에 주목하면 더 수월하게 읽을 수 있을 것이라 생각합니다. 







<br>

## Summary

내용을 정리합니다. 

* FCN은 end-to-end segmentation model의 기초가 되는 모델
  * pretrained VGG-16 backbone 사용
  * FC layer 대신 1x1 convolution layer 사용
  * pixel wise prediction을 위해 transposed convolution으로 upsampling
* 가장 기본적인 FCN-32s와 중간 단계의 feature map들을 활용하여 정보를 보강하는 FCN-16s, FCN-8s 존재
* 실제 논문에서는 7x7 convolution을 사용했음







<br>

## 실습) FCN-32s, FCN-16s, FCN-8s

실습에서는 FCN을 구현해봅니다. 

**FCN-32s**

Feature map의 width&height, channel 수에 유의하며 코드를 작성합니다. 

```python
import torch
import torch.nn as nn
class FCN32s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN32s, self).__init__()
        self.relu    = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(2, stride=2)
        
        # TODO
        # o = (i + 2p - k) // s + 1
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            self.relu,
            nn.Conv2d(64, 64, 3, 1, 1),
            self.relu,
            self.maxpool
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1),
            self.relu,
            nn.Conv2d(128, 128, 3, 1, 1),
            self.relu,
            self.maxpool
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, 1, 1),
            self.relu,
            nn.Conv2d(256, 256, 3, 1, 1),
            self.relu,
            nn.Conv2d(256, 256, 3, 1, 1),
            self.relu,
            self.maxpool
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, 1, 1),
            self.relu,
            nn.Conv2d(512, 512, 3, 1, 1),
            self.relu,
            nn.Conv2d(512, 512, 3, 1, 1),
            self.relu,
            self.maxpool
        )
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, 1, 1),
            self.relu,
            nn.Conv2d(512, 512, 3, 1, 1),
            self.relu,
            nn.Conv2d(512, 512, 3, 1, 1),
            self.relu,
            self.maxpool
        )
        self.fc6 = nn.Sequential(
            nn.Conv2d(512, 4096, 1, 1, 0),
            nn.Dropout2d()
        )
        self.fc7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1, 1, 0),
            nn.Dropout2d()
        )
        self.score = nn.Conv2d(4096, num_classes, 1, 1, 0)
        self.up_score = nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16)
            
        self.fcn = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            self.fc6,
            self.fc7,
            self.score,
            self.up_score
        )
        
        # or
        self.vgg = torchvision.models.vgg16(pretrained=True)
        self.vgg.avgpool = nn.Sequential( # fc6
            nn.Conv2d(512, 4096, 1, 1, 0),
            nn.Dropout2d()
        )
        self.vgg = nn.Sequential(*list(self.vgg.children())[:-1]) # drop classifier
        self.fcn = nn.Sequential(
            self.vgg, 
            nn.Conv2d(4096, 4096, 1, 1, 0),
            nn.Dropout2d(),
            nn.Conv2d(4096, num_classes, 1, 1, 0),
            nn.ConvTranspose2d(num_classes, num_classes, kernel_size=64, stride=32, padding=16) # x32
        )

    def forward(self, x):
        # TODO
        output = self.fcn(x)
        
        return output
```

<br>

**FCN-16s**

FCN-16s에서는 중간 단계의 feature map과 합하는 과정이 필요합니다. 

FCN-32s와 동일한 코드는 생략하겠습니다. 

```python
import torch
import torch.nn as nn
class FCN16s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN16s, self).__init__()
        # TODO
        # ...
        self.score = nn.Conv2d(4096, num_classes, 1, 1, 0)
        self.up_score_conv5 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1) # x2
        self.conv1x1_conv4 = nn.Conv2d(512, num_classes, 1, 1, 0)
        self.up_score = nn.ConvTranspose2d(num_classes, 
                                            num_classes, 
                                            kernel_size=32,
                                            stride=16,
                                            padding=8) # x16

    def forward(self, x):
        # TODO
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        
        x5 = self.fc6(x5)
        x5 = self.fc7(x5)
        x5 = self.score(x5)
        x5 = self.up_score_conv5(x5)
        
        x4 = self.conv1x1_conv4(x4)
        
        output = self.up_score(x4+x5)
        
        return output
```



<br>

**FCN-8s**

FCN-8s도 마찬가지로 FCN-32s와 동일한 부분은 생략합니다. 

```python
import torch
import torch.nn as nn
class FCN8s(nn.Module):
    def __init__(self, num_classes=21):
        super(FCN8s, self).__init__()
        # TODO
        # ...
        self.score = nn.Conv2d(4096, num_classes, 1, 1, 0)
        self.up_score_conv5 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1) # x2
        self.conv1x1_conv4 = nn.Conv2d(512, num_classes, 1, 1, 0)
        self.up_score_conv4_5 = nn.ConvTranspose2d(num_classes, 
                                            num_classes, 
                                            kernel_size=4,
                                            stride=2,
                                            padding=1) # x2
        self.conv1x1_conv3 = nn.Conv2d(256, num_classes, 1, 1, 0)
        self.up_score = nn.ConvTranspose2d(num_classes, 
                                           num_classes,
                                           kernel_size=16,
                                           stride=8,
                                           padding=4) # x8

    def forward(self, x):
        # TODO
        x1 = self.conv1(x)
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        
        x5 = self.fc6(x5)
        x5 = self.fc7(x5)
        x5 = self.score(x5)
        x5 = self.up_score_conv5(x5)
        
        x4 = self.conv1x1_conv4(x4)
        x4_5 = self.up_score_conv4_5(x4+x5)
        
        x3 = self.conv1x1_conv3(x3)
        
        output = self.up_score(x3+x4_5)
        
        return output
```

<br>

**Tip**

작성하면서 가장 신경 써야 했던 부분은 feature map의 size와 channel이었습니다. 

_Convolution_

![image-20220425142828639](https://user-images.githubusercontent.com/70505378/165026047-4b68818b-ae75-48a6-b6ae-4a3174638ea8.png)

기본적으로 size를 유지하는 연산을 할 때 3x3 conv와 1x1 conv는 다음과 같이 작성합니다. 

```python
# 3x3 conv
nn.Conv2d(3, 64, 3, 1, 1),
# 1x1 conv
nn.Conv2d(512, 4096, 1, 1, 0)
```

_Transposed convolution_

![image-20220425142838912](https://user-images.githubusercontent.com/70505378/165026044-44ac4895-5696-4cf9-b3db-b7f932071d29.png)

Convolution 식에서 input과 output을 바꿔주고 output에 대해 식을 정리한 것과 동일합니다. 

Transposed convolution에서 resolution을 k만큼 키우고 싶을 때 stride, kernel_size, padding은 아래와 같이 정의됩니다. 

* stride = k
* kernel_size = 2s
* padding = s/2

예를 들어 2배 만큼 키울 때는 아래와 같습니다. 

```python
resolution
self.up_score_conv5 = nn.ConvTranspose2d(num_classes,
                                           num_classes,
                                           kernel_size=4,
                                           stride=2,
                                           padding=1) # x2
```

































<br>

<br>

# 참고 자료

* 
