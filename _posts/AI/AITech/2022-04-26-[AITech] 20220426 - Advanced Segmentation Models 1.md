---
layout: single
title: "[AITech][Semantic Segmentation] 20220426 - Advanced Segmentation Models 1"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['DeconvNet', 'SegNet', 'FC DenseNet', 'UNet', 'DeepLab v1', 'DilatedNet']
---



<br>

_**본 포스팅은 KAIST의 '김현우' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Advanced Segmentation Models 1

이번 포스팅에서는 FCN의 한계점을 보며 이를 극복한 모델들에 대해 보도록 하겠습니다. 

크게 Decoder를 개선한 모델, Skip connection을 적용한 모델, Receptive field를 확장한 모델로 나뉩니다. 

## FCN의 한계점

FCN이 가지는 한계점에 대해 알아보겠습니다. 

**1. 객체의 크기가 크거나 작은 경우 예측을 잘 하지 못 하는 문제**

* 큰 object의 경우 지역적인 정보만으로 예측
  * kernel의 크기에서 오는 receptive field의 한계
  * 큰 object를 하나의 object로 보지 못하고 다르게 labeling
* 작은 object의 경우 무시됨
  * pooling 과정에서 세부적인 정보가 사라짐

![image-20220425150730946](https://user-images.githubusercontent.com/70505378/165226464-1c02455f-ca0f-488e-81d0-310c99611df8.png)

**2. Object의 디테일한 모습이 사라지는 문제**

* Deconvolution의 절차가 단순해 경계를 학습하기 어려움

![image-20220425150937393](https://user-images.githubusercontent.com/70505378/165226465-0390f235-04be-43cb-95f1-2b1d5729f9be.png)

위 그림에서 (c)는 바로 뒤이어 소개할 DeconvNet 모델입니다. 









<br>

<br>

## Decoder를 개선한 models

### DeconvNet

![image-20220425152042332](https://user-images.githubusercontent.com/70505378/165226467-f74c1c6e-5ffd-4c1f-949c-96f0067840e1.png)

2015년 발표된 `DeconvNet`은 Decoder를 Encoder와 대칭으로 설계한 구조를 가집니다. 

Convolution Network로는 VGG-16을 사용합니다. 

* 13개 층으로 구성
* Conv block - Convolution + BatchNorm + ReLU
* 7x7 conv 및 1x1 conv 활용

Deconvolution Network에서는 크게 Unpooling과 Deconvolution block을 사용합니다. 

* Deconv block - Transposed Convolution + BatchNorm + ReLU
* Feature map의 size 변화는 unpooling에서 일어나고 deconv에서는 유지됩니다. 

<br>

DeconvNet에서 특징적인 점은 **Unpooling**을 사용한다는 것입니다. FCN에서는 Deconvolution 만을 사용하였는데, 이는 decoder에서 feature map의 크기 복원 시 **구조적인 정보는 복원하지 못한다**는 문제가 있습니다. 

DeconvNet에서는 Unpooling과 Deconvolution을 동시에 사용하여, unpooling으로 구조적인 정보를 복원(디테일한 경계를 포착)하고 deconvolution으로 내용적인 정보를 복원(전반적인 모습을 포착)하도록 했습니다. 

말로만 들으면 추상적일 수 있으나, unpooling과 transposed convolution의 input-output 관계를 보면 이해를 할 수 있을 것입니다. 

**Unpooling**

![image-20220425152724717](https://user-images.githubusercontent.com/70505378/165226468-3add1406-9c89-49e1-ab25-3f0dd535f989.png)

**Transposed convolution**

![image-20220425152755230](https://user-images.githubusercontent.com/70505378/165226470-cbf38164-63aa-4952-84a7-9cb55e8f8421.png)

<br>

위 그림과 같이, unpooling의 경우 pooling을 통해 작아진 map의 크기를 복원해주는데, 그 과정에서 앞서 pooling 시 저장했던 max value의 index 값을 사용합니다. 그리고 max value 이외의 자리들은 0으로 채웁니다. 

Unpooling을 거치고 나면 상당히 sparse한 feature map을 얻게 됩니다. 이는 구조적으로 디테일한 경계를 포착하는 것으로 이해할 수 있습니다. 

따라서 여기에 transposed convolution을 적용하여 sparse한 matrix를 dense하게 만들어줍니다. 이는 전반적인 내용을 복원하는 것으로 이해할 수 있습니다. 

이렇게 unpooling과 transposed convolution을 동시에 사용함으로써 훨씬 디테일한 경계 정보를 포착할 수 있게 되었습니다. 

![image-20220425153220860](https://user-images.githubusercontent.com/70505378/165226473-8c66859c-cded-4048-a2e5-36fc47235f80.png)

<br>

이번에는 코드 레벨에서 모델 구조를 보도록 하겠습니다. 

![image-20220425152042332](C:\Users\user\AppData\Roaming\Typora\typora-user-images\image-20220425152042332.png)

하나의 conv block은 아래의 코드로 구현할 수 있습니다. 

![image-20220425153414695](https://user-images.githubusercontent.com/70505378/165226476-8b7d2af3-1774-4d0d-88ba-ecbc92fe158e.png)

마찬가지로 하나의 deconv block은 아래와 같이 구현합니다. 

![image-20220425153443316](https://user-images.githubusercontent.com/70505378/165226481-80dbedaa-978d-486b-b68e-b27f3c62de48.png)

Encoder와 Decoder는 대칭적인 구조를 띠는데, 이때 encoder에서 pooling 시 `return_indices=True`로 설정하여 max value의 index 값을 기록하고, decoder에서 unpooling 시 해당 index를 사용합니다. 

![image-20220425153941060](https://user-images.githubusercontent.com/70505378/165226482-1202c626-c82a-4c2a-af6e-7d078d983777.png)

![image-20220425153959118](https://user-images.githubusercontent.com/70505378/165226484-6da05fed-a3f2-443f-bf02-8f45f5b244d3.png)

![image-20220425154114028](https://user-images.githubusercontent.com/70505378/165226485-c7d32d39-0321-435a-85d8-4b9d4828c0ab.png)

Decoder까지 지나고 나면 1x1 conv 연산으로 score를 구합니다. 

![image-20220425154221463](https://user-images.githubusercontent.com/70505378/165226488-52f845c8-1e6e-4c43-874a-727c9d6d6012.png)

<br>

### SegNet

또다른 Encoder-Decoder 구조를 사용하는 `SegNet` 모델은 2016년 발표되었으며, 성능을 높이면서도 **속도**에 초점을 맞춘 모델입니다. Real-time segmentation을 수행할 수 있을 만큼 빠른 속도를 가졌습니다. 

SegNet의 구조는 DeconvNet과 비슷하면서도 다릅니다. 

![image-20220425154603735](https://user-images.githubusercontent.com/70505378/165226491-fd53790a-4540-412b-b4a7-4275e4af3c74.png)

SegNet은 DeconvNet에 비해 몇 가지 연산을 제거 및 대체함으로써 속도를 높였습니다. 

* Encoder-Decoder를 연결하는 1x1 conv/deconv 연산 제거
* Deconv 대신 conv 연산 사용
  * Deconv에서 size를 키우는 연산을 하지 않기 때문에 대체 가능
* 마지막 Deconv block을 사용하지 않고 score를 구할 때 3x3 conv를 사용





<br>

<br>

## Skip Connection을 적용한 models

### FC DenseNet

2017년 발표된 `FC DenseNet`은 densenet에서 사용한 **Dense block**을 사용한 모델입니다. 

DenseNet에서 사용한 Dense block과 skip connection 구조는 아래와 같습니다. 

![image-20220426112949754](https://user-images.githubusercontent.com/70505378/165226493-f9613fe6-4688-4547-8278-469f6b6477af.png)

이와 유사하게 FC DenseNet은 아래의 구조를 가집니다. 

![image-20220426113019716](https://user-images.githubusercontent.com/70505378/165226495-0c5c4a74-f674-43e5-ad68-2086f8dda98b.png)

<br>

### Unet

2015년 발표된 `Unet`은 encoder와 decoder가 대칭을 이루며 4 개의 skip connection을 통해 대칭 형태로 정보를 전달해주는 구조입니다. 

Unet은 상당히 중요한 논문이기 때문에 이후 포스팅에서 깊게 다룰 예정입니다. 

![image-20220426113203959](https://user-images.githubusercontent.com/70505378/165226497-32a9874e-f929-4868-952b-894d2bca33c5.png)











<br>

<br>

## Receptive Field를 확장시킨 models

### DeepLab v1

Receptive field는 현재 feature map이 보고있는(의미를 담고 있는) 입력 이미지 영역으로, 이것이 넓어야 더 많은 정보를 이용한 추론이 가능해집니다. 

FCN에서는 이 receptive field의 영역의 크기가 제한적이기 때문에 아래와 같이 하나의 bus를 하나의 물체로 인식하지 못 하고 여러 물체로 예측하였습니다. 

![image-20220426113627632](https://user-images.githubusercontent.com/70505378/165226498-7785baee-e18c-454b-8068-02369746bfc5.png)

그래서 이 receptive field를 넓히기 위해 여러 방법들이 고안되어 왔습니다. 

하나의 예시로 **conv와 conv 연산 사이에 pooling 연산을 추가**하여 conv 연산만 연속적으로 취했을 때에 비해 더 적은 메모리를 요구하면서도 동일한(혹은 더 큰) receptive field를 가져갈 수 있는 방법이 있습니다. 

![image-20220426114239426](https://user-images.githubusercontent.com/70505378/165226500-d89cbe9a-1270-4b46-bcb3-855c8386cadd.png)

그러나 pooling에 의해 정보 소실이 일어나고 feature map의 크기가 많이 작아지기 때문에 resolution 측면에서 low feature resolution을 가지는 문제점이 발생합니다. 

이를 극복하기 위해 **이미지의 크기는 많이 줄이지 않고, 파라미터의 수도 변함이 없는 채로 receptive field만 넓히는 방법**을 고안하던 중 나온 방법이 **dilated convolution (atrous convolution)**입니다. 

Dilated convolution은 아래와 같이 kernel 사이에 zero padding을 추가함으로써 목적을 달성할 수 있었습니다. 

![image-20220426114815742](https://user-images.githubusercontent.com/70505378/165226503-cac8c7e1-f386-402c-b9d9-5ddea4e3ee0b.png)

![image-20220426114826468](https://user-images.githubusercontent.com/70505378/165226504-7f99d228-5658-454d-9414-8139d2c2e99e.png)

<br>

`Deeplab v1(2016)`은 바로 이 dilated convolution을 사용하여 receptive field를 넓힌 모델입니다. Deeplab v1의 구조는 아래와 같이 단순화 할 수 있습니다. 

![image-20220426115405851](https://user-images.githubusercontent.com/70505378/165226506-07f50110-24cb-4481-aa74-8fbb505a7e37.png)

![image-20220426115810059](https://user-images.githubusercontent.com/70505378/165226507-7fc8478b-8c0b-4472-8347-bb54b26375db.png)

conv1 ~ conv3에서는 dilation=1로 지정하여 일반 convolution과 동일한 연산을 취합니다. 3x3 MaxPool을 취한다는 점이 특징적이며, 이때 stride=2, padding=1로 지정하여 feature map의 크기가 x(1/2)이 되도록 합니다. 

 conv4에서도 dilation=1로 지정하는데, Maxpool에서 stride=1, padding=1로 지정하여 feature map의 크기를 고정합니다. 

conv5에서는 dilation=2로 지정하여 dilated convolution 연산을 가하고, Maxpool에서는 conv4와 마찬가지로 stride=1. padding=1로 지정하여 feature map의 크기를 바꾸지 않습니다. 

FC6, FC7, Score에서의 연산은 아래와 같은 코드로 구현할 수 있습니다. 

![image-20220426120115689](https://user-images.githubusercontent.com/70505378/165226509-158ddf94-4102-474a-8276-5146d33782ce.png)

마지막으로 conv1~conv3를 통해 입력 이미지보다 x(1/8) 배 된 feature map을 원래 크기로 upsampling해줍니다. 

Upsampling에서는 'bilinear interpolation'을 사용합니다. 파이토치의 `F.interpolate` 함수를 사용하여 구현할 수 있고, 인자로는 input, size, mode, align_corners 등이 있습니다. 

![image-20220426133322753](https://user-images.githubusercontent.com/70505378/165226510-1f0a9540-f39f-4786-900c-957bcc8f9942.png)

<br>

하지만 Bilinear interpolation은 픽셀 단위의 정교한 segmentation을 하기에는 부족합니다. 이를 개선하기 위해 Deeplab v1에서는 후처리 기법으로 **Dense CRF**를 사용합니다. 

Dense CRF는 입력으로 원본 이미지와 score map을 받아서 더 정교화된 score map을 출력으로 반환합니다. 

CRF란 단순히 말하면 아래의 과정을 수행한다고 할 수 있습니다. 

* 색상이 유사한 픽셀이 가까이 위치하면 같은 범주에 속함
* 색상이 유사해도 픽셀의 거리가 멀다면 같은 범주에 속하지 않음

Dense CRF는 위 과정을 모든 픽셀 쌍에 대해 수행하는 것을 말합니다. Dense CRF는 각 클래스에 대해 개별적으로 수행되며, 논문에서는 총 10번의 iteration을 반복했습니다. 

여러 번의 Dense CRF를 반복함으로써 score map은 아래와 같이 점차 정교화됩니다. 

![image-20220426134127662](https://user-images.githubusercontent.com/70505378/165226439-f7dd2f78-6302-4a6c-811b-8a59235cae96.png)

그리고 이를 모든 클래스에 대해 수행하면 아래와 같이 Dense CRF를 수행하지 않았을 때보다 훨씬 정교화된 결과를 얻을 수 있습니다. 

![image-20220426134233780](https://user-images.githubusercontent.com/70505378/165226441-f56b9209-8433-48bb-b3cc-b2a92f3be7e8.png)

<br>

아래 그림은 Deeplab v3의 전체 pipeline을 나타낸 것입니다. 

![image-20220426134305312](https://user-images.githubusercontent.com/70505378/165226443-c9c0f212-c120-4c8c-9075-978607c7732d.png)





<br>

### DilatedNet

Deeplab v1 이후에 발표된 `DilatedNet(2016)` 모델은 동일하게 dilated convolution을 사용하지만, 좀 더 효율적으로 사용한 모델입니다. 

![image-20220426134552646](https://user-images.githubusercontent.com/70505378/165226444-b43f654e-7438-4429-a72e-cd4ce8385700.png)

Deeplab v1과 비교하자면 conv1 ~ conv3에서 3x3 maxpool 대신에 2x2 maxpool을 사용했습니다. 다만 매 conv block마다 feature map의 크기를 x(1/2) 배 한다는 점은 동일합니다. 

특징적인 점은 conv4와 conv5에서 Maxpool을 사용하지 않는다는 것입니다. 여기서는 feature map의 크기가 변동되지 않습니다. 

FC6, FC7, Score 부분에서의 차이점은 FC6에서 7x7 convolution을 dilated=4로 지정하여 수행한다는 것입니다. 마찬가지로 feature map의 크기가 변동되지 않도록 padding을 지정합니다. 

![image-20220426135024345](https://user-images.githubusercontent.com/70505378/165226446-ab275874-f582-4aad-bb7e-00f2fb09977f.png)

마지막 Upsampling에서는 Deeplab v1과 달리 Transposed Convolution을 사용하며, feature map의 크기를 x8 배하여 원본 이미지와 동일한 크기로 만들어줍니다. 

![image-20220426135156262](https://user-images.githubusercontent.com/70505378/165226447-48562434-1ffa-4637-a4f2-92a58a4347f5.png)

<br>

여기까지 살펴본 모델은 DilatedNet의 Front-End module에 해당하며, 본 논문에서는 이에 더해 **Basic Context module**이라는 것을 추가로 제안합니다. 

![image-20220426135509780](https://user-images.githubusercontent.com/70505378/165226453-c66b43f9-0519-4d24-ab8a-b14ad511ac82.png)

![image-20220426135519620](https://user-images.githubusercontent.com/70505378/165226455-30db1fa9-0f5b-436a-8854-80e44fdf9333.png)

위에서 보시는 것 같이 Basic Context module은 Score map과 Upsampling 사이에 삽입되며, 이미지 크기가 변하지 않도록 설계되었기 때문에 Deconv 부분의 코드는 변하지 않습니다. 

<br>

본 논문에서는 DilatedNet Front End module만을 사용하는 것 만으로 기존 Deeplab v1의 성능을 크게 뛰어넘었다고 주장했고, Context module을 함께 사용함으로써 더 정교한 에측이 가능했다고 합니다. 여기에 CRF 후처리 과정을 추가함으로써 성능을 더욱 높일 수 있다고 합니다. 

![image-20220426140007363](https://user-images.githubusercontent.com/70505378/165226456-907b7787-047e-488a-b838-3c9a263cc26f.png)

![image-20220426140055808](https://user-images.githubusercontent.com/70505378/165226462-e3689617-bd42-4ca6-aab6-0c83110b5dce.png)

<br>

<br>

## 실습) DeconvNet, DeepLabv1

### DeconvNet

`DeconvNet`에서는 대칭적인 encoder-decoder 구조와 unpooling이 특징적입니다. 

```python
'''
reference 
http://cvlab.postech.ac.kr/research/deconvnet/model/DeconvNet/DeconvNet_inference_deploy.prototxt
'''
import torch
import torch.nn as nn
from torchvision import models

class DeconvNet(nn.Module):
    def __init__(self, num_classes=21):
        super(DeconvNet, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        
        def CBR(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, 
                          out_channels=out_channels,
                          kernel_size=kernel_size, 
                          stride=stride, 
                          padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())
        
        def DCB(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels=in_channels, 
                                   out_channels=out_channels,
                                   kernel_size=kernel_size, 
                                   stride=stride,
                                   padding=padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU())        
        
        # conv1
        self.conv1_1 = CBR(3, 64, 3, 1, 1)
        self.conv1_2 = CBR(64, 64, 3, 1, 1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) # 1/2
        
        # conv2 
        self.conv2_1 = CBR(64, 128, 3, 1, 1)
        self.conv2_2 = CBR(128, 128, 3, 1, 1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) # 1/4
        
        # conv3
        self.conv3_1 = CBR(128, 256, 3, 1, 1)
        self.conv3_2 = CBR(256, 256, 3, 1, 1)
        self.conv3_3 = CBR(256, 256, 3, 1, 1)        
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) # 1/8
        
        # conv4
        self.conv4_1 = CBR(256, 512, 3, 1, 1)
        self.conv4_2 = CBR(512, 512, 3, 1, 1)
        self.conv4_3 = CBR(512, 512, 3, 1, 1)        
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True) # 1/16
        
        # conv5
        self.conv5_1 = CBR(512, 512, 3, 1, 1)
        self.conv5_2 = CBR(512, 512, 3, 1, 1)
        self.conv5_3 = CBR(512, 512, 3, 1, 1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True, return_indices=True)
        
        # fc6
        self.fc6 = CBR(512, 4096, 7, 1, 0)
        self.drop6 = nn.Dropout2d(0.5)
        
        # fc7
        self.fc7 = CBR(4096, 4096, 1, 1, 0)
        self.drop7 = nn.Dropout2d(0.5)
        
        # fc6-deconv
        self.fc6_deconv = DCB(4096, 512, 7, 1, 0)
        
        # unpool5
        self.unpool5 = nn.MaxUnpool2d(2, stride=2)
        self.deconv5_1 = DCB(512, 512, 3, 1, 1)
        self.deconv5_2 = DCB(512, 512, 3, 1, 1)
        self.deconv5_3 = DCB(512, 512, 3, 1, 1)
        
        # unpool4
        self.unpool4 = nn.MaxUnpool2d(2, stride=2)
        self.deconv4_1 = DCB(512, 512, 3, 1, 1)
        self.deconv4_2 = DCB(512, 512, 3, 1, 1)
        self.deconv4_3 = DCB(512, 256, 3, 1, 1)        

        # unpool3
        self.unpool3 = nn.MaxUnpool2d(2, stride=2)
        self.deconv3_1 = DCB(256, 256, 3, 1, 1)
        self.deconv3_2 = DCB(256, 256, 3, 1, 1)
        self.deconv3_3 = DCB(256, 128, 3, 1, 1)                          
        
        # unpool2
        self.unpool2 = nn.MaxUnpool2d(2, stride=2)
        self.deconv2_1 = DCB(128, 128, 3, 1, 1)
        self.deconv2_2 = DCB(128, 64, 3, 1, 1)

        # unpool1
        self.unpool1 = nn.MaxUnpool2d(2, stride=2)
        self.deconv1_1 = DCB(64, 64, 3, 1, 1)
        self.deconv1_2 = DCB(64, 64, 3, 1, 1)
        
        # Score
        self.score_fr = nn.Conv2d(64, num_classes, 1, 1, 0, 1)

    def forward(self, x):
        
        h = self.conv1_1(x)
        h = self.conv1_2(h)
        h, pool1_indices = self.pool1(h)
        
        h = self.conv2_1(h)
        h = self.conv2_2(h)
        h, pool2_indices = self.pool2(h)
        
        h = self.conv3_1(h)
        h = self.conv3_2(h)
        h = self.conv3_3(h)        
        h, pool3_indices = self.pool3(h)
        
        h = self.conv4_1(h)
        h = self.conv4_2(h)
        h = self.conv4_3(h)        
        h, pool4_indices = self.pool4(h) 
        
        h = self.conv5_1(h)
        h = self.conv5_2(h)
        h = self.conv5_3(h)        
        h, pool5_indices = self.pool5(h)
        
        h = self.fc6(h)
        h = self.drop6(h)
        
        h = self.fc7(h)
        h = self.drop7(h)
        
        h = self.fc6_deconv(h)     
        
        h = self.unpool5(h, pool5_indices)
        h = self.deconv5_1(h)        
        h = self.deconv5_2(h)                
        h = self.deconv5_3(h)                

        h = self.unpool4(h, pool4_indices)
        h = self.deconv4_1(h)        
        h = self.deconv4_2(h)                
        h = self.deconv4_3(h)                       

        h = self.unpool3(h, pool3_indices)
        h = self.deconv3_1(h)        
        h = self.deconv3_2(h)                
        h = self.deconv3_3(h)                            
        
        h = self.unpool2(h, pool2_indices)
        h = self.deconv2_1(h)        
        h = self.deconv2_2(h)                                         

        h = self.unpool1(h, pool1_indices)
        h = self.deconv1_1(h)        
        h = self.deconv1_2(h)                                    
            
        
        h = self.score_fr(h)           
        return h
```





<br>

### DeepLabv1

Dilated Convolution과 3x3 Maxpooling이 특징적입니다. 

CRF 프로세스는 추가되지 않았습니다. 

```python
import torch
import torch.nn as nn
from torch.nn import functional as F

def conv_relu(in_ch, out_ch, size=3, rate=1):
    conv_relu = nn.Sequential(nn.Conv2d(in_ch, 
                                        out_ch, 
                                        kernel_size=size, 
                                        stride=1,
                                        padding=rate,
                                        dilation=rate),
                             nn.ReLU())
    return conv_relu


class VGG16(nn.Module):
    def __init__(self):
        super(VGG16, self).__init__()
        self.features1 = nn.Sequential(conv_relu(3, 64, 3, 1),
                                      conv_relu(64, 64, 3, 1),
                                      nn.MaxPool2d(3, stride=2, padding=1))
        self.features2 = nn.Sequential(conv_relu(64, 128, 3, 1),
                                      conv_relu(128, 128, 3, 1),
                                      nn.MaxPool2d(3, stride=2, padding=1))
        self.features3 = nn.Sequential(conv_relu(128, 256, 3, 1),
                                      conv_relu(256, 256, 3, 1),
                                      conv_relu(256, 256, 3, 1),
                                      nn.MaxPool2d(3, stride=2, padding=1))
        self.features4 = nn.Sequential(conv_relu(256, 512, 3, 1),
                                      conv_relu(512, 512, 3, 1),
                                      conv_relu(512, 512, 3, 1),
                                      nn.MaxPool2d(3, stride=1, padding=1))
                                      # and replace subsequent conv layer r=2
        self.features5 = nn.Sequential(conv_relu(512, 512, 3, rate=2),
                                      conv_relu(512, 512, 3, rate=2),
                                      conv_relu(512, 512, 3, rate=2),
                                      nn.MaxPool2d(3, stride=1, padding=1), 
                                      nn.AvgPool2d(3, stride=1, padding=1)) # 마지막 stride=1로 해서 두 layer 크기 유지 
    def forward(self, x):
        out = self.features1(x)
        out = self.features2(out)
        out = self.features3(out)
        out = self.features4(out)
        out = self.features5(out)
        return out

class Classifier(nn.Module):
    def __init__(self, num_classes): 
        super(Classifier, self).__init__()
        self.classifier = nn.Sequential(conv_relu(512, 1024, 3, rate=12), 
                                       nn.Dropout2d(0.5), 
                                       conv_relu(1024, 1024, 1, 1), 
                                       nn.Dropout2d(0.5), 
                                       nn.Conv2d(1024, num_classes, 1)
                                       )
    def forward(self, x): 
        out = self.classifier(x)
        return out 

class DeepLabV1(nn.Module):
    def __init__(self, backbone, classifier, upsampling=8):
        super(DeepLabV1, self).__init__()
        self.backbone = backbone
        self.classifier = classifier
        self.upsampling = upsampling

    def forward(self, x):
        x = self.backbone(x)
        _, _, feature_map_h, feature_map_w = x.size()
        x = self.classifier(x)
        out = F.interpolate(x, size=(feature_map_h * self.upsampling, feature_map_w * self.upsampling), mode="bilinear")
        return out
```

```python
backbone = VGG16()
classifier = Classifier(num_classes=11)
model = DeepLabV1(backbone=backbone, classifier=classifier)
```









<br>

<br>

# 참고 자료

* 
