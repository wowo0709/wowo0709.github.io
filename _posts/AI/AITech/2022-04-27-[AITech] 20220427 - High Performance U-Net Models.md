---
layout: single
title: "[AITech][Semantic Segmentation] 20220427 - High Performance U-Net Models"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['U-Net', 'U-Net++', 'U-Net 3+']
---



<br>

_**본 포스팅은 KAIST의 '김현우' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# High Performance U-Net Models

이번 포스팅에서는 U-Net을 포함해 U-Net의 구조를 차용해 발전된 모델들에 대해 보도록 하겠습니다. 

U-Net은 그 논문의 인용수가 현시점에서 40,000회 이상을 기록(YOLO가 약 24,000회)할 정도로 segmentation에서 큰 족적을 남긴 모델입니다. 

![image-20220427151845643](https://user-images.githubusercontent.com/70505378/165674432-85ad2e7a-9c18-4ae8-9d39-88de5bed3846.png)

## U-Net

`U-Net`은 의료분야 segmentation task에서 사용하기 위해 나온 모델이지만, 그 구조와 성능의 강력함으로 여러 분야의 segmentation 모델들에서 차용된 모델입니다. 

의료 분야는 특히나 사용 가능한 데이터의 수가 적고, 라벨링도 일반인이 하기에는 어렵다는 점 때문에 많은 학습 데이터를 확보하기 어렵습니다. 특히, cell segmentation 작업의 경우 같은 클래스가 인접해 있는 셀 사이의 경계를 구분할 필요가 있는데 이 문제는 일반적인 semantic segmentation으로는 불가능합니다. 

따라서 U-Net에서는 대칭 형태를 이루는 Contracting Path(Encoder)와 Expanding Path(Decoder)를 사용함으로써 이러한 문제들을 해결하기 위해 등장하였습니다. 

![image-20220427152414840](https://user-images.githubusercontent.com/70505378/165674436-c2054a87-05c4-4502-b983-9991160ee64b.png)

구조적 특징에 대해 보도록 하겠습니다. 

* 파란색 화살표
  * 3x3 Conv - (BN) - ReLU
  * zero padding을 적용하지 않아 feature map의 크기가 감소
  * 각 level의 첫번째 파란색 화살표: Contracting path에서는 채널의 수가 2배로 증가 (입력부 제외), Expanding path에서는 채널의 수가 2배로 감소
* 회색 화살표
  * 같은 계층(level)의 Encoder 출력물과 Decoder의 up-conv 결과를 concatenate
  * Resolution이 서로 동일하지 않기 때문에 encoder의 출력물을 center crop하여 resolution을 맞춰줌
  *  이러한 문제 때문에 구현체에 따라 padding=1로 지정하여 resolution을 동일하게 유지하는 경우도 있음
* 빨간색 화살표
  * maxpooling으로 feature map의 resolution을 2배로 감소
* 초록색 화살표
  * up-conv(transposed conv)로 feature map의 resolution을 2배로 증가
* 청록색 화살표
  * 1x1 conv를 적용하여 최종 score map 출력

<br>

U-Net의 contribution은 아래와 같습니다. 

1. Encoder가 확장됨에 따라 채널의 수를 1024까지 증가시켜 좀 더 고차원에서 정보를 매핑

2. 각기 다른 계층의 encoder의 출력을 decoder와 결합시켜서 이전 레이어의 정보를 효율적으로 활용

3. Random Elastic deformation을 통해 augmentation 수행

   * Model이 invariance와 robustness를 학습할 수 있도록 하는 방법
   * 의료 분야라는 특수성 때문에 사용

   ![image-20220427154159475](https://user-images.githubusercontent.com/70505378/165674437-ff0462b9-c768-4fa2-9b20-3a9a94bff195.png)

4. Pixel-wise loss weight를 계산하기 위한 weight map 생성

   * 같은 클래스를 가지는 인접한 셀을 분리하기 위해 해당 경계 부분에 가중치를 제공

   ![image-20220427154215370](https://user-images.githubusercontent.com/70505378/165674440-f1c8179c-b611-4d8d-a918-b271a6e545ea.png)

<br>

다음으로 U-Net의 한계점에 대해 보도록 하겠습니다. 

1. U-Net은 기본적으로 깊이가 4로 고정
   * 데이터셋마다 최고의 성능을 보장하지 못 함
   * 최적 깊이 탐색 비용 증가
2. 단순한 Skip Connection
   * 동일한 깊이를 가지는 encoder와 decoder만 연결되는 제한적인 구조











<br>

<br>

## U-Net++

`U-Net++`은 U-Net의 두가지 한계점을 극복하기 위해 새로운 형태의 아키텍쳐를 제시했습니다. 

![image-20220427155320850](https://user-images.githubusercontent.com/70505378/165674441-5428d5e5-3f3d-443f-8fbd-bf2c8f7a6e7b.png)

* Encoder를 공유하는 다양한 깊이의 U-Net을 생성
  * Encoder<sub>depth=1</sub> ~ Encoder<sub>depth=4</sub>
* Skip connection을 동일한 깊이에서의 Feature map들이 모두 결합되도록 유연한 feature map 생성

U-Net++의 특징적인 아이디어로는 3가지를 말할 수 있는데요, 각각에 대해 살펴보도록 하겠습니다. 

### Dense Skip Connection

 ![image-20220428104216423](https://user-images.githubusercontent.com/70505378/165674443-4155082c-4237-4397-9536-daf23e7ab38c.png)

각 level의 feature map들은 dense connection을 통해 같은 level에 전달됩니다. Skip connection 시에는 단순히 feature map들을 concat합니다. 

예를 들어 X<sup>0, 4</sup>는 아래와 같이 나타낼 수 있습니다. (H는 convolution을 나타냅니다)

![image-20220428104553155](https://user-images.githubusercontent.com/70505378/165674445-9a04fc5b-3731-4d34-ad78-efd75b522d99.png)

### Ensemble

그리고 여러 depth의 feature map들을 직접 추론 결과로 사용함으로써 다양한 모델들을 앙상블하는 효과를 얻을 수 있습니다. 

![image-20220428104748346](https://user-images.githubusercontent.com/70505378/165674447-82e106f1-56be-4ea6-b291-ba6970ecc8e4.png)





### Deep Supervision

또한 각 depth의 feature map들은 추론에 사용하는 것 뿐 아니라 loss 계산 시에도 사용되어 Deep supervision 학습을 진행합니다. 

각 depth에 대한 손실함수 값을 계산한 후 이를 평균을 취해 최종 손실 값으로 사용합니다. 

![image-20220428105101667](https://user-images.githubusercontent.com/70505378/165674450-1d7d94c2-ca81-4c47-b481-a404458949c6.png)

위 Loss 수식의 L(Y, P)는 아래와 같습니다. Pixel-wise cross entropy(빨간색)와 Soft dice coefficient(초록색)를 사용합니다. 

![image-20220428105352039](https://user-images.githubusercontent.com/70505378/165674452-58d666d2-0687-4773-96ff-40aa21e00e10.png)

* 𝑁 : Batch size 내의 픽셀 개수
* 𝐶 : class 개수
* 𝑦<sub>n, c</sub> :targetlabel
* 𝑝<sub>n, c</sub> : predict label  



<br>

이러한 U-Net++의 한계점으로는 아래와 같은 점들이 있습니다. 

* 복잡한 connection으로 인한 parameter 증가
* 많은 connection으로 인한 메모리 증가
* Encoder-Decoder 사이에서의 connection이 동일한 크기를 갖는 feature map에서만 진행됨
  * 즉, full scale에서 충분한 정보를 탐색하지 못해 위치와 경계를 명시적으로 학습하지 못 함







<br>

<br>

## U-Net 3+

![image-20220428111015879](https://user-images.githubusercontent.com/70505378/165674453-3f1412f0-23f9-4196-89ae-b5d5a744986c.png)

마찬가지로 `U-Net 3+`의 아이디어도 크게 3가지로 보도록 하겠습니다. 

### Full-scale Skip Connection

U-Net과 U-Net++에서 존재했던 skip connection에서의 feature map scale의 문제를 극복하기 위해 U-Net 3+에서는 이를 **(conventional + inter + intra) skip connection**으로 다양하게 구성하였습니다. 

* Conventional skip connection
  * Encoder layer로부터 same-scale의 feature map을 전달받음
* Inter skip connection
  * Encoder layer로부터 smaller-scale의 low-level feature map 을 전달받음
    * 여기서 smaller scale이란 resolution이 작다는 것이 아니라 하나의 pixel이 담고 있는 공간 정보가 적다는 것
  * 풍부한 공간 정보를 통해 경계 강조
* Intra skip connection
  * Decoder layer로부터 larger-scale의 high-level feature map 을 전달받음
    * 마찬가지로 larger-scale이란 하나의 pixel이 담고 있는 공간 정보가 많다는 것
  * 어디에 위치하는 지 위치 정보 구현

예를 들어 X<sub>De</sub><sup>3</sup>가 만들어지는 과정은 아래와 같습니다. 

![image-20220428111505521](https://user-images.githubusercontent.com/70505378/165674456-2066aeef-4c59-447a-9cdd-a6a90465a735.png)

또한, U-Net 3+에서는 파라미터 수를 줄이기 위해 모든 decoder layer의 channel 수를 320으로 통일하였습니다. 이를 통일하기 위해 skip connection 시 64 channel(# of kernels), 3x3 conv를 동일하게 적용하여 concat(64x5=320)합니다. 

U-Net 3+은 Full-scale skip connection을 통해 파라미터 수를 줄이면서도 성능 향상을 얻을 수 있었습니다. 

![image-20220428112231052](https://user-images.githubusercontent.com/70505378/165674457-7cef7770-d60b-4916-95f8-2b6fddcb0e0f.png)

<br>

### Classification-guided Module (GCM)

Low-level layer에 남아있는 background의 noise가 발생하여 다수의 false-positive 문제가 발생할 수 있습니다. 

U-Net 3+에서는 정확도를 높이고자, extra classification task를 진행하였습니다. 

* High-level feature map인 **X<sub>De</sub><sup>5</sup>**를 활용
  * Dropout, 1x1 conv, AdaptiveMaxPool, Sigmoid 통과
    * 확률값에 대한 Binary cross entropy loss 값 계산
  * Argmax를 통해 Organ(물체)이 없으면 0, 있으면 1로 출력
  * 위에서 얻은 결과와 각 low-layer마다 나온 결과를 곱
    * 0으로 분류 시 모든 false positive 제거

![image-20220428113028203](https://user-images.githubusercontent.com/70505378/165674458-933b315d-ca68-4537-9959-7b3f50c4c87f.png)







<br>

### Full-scale Deep Supervision (Loss funciton)

최종적으로 경계 부분을 잘 학습하기 위해 여러 Loss를 결합합니다. 

![image-20220428113305821](https://user-images.githubusercontent.com/70505378/165674459-60f1b7a4-20cd-44da-ad08-7da5fe09882a.png)

* Focal loss: 클래스의 불균형 해소
* ms-ssim Loss: Boundary 인식 강화
* IoU: 픽셀의 분류 정확도를 상승

최종적으로 아래와 같은 SOTA 성능을 달성할 수 있었습니다. 

![image-20220428113602870](https://user-images.githubusercontent.com/70505378/165674461-d76d884d-29dd-4097-aaff-dd7e493730b7.png)

<br>

<br>

## Another version of the U-Net

마지막으로 U-Net을 개선한 또 다른 세 가지 모델들에 대해 보도록 하겠습니다. 

### Residual U-Net

`Residual U-Net`은 encoder와 decoder 부분의 block마다 **residual unit with identity mapping**을 적용하여 만든 네트워크입니다. 

![image-20220428114053631](https://user-images.githubusercontent.com/70505378/165674464-adfbacf6-d880-4995-b38b-5a687b770faf.png)







<br>

### Mobile U-Net

`Mobile U-Net`은 backbone 부분에 mobile network를 적용하여 속도를 개선한 네트워크입니다. 

![image-20220428114147121](https://user-images.githubusercontent.com/70505378/165674467-9bb3bc91-be93-4af6-9db4-fb67dced2501.png)









<br>

### Eff-UNet

`Eff-UNet`은 Encoder로 EfficientNet을 사용하여 성능 향상을 달성한 네트워크입니다. 

Encoder 부분에서는 MBConv(Mobile inverted Bottleneck Convolution)라는 연산을 사용합니다. 

![image-20220428114326638](https://user-images.githubusercontent.com/70505378/165674470-296a6314-a889-4219-bf58-76f7895167b8.png)

아래는 전체 구조입니다. 

![image-20220428114522463](https://user-images.githubusercontent.com/70505378/165674474-f2593b64-220f-4e89-9c73-063aabf8efd6.png)



<br>

<br>

## 실습) U-Net, U-Net++

### U-Net

![image-20220428120146710](https://user-images.githubusercontent.com/70505378/165674476-dd16edbe-915b-4ee0-a7e1-37e40a57eac1.png)

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, num_classes=11):
        super(UNet, self).__init__()
        def CBR2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
            layers = []
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                 kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
            layers += [nn.BatchNorm2d(num_features=out_channels)]
            layers += [nn.ReLU()]

            cbr = nn.Sequential(*layers)
            return cbr

        # Contracting path 
        self.enc1_1 = CBR2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc1_2 = CBR2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)     
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.enc2_1 = CBR2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc2_2 = CBR2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
 
        self.enc3_1 = CBR2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc3_2 = CBR2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2)    

        self.enc4_1 = CBR2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc4_2 = CBR2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2)    

        self.enc5_1 = CBR2d(in_channels=512, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.enc5_2 = CBR2d(in_channels=1024, out_channels=1024, kernel_size=3, stride=1, padding=1, bias=True)
        self.unpool4 = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec4_2 = CBR2d(in_channels=1024, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True) 
        self.dec4_1 = CBR2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1, bias=True) 

        self.unpool3 = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec3_2 = CBR2d(in_channels=512, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True) 
        self.dec3_1 = CBR2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=True) 

        self.unpool2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec2_2 = CBR2d(in_channels=256, out_channels=128, kernel_size=3, stride=1, padding=1, bias=True)  
        self.dec2_1 = CBR2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True)  

        self.unpool1 = nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=2, stride=2, padding=0, bias=True)

        self.dec1_2 = CBR2d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True) 
        self.dec1_1 = CBR2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, bias=True) 
        self.score_fr = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True) # Output Segmentation map 

    def forward(self, x):
        enc1_1 = self.enc1_1(x)
        enc1_2 = self.enc1_2(enc1_1)
        pool1 = self.pool1(enc1_2)

        enc2_1 = self.enc2_1(pool1)
        enc2_2 = self.enc2_2(enc2_1)
        pool2 = self.pool2(enc2_2)

        enc3_1 = self.enc3_1(pool2)
        enc3_2 = self.enc3_2(enc3_1)
        pool3 = self.pool3(enc3_2)

        enc4_1 = self.enc4_1(pool3)
        enc4_2 = self.enc4_2(enc4_1)
        pool4 = self.pool4(enc4_2)

        enc5_1 = self.enc5_1(pool4)
        enc5_2 = self.enc5_2(enc5_1)

        unpool4 = self.unpool4(enc5_2)
        cat4 = torch.cat((unpool4, enc4_2), dim=1) 
        dec4_2 = self.dec4_2(cat4)
        dec4_1 = self.dec4_1(dec4_2)

        unpool3 = self.unpool3(dec4_1)
        cat3 = torch.cat((unpool3, enc3_2), dim=1) 
        dec3_2 = self.dec3_2(cat3)
        dec3_1 = self.dec3_1(dec3_2)

        unpool2 = self.unpool2(dec3_1)
        cat2 = torch.cat((unpool2, enc2_2), dim=1) 
        dec2_2 = self.dec2_2(cat2)
        dec2_1 = self.dec2_1(dec2_2)

        unpool1 = self.unpool1(dec2_1)
        cat1 = torch.cat((unpool1, enc1_2), dim=1) 
        dec1_2 = self.dec1_2(cat1)
        dec1_1 = self.dec1_1(dec1_2)

        output = self.score_fr(dec1_1) 
        return output
```

<br>

### U-Net++

![image-20220428120307676](https://user-images.githubusercontent.com/70505378/165674478-6b93de4d-a124-41ee-a91c-d0f98db9cb71.png)

```python
# 출처 : https://jinglescode.github.io/2019/12/02/biomedical-image-segmentation-u-net-nested/
import torch
import torch.nn as nn

class conv_block_nested(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(conv_block_nested, self).__init__()
        self.activation = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1, bias=True)
        self.bn1 = nn.BatchNorm2d(mid_ch)
        self.conv2 = nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1, bias=True)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activation(x)

        x = self.conv2(x)
        x = self.bn2(x)
        output = self.activation(x)
        return output

class UNetPlusPlus(nn.Module):

    def __init__(self, in_ch=3, out_ch=1, n1=64, height=512, width=512, supervision=True):
        super(UNetPlusPlus, self).__init__()

        filters = [n1, n1 * 2, n1 * 4, n1 * 8, n1 * 16]

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.Up = nn.ModuleList([nn.Upsample(size=(height//(2**c), width//(2**c)), mode='bilinear', align_corners=True) for c in range(4)])
        self.supervision = supervision

        self.conv0_0 = conv_block_nested(in_ch, filters[0], filters[0])
        self.conv1_0 = conv_block_nested(filters[0], filters[1], filters[1])
        self.conv2_0 = conv_block_nested(filters[1], filters[2], filters[2])
        self.conv3_0 = conv_block_nested(filters[2], filters[3], filters[3])
        self.conv4_0 = conv_block_nested(filters[3], filters[4], filters[4])

        self.conv0_1 = conv_block_nested(filters[0] + filters[1], filters[0], filters[0])
        self.conv1_1 = conv_block_nested(filters[1] + filters[2], filters[1], filters[1])
        self.conv2_1 = conv_block_nested(filters[2] + filters[3], filters[2], filters[2])
        self.conv3_1 = conv_block_nested(filters[3] + filters[4], filters[3], filters[3])

        self.conv0_2 = conv_block_nested(filters[0]*2 + filters[1], filters[0], filters[0])
        self.conv1_2 = conv_block_nested(filters[1]*2 + filters[2], filters[1], filters[1])
        self.conv2_2 = conv_block_nested(filters[2]*2 + filters[3], filters[2], filters[2])

        self.conv0_3 = conv_block_nested(filters[0]*3 + filters[1], filters[0], filters[0])
        self.conv1_3 = conv_block_nested(filters[1]*3 + filters[2], filters[1], filters[1])

        self.conv0_4 = conv_block_nested(filters[0]*4 + filters[1], filters[0], filters[0])

        self.seg_outputs = nn.ModuleList([nn.Conv2d(filters[0], out_ch, kernel_size=1, padding=0) for _ in range(4)])

    def forward(self, x):
        seg_outputs = []
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.Up[0](x1_0)], 1))
        seg_outputs.append(self.seg_outputs[0](x0_1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.Up[1](x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.Up[0](x1_1)], 1))
        seg_outputs.append(self.seg_outputs[1](x0_2))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.Up[2](x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.Up[1](x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.Up[0](x1_2)], 1))
        seg_outputs.append(self.seg_outputs[2](x0_3))

        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.Up[3](x4_0)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.Up[2](x3_1)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.Up[1](x2_2)], 1))
        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.Up[0](x1_3)], 1))
        seg_outputs.append(self.seg_outputs[3](x0_4))

        if self.supervision: 
            return seg_outputs
        else:
            return seg_outputs[-1]
```













<br>

<br>

# 참고 자료

* 
