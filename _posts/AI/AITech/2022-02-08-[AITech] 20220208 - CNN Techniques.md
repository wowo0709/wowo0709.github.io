---

layout: single
title: "[AITech] 20220208 - CNN Techniques"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['AlexNet', 'VGGNet', 'GoogLeNet', 'ResNet', 'DenseNet']
---



<br>

## 학습 내용

이번 포스팅에서는 ImageNet을 이용한 이미지 분류 대회인 `ILSVRC`에서 뛰어난 업적을 남긴 네트워크들이 차용한 테크닉들을 살펴보겠습니다. 

### AlexNet

`AlexNet`은 지금 생각하면 당연한 것들을 사용한 수준이지만, 그 때 당시에는 혁신적이었으며 최초로 딥러닝 모델이 이 대회 우승을 차지하였습니다. 

![image-20220208142213924](https://user-images.githubusercontent.com/70505378/152952223-d0adaf03-ec82-45b2-a5a0-81c673c64d0a.png)

AlexNet의 주요 아이디어는 다음과 같습니다. 

* ReLU activation
  * 선형 모델의 특성을 보존
  * 경사하강법으로 최적화하기 용이
  * 높은 일반화 성능
  * 기울기 소실 문제를 극복
* Multi-GPU(2 GPUs, Model Parallel)
* Local response normalization, Overlapping pooling
* Data augmentation
* Dropout





<br>

### VGGNet

`VGGNet`의 가장 큰 특징은 **3x3 크기의 필터만 사용함으로써 네트워크의 깊이를 효과적으로 늘린 것**입니다. 또한 최종 분류층으로는 FC 층을 사용하는 대신에 1x1 Convolution을 사용하였습니다. (이 1x1 convolution의 특징은 다음 부분인 GoogleNet에서 설명합니다)

![image-20220208143000062](https://user-images.githubusercontent.com/70505378/152952228-3194c4f9-f897-40b4-b8b6-dbcb67f46599.png)

그렇다면 3x3 커널만을 사용하는 것이 가져오는 장점을 알아야겠죠? 

1. 3x3 커널을 여러 번 사용하면 5x5(2번), 7x7(3번) 커널과 같은 receptive field(수용 영역)를 가질 수 있다. 
2. 3x3 커널을 여러 번 사용하는 것이 5x5, 7x7 커널을 사용하는 것보다 파라미터 수가 적다. 

![image-20220208142910341](https://user-images.githubusercontent.com/70505378/152952226-558ad106-a285-4fd2-822c-578d95f97aa8.png)



<br>

### GoogLeNet

`GoogLeNet`의 가장 큰 특징은 앞에서 말했듯이 **1x1 convolution을 이용하여 파라미터의 개수를 줄인 것**입니다. 이는 FC층을 1x1 convolution 층으로 대체한 VGGNet의 목적과는 다르죠. 

![image-20220208143128662](https://user-images.githubusercontent.com/70505378/152952231-abe6dcfc-5e28-4e71-9e18-934f53d78b61.png)

위 구조를 보면 전의 네트워크들과는 달리 layer가 직렬적으로 연결된 부분 외에 **병렬적으로 연결**된 부분들이 눈에 띕니다. 이 블록을 **Inception block**이라고 합니다. 

![image-20220208143252448](https://user-images.githubusercontent.com/70505378/152952234-eaef2362-6629-4ce4-9a30-e1744cb0e3e0.png)

Inception block에서는 1x1 convolution 연산을 사용하여 채널 방향의 차원을 줄임으로써 파라미터의 개수를 효과적으로 감소시킵니다. 

우리는 kernel의 개수가 곧 피쳐맵의 channel-wise dimension이 되는 것을 알고 있습니다. 그렇다면 1x1 convolution을 사용하면 (w, h) 크기는 일정하게 되고 커널의 개수를 조정함으로써 같은 크기의 피쳐맵을 채널 방향으로 축소시킬 수 있습니다. 이는 실제로 다음과 같이 사용됩니다. 

![image-20220208143642481](https://user-images.githubusercontent.com/70505378/152952239-1635a48c-552d-416e-8536-363f4eaa83a5.png)

**receptive field, output size 모두 동일한 데 parameter의 수를 획기적으로 줄일 수 있습니다.** 이것이 GoogLeNet의 Inception block에서 사용하는 1x1 convolution의 강점입니다. 

실제로, AlexNet, VGGNet, GoogLeNet의 각각의 layer의 개수와 parameter의 개수는 다음과 같습니다. 

| Model     | layers | parameters |
| --------- | ------ | ---------- |
| AlexNet   | 8      | 60M        |
| VGGNet    | 19     | 110M       |
| GoogLeNet | 22     | 4M         |





<br>

### ResNet

`ResNet`의 가장 큰 특징은 **redisual module(잔차 모듈)을 사용함으로써 더 깊은 모델의 성능을 올린 것**입니다. 

이게 무슨 말이냐 하면, 모델의 깊이가 깊어지다보면(layer가 많아지다보면) 어느 순간 성능이 더 이상 좋아지지 않는 한계가 발생하고, 일정 개수 이상부터는 오히려 성능이 나빠집니다. 아래 그림처럼요. 

![image-20220208144317645](https://user-images.githubusercontent.com/70505378/152952242-c8d71b6a-2df6-4564-8458-224bdc7e0a8c.png)

이는 네트워크의 깊이가 깊어짐에 따라 역전파 시 기울기가 제대로 전달되지 않는 문제가 발생하게 되고 따라서 학습이 제대로 되지 않아서 그렇습니다. 

ResNet은 이를 skip connection이라는 것을 이용해, **연산이 적용된 정보와 더불어 기존 정보도 추가로 더해서 전달(f(x) + x)**하는 형태로 이 문제를 극복했습니다. 

![image-20220208144515759](https://user-images.githubusercontent.com/70505378/152952243-4142a7d1-2b6b-4764-b08e-4dda9367f639.png)

이렇게 함으로써 ResNet 네트워크는 다른 네트워크들과 달리 layer를 효과적으로 더 깊이 쌓을 수 있게 되었고, 결과적으로 깊은 모델의 성능이 더 좋아질 수 있도록 만들었습니다. 

![image-20220208144815085](https://user-images.githubusercontent.com/70505378/152952244-5e1e789f-b1eb-4ea9-9a16-20ab2274c825.png)

추가적으로, 위에서 본 skip connection에 대한 얘기를 조금 더 하겠습니다. skip connection을 수행하는 부분을 **Shortcut**이라고 하고,  Simple shortcut과 Projected shortcut이 있습니다. Projected shortcut은 Simple shortcut에 1x1 convolution이 더해진 형태인데, 이는 연산이 수행된 결과 피쳐맵과 기존의 입력 피쳐맵의 채널 방향의 차원 수가 같아야하기 때문에 이를 맞춰주기 위해서 존재합니다. 

![image-20220208145438604](https://user-images.githubusercontent.com/70505378/152952255-5b6912af-d693-4e4a-953c-cf2ca153027f.png)

그리고 채널 방향 차원 수를 맞춰주기 위해 다음과 같은 Bottleneck architeture도 사용합니다. 

![image-20220208145447581](https://user-images.githubusercontent.com/70505378/152952260-ac483eab-3aa3-482c-a001-360a8ddc468c.png)









<br>

### DenseNet

마지막으로 `DenseNet`입니다. DenseNet은 ResNet에서 연산이 적용된 피쳐맵과 기존 입력 피쳐맵 사이에 addition을 했던 것 대신에, **concatenation**을 사용합니다. 즉, 두 피쳐맵을 더하는 것 대신에 채널 방향으로 이어붙이는 것입니다. 그리고 이를 **Dense block**이라고 합니다. 

이는 필연적으로 채널 방향 차원 수의 급증을 발생시키는데요, 이를 해결하고자 채널 방향 차원 수를 감소시키는 층을 사용하고 이를 **Transition block**이라고 합니다. 

* **Dense Block**
  * 이전 앞선 layer들에서 발생한 feature map들을 모두 이어붙여서 사용한다. 
  * 채널 방향 차원 수가 기하급수적으로 증가한다. 
* **Transition Block**
  * Batch Norm -> 1x1 Conv -> 2x2 AvgPooling 층을 사용한다. 
  * 차원 수를 감소시킨다. 

![image-20220208150525330](https://user-images.githubusercontent.com/70505378/152952216-4dc82a71-894d-4a1b-9e3c-e5c8b349790d.png)

<br>

### Summary

* **VGG**: repeated 3x3 blocks
* **GoogLeNet**: 1x1 convolutoin
* **ResNet**: skip-connection
* **DenseNet**: concatenation







<br>

<br>

## 참고 자료

* 

















<br>
