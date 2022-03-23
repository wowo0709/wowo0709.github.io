---
layout: single
title: "[AITech][Object Detection] 20220323 - EfficientDet"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 '송원호' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# EfficientDet

이번 포스팅에서는 Image Classification을 위한 모델인 EfficientNet의 Object Detection 버전인 `EfficientDet`에 대해 알아보겠습니다. 

## Efficient in Object Detection

**Efficient**라는 것은 무엇일까요?

지금까지 모델들은 3가지 측면에서 모델의 크기를 키우는 Model Scaling을 진행했습니다. 

* Width Scaling: 채널 수를 늘린다. 
* Depth Scaling: 모델 층 수를 늘린다. 
* Resolution Scaling: 입력 크기를 늘린다. 

Google의 EfficientNet 팀은 아래와 같은 이야기를 하였습니다. 

> _EfficientNet팀의 연구는 네트워크의 폭(width), 깊이(depth), 해상도(resolution) 모든 차원에서의 균형을 맞추는 것이 중요하다는 것을 보여주었다. 그리고 이러한 균형은 각각의 크기를 일정한 비율로 확장하는 것으로 달성할 수 있었다._

위와 같이 **세 가지 측면을 효율적으로 조절해서 더 높은 정확도와 효율성을 갖도록 하는 것**이 여기서 말하는 **Efficient**입니다. 

아래는 EfficientDet의 성능을 나타낸 그래프입니다. 

![image-20220323141406908](https://user-images.githubusercontent.com/70505378/159639917-966c3a1e-439e-4540-90f0-3cf5b0565ac6.png)

<br>

## EfficientNet

`EfficientNet`의 등장배경은 다음과 같습니다. 

Model Scaling에 따라, 모델의 성능이 좋아짐과 동시에 모델의 파라미터 수 또한 크게 늘어났습니다. 하지만, 이를 실제로 적용하기 위해서는 어느정도 경량화되고 추론 속도가 빠른 모델이 필요합니다. 

EfficientNet은 이러한 요구 사항에 맞추어, 세 가지 모델 스케일링을 가장 효율적으로 조절함으로써 같은 파라미터 수 대비 훨씬 좋은 성능을 낼 수 있었습니다. 

![image-20220323142319993](https://user-images.githubusercontent.com/70505378/159639922-e7609cd1-4012-4b0b-a2cb-f69482696567.png)

### Model Scaling

**Width Scaling**

* 네트워크의 width를 스케일링하는 방법은 작은 모델에서 주로 사용됨 (ex. MobileNet, MnasNet)
* 더 wide한 네트워크는 미세한 특징을 잘 잡아내는 경향이 있고, 학습도 쉬움
* 하지만, 극단적으로 넓지만 얕은 모델은 high-level 특징들을 잘 잡지 못 하는 경향이 있음

![image-20220323142518418](https://user-images.githubusercontent.com/70505378/159639925-afe2c2c2-6212-4429-a5bc-f2e7deaef2f6.png)

**Depth Scaling**

* 네트워크의 깊이를 스케일링하는 방법은 많은 ConvNet에서 쓰이는 방법 (ex. DenseNet, Inception-v4)
* 깊은 ConvNet은 풍부하고 복잡한 특징들을 잡아낼 수 있고, 새로운 테스크에도 잘 일반화 됨
* 하지만 깊은 네트워크는 graident vanishing 문제가 있어 학습이 어려움

![image-20220323142806358](https://user-images.githubusercontent.com/70505378/159639930-2ae77b8b-9c79-489f-ba7a-6854ae9a7ab0.png)

**Resolution Scaling**

* 고화질의 input 이미지를 이용하면 ConvNet은 미세한 패턴을 잘 잡아낼 수 있음
* Gpipe는 480x480 이미지를 이용하여, ImageNet에서 SOTA를 달성

![image-20220323142755351](https://user-images.githubusercontent.com/70505378/159639929-c349ca34-cb57-42a0-b67b-121baa2df581.png)

마지막으로 세 가지 scaling 기법을 모두 사용하는 compound scaling이 있을 수 있습니다. 

![image-20220323142744848](https://user-images.githubusercontent.com/70505378/159639927-7bf6fbc3-875a-4e02-801f-15e774191083.png)

### Accuracy & Efficiency

EfficientNet의 object function은 아래와 같습니다. d, w, r은 각각 depth, width, resolution에 해당하는 scale factor입니다. 

![image-20220323143350753](https://user-images.githubusercontent.com/70505378/159639933-ee32689d-7d1f-4776-9871-83f007e45f0d.png)

![image-20220323143358638](https://user-images.githubusercontent.com/70505378/159639935-5964875f-1b15-4f64-b846-2d12b973cede.png)

EfficientNet 연구팀은 위의 object function을 가지고 여러 실험을 반복하여, 아래의 결론들을 이끌어 냈습니다. 

**Observation 1**

_네트워크의 폭, 깊이, 혹은 해상도를 키우면 정확도가 향상된다. 하지만 더 큰 모델에 대해서는 정확도 향상 정도가 감소한다._

![image-20220323143829524](https://user-images.githubusercontent.com/70505378/159639936-8dc8b3d6-96a9-4716-a0b6-60639e6072bf.png)

**Observation 2**

_더 나은 정확도와 효율성을 위해서는, ConvNet 스케일링 과정에서 네트워크의 폭, 깊이, 해상도의 균형을 잘 맞춰주는 것이 중요하다._

![image-20220323143905543](https://user-images.githubusercontent.com/70505378/159639940-fd1c3e3f-acbd-4071-af0f-0ce46aa9d6b4.png)

**Compond Scaling Method**

그래서 EfficientNet 연구팀은 아래 조건을 만족하면서 d, w, r 를 모두 바꾸는 Compund scaling method를 제안합니다. 

![image-20220323144141243](https://user-images.githubusercontent.com/70505378/159639943-94f1fdb9-0ee6-449b-aa00-4c49fb2b7bca.png)

### EfficientNet

이제는 앞의 조건들을 만족하는 alpha, beta, gamma와 모델 구조 F를 결정해야 합니다. 

**EfficientNet-B0**

가장 초기 EfficientNet 모델은 EfficientNet-B0 모델입니다. 

이 모델은 Accuracy와 FLOPs를 고려하여 NAS로 최적의 scaling factor와 모델 구조를 찾았습니다. 

![image-20220323144817066](https://user-images.githubusercontent.com/70505378/159639947-5d810254-6789-4604-918d-9c3a45b5ff46.png)

**EfficientNet-B1 ~ B7**

이후에는 모델 구조는 고정하고, 𝜙 = 1 로 고정하여 𝛼, 𝛽, 𝛾를 small grid search를 통해 찾았습니다. 그 결과 **𝛼 = 1.2, β = 1.1, 𝛾 = 1.15 under constraint of α ∙ 𝛽<sup>2</sup>∙ 𝛾<sup>2</sup> ≈ 2**라는 결과를 얻었습니다. 

위에서 찾은 𝛼, 𝛽, 𝛾를 상수로 고정하고, 𝜙를 1, 2, ..., 7로 scale up 했을 때의 모델 결과가 바로 EfficientNet-B1 ~ B7에 해당합니다. 

![image-20220323145225221](https://user-images.githubusercontent.com/70505378/159639950-fd2851b3-233e-45a5-afa6-5fe2902a906f.png)

<br>

결과적으로 EfficientNet은 당시에 동일한 FLOPs 대비 다른 모델들보다 훨씬 뛰어난 성능을 보여줬습니다. 

![image-20220323145328685](https://user-images.githubusercontent.com/70505378/159639951-81ba1083-1efa-44f3-957b-20af392ef4e2.png)

<br>

## EfficientDet

`EfficientDet`은 EfficientNet과 같이 compound scaling을 이용해 detection task에서의 최적의 model scaling을 찾으려는 시도를 한 모델입니다. 

Object Detection 에서는 특히나 모델의 사이즈와 연산량이 중요한데요, 1 stage detector는 속도는 빠르지만 accuracy가 너무 낮습니다. 따라서 자원의 제약이 있는 상태에서 더 높은 정확도와 효율성을 가지는 detection 구조를 만드려는 시도가 필요했고, 이것을 해결한 것이 EfficientDet 모델입니다. 

EfficientDet은 backbone, FPN, box/class prediction head를 모두 고려하여 model scaling을 진행했습니다. 

### Efficient multi-scale feature fusion

첫번째로 EfficientDet에서 조절한 것은 Neck의 구조입니다. 기존 FPN에서는 두 feature map을 단순히 summation 하는 형태로 fusion을 수행했습니다. 

![image-20220323151115498](https://user-images.githubusercontent.com/70505378/159639955-6a1bd08f-4d80-4664-86cb-8b651354efea.png)

EfficientDet에서는 기존의 FPN 대신, 개선된 구조의 BiFPN을 사용합니다. **불필요한 연결을 끊어버리고, residual connection을 추가**한 형태입니다. 

![image-20220322143549600](https://user-images.githubusercontent.com/70505378/159451754-fe1c2bc1-612f-4049-9a3e-652a9558ed0a.png)

그리고 이로 인해 불필요한 연산이 줄어서, BiFPN에서는 위의 repeated block 구조를 반복하여 Neck을 설계했습니다. 

또한 중요한 것은, BiFPN에서는 feature map을 합칠 때 단순 summation이 아닌 **Weighted Feature Fusion**을 수행합니다. 이는 feature map이 더해질 때 각 feature map에 가중치를 두어 weighted sum을 하는 형태이고, 이 가중치도 학습 가능한 파라미터로 두어 학습이 가능하게 합니다. 

이것으로 중요한 feature를 강조하여 성능 향상을 이루면서도, 모델 사이즈의 증가는 거의 없다고 합니다. 

![image-20220322143946890](https://user-images.githubusercontent.com/70505378/159451758-4ef0732c-53a4-4114-8676-a5fdfe2bc690.png)

위 그림에서 위첨자에 in이 있는 것은 첫번째 layer, td(top-down)가 붙어있는 것은 중간 layer, out이 붙어있는 것은 마지막 layer의 feature map을 가리킵니다. 

### Model Scaling

Model Scaing은 아래와 같이 진행되었습니다. 

* EfficientNet B0~B6를 backbone으로 사용

* BiFPN

  * 네트워크의 width(=# channels)와 depth(=# layers)를 compound 계수에 따라 증가시킴

    ![image-20220323152003251](https://user-images.githubusercontent.com/70505378/159639958-d23427b6-19a8-4888-a5d9-a8f07f4b1372.png)

* Box/class prediction network

  * Width는 고정, depth를 다음과 같은 식에 따라 증가

    ![image-20220323152042079](https://user-images.githubusercontent.com/70505378/159639960-94d7a9f3-14ed-4101-a9d5-16b5e02ed9e1.png)

* Input image resolution

  * Resolution을 다음과 같이 선형적으로 증가

    ![image-20220323152117016](https://user-images.githubusercontent.com/70505378/159639962-89e1e712-19a7-46e5-bdfe-7fae02871efd.png)

그리고 아래와 같이 EfficientDet D0 ~ D7을 만들어냈습니다. 

![image-20220323152222115](https://user-images.githubusercontent.com/70505378/159639966-216953a0-74e2-488a-82f9-9064066ea89f.png)

<br>

아래는 여러 방면에서 EfficientDet과 다른 모델들을 비교한 그래프입니다. 

![image-20220323152301598](https://user-images.githubusercontent.com/70505378/159639972-0d4efe80-5390-489c-9adf-db71c35c7be6.png)





















<br>

<br>

# 참고 자료

* 

