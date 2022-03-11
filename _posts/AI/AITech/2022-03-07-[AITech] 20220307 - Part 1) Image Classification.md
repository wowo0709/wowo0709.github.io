---
layout: single
title: "[AITech][CV] 20220307 - Part 1) Image Classification & Data Efficient Learning"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['CNN Architectures', 'Data Efficient Learning', 'ResNet34 구현']
---

_**본 포스팅은 POSTECH '오태현' 강사 님의 강의를 바탕으로 작성되었습니다. **_

<br>

# Image Classification

Part 1 `Image Classification` 에서는 ILSVRC 대회에서 우수한 성적을 보인 모델들을 중심으로 CNN 구조의 발전사에 대해 살펴보고, Annotation data efficient learning에 대해 알아보겠습니다. 

## CNN Architectures for image classification

![image-20220311103435818](https://user-images.githubusercontent.com/70505378/157833199-ec7ff7ab-5abb-4d42-91b8-6530d86f1a8d.png)

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
* Local response normalization(LRN), Overlapping pooling
* Data augmentation
* Dropout
* Large kernel size

그 중 현재는 쓰이지 않는 LRN과 Large kernel size 기법들이 왜 쓰였는지, 현재는 무엇으로 대체되었는지 보겠습니다. 

* LRN

  * 이미지에서 명암을 normalize하는 역할을 합니다. 
  * 적용함으로써 명암 외의 이미지 자체에 더 민감하게 반응하는 효과를 얻을 수 있습니다. 
  * 현재는 **Batch Normalization**으로 대체

  ![image-20220311104547539](https://user-images.githubusercontent.com/70505378/157833136-5fe88195-d501-46c0-b8fd-d0d11359315f.png)

* 11x11 convolution filter

  * 이미지 크기가 커짐에 따라 큰 크기의 커널이 요구
  * 큰 크기의 커널로부터 얻을 수 있는 이점은 receptive field가 커진다는 것
    * receptive field란 input image에서 우리가 관찰하고 있는(특징을 얻는) 범위
    * KxK convolution filter, stride 1, PxP pooling layer를 사용할 때 receptive field는 (P+K-1)x(P+K-1)
  * 현재는 3x3 크기의 커널을 여러 개 사용함으로써 큰 커널을 사용한 것과 동일한 receptive field를 가지는 효과를 얻으면서, 가중치의 수를 크게 줄임

  ![image-20220311105216095](https://user-images.githubusercontent.com/70505378/157833142-52ed19c2-f507-4de9-b871-ea960ef3843d.png)

<br>

### VGGNet

`VGGNet`의 가장 큰 특징은 **3x3 크기의 필터(와 2x2 크기의 풀링)만 사용함으로써 네트워크의 깊이를 효과적으로 늘린 것**입니다. 또한 최종 분류층으로는 FC 층을 사용하는 대신에 1x1 Convolution을 사용하였습니다. (이 1x1 convolution의 특징은 다음 부분인 GoogleNet에서 설명합니다)

![image-20220208143000062](https://user-images.githubusercontent.com/70505378/152952228-3194c4f9-f897-40b4-b8b6-dbcb67f46599.png)

그렇다면 3x3 커널만을 사용하는 것이 가져오는 장점을 알아야겠죠? 

1. 3x3 커널을 여러 번 사용하면 5x5(2번), 7x7(3번) 커널과 같은 receptive field(수용 영역)를 가질 수 있다. 
2. 3x3 커널을 여러 번 사용하는 것이 5x5, 7x7 커널을 사용하는 것보다 파라미터 수가 적다. 

![image-20220208142910341](https://user-images.githubusercontent.com/70505378/152952226-558ad106-a285-4fd2-822c-578d95f97aa8.png)

VGGNet을 AlexNet에 비교하면 다음과 같습니다. 

* Deeper architecture
* Simpler architecture
* Better performance
* Better generalization

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

또, GoogLeNet은 추가적으로 네트워크의 중간 단에 **Auxiliary classifier**를 사용하여 추가적으로 gradient를 전달함으로써, 최종 classifier 단에서 최초 input 단까지 gradient가 제대로 전달되지 못 하고 vanishing되는 문제를 해결하려 했습니다. 최종 classifier에서는 single FC layer를 사용합니다. 

![image-20220311110815581](https://user-images.githubusercontent.com/70505378/157833143-73c03f36-2f8b-4805-8ef4-7f7c22fefc79.png)

Auxiliary classifier는 train시에만 동작하고 inference 시에는 동작하지 않습니다. 



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

여기서 짚고 넘어가야 할 것이 있습니다. 아래는 ResNet34의 구조입니다. 

![image-20220311111633529](https://user-images.githubusercontent.com/70505378/157833145-7b1cb0e6-8296-4ca5-af5f-877a93ebd290.png)

크게 5개의 conv 부분이 있는 것을 볼 수 있습니다(색깔로 구분). 여기서 각 conv part의 맨 앞 단 블록을 보면 `/2`와 같이 표시되어 있는 것이 있습니다. 이는 stride로, stride=2로 지정하여 피쳐맵의 (h, w)를 줄이는 것입니다. 

문제는, conv 연산을 거쳐 나오는 feature map의 (h, w)는 1/2로 줄어드는 데 반해, shortcut(1x1 convolution)을 거친 identity x 값은 channel 방향 조정한 일어나고 (h, w) 방향으로는 변화가 없다는 것입니다. 차원의 크기가 맞지 않아 tensor 간 연산을 수행할 수 없습니다. 

따라서 ResNet은 이러한 문제가 발생하는 맨 앞 conv layer에 한해서 identity x의 (h, w) 방향 크기를 1/2로 줄이기 위해 다음의 방법들을 제안합니다. 

* Pooling 연산
* 1x1 convolution 연산에서 stride=2로 지정
* 크기를 1/2로 줄여줄 수 있는 trainable matrix 정의

이 중 실제로 구현 시 사용되는 방법은 2번째 방법인 1x1 convolution 연산 시 stride=2로 지정하는 방법입니다. 

<br>

### DenseNet

`DenseNet`은 ResNet에서 연산이 적용된 피쳐맵과 기존 입력 피쳐맵 사이에 addition을 했던 것 대신에, **concatenation**을 사용합니다. 즉, 두 피쳐맵을 더하는 것 대신에 채널 방향으로 이어붙이는 것입니다. 그리고 이를 **Dense block**이라고 합니다. 

이는 필연적으로 채널 방향 차원 수의 급증을 발생시키는데요, 이를 해결하고자 채널 방향 차원 수를 감소시키는 층을 사용하고 이를 **Transition block**이라고 합니다. 

* **Dense Block**
  * 이전 앞선 layer들에서 발생한 feature map들을 모두 이어붙여서 사용한다. 
  * 채널 방향 차원 수가 기하급수적으로 증가한다. 
* **Transition Block**
  * Batch Norm -> 1x1 Conv -> 2x2 AvgPooling 층을 사용한다. 
  * 차원 수를 감소시킨다. 

![image-20220208150525330](https://user-images.githubusercontent.com/70505378/152952216-4dc82a71-894d-4a1b-9e3c-e5c8b349790d.png)

<br>

### SENet

`SENet`은 모델의 깊이를 늘리거나 연결을 다른 방법으로 하기보다는, 주어진 activation 간의 관계가 더 명확해질 수 있도록 channel 간의 관계를 모델링하고, 중요도를 파악해 weight update에 있어 channel-wise information에 attention하는 모델입니다. 

SENet에서는 2가지 주된 기법을 사용합니다. 

* **Squueze**: 피쳐맵에 gap(global average pooling)를 적용해 1x1xC 형태의 벡터를 생성합니다. 
* **Excitation**: FC layer를 통해 얻어진 channel-wise attention weights 행렬과 앞서 얻은 1x1xC 크기의 벡터를 곱해 new weighted된 1x1xC 크기의 벡터를 얻어냅니다. 이 때의 생성된 벡터를 attention score라고 합니다. 

![image-20220311113827671](https://user-images.githubusercontent.com/70505378/157833150-51e98205-e754-4f69-82ed-70a7d09669fd.png)



<br>

### EfficientNet

`EfficientNet`은 네트워크의 **depth, width, resolution**을 trainable value로 하여 최적의 값을 찾아내는 모델입니다. 

![image-20220311114015373](https://user-images.githubusercontent.com/70505378/157833151-7f03f185-2ac7-4a97-832c-90feff1f4db4.png)

<br>

### Deformable convolution

**Deformable convolution**은 기존의 standard convolution과 조금 다른 방법으로 convolution 연산을 적용하는 것입니다. 기존의 standard convolution은 이미지에 convolution이 적용되는 모양(형태)이 직사각형 형태로 같았다면, deformable convolution은 여기에 offset을 추가하여 물체의 형태의 조금 더 맞는 형태로 convolution을 적용합니다. 

Standard CNN과 2D offset(grid sampling)을 통해 구현할 수 있습니다. 

![image-20220311114824711](https://user-images.githubusercontent.com/70505378/157833155-e0cd4f08-8b31-4ec6-8bc2-d69085fe78b5.png)





<br>

<br>

## Annotation Data Efficient Learning

### Data augmentation

이미지 분류 모델을 학습시킨다는 것은 **이미지의 분포를 모델링(또는 이미지의 분포를 나타내는 함수를 추정)**한다는 것입니다. 

우리가 모든 데이터를 가지고 있다면 모든 이미지를 완벽하게 분류할 수 있겠지만, 이는 불가능하죠. 사용하는 training data와 real data 사이에는 항상 gap이 있기 마련입니다. 이는 다른 말로 하면, training data는 전체 데이터의 일부분만을 가지고 있기 때문에, data distribution을 완벽하게 표현하지 않는다는 것입니다. 

![image-20220311131948257](https://user-images.githubusercontent.com/70505378/157833157-4a62e85d-d446-4b4c-bdbb-ac86d6f9932d.png)

여기서 **Data augmentation**의 필요성이 대두됩니다. Data augmentation(데이터 증강)은 가지고 있는 training data에 변형을 가함으로써, 우리의 데이터가 좀 더 넓은 범위(더 다양한)의 data를 갖게 되고, 따라서 distrubution을 더 잘 표현하도록 해줍니다. 

![image-20220311132047655](https://user-images.githubusercontent.com/70505378/157833158-a2c35f2e-4a5f-499f-925a-fab7bcaec2a0.png)

Data augmentation에는 정말 많은 종류가 있습니다. 그것을 모두 외운다거나 하는 것은 큰 의미가 없습니다. 중요한 것은, **우리의 데이터에 따라 적용할 augmentation 기법을 적절히 선택**할 수 있는 능력을 기르는 것입니다. 예컨대, 도로 위의 사람들을 detection하는 이미지 데이터의 경우 이를 수직으로 뒤집는 VerticalFlip 연산은 오히려 모델 학습에 혼란만 가중시킬 수 있겠죠. 

다시 한 번 말하지만, Data augmentation의 최종 목표는 **training data distribution을 real world data distribution과 유사하게 만드는 것**입니다. 

전통적으로 많이 사용해 온 augmentation 기법에는 Crop, Shear, Brightness, Perspective(Resize), Rotate, Flip, ColorJitter, Noise 등이 있으며, 최근 떠오르는 기법으로는 **Cutmix**가 있습니다. 

![image-20220311132741340](https://user-images.githubusercontent.com/70505378/157833160-49cecb6c-8318-4354-9f3b-751286b697e7.png)

Cutmix는 원본 이미지의 일부에 다른 클래스의 이미지를 섞고 라벨도 그에 맞게 one-hot 형태가 아닌 logit 형태로 바꾸는 것입니다. 이 기법으로 모델의 일반화 성능 향상을 기대할 수 있습니다. 

마지막으로, augmentation 기법은 모든 training data에 대해 동일하게 적용하는 것보다는 매번 random하게 몇 개만 적용하는 방식으로 사용하는 것이 더 좋다고 합니다. 







<br>

### Leveraging pre-trained information

**Transfer Learning**

다들 **transfer learning**에 대해서는 잘 알고 있으실 겁니다. 좋은 품질의 대용량의 데이터셋으로 기학습된 모델을 가져와, 우리의 target task에 맞게 재학습을 시킴으로써 쉽게 좋은 성능의 모델을 얻어내는 것이죠. 

Transfer learning에는 크게 두 가지 학습 방법이 있습니다. 

* Approach 1: Transfer knowledge from a pre-trained task to a new task **(Feature Extraction)**

  * 교체된 FC layer만 학습시키고 feature extraction 부분(CNN backbone)은 freeze 시키는 것

  ![image-20220311135041424](https://user-images.githubusercontent.com/70505378/157833161-a2c19474-ac05-4bd6-a9c1-99e785336bac.png)

* Approch 2: Fine-tuning the whole model **(Fine tuning)**

  * 교체된 FC layer를 포함하여 전체 모델을 재학습시키는 것
  * CNN 부분은 low learning rate, FC layer 부분은 high learning rate

  ![image-20220311135218028](https://user-images.githubusercontent.com/70505378/157833162-a2e7824c-9eaa-4ae0-bc7b-7a562cffc9fe.png)

어떨 때 각각의 방법을 사용할 지는 **우리가 가지고 있는 데이터**에 달려있습니다. 

* Case 1. 문제를 해결하기 위한 학습 데이터가 충분하다.

![image-20220223173901422](https://user-images.githubusercontent.com/70505378/155285614-eb98e48f-e08d-4abf-a479-7f22b787f144.png)



* Case 2. 학습 데이터가 충분하지 않다.

![image-20220223173912593](https://user-images.githubusercontent.com/70505378/155285595-986dc563-023f-4712-adcd-7d529dcc3814.png)



**Knowledge Distillation**

**Knowledge distillation**에 대해서는 잘 알지 못하시는 분들이 조금 더 있을거라 생각합니다. Knowledge distillation은 **큰 모델의 학습 정보를 작은 모델에 전달(이식)하는 것**을 말합니다. 이를 **teacher-student learning**이라고 하기도 합니다. 

![image-20220311135847802](https://user-images.githubusercontent.com/70505378/157833166-adc64016-885c-4698-92f9-062b1f7427c1.png)

Knowledge distillation의 목적은 크게 두 가지입니다. 

1. Model compression (Mimicking what a larger model knows)
2. Pseudo-labeling (Generating pseudo-labels for an unlabeled dataset)

Knowledge distillation은 모델의 가중치를 그대로 가져오는 형태로 사용할 수 없기 때문에, teacher-student network structure라는 특별한 네트워크 형태를 사용합니다. 

![image-20220311140033117](https://user-images.githubusercontent.com/70505378/157833169-66e52d3e-c42a-4487-b5dd-b3c5d6487834.png)

Student Model은 Teacher Model의 예측 값과 동일하게 예측하도록 훈련됩니다. 따라서 이는 비지도 학습(unsupervised learning) 과정입니다. 

만약 라벨링된 데이터가 사용 가능하다면, 이를 학습에 활용할 수 있습니다. 이 때 모델의 구조는 아래와 같이 변경되어야 합니다. 

![image-20220311140944762](https://user-images.githubusercontent.com/70505378/157833171-f9466003-dc98-4167-a853-4b29d0487422.png)

**Distillation Loss**는 **student model이 teacher model의 예측값과 유사하게 예측하도록 학습**되게 합니다. Soft label과 Soft prediction (T=t) 사이의 KL divergence 값을 사용합니다. 

**Student Loss**는 **정답(ground truth)과 유사하게 예측하도록 학습**되게 합니다. Soft prediction(T=1)과 ground truth 사이의 CrossEntropy 값을 사용합니다. 

<br>

Hard label과 Soft label, Softmax with temperature가 무엇인지 보겠습니다. 

**Hard label**은 원-핫 벡터 형태의 라벨로, 일반적으로 데이터셋에서 사용하는 라벨입니다. 

반대로 **soft label**은 모델의 최종 출력으로 나오는 형태로, 각각의 클래스에 해당할 확률을 나타내죠. 

![image-20220311141556756](https://user-images.githubusercontent.com/70505378/157833176-02e403c6-f160-4206-9152-23d0883cafb3.png)

위 구조에서 soft prediction을 만들어내는 **Softmax (T=t)**는 일반 소프트맥스 연산에 최종 확률을 scaling해주는 temperature hyperparameter가 추가된 형태입니다. 

![image-20220311141846242](https://user-images.githubusercontent.com/70505378/157833178-61a18697-1f7f-4e22-a0a5-0abdeb1f360d.png)

위와 같이 T를 키우게 되면 큰 확률값과 작은 확률값 사이 difference가 작아져서 다양한 분포를 생성할 수 있도록 학습됩니다. 반대로, T값을 줄인다면 큰 값을 선택할 확률이 더더욱 높아지겠죠. 다양한 분포를 생성한다는 것은 모델의 일반화 성능을 높일 수도 있지만, 정답을 맞힐 확률을 그만큼 감소시킬 수도 있습니다. 

이렇게 labeling된 data가 있는 경우의 teacher-student network의 학습은 'Teacher model의 예측값과 Student model의 예측값 사이 KL Divergence 값인 Distillation loss'와 'Student model의 예측값과 실제 ground truth 값 사이 CrossEntropy 값인 Student loss' 의 weighted sum을 이용해 학습이 진행됩니다. 









<br>

### Leveraging unlabeled dataset for training

우리 세상에는 labeling된 data보다 unlabeled data가 훨씬 더 많습니다. 이번 섹션에서는 모델 학습에 어떻게 unlabeled data를 활용할 수 있는지에 대해 알아보겠습니다. 

**Semi-supervised learning**

Semi-supervised learning의 개념은 간단합니다. **Labeled dataset**과 **Pseudo-labeled dataset**(pretrained model의 예측으로 labeling된 unlabeled dataset)을 이용하여 모델을 학습시키는 것입니다. 

![image-20220311142951092](https://user-images.githubusercontent.com/70505378/157833180-cac932c5-6ec1-45c5-ac01-745825d3409e.png)

**Self-training**

Self-training은 앞서 살펴본 data efficient training 기법들인 Augmentation, Teacher-Student network, Semi-supervised learning을 모두 활용하여 모델을 학습시키는 것입니다. 

2019년 발표된 논문인 [SOTA ImageNet classification]에서는 기존에 가장 좋은 성능을 보이던 EfficientNet 모델에 **Noisy Student Training** 기법을 적용시켜 모델의 성능을 더욱 끌어올렸다고 발표했습니다. 

![image-20220311143418784](https://user-images.githubusercontent.com/70505378/157833185-5d208846-babd-4950-96a8-bcb2a1202a58.png)

네트워크의 구조는 아래와 같습니다.

![image-20220311143720065](https://user-images.githubusercontent.com/70505378/157833191-7ba28e13-0b11-4aee-904f-1f187e09de7d.png)

학습 과정을 간단히 정리하면 아래와 같습니다. 

1. Initial Teacher model을 1M 개의 ImageNet dataset으로 학습시킵니다. 
2. 학습된 Teacher model을 300M 개의 unlabeled dataset에 대해 예측을 수행해 300M 개의 pseudo-labeled dataset을 생성합니다. 
3. 1M 개의 ImageNEt dataset, 300M 개의 pseudo-labeled dataset과 random augmentation을 사용하여 student model을 학습시킵니다. 
4. 3에서 학습된 student model을 다음 iteration의 teacher model로 사용하고, 다음 student model을 학습시킵니다. 이 때 매 iteration마다 teacher model의 크기는 커집니다. 
5. 2~4 단계의 과정을 매 iteration마다 반복합니다. 

<br>

<br>

## 실습) ResNet34 구현하기

이번 포스팅에 해당하는 실습은 ResNet34 구현입니다. 

모델을 구현할 때는 작은 블록부터 큰 블록 순으로 차례로 구현하여 쌓아 올리는 것이 좋습니다. 

![image-20220311172618602](https://user-images.githubusercontent.com/70505378/157833192-07fc857c-b7e7-456d-a0d6-44652101aefb.png)

**ConvBlock**

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()

        self.layers = []

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.layers.append(torch.nn.Conv2d(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding))
        self.layers.append(torch.nn.BatchNorm2d(num_features=self.out_channels))
        self.layers.append(torch.nn.ReLU())

        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        x = self.net(x)
        return x
```

**ResBlock**

![image-20220311172831162](https://user-images.githubusercontent.com/70505378/157833197-1dafdaa4-0d89-4dbe-97ca-d624197b7fb8.png)

```python
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, downsample=1):
        super().__init__()

        self.layers = []

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.downsample = downsample

        self.layers.append(ConvBlock(self.in_channels, self.out_channels, self.kernel_size, self.stride, self.padding)) 
        self.layers.append(ConvBlock(self.out_channels, self.out_channels, self.kernel_size, 1, self.padding)) # 2번째 ConvBlock의 stride는 항상 1!!

        self.shortcut = nn.Conv2d(self.in_channels, self.out_channels, 1, stride=self.downsample, padding=0)

        self.net = nn.Sequential(*self.layers)

    def resblk(self, x):
        return self.shortcut(x)

    def forward(self, x):
        identity = x

        x = self.net(x) 
        identity = self.shortcut(identity)
        out = F.relu(x+identity)

        return out
```

**ResNet**

```python
class ResNet(nn.Module):
    def __init__(self, in_channels, out_channels, nker=64, nblk=[3,4,6,3]):
        super(ResNet, self).__init__()

        self.enc = ConvBlock(in_channels, nker, kernel_size=7, stride=2, padding=1) # output: (64, 112, 112)
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # output: (64, 56, 56)

        ### 각 conv_x 층마다 첫 layer의 stride=2 -> feature map과 identity(x)의 형상을 어떻게 맞추지? -> 채널 방향은 identity에 1x1 conv, (h, w) 방향은?? -> stride=2??
        # 1. pooling 연산
        # 2. stride = 2
        # 3. 학습 가능한 matrix
        ### nn.Sequential -> *args -> self.sequential(x) / nn.ModuleList -> list -> for module in modulelist: x = layer(x)
        self.conv2_x = nn.ModuleList([self.max_pool]+[ResBlock(64, 64, kernel_size=3, padding=1) for _ in range(3)]) # output: (64, 56, 56)
        self.conv3_x = nn.ModuleList([ResBlock(64,128,kernel_size=3, stride=2, padding=1, downsample=2)] + [ResBlock(128,128,kernel_size=3, stride=1, padding=1) for _ in range(3)]) # output: (128, 28, 28)
        self.conv4_x = nn.ModuleList([ResBlock(128, 256, kernel_size=3, stride=2, padding=1, downsample=2)]+[ResBlock(256, 256, kernel_size=3, stride=1, padding=1) for _ in range(5)]) # output: (256, 14, 14)
        self.conv5_x = nn.ModuleList([ResBlock(256, 512, kernel_size=3, stride=2, padding=1, downsample=2)]+[ResBlock(512, 512, kernel_size=3, stride=1, padding=1) for _ in range(2)]) # output: (512, 7, 7)
        self.avg_pool = nn.AvgPool2d(kernel_size=7) # (512, 1,1) -> 512

        self.fc = nn.Linear(nker*2*2*2, out_channels) # 512 -> out_channels


    def forward(self, x):
        x = self.enc(x)
        for conv in self.conv2_x:
            x = conv.forward(x)
        for conv in self.conv3_x:
            x = conv.forward(x)
        for conv in self.conv4_x:
            x = conv.forward(x)
        for conv in self.conv5_x:
            x = conv.forward(x)
        x = torch.squeeze(self.avg_pool(x))

        out = self.fc(x)

        return out
```

맨 위 모델 구조 표에는 stride나 shortcut 구조에 대한 구체적인 가이드는 제시되어 있지 않습니다. 따라서 모델을 직접 구현할 때는 각 부분에 대한 충분한 이해가 필요하고, 그에 따라 구현할 수 있는 프로그래밍 역량이 요구됩니다. 

### 배운 점

* 모델을 구현할 때는 작은 블록 -> 큰 블록 순으로 구현하며 쌓아올린다. 
* 모듈 내의 layers는 nn.module을 상속받은 상태여야 한다. 
  * nn.module을 상속받지 않으면 model weight를 cuda에 올릴 수 없다. 
  * nn.Sequential(*args) 나 nn.ModuleList(list)를 사용한다. 
    * forward 메서드 정의 시, nn.Sequential은 한 번에 `x = self.sequential(x)`와 같이 수행하고, nn.ModuleList는 각 layer에 대해 `for module in modulelist: x = module(x)`와 같이 수행한다. 
* 총 5개의 conv part 의 맨 앞 conv block에는 stride를 2로 가지는 layer가 온다. 
* conv block을 거친 x와 연산을 거치지 않은 identity를 더할 때는 identity의 차원을 x에 맞춰줘야 한다. 
  * 채널 방향 차원 조절을 위해 1x1 convolution을 사용한다. 
  * (h, w) 방향 차원 조절(x(1/2))을 위해 다음의 세 가지 방법을 고려할 수 있다. 
    * Pooling 연산
    * 1x1 convolution 연산에서 stride=2로 지정
    * 크기를 1/2로 줄여줄 수 있는 trainable matrix 정의
  * 세 가지 방법 중 구현 시에는 두번째 방법(stride=2)을 주로 사용한다. 









<br>

<br>

# 참고 자료

* Data Augmentation
  * Yun et al., CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features, ICCV 2019
  * Cubuk et al., Randaugment: Practical automated data augmentation with a reduced search space, CVPRW 2020  
  * [CutMix paper](https://arxiv.org/abs/1905.04899)
* Leveraging pre-trained information
  * Ahmed et al., Fusion of local and global features for effective image extraction, Applied Intelligence 2017
  * Oquab et al., Learning and Transferring Mid-Level Image Representations using Convolutional Neural Networks,
    CVPR 2015
  * Hinton et al., Distilling the Knowledge in a Neural Network, NIPS deep learning workshop 2015
  * Li & Hoiem, Learning without Forgetting, TPAMI 2018  
* Leveraging unlabeled dataset for training
  * Lee, Pseudo-label : The simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks, ICML
    Workshop 2013
  * Xie et al., Self-training with Noisy Student improves ImageNet classification, CVPR 2020  









<br>
