---

layout: single
title: "[AITech] 20220208 - CNN Applications"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['SementicSegmentation', 'ObjectDetection']
---



<br>

## 학습 내용

이번 포스팅에서는 CNN 구조의 적용 및 확장 형태인 **의미론적 분할**과 **객체 탐지** 네트워크의 아이디어에 대해 간단히 살펴보겠습니다 😊

### Sementic Segmentation

`Sementic Segmentation(의미론적 분할)`은 분류 네트워크가 각 이미지에 대해 추론을 수행했다면, 이를 각 픽셀 별로 추론을 수행하는 것으로 바꾼 것입니다. 

예를 들면 다음과 같이 말이죠. 

![image-20220208151328641](https://user-images.githubusercontent.com/70505378/152952763-d2969458-e7f5-4599-90f7-84bf18bd1e4b.png)

그렇다면 어떻게 이를 수행할 수 있을까요?

물론 많은 발전된 아이디어들이 존재하지만, 가장 기본적인 아이디어는 **마지막 분류기를 Fully Connected layer에서 Convolution layer로 바꾸는 것**입니다. 그리고 이를 **Convolutionalization**이라고 합니다. 

![image-20220208151556370](https://user-images.githubusercontent.com/70505378/152952766-b5a5c37c-bd3d-40da-bf6e-e77eef743958.png)

위의 두 형태의 네트워크는 정확히 2,560(4x4x16x10)개의 똑같은 개수의 파라미터를 가지고, 같은 형태의 output을 만들어냅니다. 

하지만 전체가 convolution layer로 구성되어 있는 FCN(Fully Convolutional Network)을 사용하게 되면 기존의 dense 층이 각 클래스에 속할 확률만을 출력했던 것과 달리 **spatial feature를 보존하며 heapmap 형태의 정보를 얻을 수 있습니다.** 또한, 이로부터 얻는 이점은 dense 층을 사용할 경우 입력 크기가 다르다면 사용할 수 없지만 FCN의 경우 입력 크기가 달라져도 **뒤로 쌓는 feature map의 개수만이 달라지게 됩니다.**

![image-20220208152132807](https://user-images.githubusercontent.com/70505378/152952767-4da5f4d8-5b46-4c04-bcaf-628ceea51362.png)

놓치지 말아야 할 것이 있습니다! 입력 이미지가 CNN을 통과하게 되면 그 크기가 작아집니다. 그런데 이 결과를 이용해 원본 이미지에 대해 semantic segmentation을 수행하려면 당연히 **최종 결과를 원본 이미지 크기로 복원하는 과정(Upsampling)**이 필요합니다. 

이 upsampling을 위해 여러 방법을 사용할 수 있는데요, 여기서는 FCN에서 사용하는 **Deconvolution**에 대해 설명합니다. 

Deconvolution은 convolution 연산의 역연산이지만, **엄밀히 말하면 결과적으로 역연산이라고는 볼 수 없습니다(전치 합성곱, Transposed convolution이라고 합니다).** 선형 결합된 정보들의 결과물에서 다시 그 선형 결합된 정보들을 똑같이 얻어내는 것은 불가능하기 때문이죠. 하지만, 이를 역연산으로 생각하면 쉬운 이유는 어찌됐든 연산 이전의 크기로 피쳐맵을 되돌려놓고, 코딩 시에 파라미터를 전달할 때 대칭적으로 전달하면 되기 때문입니다. 

```python
input = torch.randn(1, 16, 12, 12)

downsample = nn.Conv2d(16, 8, 3, stride=2, padding=1)
upsample = nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1)

h = downsample(input)
print(h.size())
# torch.Size([1, 8, 6, 6])
output = upsample(h, output_size=input.size())
print(output.size())
# torch.Size([1, 16, 12, 12])
```

아래는 저의 다른 포스팅 중 **전치 합성곱**에 대한 설명 부분을 발췌한 내용입니다. 

> * **전치 합성곱(디컨볼루션)**
>
> 합성곱 계층에서 출력 형상은 다음과 같이 계산할 수 있다. 
>
> ![KakaoTalk_20210807_154656167](https://user-images.githubusercontent.com/70505378/128715869-b939d19e-330f-4439-b959-ecdb4773e8ea.png)
>
> 합성곱의 공간 변환을 역으로 수행하는 계층을 개발해야 한다고 가정하다. 즉, 형상이 (H<sub>o</sub>, W<sub>o</sub>, N)인 특징맵과 파라미터 k, D, N, p, s가 주어졌을 때 형상이 (H, W, D)인 텐서를 찾는 것이다. 
>
> 이는 위 수식을 H, W에 대해 정리함으로써 간단히 얻을 수 있다! 
>
> 이것이 **전치합성곱**이 정의된 방식이다. 
>
> 이 계층은 kxkxKxN(FH=FW=k)의 커널 스택을 이용해 H<sub>o</sub>xW<sub>o</sub>xN 텐서에 합성곱을 적용해 HxWxD의 맵으로 변환한다. 이를 달성하기 위해 먼저 입력 텐서는 팽창(dilation) 과정을 거쳐야 한다. 
>
> * 비율 d에 의해 정의된 팽창 연산은 입력 텐서의 행과 열의 쌍 사이에 (개별적으로) **d-1개의 0으로 채워진 행과 열을 추가**하는 것으로 구성된다. 전치 합성곱에서 **d=s** 로 설정된다. 
> * 이 재표본 추출 후에 텐서는 **p' = k-p-1 에 의해 패딩**된다. 
> * 그런 다음 보폭 **s'=1**을 사용해 실제 텐서를 해당 계층의 필터와 합성곱을 수행하고 그 결과 **HxWxD의 결과**를 얻게 된다. 
>
> ![KakaoTalk_20210819_174615589](https://user-images.githubusercontent.com/70505378/130202384-c48d65e6-c0a5-4e91-861b-891f0d9a6c8f.png)
>
> ![KakaoTalk_20210820_151725578](https://user-images.githubusercontent.com/70505378/130202449-d0f80d39-66f5-45ce-a414-a1dc5a44f92d.png)
>
> 프로세스를 이해하기 어렵다면, 전치 합성곱 계층은 일반적으로 표준 합성곱 계층을 반전시켜 특징 맵의 콘텐츠와 훈련 가능한 필터 사이의 합성곱을 통해 특징 맵의 공간 차원을 증가시키기 위해 사용된다는 점만 기억해도 충분하다. 

'최대 언풀링, 평균 언풀링, 아트루스 합성곱' 등의 다른 역연산에 대해 궁금하신 분들은 [저의 다른 포스팅](https://wowo0709.github.io/ai/computervision/Computer-Vision-6(1).-%EC%9D%B8%EC%BD%94%EB%8D%94-%EB%94%94%EC%BD%94%EB%8D%94%EB%A1%9C-%EC%9D%B4%EB%AF%B8%EC%A7%80-%EB%B3%80%ED%99%98/) ([AI - Computer Vision - 6(1). 인코더-디코더로 이미지 변환]에 있습니다)을 참고하는 것을 추천드립니다. 

* FCN Architecture

![KakaoTalk_20210820_164748900](https://user-images.githubusercontent.com/70505378/130202602-744e8ae6-cead-42e3-bb5e-edbc95cbb083.png)







<br>

### Object Detection

#### R-CNN

객체 탐지 부분에서 처음으로 소개된 네트워크는 R-CNN 계열의 네트워크들입니다. R-CNN, Fast R-CNN, Faster R-CNN 순으로 발전해왔고 유사하게 SPPNet도 있습니다. 

![image-20220208175442234](https://user-images.githubusercontent.com/70505378/152952758-0c06a06e-257c-4583-8b4b-9762c304bdbd.png)

R-CNN 계열의 detection model들은 2-stage detector이며, 이는 region proposal(영역 제안)과 classification(분류)을 따로 수행하기 때문에 그렇습니다. R-CNN 모델들의 아이디어는 쉽지 않기 때문에, 제가 정리해놓은 논문 리뷰 포스팅을 보는 것을 추천드립니다. 

* [RCNN 시리즈 논문 분석](https://wowo0709.github.io/ai/computervision/Computer-Vision-RCNN-%EC%8B%9C%EB%A6%AC%EC%A6%88-%EB%85%BC%EB%AC%B8-%EB%B6%84%EC%84%9D/)
  * [AI - Computer Vision - RCNN 시리즈 논문 분석]에 있습니다. 

#### YOLO

YOLO는 R-CNN과 반대로 region proposal과 classification을 동시에 수행하는 1-stage detector 모델입니다. 이 덕분에 기본 45fps ~ 최대 155fps의 추론 속도를 낼 수 있습니다. 

![image-20220208173303961](https://user-images.githubusercontent.com/70505378/152952768-d83e07c8-4297-4083-b81f-03ceb4271bc1.png)



아래 그림은 YOLO가 detection을 수행하는 과정을 간단히 흐름에 따라 정리한 것입니다. 

![image-20220208174633482](https://user-images.githubusercontent.com/70505378/152952771-d80f5979-3353-4aaf-a086-3d17553474c2.png)







<br>

<br>

## 참고 자료

* 

















<br>
