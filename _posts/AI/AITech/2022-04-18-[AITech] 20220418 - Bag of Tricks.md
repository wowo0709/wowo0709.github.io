---
layout: single
title: "[AITech][Data Annotation] 20220418 - Bag of Tricks"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Synthetic Data', 'Data Augmentation', 'Multi-scale training']
---



<br>

_**본 포스팅은 Upstage의 '이활석' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Bag of Tricks

이번 포스팅에서는 text detection 뿐 아니라 다양한 분야에서 사용할 수 있는 데이터 관련한 trick들을 소개합니다. 

## Synthetic Data

**합성 데이터**는 실제 데이터를 충분히 모으기 어렵거나, pretraining set이 필요한 경우 매우 유용하게 사용할 수 있습니다. 

### SynthText

`SynthText`는 가장 대표적인 글자 검출 합성 데이터셋입니다. CVPR 2016 논문(Synthetic Data of Text Localization in Natural Image)과 함께 800K 크기의 데이터셋과 글자 이미지 합성 코드가 공개되어 직접 데이터셋을 사용하고 생성할 수도 있습니다. 

![image-20220423231429882](https://user-images.githubusercontent.com/70505378/164912352-9b12551e-4aca-4c96-a73c-ce5bf36a2b9d.png)

SynthText는 실제 배경 이미지에 글자를 합성하는 방식을 사용합니다. 이때 배경 이미지는 단 하나의 글자도 포함하지 않은 이미지만을 사용합니다. 

그런데 이미지에 무작위로 text를 배치하게 되면 실제 데이터와 매우 동떨어진 데이터를 얻게 됩니다. 따라서 실제와 같은 합성 이미지를 생성하기 위해 depth estimation을 통해 text region을 구분하여 하나의 글자 영역은 하나의 text region에만 속하도록 글자를 합성합니다. 

![image-20220423231414822](https://user-images.githubusercontent.com/70505378/164912351-2f9ac20d-9dfe-4fdc-a444-c0c2d3d4b7e3.png)

### SynthText3D

SynthText3D는 3D 가상 엔진을 사용하여 제작한 합성 데이터셋입니다. 제작 과정은 아래와 같습니다. 

1. Word, illumination, camera view를 결정
2. 정해진 view에서 합성할 영역을 결정
3. 합성할 글자를 결정하여 합성할 영역에 배치
4. Unreal 엔진을 이용하여 이미지 랜더링

![image-20220423231914330](https://user-images.githubusercontent.com/70505378/164912353-f397ca15-c2dd-44df-b1bb-aedc0a6b9d35.png)

이렇게 3D 엔진을 사용해 합성 데이터셋을 만들면 기존 2D 합성 데이터셋 대비 장점이 있습니다. 

* Geometry 정보를 이용하여 글자의 위치를 더 정확하고 사실적으로 합성
* Illumination과 view의 조정을 통해 같은 장면에서도 다양한 이미지 생성
* 2D 이미지에 비해 데이터셋 규모 대비 사람의 수작업이 덜 함
* 가려짐, 그늘짐 등 특수한 경우의 이미지도 생성 가능

### UnrealText

마지막으로 소개할 합성 데이터셋인 `UnrealText` 또한 3D 가상 엔진을 이용하여 생성되었습니다. 

UnrealText와 SynthText3D의 가장 큰 차이점은 camera view의 설정입니다. SynthText3D에서는 부자연스러운 view를 방지하기 위해 사람이 미리 정해놓은 view를 사용했다면 UnrealText에서는 자연스러운 view를 찾는 자동화된 방식을 고안하여 이 목적을 달성했습니다. 

또한 view를 찾는 방식을 자동화함으로써 UnrealText에서는 보다 다양하고 많은 데이터셋을 생성할 수 있었습니다. 

![image-20220423232549357](https://user-images.githubusercontent.com/70505378/164912355-df79bf9c-6dc1-4eba-9f43-9118309f5c40.png)

### How to Use

합성 데이터를 사용하는 가장 일반적인 방법은 pretraining set으로 사용하는 것입니다. 

예를 들어 Image classification에서 ImageNet으로 pretrained된 모델을 fine-tuning할 때 합성 데이터로 한 번 더 pretrain 한 뒤 target data로 fine-tuning하여 더 높은 성능을 기대할 수 있습니다. 

![image-20220423232747777](https://user-images.githubusercontent.com/70505378/164912357-f3ff298c-b37c-46f6-957c-d81322fedab3.png)

실제로 UnrealText 논문에서는 real data에 더해 본인들의 UnrealText 데이터를 함께 사용했을 때 가장 높은 성능을 기록했다고 주장했습니다. 

![image-20220423232900374](https://user-images.githubusercontent.com/70505378/164912358-64968618-ecaa-4f1b-86d6-625d15fb9bd9.png)

또 다른 사용 방법으로는 character-level detection을 요구하는 모델에 사용할 수 있습니다. 

이러한 모델로는 CRAFT나 TextFuseNet 모델이 대표적이며, real dataset은 대부분 word-level annotation만 포함하기 때문에 character level annotation이 필요합니다. 하지만 character level annotation을 직접 만드는 것은 매우 힘들고 어려운 일입니다. 

이러한 경우에 weakly supervised learning 방식을 사용하여 합성 데이터를 활용할 수 있습니다. 

1. 글자 단위 annotation을 제공하는 합성 데이터로 pretraining된 모델을 생성
2. Pretrained 모델을 real dataset에 적용해서 글자 단위 pseudo annotation 확보
3. 글자 단위 pseudo annotation과 실제 word annotation을 함께 사용하여 target model을 학습

<br>

여기서 우리가 알아가야 할 것은 합성 데이터를 사용함으로써 모델의 성능 향상을 기대할 수 있지만, task와 real data, model에 따라 실제로 가장 좋은 성능을 보이는 합성 데이터셋은 다를 수 있으니 실험을 통해 가장 나은 합성 데이터를 찾는 과정이 필요하다는 것입니다. 

## Data Augmentation

Data Augmentation은 불균형하고 제한된 분포(다양성)를 가지는 데이터셋에 적절한 변형을 가함으로써 균등하고 다양한 데이터셋으로 만들어주는 기법이라고 할 수 있습니다. 

Image Data Augmentation은 크게 **Geometric Transformation**, **Style Transformation**, **Others**로 구분할 수 있습니다. 

* Geometric Transformation

  * Global level의 변화를 주는 변형
  * Random crop, resize, rotate, flip, shear 등

  ![image-20220423234500238](https://user-images.githubusercontent.com/70505378/164912359-f06a4cd6-38b8-47b7-8369-474c8aa503de.png)

* Style Transformation

  * Local level의 변화를 주는 변형
  * Color Jitter, channel shuffle, noise filter 등

  ![image-20220423234510275](https://user-images.githubusercontent.com/70505378/164912361-c22178de-a9cd-41f8-876b-459494daa54a.png)

* Others

  * Geometric/Style Transformation에 속하지 않는 변형
  * Grid distortion, Elastic transformation, Cutout/Cutmix, Mosaic 등

  ![image-20220423234547125](https://user-images.githubusercontent.com/70505378/164912363-3ce82077-e4f7-498e-86d7-5194b395a1c8.png)

<br>

이러한 augmentation을 적용할 때 중요한 것은 **task에 적절한 변형을 가해야 한다**는 것입니다. 

그리고 text detection의 경우 아래와 같은 변형들은 적절하지 않는 변형에 해당합니다. 

* 글자를 포함하지 않는 경우
* 글자가 잘려서 일부만 나타나는 경우

이러한 변형들은 사용하지 않거나, 또는 rule을 도입하여 적절한 이미지의 형태로 정제해주는 과정으로 극복할 수 있습니다. 

Data augmentation에서 중요한 것은 도메인의 특징에 따라 다양한 문제가 발생할 수 있고, 이에 따라 적절한 augmentation의 종류가 다르다는 것입니다. 

다만 처음에는 일반적인 방법으로 시작해서 상황에 맞게 특화해나가는 방식으로 적절한 augmentation을 탐색하는 방법이 좋습니다. 

* 실제로 모델에 입력되는 이미지 관찰
* Augmentation을 가한 데이터 입력 후 모델의 성능 모니터링
* Loss가 크게 발생하는 영역들을 분석하여 rule을 업데이트











<br>

## Multi-Scale Training & Inference

이미지에서 글자는 매우 다양한 크기로 나타나며, 이러한 scale variation은 난이도를 높이는 주요 원인입니다. 

* 작은 글자들: miss detection
* 큰 글자들: broken/partial detection

이를 위해 이미지를 다양한 크기로 바꿔가면서 입력해주는 것이 좋은 방법일 수 있습니다. 

* Crop & resize augmentation
* Image pyramid

하지만 위 방법들의 경우 원래 작은 글자가 더욱 작아지거나, 원래 큰 글자가 더욱 커지는 문제점을 발생할 수 있고, 이는 곧 성능의 저하로 이어질 수 았습니다. 

![image-20220424000122452](https://user-images.githubusercontent.com/70505378/164912366-87f4ea3d-4249-4743-b0b5-1205906bd112.png)

이에 대한 대응 방법으로 `SNIP(Scale Normalization for Image Pyramid)`이라는 학습법이 제안되었습니다. 해당 방법은 scale augmentation을 적용하되 개체의 크기에 대한 범위를 정해놓고 해당 범위를 벗어나는 개체는 학습에서 무시하는 방법입니다. 

![image-20220424000106858](https://user-images.githubusercontent.com/70505378/164912364-14f97ba0-eec2-4eac-9a99-8db493dd4efe.png)

SNIP 방법은 inference 시에도 사용할 수 있는데, 이 때에는 multi-scale inference를 적용하여 검출 결과 중 크기가 적정 범위에 있는 bbox 영역들에 대해서만 NMS를 적용하여 최종 에측을 수행합니다. 

실제로 SNIP 방식은 일반적인 multi-scale training 방식에 비해 다양한 크기의 객체 검출에서 모두 더 나은 성능을 보여줍니다. 

![image-20220424000434489](https://user-images.githubusercontent.com/70505378/164912367-1cce1f31-7d9f-45de-a883-03d49e012190.png)

<br>

추론 시에는 이미지를 여러 크기로 조절하여 모델에 입력하는 multi-scale inference 방식을 사용할 수 있습니다. 

하지만 일반적인 multi-scale inference 방식은 계산량이 많아지기 때문에 비효율적이고, 여러 크기에서의 False Positive가 누적될 수 있습니다. 

`Adaptive Scaling` 방식은 대신에 이미지에서 글자가 있을 만한 영역만 글자가 적정 크기로 나타나도록 크기를 조정하여 모델에 입력할 것을 제안합니다. 

이를 위해서 모델에 이미지를 두 차례 입력합니다. 

1. 먼저 글자의 위치와 크기를 대략적으로 예측하고, 
2. 그 결과를 기반으로 재구성한 이미지를 다시 입력해서 최종 결과를 얻는다. 

![image-20220424000927663](https://user-images.githubusercontent.com/70505378/164912348-a2253d8e-cae8-4314-9e82-ec144b0661c4.png)

대략적인 과정은 아래와 같습니다. 

1. Downsizing으로 축소된 버전의 이미지를 만든다. 
2. 축소된 이미지를 모델에 입력해서 scale mask와 seg mask를 예측한다. 
3. 이미지를 재구성(객체 영역들만 이어붙임)하여 'Canonical knapsack'을 만든다. 
4. 'Canonical knapsack'을 모델에 입력, 통상적인 text detection을 수행한다. 

이를 통해 훨씬 효율적인 multi-scale inference가 가능해집니다. 

* 배경 영역에 대한 계산을 하지 않아 경제적이다. 
* 글자들의 크기가 적정 크기로 통일되어 scale variation으로 인한 성능 저하가 없다. 

<br>

<br>

# 참고 자료

* 
