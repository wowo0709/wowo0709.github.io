---
layout: single
title: "[AITech][CV] 20220308 - Part 7) Multi-modal Learning"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 POSTECH '오태현' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# Multi-modal Learning

이번 포스팅의 내용은 `Multi-modal Learning`입니다. Multi-modal learning이란 image, text, audio 등의 다른 modality의 데이터들을 같이 학습에 이용하는 것을 말합니다. 

다른 modality의 데이터를 어떻게 활용할 것인가에 따라 모델을 구성하는 방식과 모델을 학습시키는 방법들이 어떻게 달라지는 지에 특히 집중하여, 본인이 향후 풀어내고자 하는 문제에 어떻게 적용할 수 있을지에 대해 생각하며 포스팅을 읽어보는 것을 추천드립니다. 

## Overview of multi-modal learning

앞에서 말했듯이, Multi-modal learning은 여러 다른 modality의 data들을 함께 모델 학습에 사용하는 것입니다. 우리가 공부하고 있는 분야가 CV인 만큼, Visual data와 함께 text data, audio data 등이 함께 사용될 수 있습니다. 

실셰게의 감각들은 복합적으로 작용하기 때문에, 인간과 비슷한 모델을 만들기 위해서는 multi-modal learning이 중요한 역할을 할 수 있다고 말할 수 있습니다. 다만, multi-modal learning에는 여러 어려움들이 존재하기도 합니다. 

* 서로 다른 modality의 데이터들을 그 표현 방식이 다르다. 
* 하나의 modality의 데이터는 다른 modality의 데이터와 항상 1 대 1 matching 되지는 않는다. 
* 여러 modality의 데이터를 모델에 넣어줘도, 모델이 각 modality의 데이터를 균형있게 잘 사용해서 잘 학습할 지는 미지수이다. 

위 어려움들에도 불구하고, Multi-modal learning은 더 나은 모델을 만들고, 우리가 풀 수 있는 문제를 확장하기에 필수적인 학습 방법입니다. 

서로 다른 modality의 데이터를 대응시키는 **matching**, 하나의 modality의 데이터를 다른 modality의 데이터로 변환하는 **translating**, 특정  modality의 모델을 더욱 잘 학습시키기 위해 다른 modality의 데이터를 함께 사용하는 **referencing** 등 우리는 Multi-modal learning으로부터 많은 것을 얻을 수 있습니다. 

![image-20220315175559423](https://user-images.githubusercontent.com/70505378/158545437-979b572f-31df-46dd-8bbc-1f2a5e1be4b3.png)







<br>

## Multi-modal tasks(1) - Visual data & Text

### Text Representation

Text data를 사용할 때는 word(단어)를 **Embedding vector** 형태로 표현합니다. 이를 수행해주는 모델을 **word2vec** 모델이라고 하고, skip-gram(또는 CBOW) 방법을 사용해서 중심 단어와 주변 단어와의 분포를 학습하여 각 단어를 특정 길이의 벡터로 임베딩 해줍니다. 

Word embedding된 벡터는 다음과 같은 특성들을 가지고 있습니다. 

* 비슷한 단어들은 공간 상에서 가까이 위치한다. 
* 관계가 비슷한 단어 쌍들은 비슷한 벡터를 갖는다. 

![image-20220315180923953](https://user-images.githubusercontent.com/70505378/158545441-39fddd18-903b-46a7-8d60-ffb026126f24.png)

자연어 처리에 대한 개괄적인 내용은 [이 포스팅](https://wowo0709.github.io/ai/deeplearning/DeepLearning-%EC%9E%90%EC%97%B0%EC%96%B4-%EC%B2%98%EB%A6%AC/)을 읽어보는 것을 추천드립니다. 





### Joint Embedding

`Joint Embedding`은 Matching task에서 사용되는 기법입니다. 

![image-20220315205011028](https://user-images.githubusercontent.com/70505378/158545445-d5967f0b-fd9b-4446-a76f-aa5ff1c36201.png)

대표적인 Matching task중 하나는 **Image tagging** task입니다. Image와 대응되는 text(또는 반대)를 찾는 것이죠. 

![image-20220315205744737](https://user-images.githubusercontent.com/70505378/158545454-f5935840-30ff-4e50-be6e-0b974c2373dd.png)

이때 사용되는 것이 Joint Embedding입니다. 다른 modality의 두 Unimodal model이 각각 고정된 크기의 feature vector를 만들어냅니다. Joint Embedding은 두 feature vector 간 관계를 잘 mapping할 수 있는 embedding space를 만들어 내도록 학습됩니다. 

![image-20220315205154592](https://user-images.githubusercontent.com/70505378/158545447-b5321203-81e1-4e23-853f-b5e4daba085e.png)

Matching task는 Image tagging 외에도 Food image-Recipe retrieval과 같이 서로 다른 modality의 데이터를 대응시켜 주는 task라면 그 활용 가능성이 매우 높습니다. 

학습된 joint embedding layer는 아래와 같이 서로 다른 modality의 데이터 간의 관계를 잘 나타내는 것을 볼 수 있습니다. 

![image-20220315205722527](https://user-images.githubusercontent.com/70505378/158545450-d82caecc-9a0f-4aa2-a1c5-bca7df3437ad.png)

### Cross modal translation

 이번에는 translation task에 대해 살펴보도록 하겠습니다. Translation task에는 대표적으로 Image captioning과 그 반대인 Text-to-Image translation이 있습니다. 

먼저 **Image captioning**을 보겠습니다. Image captioning은 주어진 이미지로부터 적절한 텍스트를 생성해 문장을 만들어주는 task입니다. 

![image-20220315211218512](https://user-images.githubusercontent.com/70505378/158545456-bf85295a-8b8e-42f8-a881-2acd55e8e585.png)

당연하게도, 입력 이미지를 처리할 image model과 출력 텍스트를 생성할 text model이 필요합니다. 모델의 전체 구조는 아래와 같은 형태를 보이며, 이를 **Show and tell**이라고 합니다. 

Show and tell 형태의 모델은 입력 이미지로부터 image model이 일정 크기의 feature vector를 생성하고, 이 feature vector를 이용해 text model이 문장을 생성 해냅니다. 

![image-20220315211424855](https://user-images.githubusercontent.com/70505378/158545458-ec19a8bc-31e0-4226-a896-f63cbdb28292.png)

여기서 한 단계 더 나아가, 이미지에 각 부분에 주목하며 텍스트를 만들어내는 방법인 **Show, attend, and tell**이라는 방법도 있습니다. 이 방법은 Attention 메커니즘을 이용하여 이미지의 각 부분에 알맞은 텍스트를 생성해냅니다. 

![image-20220315211628094](https://user-images.githubusercontent.com/70505378/158545459-55e1ec68-f05a-4868-a942-12ed4caa195f.png)

조금 더 자세히 알아보겠습니다. 

1. 입력 이미지는 image model을 거쳐 공간 정보가 남아있는 feature map 형태가 됩니다 (마지막 FC layer나 Softmax는 생략합니다).
2. Feature map을 text model에 전달해 initial hidden state `h0`을 생성합니다. `h0`로부터 이미지의 어떤 부분에 주의를 해야 할 지에 대한 정보를 갖는 spatial attention `s1`를 생성합니다. 
3. 앞서 생성한 feature map과 spatial attention `s1`과의 inner product를 계산해 하나의 fixed dimensional vector `z1`을 구합니다. 
4. Feature map에서 주의를 기울이고 있는 공간 정보가 들어있는 `z1`과 start word token `y1`을 입력으로 하여 hidden state `h1`을 생성하고, 다음 단어 생성을 위한 spatial attention `s2`와 출력 단어 `d1`을 생성합니다. 
5. 이후부터(n >= 2)는 3~4 과정을 반복하여 문장을 만들어냅니다. 

![image-20220315213831020](https://user-images.githubusercontent.com/70505378/158545461-35f32394-165b-443d-a0a8-403c510bd52d.png)

문장을 생성할 때는 Beam Search라는 탐색 알고리즘을 사용하는데, 이에 대해서는 설명하지 않겠습니다. 

<br>

이번에는 **Text-to-image translation**에 대해 간단히 알아보겠습니다. 기본적으로 Text로부터 생성될 수 있는 Image의 종류는 너무나 많기 때문에, text-to-image translation task에서는 generative model을 사용합니다. 

![image-20220315214021481](https://user-images.githubusercontent.com/70505378/158545466-546119e8-66bb-4ca0-a814-225a3d1cb1b3.png)

* Generator Network
  * Text를 fixed dimensional feature vector로 embedding합니다. 
  * feature vector에 random gaussian noise vector를 concat합니다. 
  * concat한 vector는 decoder를 거쳐 새로운 image가 생성되게 됩니다. 
* Discriminator Network
  * Generator가 생성한 image를 encoder를 거쳐 feature vector로 생성합니다. 
  * Image feature vector에 앞서 입력에서 생성한 text feature vector를 concat합니다. 
  * concat한 vector를 다시 encoding하고, 이 encoding된 정보가 real data와 유사한 지(make sense한 지) 검사합니다. 

![image-20220315214710460](https://user-images.githubusercontent.com/70505378/158545469-dee1a3b6-ca2b-49d5-a6af-9f8b02e46ee0.png)











### Cross modal reasoning

Cross modal reasoning의 대표적인 task는 **Visual question answering**입니다. 해당 task에서는 Image stream과 Question stream이 존재하게 됩니다. 

Image stream과 Question stream은 각각 input image와 question text를 fixed dimensional vector로 인코딩하고, 이를 point-wise multiplication을 통해 두 vector 간 관계를 만들어냅니다. 이 vector는 FC layer와 Softmax를 거쳐 정답 클래스를 예측하게 됩니다. 

여기서 헷갈리지 말아야 할 것은, **Visual question answering에서 최종 출력**은 text generation이 아니라 **classification**이라는 것입니다. 

![image-20220315215334366](https://user-images.githubusercontent.com/70505378/158545470-3ef72eac-4d0c-4ada-aa3e-0c18713c9c5f.png)

위와 같은 구조에서 발전하여, 'show, attend, and tell' 기법과 같이 attention 메커니즘을 이용하여 더욱 좋은 성능을 보이는 네트워크를 만들 수도 있습니다. 





<br>

## Multi-modal tasks(2) - Visual data & Audio

### Sound Representation

이번에는 sound(audio)와의 multi-modal learning에 대해 알아봅시다. 

Sound data를 neural net에서 사용하기 위해서는 그 형태를 적절히 변환해야 하는데, 기본적으로 **Spectogram** 형태가 있습니다. 

* Waveform: 초기 Sound data가 생긴 모양 (시간에 따른 변화)
* Power spectrum: 일정 시간 구간에서의 주파수 성분 (주파수에 따른 변화)
* Spectrogram: 시간에 따른 주파수 성분 변화 (x축: 시간, y축: 주파수)

![image-20220316103853531](https://user-images.githubusercontent.com/70505378/158545472-bf98fcd6-59e8-4c5b-8756-7a2a5940b1b5.png)

Waveform에서 Power spectrum을 만들 때는 일정 시간 간격마다 STFT(Short-Time Fourier Transform)를 적용하여 그 시간 구간 내에 있는 주파수 성분들을 얻습니다. 이를 sliding window 형식으로 매 시간 간격마다 수행합니다. 

모든 시간 간격에 대해 얻은 Power spectrum을 이어 붙인 것이 Spectogram입니다. Spectogram의 x축은 시간이고, y축은 주파수 성분입니다. 

이 Spectogram을 바로 사용할 수도 있지만, 이를 Melspectogram이나 MFCC 형태로 변환해서 사용하는 경우도 많다고 합니다. 해당 내용에 대해서는 다루지 않습니다. 

### Joint embedding

![image-20220316104915879](https://user-images.githubusercontent.com/70505378/158545477-dcd61a5b-74f0-4051-b474-eac3984feb8d.png)

Visual data & Audio 를 사용하는 task 중 대표적인 것은 'Scene recognition by sound' task입니다. 이는 이름에서 같이 sound data로부터 그 공간이 어디인지 scene을 출력하는 task입니다. 

**SoundNet**

이 Scene recognition by sound를 수행하는 모델로는 2016년 발표된 SoundNet이 있습니다. 

SoundNet은 teacher-student learning 기법을 사용합니다. video로부터 pre-trained image model은 각 프레임에 대해 object distribution과 scene distribution을 뽑아내고, sound model은 raw waveform으로부터 sound feature를 뽑아냅니다. 이 때 특징적인 것은 sound model의 구조 또한 CNN 구조를 사용한다는 것입니다. 

Sound model의 마지막 부분에서는 2개의 conv layer로 갈라지고, 각각은 image model이 뽑아낸 object distribution과 scene distribution에 matching되어 유사성을 학습하게 됩니다. 

![image-20220316105658393](https://user-images.githubusercontent.com/70505378/158545479-c06e9d51-79eb-4a94-a129-c9425c862a04.png)

추가적으로, generalized sound information을 사용하고 싶은 경우 pool5 layer에서 feature map을 가져와 사용하면 된다고 합니다. 







### Cross modal translation

![image-20220316110730680](https://user-images.githubusercontent.com/70505378/158545481-158c3325-3235-4d24-b5bb-661a89f22568.png)

**Speech2Face**

대표적인 translation task로는 speech2face가 있습니다. 말그대로 speech가 들어왔을 때 face를 출력해주는 task입니다. 

![image-20220316110855664](https://user-images.githubusercontent.com/70505378/158545487-485da95b-4104-4c98-af1a-b6078c939e2a.png)

2019년 발표된 Speech2Face 모델은 teacher-student learning 방식을 이용하며, 이미지에서 뽑아낸 face feature와 소리에서 뽑아낸 sound feature가 서로 잘 match되도록 학습합니다. 이 때 마찬가지로 image model은 pretrained이며, 학습되지 않는 fixed 구조를 가집니다. 

Speech2Face 모델은 따로 데이터 라벨링 필요없이 face와 sound가 온전히 들어가 있는 video를 입력으로 주면 되기 때문에, self-supervised learning이라고 할 수 있습니다. 

**Image-to-speech synthesis**

Speech에서 Image를 생성해내는 것이 가능하다면 그 반대도 가능하겠죠. 또 다른 translation task로는 Image-to-speech synthesis task가 있습니다. 

![image-20220316111331505](https://user-images.githubusercontent.com/70505378/158545492-594a92c6-114e-40fa-a754-f9e57738b2c1.png)

모델 구조는 Speech2Face와는 다릅니다. 2020년 발표된 모델은 Module network를 사용하며, 앞 단에는 **Image-to-Unit Model(Show, Attend, and Tell)** 을 사용하고 뒷 단에는 **Unit-to-Speech Model(Tacotron 2)**을 사용합니다. 이 때 Unit이란 자연어와 같이 임베딩된 특징값입니다. 

![image-20220316111705079](https://user-images.githubusercontent.com/70505378/158545494-6b131e9a-d99b-4949-bbf9-7fb39e68d9a2.png)





### Cross modal reasoning

![image-20220316111849393](https://user-images.githubusercontent.com/70505378/158545499-0b9622c5-1fea-404f-bb03-0b5d02e60e9f.png)

**Sound source localizaton**

Reference 방법을 사용하는 첫번째 task로는 'Sound source localization' task가 있습니다. 

![image-20220316112818071](https://user-images.githubusercontent.com/70505378/158545500-fff68069-abc2-44ba-9be1-e124c371a03c.png)

이 모델은 특이하게 supervised version, unsupervised(self-supervised) version, semi-supervised version으로 모두 학습이 가능합니다. 

Supervised learning의 경우 아래와 같이 Localization score를 ground truth와 비교합니다. Visual net과 Audio net에는 모두 CNN 구조의 모델을 사용했습니다. 

![image-20220316113055357](https://user-images.githubusercontent.com/70505378/158545503-6baa53d1-77b5-473d-99a3-3bc5e3c63c8e.png)

Unsupervised learning을 할 경우 Localization score를 Visual net의 출력인 feature map과 다시 한 번 곱하고 채널 단위로 합하여 attended visual feature vector를 만들어냅니다. 이렇게 계산한 feature vector를 Audio net의 출력인 feature vector와 비교하여 두 vector가 비슷해지도록 학습합니다. 

![image-20220316113458374](https://user-images.githubusercontent.com/70505378/158545505-85be5b54-5064-4d07-9dfd-e7d2c5e351a4.png)

그리고 semi-supervised learning을 할 때는 두 가지 loss를 모두 사용해서 학습합니다. 

**Looking to listen at the cocktail party**

Looking to listen at the cocktail party란 여러 소리들이 합성되어 있는 video에서 각 sound 성분이 어디에서 발생하는 지 추출해내는 task입니다. 이는 아래와 같이 video frame의 각 object와 video sound를 입력으로 넣어줌으로써 달성할 수 있습니다. 

![image-20220316114303876](https://user-images.githubusercontent.com/70505378/158545510-e16722e1-4669-4dd5-8f65-63449b45234b.png)

여기서 문제는, 모델의 최종 결과인 enhanced spectogram의 loss 값 계산을 위해 기존 video에서 각 사람의 speech를 따로 분리한 clean spectogram이 ground truth로 필요하다는 것입니다. 그런데 여러 sound가 합성된 video에서 각 object가 내는 sound를 따로 분리하여 spectogram을 만들어내는 것은 불가능에 가깝습니다. 

따라서 실제로 학습을 시킬 때는, 서로 다른 video에서 각각의 clean spectogram을 뽑아놓고 모델의 input으로는 그 video들을 합친 합성된 video를 주는 것입니다. video는 단순히 옆으로 이어붙일 수 있고, sound는 단순한 더하기 연산으로 할 수 있습니다. 

**Lip movements generation**

또 다른 task로, image로 입술의 모양을 넣어준다면, lip movements generation도 할 수 있습니다. 아래 영상을 보시길 바랍니다. 

* https://youtu.be/9Yq67CjDqvw

![image-20220316115308891](https://user-images.githubusercontent.com/70505378/158545511-20571745-57ee-4519-a5ae-8b6aa868c20a.png)























<br>

## 실습) Pre-trained multi-modal model applications with CLIP

### CLIP이란?

OpenAI에서 만든 멀티모달 모델을 지칭하는 단어로, `Contrastive Language-Image Pre-Training`의 약자이기도 합니다.

직역해보자면 "언어와 이미지 데이터를 Constrastive Learning으로 사전 학습 시키기" 라고 할 수 있는데, 이름에서 알수있듯 "이미지"와 "텍스트" 두 가지 다른 타입의 데이터로 학습을 하기 때문에 멀티모달 모델로 분류됩니다. CLIP을 활용하면 이미지와 텍스트로부터 유용한 공통 특징 (feature) 공간의 feature를 추출할 수 있고 이를 활용하여 다양한 task를 수행할 수 있습니다.

모델의 전체적인 학습 과정은 아래와 같이 세 단계로 이루어집니다. 

![image-20220316145439466](https://user-images.githubusercontent.com/70505378/158545515-cd1f2e74-b2ad-40c2-b69d-da64929d4438.png)

(1) CLIP에서 수행한 Constrastive learning을 보여줍니다. 이미지 데이터와 텍스트 데이터를 encoding하여 생성한 feature vector의 내적값들을 matrix로 표현합니다. 대각 성분(하늘색)은 이미지와 텍스트의 의미가 맞는 sample(positive sample)을 의미하며, 그 외의 성분은 모두 다른 의미를 가지고 negative sample이라고 합니다. 

(2) Text Encoder에 각 label text dataset을 넣어서 각각의 feature vector를 뽑아냅니다. Label test data는 단어가 될 수도 있고 문장이 될 수도 있습니다. 

(3) Image Encoder에 원하는 input image를 넣고, text encoder에서 나온 각각의 feature vector와 비교하여 similarity를 구합니다. 이 중 가장 높은 similarity를 보이는 문장이 바로 input image와 가장 유사한 의미의 text라고 할 수 있습니다. Zero-shot prediction이라고 표현한 이유는, 추가적 학습없이 임의의 image와 text를 비교하는 task이기 때문입니다. 

**Contrastive Learning이란?**

Constrastive learning은 단어에서 유추할 수 있듯 서로 다른 경로에서 온 데이터 Feature를 서로 대조하여 그 '차이'를 학습하는 것을 Contrastive Learning이라 합니다.

아래의 그림은 강의에서 다룬 내용인데요, Contrastive learning의 한 학습 예시를 보여줍니다.

![image-20220316151016740](https://user-images.githubusercontent.com/70505378/158545520-99048709-7fb2-4864-9787-0aeed2d23ab1.png)



예시 그림에는 동일한 토끼 그림 2개와 각각 짝을 이루는 텍스트(어구 혹은 문장)가 주어졌습니다.

이 때 model은 주어진 각 그림과 텍스트로부터 나온 feature vector를 비교하여 서로 동일한 의미를 지니는 벡터끼리는 Joint Embedding 공간에서 거리를 더욱 가깝게, 서로 다른 의미를 가지면 거리를 더욱 멀어지게 학습하게 됩니다.

따라서 Contrastive Learning에서는 '다르다' 혹은 '유사하다' 라는 기준을 어떻게 잡을지(=labeling 이슈), 비교를 위한 metric은 무엇으로 정의할지(예를 들어 Euclidean distance, Cosine similarity 등등) 등이 중요 사안입니다.







### 모델 불러오기 및 간단한 테스트

CLIP 모델을 사용하기 위해서는 github repo를 clone하고 라이브러리를 설치해야 합니다. 

```
git clone https://github.com/openai/CLIP.git
pip install git+https://github.com/openai/CLIP.git
```

**모델 불러오기**

아래와 같이 모델을 불러 올 수 있습니다. `clip.load` 함수를 호출하면 model과 input preprocess 함수를 반환합니다. 

```python
device = "cuda" if torch.cuda.is_available() else "cpu"

# GPU 메모리 약 1.5 GB 필요 --> 만일 부족하다면 clip.available_models() 명령어를 통해 가지고 오는 모델을 바꿀 수 있습니다
# model과 input preprocess 함수를 반환
model, preprocess = clip.load("ViT-B/32", device=device) 
```

**이미지 가져와서 전처리하기**

```python
# Test 이미지를 불러오기
image_path = "./CLIP/CLIP.png" 
plt.figure(figsize=(20, 20))  
plt.imshow(Image.open(image_path)) 
plt.show()  
```

![image-20220316152758084](https://user-images.githubusercontent.com/70505378/158545525-a787b447-4708-46e6-8613-af040142a002.png)

모델 load시 반환 받았던 `preprocess` 함수를 사용합니다. 

```python
# image encoder에 넣을 수 있도록 전처리를 합니다.
# preprocess 함수는 (224, 224)로 Resize를 해주고 Normalization을 해줍니다.
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device) 
```

**label text dataset 토큰화하기**

clip의 `tokenize` 함수를 사용합니다. 

```python
# Test 하고자 하는 임의의 문장 or 어구
text_dataset = ["a diagram", "a dog", "a cat"]

# CLIP 모델 안에는 tokenizer가 내장되어있습니다. 각 단어를 숫자로 변환해줍니다.
text = clip.tokenize(text_dataset).to(device) 
```

**이미지와 텍스트 사이 유사도 구하기**

전처리된 image data와 토큰화된 text data를 model의 input으로 전달합니다. 

```python
with torch.no_grad():
    # 모델에 image와 text 둘 다 input으로 넣고, 각 text와 image와의 유사도를 구합니다. 값이 클수록 유사합니다.
    logits_per_image, _ = model(image, text)
    
    # 확률값으로 표현하기 위해 softmax를 값을 구합니다.
    probs = logits_per_image.softmax(dim=-1).cpu().numpy().flatten()

print("- Text와 image의 유사도 값 -")
for idx in range(len(text_dataset)):
    print("  "+text_dataset[idx] + ":", logits_per_image.cpu().numpy().flatten()[idx])
    
print("\n- 각 Text가 image와 일치할 확률 -")
for idx in range(len(text_dataset)):
    print("  "+text_dataset[idx] + ":", round(probs[idx]*100, 3) , "%")
    
'''
- Text와 image의 유사도 값 -
  a diagram: 25.552788
  a dog: 20.08989
  a cat: 19.749449

- 각 Text가 image와 일치할 확률 -
  a diagram: 99.279 %
  a dog: 0.421 %
  a cat: 0.3 %
'''
```

**Encoder를 이용해 feature vector 구하기**

앞선 과정은 model에 image와 text를 바로 input으로 넣어서 유사도를 구했습니다. 그렇지만 때로는 input 데이터의 feature를 추출해야 할 때도 있습니다. 

CLIP 모델은 이를 위한 `encode_image()`와 `encode_text()` 함수를 가지고 있습니다. 

```python
image_path = "./CLIP/CLIP.png" 
image = preprocess(Image.open(image_path)).unsqueeze(0).to(device) 
text_dataset = ["a diagram", "a dog", "a cat"]
text = clip.tokenize(text_dataset).to(device) 

# model의 image encoder와 text encoder에 각각의 데이터를 넣어줍니다.
with torch.no_grad():
    image_features = model.encode_image(image) # 이미지 feature 추출
    text_features = model.encode_text(text)    # 텍스트 feature 추출
print(image_features.shape, text_features.shape) 
'''torch.Size([1, 512]) torch.Size([3, 512])'''
```

구한 feature vector로부터 유사도와 확률을 구해보면 앞서 모델에게 바로 image/text를 전달했을 때와 같은 결과를 얻을 수 있는 것을 확인할 수 있습니다. 

```python
# feature vector로부터 유사도 계산을 위해 Cosine Similarity 함수를 활용하겠습니다. 
cos_sim = CosSim(dim=1)
logits = cos_sim(image_features, text_features)*100 # similarity의 최대값을 100점처럼 표현하기 위해 편의상 CLIP에서는 100을 곱합니다. 
probs = logits.softmax(dim=-1).cpu().numpy().flatten()

print("- Text와 image의 유사도 값 -")
for idx in range(len(text_dataset)):
    print("  "+text_dataset[idx] + ":", logits.cpu().numpy().flatten()[idx])
    
print("\n- 각 Text가 image와 일치할 확률 -")
for idx in range(len(text_dataset)):
    print("  "+text_dataset[idx] + ":", round(probs[idx]*100, 3) , "%")
    
'''
- Text와 image의 유사도 값 -
  a diagram: 25.552782
  a dog: 20.089878
  a cat: 19.749443

- 각 Text가 image와 일치할 확률 -
  a diagram: 99.279 %
  a dog: 0.421 %
  a cat: 0.3 %
'''
```









### Zero-shot Image Classification

Image와 유사도가 높은 K개의 text를 찾아낼 수도 있습니다. 이는 classification task입니다. 

여기서는 Cifar100 dataset을 사용합니다. 

```python
# CIFAR100 dataset을 다운로드 받습니다.
cifar100 = CIFAR100(root=os.path.expanduser("~/.cache"), download=True, train=False)
# CIFAR 100 모든 classes의 어구를 만들고 토큰화합니다.
text_inputs = torch.cat([clip.tokenize(f"a photo of a {c}") for c in cifar100.classes]).to(device)

# 임의의 input 이미지를 선정.
image_index = 100      #  image_index 값은 0~9999 까지 입력이 가능합니다. 자유롭게 선택해보세요.
image, _ = cifar100[image_index]
plt.imshow(image)
plt.show()

# 이미지 전처리 및 feature 추출
image_inputs =  preprocess(image).unsqueeze(0).to(device)      # CLIP 모델의 전처리 모듈 사용 (위 코드 참조) 

with torch.no_grad():
    image_features = model.encode_image(image_inputs)         # 이미지 feature 추출
    text_features = model.encode_text(text_inputs)          # 텍스트 feature 추출

# Cosine Silmilarity 계산
cos_sim = CosSim(dim=1)
similarity = cos_sim(image_features, text_features)*100      # similarity의 최대값을 100점처럼 표현하기 위해 편의상 CLIP에서는 100을 곱합니다. (위 코드 참조)

# softmax 함수로 가장 높은 K개의 확률값 구하기 
K = 5    
probs = similarity.softmax(dim=-1).cpu().numpy().flatten()
values = sorted(probs)[::-1][:K]
indices = np.argsort(probs)[::-1][:5]
print(f"\nTop {K} predictions:\n") 
for value, index in zip(values, indices):
    print(f"{cifar100.classes[index]:>16s}: {100*value.item():.2f}%") 
```

![image-20220316153922208](https://user-images.githubusercontent.com/70505378/158545536-df25b29f-9a6c-42f8-8fd1-221249e164f2.png)







### Text2Image

clip 모델을 사용하여 Text2Image generation task도 수행할 수 있습니다. 

아래 그림은 'family taking a walk at night'라는 text를 입력으로 주었을 때 모델이 생성해내는 image를 iter 순서에 따라 배치한 것입니다. 이는 500번의 iter를 수행한 것으로, 더 많이 돌리면 훨씬 더 정교하고 실제같은 이미지를 얻을 수 있을 것입니다. 

![image-20220316152211880](https://user-images.githubusercontent.com/70505378/158545522-085a0384-c542-48f5-af17-f00fd8617ee7.png)

![image-20220316152841178](https://user-images.githubusercontent.com/70505378/158545527-4cfd9b40-cc80-4f27-84e7-14d7eb0dda69.png)

![image-20220316153609505](https://user-images.githubusercontent.com/70505378/158545532-c7ff2ce7-7843-434e-a59d-222ab422e80e.png)

![image-20220316155328028](https://user-images.githubusercontent.com/70505378/158545537-45c0197e-1c53-4870-bda2-001b90453ced.png)





<br>

<br>

# 참고 자료


* Multi-modal learning overview

  * Wang et al., What Makes Training Multi-Modal Classification networks Hard?, CVPR 2020  
* Multi-modal (1) – Text

  * Srivastava and Slakhutdinov, Multimodal Learning with Deep Boltzmann Machines, JMLR 2014
  * Marin et al., Recipe 1M+: A Dataset for Learning Cross-Modal Embeddings for Cooking Recipes and Food Images, TPAMI 2019
  * Vinyals et al. , Show and Tell: A Neural Image Caption Generator, CVPR 2015
  * Xu et al., Show, attend and tell: Neural image caption generation with visual attention, ICML 2015
  * Reed et al., Generative Adversarial Text to Image Synthesis, ICML 2016
  * Antol et al., VQA: Visual Question Answering, ICCV 2015  
* Multi-modal (2) – Audio

  * Aytar et al., SoundNet: Learning Sound Representations from Unlabeled Video, NIPS 2016
  * Senocak et al., Learning to Localize Sound Sources in Visual Scenes: Analysis and Applications, CVPR 2018
  * Oh et al., Speech2Face: Learning the Face Behind a Voice, CVPR 2019
  * Hsu et al., Text-Free Image-to-Speech Synthesis Using Learned Segmental Units, arXiv 2020
  * Ephrat et al., Looking to Listen at the Cocktail Party: A Speaker-Independent Audio-Visual Model for Speech
    Separation, SIGGRAPH 2018
  * Suwajanakorn et al., Synthesizing Obama: Learning Lip Sync from Audio, SIGGRAPH 2017
  * Chen et al., Lip Movements Generation at a Glance, ECCV 2018  




<br>

