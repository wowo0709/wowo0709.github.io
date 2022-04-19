---
layout: single
title: "[AITech][Data Annotation] 20220412 - OCR Technology and Services"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['OCR']
---



<br>

_**본 포스팅은 Upstage의 '이활석' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# OCR Technology&Services

이번 포스팅은 `OCR 기술과 서비스`에 대한 내용입니다. 

## OCR Technology

우리가 말하는 OCR 기술은 엄밀히 말하면 OCR(Optical Character Recognition)과 STR(Scene Text Recognition)로 나눌 수 있습니다. OCR은 문서와 같은 종이 상에 적혀있는 문자들을 인식하는 것이고, STR은 일반적인 장면 상에 있는 문자들을 인식하는 task입니다. 

일반적으로 두 task를 묶어서 OCR이라고 말합니다. 

![image-20220419184158898](https://user-images.githubusercontent.com/70505378/164005857-356fb1df-a6d4-4155-a2f0-95531f12ae36.png)

OCR은 크게 **Text Detection**과 **Text Recognition**이 결합된 CV와 NLP가 융합된 task라고 할 수 있습니다. 여기에 추가적인 서비스 제공을 위해 Serializer와 Text Parser가 연결될 수 있습니다. 

![image-20220419184311853](https://user-images.githubusercontent.com/70505378/164005859-0b02b2c2-fc40-418b-9a09-f1d89f8114b3.png)

또, OCR은 그 인식 방법에 따라 **Offline Handwriting**과 **Online Handwriting**으로 나눌 수 있습니다. 이미지에서 텍스트를 인식하는 작업에서는 Offline 방법을 사용하며, 이번 포스팅에서는 이에 대한 내용을 다룹니다. 

(Online OCR은 주로 태블릿과 같은 기기를 이용해 글씨를 작성할 때 사용되는 기술입니다)

![image-20220419184605389](https://user-images.githubusercontent.com/70505378/164005861-bf87aa6d-c890-4900-9887-6f57aa8d9a58.png)

OCR이 무엇인지 알았으니, 아래부터는 OCR의 각 단계에 대해 설명해보도록 하겠습니다. 

### Text Detector

글자 검출(Text Detection)이 객체 검출(Object Detection)과 다른 점으로 크게 2가지를 뽑을 수 있습니다. 

* 영역의 종횡비: Text는 가로 또는 세로로 아주 긴 영역을 차지할 수 있습니다. 

  ![image-20220419210242299](https://user-images.githubusercontent.com/70505378/164005862-fdbeb06b-3ca8-4ce4-96a8-889738909986.png)

* 객체 밀도: Text는 일반적인 object에 비해 상당히 밀도 높게 위치합니다. 

  ![image-20220419210308911](https://user-images.githubusercontent.com/70505378/164005866-93eb4b13-5715-4b78-81d1-de36389b5e50.png)







<br>

### Text Recognizer

인식기의 입력으로는 검출기에서 추출된 **각각의 글자 영역**이 주어집니다. 

![image-20220419210737814](https://user-images.githubusercontent.com/70505378/164005869-6a9a9f50-60d5-44cd-8792-80cd94c1d5b7.png)

인식기는 CV와 NLP의 교집합 영역입니다. 즉, 글자 영역은 CV model과 NLP model을 거쳐서 이미지에 있는 텍스트가 인식됩니다. 

![image-20220419210834340](https://user-images.githubusercontent.com/70505378/164005875-3490c281-9358-4680-8550-26a53b7a4fad.png)







<br>

### Serializer

요즘 OCR은 Text Detector와 Text Recognizer에 더해 `Serializer`까지 포함하여 하나의 OCR 모듈을 구성합니다. 

Serializer는 인식된 text들을 사람이 읽기 자연스러운 순서로 이어주는 역할을 합니다. 

![image-20220419211048736](https://user-images.githubusercontent.com/70505378/164005877-9ae7e9aa-d341-46c0-a923-f89d6ae7ac5c.png)

Serializer는 아래 과정을 따라 텍스트의 순서를 정렬합니다. 

1. 단락끼리 묶음
2. 단락 간 정렬
3. 단락 내 정렬 (좌상단에서 우하단으로)



<br>

### Text Parser

Text Parser는 OCR 모듈에 추가적인 기능을 더해줍니다. Text Parser에서는 각 글자(value)를 미리 정해놓은 카테고리(key)에 매핑하는 기능을 제공합니다. 

예를 들어 영수증에서 \{Store, Menu\} 등의 카테고리를 미리 정의하고 학습한 뒤, 추론 시 Serializer가 반환한 output을 각 카테고리에 매핑할 수 있습니다. 

![image-20220419212456837](https://user-images.githubusercontent.com/70505378/164005883-80c99c59-44a4-40a9-8227-dea923635fe8.png)

<br>

본 기능을 구현하기 위해 **BIO Tagging**을 주로 사용합니다. BIO는 각각 Begin, Inside, Outside를 가리킵니다. 

문장을 특정 단위로 나누고, 최소 단위 토큰들을 '특정 카테고리에 해당하는 단어의 시작 토큰인지, 또는 중간 토큰(마지막 토큰을 포함)인지, 또는 카테고리에 해당하지 않는 토큰인지'를 구분합니다. 

아래 예시에서는 '해리포터 보러 메가박스 가자'라는 문장에서 각 글자를 최소 단위로 설정하여, 미리 정의된 \{영화명, 영화관명\}에 대한 BIO tagging을 수행하고 있습니다. 

![image-20220419212019725](https://user-images.githubusercontent.com/70505378/164005879-49122886-a45b-43c6-ae56-14143aba56f1.png)





<br>

최종적인 예시로서, 강의에서는 명함에서 OCR을 수행하는 과정을 각 단계에 따라 보여주고 있습니다. 

![image-20220419212224882](https://user-images.githubusercontent.com/70505378/164005882-11ded014-95de-41ab-99b8-73c2ab163911.png)





<br>

<br>

## OCR Services

OCR을 이용해 제공할 수 있는 서비스는 아주 다양합니다. 

**Image text copy&paste**

* 외국어 입력
* 다량의 글자 입력
* Wifi 비밀번호 입력

![image-20220419213249320](https://user-images.githubusercontent.com/70505378/164005885-7e57bd38-9c00-4e50-89d6-720bfcd5b749.png)

**Photo Search**

사진에 있는 글자를 이용해 사진을 검색할 수 있습니다. 

![image-20220419213302449](https://user-images.githubusercontent.com/70505378/164005889-4ddefb34-80c0-4303-8d46-296cfca701a5.png)

**Move music playlist**

하나의 플랫폼에 있는 플레이리스트를 다른 플랫폼으로 옮기고 싶을 때, 플레이리스트를 캡쳐하고 옮기고 싶은 플랫폼에서 OCR 기술을 사용하여 캡쳐된 이미지로부터 음악들을 검색해 복사할 수 있습니다. 

![image-20220419213313870](https://user-images.githubusercontent.com/70505378/164005892-8b5d0ca1-7e4b-489a-92ec-fb19732d319e.png)

**Ban specific words**

광고성, 혐오성, 음란성 메시지들을 차단할 수 있습니다. 

![image-20220419213323326](https://user-images.githubusercontent.com/70505378/164005895-4734e4b0-05f1-43a3-a992-e351d721745b.png)

**Translation**

번역 시 이미지 자체에서 번역을 수행해 편의를 제공할 수 있습니다. 

![image-20220419213333340](https://user-images.githubusercontent.com/70505378/164005901-d6f34bb1-1c93-4e46-a38c-a75f445137f6.png)

**Key-Value Extractor**

특정 key에 해당하는 value를 추출할 수 있습니다. 

![image-20220419213345055](https://user-images.githubusercontent.com/70505378/164005852-60a4fb28-8353-4f5d-8e75-7ca7e5f5fc67.png)



















<br>

<br>

# 참고 자료

* 
