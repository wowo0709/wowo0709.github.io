---
layout: single
title: "[HCI][Papers] FitVid: Responsive and Flexible Video Content Adaption 논문 리뷰"
categories: ['HCI', 'HCI-Papers']
tag: []
toc: true
toc_sticky: true
---



<br>

# FitVid: Responsive and Flexible Video Content Adaption

이번에 소개할 논문은 KAIST KIXLAB에서 publish된 `FitVid: Responsive and Flexible Video Content Adaption`이라는 논문입니다. 

목차는 아래와 같이 이어집니다(아래 목차는 실제 논문의 목차와는 다릅니다). 

* Introduction
* Formative Study
* Computational Pipeline For Automated Adaptation
* Video Interface
* Pipeline Evaluation
* Conclusion

## Introduction

![image-20220325105423821](https://user-images.githubusercontent.com/70505378/160069107-c60a9474-e977-4dc4-86ac-560729973395.png)

`FitVid`는 모바일로 강좌를 수강하는 사람들이 겪는 불편에서 시작하였습니다. 

다들 모바일로 강의, 또는 영상을 보신 적이 있으실 겁니다. 모바일로 영상을 볼 때 가장 불편한 점은 여러가지가 있겠지만, 본 논문에서는 두 가지의 주요한 요구 사항이 있었다고 말합니다. 

* **More readable content**
* **Customizable video design**

즉, 우리는 모바일로 강의를 볼 때도 텍스트/이미지가 확인하기 어렵지 않을 만큼 잘 보였으면 좋겠고, 여러 요소들을 reposition/zoom-in 등으로 커스텀 할 수 있었으면 좋겠습니다. 

FitVid는 바로 여기서 착안한 `Responsive and Customizable video content`입니다. 

<br>

실제로 웹 사이트나 e-book과 같은 매체에서는 content adaptation이 어느정도 가능합니다. 하지만 여전히 video와 같은 동적 이미지에서는 한계가 있는데, 논문에서는 그 이유로 2가지를 들었습니다. 

**_첫번째. Video frame 내의 test/image와 같은 요소들의 분리_**

비디오는 수 많은 이미지 프레임의 연속입니다. 비디오를 프레임 단위로 요소들을 분리시키는 것이 최선의 방법일까요?

또, 분리의 단위는 어떻게 해야 할까요? 글자 단위, 문자 단위, 블록 단위, 의미 단위 등 수많은 방법이 있고, 어떤 것을 선택하느냐에 따라 그 접근 방법도 달라져야 할 것입니다. 

기존에는 '강의 자료'를 이용해 비디오에서 요소들을 분리하려는 시도가 있었습니다. 하지만 강의 자료라는 것은 항상 얻을 수 있다는 보장이 없습니다. 또 하나의 큰 단점은 이러한 접근은 dynamic component, 즉 강의 자료에는 등장하지 않지만 강의 영상에는 등장하는 강사, 필기, 마우스 포인터 등의 요소들을 인식하지 못 할 것입니다. 

**_두번째. 강의 디자인의 다양성_**

강의 영상을 제작하는 데에 어느 하나의 정해진 가이드라인이 있는 것이 아닙니다. 강의 디자인은 강사, 과목, 국가 별로 매우 다양합니다. 

기존에는 이를 rule-based methods로 확립하려는 시도가 있었지만, 수 많은 경우의 수들을 모두 확립하는 것은 불가능에 가깝습니다. 

<br>

그래서 본 논문에서는 이를 자동화하는 하나의 pipeline을 만들어냈고, 이는 크게 2개의 단계로 구성됩니다. 

1. Decomposition
   * Deeplearning techniques를 활용하여 raw pixels로부터 in-video element의 metadata들을 추출해냅니다.  
2. Adaptation
   * Constrained optimization과 Set of heuristices를 이용하여 자동적으로 모바일에 최적화된 영상으로 변환해줍니다.  
   * Mobile learning guideline을 참고하여 최적화를 수행합니다. 

Automated pipeline을 통과하여 변환된 영상에 더하여, FitVid에서는 유저들에게 커스텀 기능 또한 제공합니다. 이에 대한 자세한 내용은 뒤에서 다루도록 하겠습니다. 



<br>

## Formative Study

본 논문에서는 mobile learning 경험이 있는 사람들을 모집하여 인터뷰를 수행했습니다. 인터뷰 내용은 기존의 강의 영상들이 모바일 환경에 적합한지, 그렇지 않다면 어떤 요구사항이 있는지에 대한 질문들로 구성되었습니다. 

참가자들의 응답은 크게 두 섹션으로 나눌 수 있었습니다. 

**Improve readability**

* 자동화된 줌-인 기능
* 요소별 줌-인 기능
* 모바일에 최적화된 폰트 크기
* 모바일에 최적화된 텍스트 양
* 필기체를 타자기 형태로 변환
* 현재 설명하고 있는 부분 하이라이팅

**Providing customization options**

* Text-only, Image-only 모드
* Dark 모드
* 강사 이미지 끄기

문제의 심각성과 아이디어의 참신함, 응답자들의 비율을 고려하여 본 논문에서는 크게 다음의 세 가지를 design goal로 설정했습니다. 

1. Automatiacally generating responsive design for video content
   * 딥러닝을 활용하여 비디오의 각 요소들을 추출하고, 이를 mobile learning guideline에 맞춰 조정된 영상 형태로 제공
2. Supporting direct manipulation of in-video element
   * 자동적으로 조정된 영상이 사용자의 마음에 들지 않거나, 마음에 들더라고 추가적인 조정을 가하고 싶을 수 있기 때문에 각 요소 별로 사용자가 직접적인 조정이 가능하게 함
3. Providing options for content customization
   * 템플릿을 끈다거나, 강사 이미지를 지운다거나, 배경 테마를 바꾸는 등의 디자인 조정

<br>

## Computational Pipeline For Automated Adaptation

![image-20220325135917660](https://user-images.githubusercontent.com/70505378/160069113-128cc690-d7dc-49a2-8003-224f2eb90d70.png)

### Decomposition Stage

Decomposition Stage는 4단계로 구성됩니다. 

**Shot Boundary Detection**

앞서 영상으로부터 각 요소들을 추출해야 한다고 했습니다. 그런데, 영상의 각 프레임 별로 각 요소들을 추출할 수는 없는 노릇입니다. 

본 논문에서는 **슬라이드(또는 장면) 전환 시**라는 기준을 세워, 전환 직후의 프레임을 key frame이라고 명명했습니다. 이 key frame에서 각 요소들을 추출하면, 다음 key frame 이전까지는 같은 요소들을 공유하고 있을 것입니다. 

Key frame을 찾기 위해서 **HSV(Hue, Saturation, and Intensity Value) peak detection**과 **template matching techniques**를 사용했다고 합니다. 

과정이 종료되면 **sequence of shots**(key frames)가 반환되며, 각 shot은 [start time, end time, representative video frame]으로 구성됩니다. 

**Static Object Analysis**

Shot boundary detection 과정이 종료되고 sequence of shots를 얻게 되면, 반환된 representative video frames로부터 **adaptation이 가능한 요소들을 찾습니다.**

이때 딥러닝 기법을 사용하기 때문에, **모델을 학습시킬 데이터셋**이 필요합니다. 기존에 있던 데이터셋이 존재하지만, '부족한 양', '과목의 다양성 부족', '의미적 그룹핑의 부족' 등의 이유로 활용하기에는 한계가 있었다고 합니다. 

그래서 연구팀은 총 12개 클래스(title, text box, picture, chart, figure, diagram, table, schematic diagram, header, footer, handwriting, and instructor)로 이루어진 데이터셋을 만들어 사용했습니다. 본 데이터셋은 **'의미 단위'**로 구성되었으며, 이는 화면 상으로 분리된 요소들도 포함 관계, 또는 설명 관계에 기반하여 분리를 하였음을 뜻합니다. 이러한 의미 단위 그룹핑으로 강의에서 전달되는 요소들이 최대한 보존되도록 하였습니다. 

<br>

실제 학습 과정에서는 custom dataset에 더하여 pretraining dataset을 사용하였습니다. Pretraining dataset으로는 약 500K text, image, diagram 등을 포함한 문서 페이지로 구성된 'DocBank' dataset을 사용했다고 합니다. DocBank dataset으로 pretrained된 모델은 custom dataset으로 fine-tuning 되었습니다. 

Detection model로는 Faster R-CNN, SSD based on ResNet, EfficientDet, CenterNet을 사용하였으며, 그 중 **CenterNet**의 mAP(with IoU of 0.5) 값이 79%로 가장 높아 CenterNet을 채택했습니다.  

본 과정을 통해 화면 요소들의 위치를 특정함은 물론, font size/typeface/font color 등의 metadata도 추출하였고, 슬라이드의 배경 정보도 추출하였습니다. 

**Dynamic Object Analysis**

Static object analysis와 별개로 **마우스 포인터, 필기, 강사 이미지** 등의 dynamic object analysis를 위해서는 OpenCV의 motion analysis module을 이용해 추가적으로 detection을 수행했고, 이 결과를 앞서 **deeplearning model의 결과와 비교하여 영역을 특정**했다고 합니다. 

**Text-to-Script Mapping**

다음으로는 **소리와 화면 상의 텍스트 사이 관계 매핑**을 수행했습니다. 본 과정은 다시 2단계로 구성되며, rule-based mapping algorithm을 사용합니다. 

* **Alignment Stage**
  * 이 단계에서 중요한 것은 progressive disclosure(점진적 공개)와 semantic similarity(의미적 유사성)입니다. 
  * 새롭게 공개된 텍스트와 그때의 대사를 매핑합니다. 
  * 새롭게 공개되는 것이 없다면, 현재 화면의 텍스트 요소들과 대사 사이에 유사성을 구합니다. 이때 유사성을 구하기 위해서 Sentence-BERT 모델을 사용합니다. 
    * 유사성을 구하고 나면 bipartite graph mapping 알고리즘을 이용하여 가장 유사성이 높은 텍스트 요소를 찾아냅니다. 
* **Grouping Stage**
  * 한 화면에 보여져야 할 요소들을 그룹핑합니다. 
  * Linearity of lecturing(강의의 선형성)을 고려합니다. 
    * 예를 들어, 텍스트 박스 내에서 강사가 top->bottom->middle 순으로 강의를 진행한다면 전체 텍스트 박스를 하나의 요소로 취급합니다. 
  * 비슷하게, 강사가 동시에 언급하는 내용의 텍스트들은 하나의 요소로 취급합니다. 

### Adaptation Stage

Adaptation Stage에서는 Decomposition Stage에서 추출된 요소들을 mobile learning guideline에 따라 자동으로 적절히 배치합니다. 

Adaptation Stage는 3개의 module로 구성됩니다. 

**Local Content Adaptation**

강의 원본의 내용을 충분히 보존하면서도, mobile learning guideline compiliance rate가 최대한 높아지도록 화면 요소들을 조정합니다. 

**Layout Adaptation**

화면 내에 텍스트와 이미지가 동시에 있다면, 각 요소들을 얼만큼 키워야 할 지 정합니다. 이때 constrained optimization technique이 사용됩니다. 

![image-20220325140217972](https://user-images.githubusercontent.com/70505378/160069116-ed4dc15a-f8ab-4b3e-8f59-02c8b737db95.png)

`x`는 font size, `y`는 image size, `c`는 guideline의 threshold를 나타냅니다. 즉, 위 식을 최적화한다는 것은 guideline과 adapted content 사이 deviation을 최소화하는 것입니다. 

또한 이 단계에서는 column layout도 모바일 화면에 맞게 조정합니다. 

**Global Content Adaptation**

마지막 단계에서는 앞서 조정된 결과들을 강의 원본과의 디자인 일관성을 고려하여 다시 한 번 정제합니다. 

<br>

## Video Interface

![image-20220325140611786](https://user-images.githubusercontent.com/70505378/160069118-c51b1a8c-697d-4b78-bec0-7b5cd2b13df4.png)

위 이미지는 Pipeline을 통과한 FitVid 영상의 화면을 보여줍니다. 

* (a): 강의 원본
* (b): Resizing, repositioning 등의 direct manipulation
* (c): Pipeline을 통과한 FitVid에서 기본적으로 제공하는 영상
* (d): Dark mode
* (c): 강사 이미지, 템플릿 등 지우기

재생 바 중간중간에 있는 회색의 bar들은 key frame의 위치를 나타냅니다. 해당 구간에서 변환되었던 요소들은 다른 구간으로 넘어갔다가 돌아와도 그대로 유지됩니다. 

<br>

## Pipeline Evaluation

![image-20220325141102438](https://user-images.githubusercontent.com/70505378/160069120-85b18a34-fd60-4025-8b18-b249d7dde345.png)

Pipeline evaluation은 design guideline과 비교하는 정량적 평가와 사용자들의 설문을 통한 정성적 평가로 이루어집니다. 

위 표에서 `# of Target Cases`는 **원본 영상에서 design guideline에 부합하지 않는 장면의 수**, `# of Adapted Cases`는 그 중 **Pipeline을 통해 guideline에 부합하도록 변환된 장면의 수**를 나타냅니다. 

위 표를 보면 Font Size에서 guideline에 부합하지 못 한 장면이 4개 있는 것을 볼 수 있는데, 논문에서는 그 이유를 아래와 같이 이야기합니다. 

* Detection model의 탐지 오류로 발생하는 overlap
* Global content adaptation 단계에서 '본문 내용의 font size는 제목의 font size보다 커지지 않는다'라는 규칙
* 여러 텍스트 박스들이 하나의 요소로 그룹핑 된 경우, 한 화면에 보여줘야 하기 때문에 어느 정도 이상으로 font size를 키우지 못 함

물론 위의 문제에도 불구하고, FitVid에서는 사용자의 direct manipulation을 제공하기 때문에 문제를 극복할 수 있습니다. 

![image-20220325141949180](https://user-images.githubusercontent.com/70505378/160069121-8a56bc6b-9575-4384-be9b-caac8832361a.png)

<br>

정성적 평가에서는 FitVid에서 자동으로 제공하는 조정된 영상과 원본 영상의 요소들을 각각 1~7 단계의 별점으로 평가하도록 요구했습니다. 결과는 아래 표와 같고, Adapted content에서 모두 향상되었음을 확인할 수 있습니다. 

![image-20220325143012058](https://user-images.githubusercontent.com/70505378/160069122-aa65981d-5543-4e1b-9adf-8b2b2748f3c6.png)



또한, 사용자들이 실제 사용 시 어떤 direct manipulation 기능을 많이 사용했는 지 수집했습니다. 사용자들은 resizing, repoistioning, highlighting의 기능을 많이 사용했다고 합니다. 

![image-20220325143121936](https://user-images.githubusercontent.com/70505378/160069125-60ce40fc-134c-443a-b867-c3e241ed32b4.png)

각각의 direct manipulation을 수행한 이유는 아래와 같이 확인되었습니다. 

![image-20220325143227150](https://user-images.githubusercontent.com/70505378/160069128-8496383b-b3aa-4373-9eac-03866cae5aa4.png)

<br>

## Conclusion

FitVid는 mobile learners를 위한 automated content adaptation과 design customization을 제공하는 시스템입니다. 

FitVid에서 제공하는 영상은 실제로 design guideline에 더 잘 부합하는 것을 확인하였으며, 실제로 사용자들의 경험을 더 증진시켜 주었음을 확인하였습니다. 

본 논문에서는 FitVid가 발전할 수 있는 방향을 아래와 같이 이야기합니다. 

* Tablet, smartwatch, large screen 등의 다양한 매체에서의 적용
* 자동적으로 제공하는 adaptation의 정도 조절
* 사용자들이 자주 사용하는 옵션을 버튼 형태로 제공(ex. font size: [8, 12, 16])
* 질병, 나이, 국가 등 다양한 subgroup에 최적화된 adaptation 제공
* 강의 영상 이외에 tutorial, new, educational talks 등의 informational video로의 적용

FitVid는 비단 mobile learners 뿐 아니라, 강의를 촬영하고 편집하는 instructors, 영상을 제공하는 learning platform enginners 등에게도 도움을 줄 수 있을 것으로 전망합니다. 

<br>

<br>

_**논문에 대한 내용은 여기까지입니다. 아래부터는 개인적으로 새롭게 알고 느끼게 된 부분들을 정리하는 부분입니다.**_

<br>

# 새롭게 알게 된 것들

## Vocabulary

| Vocabulary              | meanings                                  |
| ----------------------- | ----------------------------------------- |
| Nonetheless             | 그럼에도 불구하고                         |
| mitigate                | 완화시키다                                |
| compliance rate         | 준수율                                    |
| corroborate             | 확증하다                                  |
| looping over            | 반복                                      |
| tailored to             | ~에 맞춰                                  |
| motion trajectory       | 운동 궤적                                 |
| codebook                | 일련의 코드들을 쉽하고 기록하는 문서 유형 |
| inter-rater reliability | 평가자 간 신뢰도                          |
| address                 | 해결하다                                  |
| discrepancy             | 불일치                                    |
| legibility              | 읽기 쉬움                                 |
| cue                     | 신호, 역할                                |
| rigor                   | 엄밀함                                    |
| reproducibility         | 재현성                                    |
| progressive disclosure  | 점진적 공개                               |
| coherent                | 일관된                                    |
| trim down               | 줄이다                                    |
| counterbalance          | 균형                                      |
| astigmatism             | 난시                                      |
| fatigue                 | 피로                                      |
| cursive                 | 필기체                                    |
| pose a load             | 부담을 가중시키다                         |
| pragmatic               | 바쁜                                      |
| readily                 | 손쉽게                                    |
| dyslexia                | 난독증                                    |
| inclusive               | 포함한                                    |
| stakeholder             | 이해관계자                                |



<br>

## Domain-specific word

_Cohen's kappa score_

![image-20220325153728824](https://user-images.githubusercontent.com/70505378/160069143-9e60e384-22b9-4ab0-8690-10a9ed7c05b8.png)

> 2명의 관찰자(또는 평가자)의 신뢰도를 확보하기 위한 확률로서 평가지표로 사용되는 상관계수. 3명 이상의 신뢰도를 얻기 위해서는 Fleiss' kappa score를 사용. 
>
> https://ko.wikipedia.org/wiki/%EC%B9%B4%ED%8C%8C_%EC%83%81%EA%B4%80%EA%B3%84%EC%88%98

> Cohen suggested the Kappa result be interpreted as follows: values ≤ 0 as indicating no agreement and 0.01–0.20 as none to slight, 0.21–0.40 as fair, 0.41– 0.60 as moderate, 0.61–0.80 as substantial, and 0.81–1.00 as almost perfect agreement.
>
> https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3900052/ 



_HSV peak detection_

![image-20220325153330532](https://user-images.githubusercontent.com/70505378/160069139-f1bd56b3-2b0a-4ecd-b68e-2493f8d7109f.png)

> HSV(Hue, Saturation, Value) 값의 급격한 변화를 이용해 화면의 전환이나 물체의 이동 등을 탐지하는 방법

_template matching technique_

![image-20220325153240348](https://user-images.githubusercontent.com/70505378/160069136-84d65b6e-ab5e-4ef5-9fa0-74940a113208.png)

> Template matching is **a technique in digital image processing for finding small parts of an image which match a template image**. It can be used in manufacturing as a part of quality control, a way to navigate a mobile robot, or as a way to detect edges in images.
>
> https://en.wikipedia.org/wiki/Template_matching

> 템플릿 매칭(template matching)은 참조 영상(reference image)에서 템플릿(template) 영상과 매칭되는 위치를 탐색하는 방법이다. 일반적으로 템플릿 매칭은 이동(translation) 문제는 해결할 수 있는 반면, 회전 및 스케일링된 물체의 매칭은 어려운 문제이다. 
>
> 템플릿 매칭에서 영상의 밝기를 그대로 사용할 수도 있고, 에지, 코너점, 주파수 변환 등의 특징 공간으로 변환하여 템플릿 매칭을 수행할 수 있으며, 영상의 밝기 등에 덜 민감하도록 정규화 과정이 필요하다.
>
> https://m.blog.naver.com/windowsub0406/220540208296



_bipartite graph matching algorithm_

![image-20220325153149430](https://user-images.githubusercontent.com/70505378/160069134-a495872b-fc8a-4df2-b1e6-227aa1e505f2.png)

> 이분 그래프 & 최대 이분 매칭 (Bipartite Graph & Maximum Bipartite Matching)
>
> https://dhpark1212.tistory.com/entry/%EC%9D%B4%EB%B6%84-%EA%B7%B8%EB%9E%98%ED%94%84-vs-%EC%9D%B4%EB%B6%84-%EB%A7%A4%EC%B9%AD

> 이분 매칭은 A 집단이 B 집단을 선택하는 방법에 대한 알고리즘입니다. 
>
> https://blog.naver.com/ndb796/221240613074

_Wilcoxon signed-rank test_

![image-20220325153028133](https://user-images.githubusercontent.com/70505378/160069133-7601609b-7992-4e67-996c-6ce6a9335a41.png)

> The Wilcoxon signed-rank test is **a non-parametric statistical hypothesis test used either to test the location of a set of samples or to compare the locations of two populations using a set of matched samples**.
>
> https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test

> 오늘은 Paired t-test와 맞대응 되는 비모수 검정의 Wilcoxon signed-rank test 이다. 
>
> https://blog.naver.com/PostView.nhn?blogId=y4769&logNo=220116970296

_Likert scale question_

![image-20220325152959962](https://user-images.githubusercontent.com/70505378/160069130-7c09da3d-6df4-40a9-98b5-19db1f6783aa.png)

> Likert Scale questions are **a form of closed question and one of the most widely used tools in researching popular opinion**. They use psychometric testing to measure beliefs, attitudes and opinion. The questions use statements and a respondent then indicates how much they agree or disagree with that statement.
>
> https://www.smartsurvey.co.uk/survey-questions/likert-scale

> 어떠한 사안에 대해 얼마나 동의를 하고 얼마나 동의하지 않는지에 대해 묻는 질문에 답변해 보신 적이 있나요?
>
> https://ko.surveymonkey.com/mp/likert-scale/ 

<br>

## Others

AI 분야의 기술 논문과는 다르게, 논문에서 설문 과정, 설문 결과, 사용자 경험 등이 차지하는 부분이 매우 큰 것을 느낄 수 있었다. 

설문 과정을 기술하는 부분에서 인상적으로 느꼈던 부분들을 조금 정리해본다. 

* 질문으로부터 평가할 요소들을 확실히 정한다. 
* 평가에 필요한 요소 이외의 요소들은 통제한다. 
* 적절한 상태, 경험을 가지고 있는 참가자들을 모집한다. 
* 사용자의 응답이 믿을 만한 것인가를 평가하기 위한 장치가 필요하다. 
* 왜 해당 질문에 그렇게 답했는지에 대해 추가 질문을 한다. 
  * 실제 사용 용례를 조사/설문한다. 

<br>

<br>













