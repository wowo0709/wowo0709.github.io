---
layout: single
title: "[HCI][Papers] User Generated Labels 논문 리뷰"
categories: ['HCI', 'HCI-Papers']
tag: []
toc: true
toc_sticky: true
---



<br>

# User Generated Labels

이번에 소개할 논문은 KAIST KIXLAB에서 publish된 `User Generated Labels`라는 논문입니다. 

목차는 아래와 같이 이어집니다(아래 목차는 실제 논문의 목차와는 다릅니다). 

* Introduction
* Formative Study
* User Generated Labels
* Evaluation
* Conclusion

## Introduction

![image-20220330233806573](https://user-images.githubusercontent.com/70505378/160979190-cb20c7ce-3469-4e15-9957-26b54c34c5bc.png)

현재 대부분의 인터넷 플랫폼은 좋아요/싫어요 시스템을 사용합니다. 다들 한 번 쯤은 기사 또는 댓글에 좋아요/싫어요를 눌러보신 경험이 있으실 겁니다. 

이러한 시스템은 사용자가 쉽고 빠르게 자신의 의견을 남길 수 있다는 장점을 가지고 있습니다. 

하지만, 간단함에서 오는 문제점들도 분명 존재합니다. 본 논문에서는 좋아요/싫어요 시스템의 문제점으로 크게 2가지를 제시합니다. 

1. 다양한 표현이 제한된다. 
2. 남들의 의견을 제대로 이해하지 못 한다(왜 좋아요/싫어요를 남겼는지 정확히 알 수 없다).

좋아요/싫어요 시스템은 사용자의 다양한 감정을 단지 2가지 선택지인 좋아요/싫어요에 투영해야 하며, 따라서 당연하게도 다른 사람이 '왜' 좋아요/싫어요를 눌렀는지 알 수가 없습니다. 

본 논문에서는 이에 대한 해결책으로 `사용자 생성 라벨`(User generated labels, UGLs)을 제안합니다. 이는 사용자가 직접 생성할 수 있는 텍스트 기반의 라벨로, 이것으로 자신의 의견을 좀 더 잘 표현하고 다른 사람들의 의견들을 더 잘 이해하기를 기대할 수 있습니다. 

![image-20220330233904795](https://user-images.githubusercontent.com/70505378/160979194-92614a7f-eefa-4db9-99ce-b0e6437876e0.png)

<br>

본 논문에서는 이러한 '리액션 시스템'에 대한 선행 연구들에 어떤 것들이 있었는지 이야기합니다. 

**_Study 1. 여러 개 리액션 버튼 중 하나 선택하기_**

단순한 좋아요/싫어요 시스템에서 조금 더 발전하여, 다양한 리액션 버튼(이모지라고 생각할 수 있습니다)들 중 사용자가 적절하게 선택할 수 있도록 하는 시스템들도 있습니다. 

하지만 이 또한 여전히, **복잡하고 복합적인 표현**을 하기에는 부족합니다. 많은 선택지를 제공한다면 이러한 부분을 해결할 수도 있겠지만, '많은 선택지 중 하나를 선택해야 한다'는 것은 사용자에게 부담이 될 수 있으며, 적절한 선택지를 찾는데 시간이 걸릴 수도 있습니다. 

이는 자연히 사람들의 참여율 하락으로 이어집니다. 

반면 UGL의 경우, 선택지를 제한하지 않고 사용자가 생성할 수 있도록 하기 때문에 표현의 자유도가 올라가고, 선택에 대한 부담 또한 없앨 수 있씁니다. 

**_Study 2. 사용자 참여에 대한 보상 디자인_**

댓글을 본 모든 사용자들이 그에 대한 리액션을 남기는 것은 아닙니다. 실제로, 리액션을 남기는 사람들은 전체 사용자들 중 소수에 불과합니다. 

이에 대해 **어떻게 하면 사용자들의 참여를 증진시킬 것인가**에 대한 연구 또한 꾸준히 이어지고 있으며, 리액션 시스템에서 중요한 부분 중 하나입니다. 

**_Study 3. 다양한 의견에 대한 이해와 양극화 문제_**

사람들은 자신과 비슷한 의견에 대해서는 과대평가하고, 다른 의견에 대해서는 과소평가하는 경향이 있습니다. 이러한 양상은 여론의 양극화를 심화시킵니다. 

이에 대한 대안으로 제시된 것은 **사람들을 다양한 관점의 의견에 노출시키는 것**입니다. 좋아요/싫어요 시스템의 경우, 그 선택지가 2개 밖에 없기 때문에 양극화 양상은 심화될 수 밖에 없습니다. 

UGL은 사용자들의 다양한 의견을 표현할 수 있는 리액션을 직접 생성할 수 있도록 하여, 사용자들이 다양한 관점의 의견을 접하도록 할 수 있습니다. 

<br>

<br>

## Formative Study

본 논문에서는 사람들이 좋아요/싫어요 시스템과 같은 **이분법적 시스템 하에서 어떻게 리액션을 남기고 해석하는지**에 대한 연구를 진행하였습니다. 

실험은 아래와 같이 진행되었습니다. 

> 1. 논쟁적인 이슈에 대해 다루는 3개의 기사와, 각 기사에서 리액션 수가 많은 상위 10개의 댓글을 초기 상태로 가져온다. 
> 2. 각 댓글들에 대한 사고를 동일하게 하도록 사전에 있던 리액션 수들은 숨긴다. 
> 3. 각 사용자들은 댓글에 대한 해석과 리액션을 생각하고, 다른 사용자와 공유해본다. 

<br>

위 실험을 통해, 본 논문에서는 크게 3가지를 알 수 있었다고 합니다. 

1. 단순한 좋아요/싫어요 시스템은 다양한 의견과 뉘앙스를 표현하지 못 한다. 
2. 본인의 의견을 자세히 표현하기 힘들다고 느끼거나 댓글에 의미있는 기여를 할 수 없다는 생각이 들면 리액션을 하려 하지 않는다. 
3. 좋아요/싫어요가 글에 대한 사용자의 선호도를 표현한다고 믿는다. 

이제 UGL이 각각의 문제를 어떻게 해결할 수 있는지 살펴봅시다. 

<br>

<br>

## User Generated Labels

![image-20220330234542645](https://user-images.githubusercontent.com/70505378/160979195-032d9007-8bdb-4576-9286-849168cc6c03.png)

본 논문에서는 UGL의 기능으로 세 가지를 이야기합니다. 

**_Feature 1. Creating UGLs_**

UGL은 사용자에 의해 생성되는 리액션입니다. 

사용자는 20자 이하의 텍스트 라벨을 생성하고, 이를 Positive/Negative로 직접 분류할 수 있습니다. 여러 개의 UGL을 생성하는 것도 가능합니다. 하지만 중복되는 리액션을 생성하는 것을 불가능합니다. 

**_Feature 2. Reading and voting on UGLs_**

사용자들은 UGL들을 읽고, 해당 리액션을 추천 할 수도 있습니다. 

**_Feature 3. Managing toxic of irrelavant UGLs_**

부적절한 UGL을 신고할 수 있으며, 3개 이상의 신고를 받은 UGL은 시스템에 의해 숨겨집니다. 몇 개 이상의 신고를 받으면 숨겨질 지는 플랫폼 종류나 사용자의 수에 따라 변경될 수 있습니다. 

<br>

그리고 UGL로 얻을 수 있는 세 가지 기대효과에 대해 말합니다. 

1. 다양하고 복합적인 리액션을 표현할 수 있다. 
2. 내 의견을 구체적으로 남기고, 유일한 리액션을 남기거나 남들에게 내 리액션이 추천받는 것을 통해 참여에 대한 흥미와 동기부여를 얻을 수 있다. 
3. 내 의견을 쉽게 표현하고, 남들의 의견을 쉽게 이해할 수 있다. 

이를 확인하기 위해 추가적인 연구를 아래에서 진행합니다. 

<br>

<br>

## Evaluation

본 논문에서는 기존의 이분법적 시스템과 UGL 시스템의 비교를 위해 또 하나의 연구를 진행했습니다. 

연구는 아래와 같이 진행되었습니다. 

> 1. 찬반이 강하게 갈리는 4개 주제의 기사와, 각 기사에서 총 6개(3 supporting, 3 opposing)의 댓글을 초기 상태로 가져온다. 
>    * 주제: 1) 사형 제도 폐지, 2) 채용 관행에서 우대 조치 금지, 3) 동물 실험 금지, 4) 소비자 데이터 사용 제재
> 2. 218 명의 참가자들은 8개 그룹(4개 주제, 2개 시스템)으로 나눠진다. 
> 3. 실험 전, 각 참가자들은 사전 설문을 진행한다. 
>    * 평소에 얼마나 자주 온라인 활동을 했는지
>    * 반대 의견을 얼마나 잘 받아들이는지
> 4. 참가자들은 자유롭게 1에서 준비된 상태에서 리액션을 남긴다. 
>    * Formative study에서와는 다르게, 기존에 있던 추천/비추천수를 지우지 않는다. 
>    * 앞선 참가자들의 댓글, 리액션 또한 모두 볼 수 있게 한다. 
> 5. 실험 후, 각 참가자들에게 실험에 대한 질문을 진행한다. 
>    * 댓글에 리액션을 남기거나 다른 사람들의 리액션을 보는 것이 어땠는지
>    * 추천/비추천을 남긴 이유가 무엇일지
>    * 시스템의 사용성, 반대 의견을 얼마나 잘 받아들이는지 등

참가자들의 설문을 취합한 결과, 총 4개의 질문에 답을 할 수 있었다고 합니다. 

**_RQ1. How well do UGLs capture opinions towards comments?_**

* 참가자들은 댓글에 대한 동의 정도, 논쟁의 세기 정도, 댓글의 스타일, 댓글을 단 사람에 대한 판단, 주제에 대한 신념이나 믿음 등에 근거하여 총 394개의 UGL을 생성

  ![image-20220331114146634](https://user-images.githubusercontent.com/70505378/160979204-02c9c134-3dbd-4581-95c1-9e84e6b8b57d.png)

* UGL 시스템에서 받은 리액션 수는 4개 주제 모두에서 이분법적 시스템에서 받은 리액션 수를 능가

  ![image-20220331111527119](https://user-images.githubusercontent.com/70505378/160979197-40fbf028-f8db-45ef-ac9f-7b54216c6e07.png)

* 일정 수준의 UGL이 모여야 UGL 생성이 활발해지는 것을 확인(소비자 데이터 관련 주제 제외)

  ![image-20220331111537974](https://user-images.githubusercontent.com/70505378/160979199-0357b664-353d-4881-a673-35065f8f002f.png)

* UGL 시스템에 참가한 109명의 참가자 중 14명의 참가자는 positive-negative 영역의 리액션을 모두 생성하며 복합적인 의견을 드러낸 것을 확인

**_RQ2. How does having UGLs affect users' experience in evaluating comments?_**

* 리액션의 의도와 특별함이라는 면에서, 참가자들은 UGL 시스템의 리액션 방식이 더 만족스러웠다고 답변
* 참가자들은 UGL을 생성할 때 댓글이나 논쟁에서 더 큰 기여도를 경험
* UGL 시스템 하에서 이분법적 시스템보다 리액션을 남기는 데 더 큰 힘이 소요
  * 하지만 비교적 쉽게 정확한 의도를 드러낼 수 있는 것에 만족

![image-20220331113008149](https://user-images.githubusercontent.com/70505378/160979201-31b80572-4617-46bb-bf39-cf32de6aec8a.png)



**_RQ3. Do UGLs allow users to better understand the multifacetedness of public evaluation of a comment?_**

* UGL 시스템 하에서 리액션의 다면성을 이해하고, 리액션을 남긴 이유에 대해 더 정확하고 다양하고 추측



**_RQ4. How do UGLs affect participants' tolerance to the opinions that do not align with theirs?_**

* 자신과 다른 의견을 받아들이는 정도는 두 시스템에서 비슷한 모습을 보임
* 하지만 UGL 시스템 하에서 자신과 의견이 다른 댓글에 더 많은 positive reaction을 남김

<br>

<br>

## Conclusion

UGL 시스템은 기존 좋아요/싫어요와 같은 이분법적 추천 시스템 하에서 발생할 수 있는 문제점을 해결할 수 있는 새로운 시스템입니다. 

UGL 시스템에서 제공하는 강력한 사용자 경험은 2가지 입니다. 

1. **자신의 생각과 의견을 더 잘 표현**
2. **리액션으로부터 그 사람의 생각과 의견을 더 잘 이해**

이는 실제로 본 논문의 실험 과정에서 확인할 수 있었습니다. 

본 논문은 본 연구의 한계와 앞으로 진행할 수 있는 추가적인 연구에 대해 말하며 글을 마칩니다. 

* UGL 시스템이 댓글 작성자에게 주는 영향
* 새로운 UGL이라는 경험 때문에 더 많은 리액션을 남겼을 수 있다. 따라서 더 긴 기간의 실험을 통해 더 믿을 만한 실험 결과를 도출할 수 있다. 
* 논쟁적인 주제 하에서의 실험 외에, 다른 유형의 플랫폼들(질문-답변 등)에서의 실효성

<br>

<br>

_**논문에 대한 내용은 여기까지입니다. 아래부터는 개인적으로 새롭게 알고 느끼게 된 부분들을 정리하는 부분입니다.**_

<br>

# 새롭게 알게 된 것들

## Vocabulary

| Vocabulary         | meanings                                  |
| ------------------ | ----------------------------------------- |
| arguably           | 틀림없이                                  |
| dichotomized       | 이분법적                                  |
| between-subjects   | 과목 간                                   |
| multifacetedness   | 다면성                                    |
| polarization       | 양극화                                    |
| hostility          | 적대심                                    |
| forgoes            | 잊다                                      |
| hinder             | 방해하다                                  |
| lurk               | 밀행, 숨다; 속임수                        |
| transparency       | 투명도                                    |
| reciprocity        | 상호 상태(관계)                           |
| organically        | 유기적으로                                |
| homogeneous        | 동종의                                    |
| abortion           | 낙태                                      |
| subsidie           | 보조금                                    |
| articulate         | 명확히 하다                               |
| skew               | 비스듬한                                  |
| contentious        | 논쟁적인                                  |
| misogynist         | 여성혐오자                                |
| rationale          | 이론적 해석                               |
| dynamism           | 원동력                                    |
| capital punishment | 사형                                      |
| affirmative        | 긍정                                      |
| pronounced         | 명백한                                    |
| scholaly           | 학술적으로                                |
| endorsement        | 배서(뒷면에 글씨를 쓰는 것. 또는 그 글씨) |
| longitudinal       | 세로(경도의)                              |



<br>

<br>

## Domain-specific word

**_Mann-Whitney(MW) U test_**

MW test는 Mann-Whitney Wilcoxon test, Wilcoxon Rank-Sum test라고도 불립니다. 

![image-20220331132757755](https://user-images.githubusercontent.com/70505378/160979206-ddc2cf12-c129-4547-b725-06b079bb8277.png)

지난 Fitvid 포스팅에서 Wilcoxon signed-rank test와 t-test에 대해 보았습니다. 세 test는 그룹 간의 평가를 비교한다는 점에서 공통적인데, 그 방식에서 아래와 같은 차이가 있습니다. 

| Test                                                         | Parametric     | Samples between groups |
| :----------------------------------------------------------- | -------------- | ---------------------- |
| T-test                                                       | Parametric     | independent            |
| Wilcoxon signed-rank test                                    | Non parametric | dependent              |
| Mann-Whitney Wilcoxon(U) test<br />(or Wilcoxon rank-sum test) | Non parametric | independent            |

비모수 검정(Non parametric test)의 특징은 모수 검정(Parametric test)보다 덜 엄격한 요구 사항이 적용된다는 점입니다. 

![image-20220331132741738](https://user-images.githubusercontent.com/70505378/160979205-42eff8bb-4a0e-481e-9439-bd25c896fc7a.png)

아래 사이트들을 참고해주세요. 

* [Conduct and Interpret a Wilcoxon Sign Test](https://www.statisticssolutions.com/free-resources/directory-of-statistical-analyses/wilcoxon-sign-test/)
* [Mann-Whitney U Test - Statstest](https://www.statstest.com/mann-whitney-u-test/)
* [Mann-Whitney U Test - Datatab](https://datatab.net/tutorial/mann-whitney-u-test)





**_chi-square test_**

![image-20220331134820422](https://user-images.githubusercontent.com/70505378/160979209-3f50350a-4452-4a78-8996-9465baf5597c.png)

> 카이제곱 검정 또는 χ² 검정은 카이제곱 분포에 기초한 통계적 방법으로, 관찰된 빈도가 기대되는 빈도와 의미있게 다른지의 여부를 검정하기 위해 사용되는 검정방법이다. 자료가 빈도로 주어졌을 때, 특히 명목척도 자료의 분석에 이용된다. 
>
> [카이제곱 검정 - 위키백과](https://ko.wikipedia.org/wiki/%EC%B9%B4%EC%9D%B4%EC%A0%9C%EA%B3%B1_%EA%B2%80%EC%A0%95)











<br>

<br>

## Others

HCI 계열의 논문들은 모두 시스템/플랫폼의 개발과 직접적으로 연관이 있고, 가설에 대해 직접 참가자들과의 실험을 통해 검증한다는 점이 인상적으로 다가왔다. 











<br>

<br>













