---
layout: single
title: "[HCI][Papers] COCOMIX: Utilizing Comments to Improve Non-Visual Webtoon Accessibility 논문 리뷰"
categories: ['HCI', 'HCI-Papers']
tag: []
toc: true
toc_sticky: true
---



<br>

# COCOMIX: Utilizing Comments to Improve Non-Visual Webtoon Accessibility

이번에 소개할 논문은 KAIST KIXLAB에서 publish된 `COCOMIX: Utilizing Comments to Improve Non-Visual Webtoon Accessibility`이라는 논문입니다. 

목차는 아래와 같이 이어집니다(아래 목차는 실제 논문의 목차와는 다릅니다). 

* Introduction
* Formative Study
* Webtoon comments analysis
* COCOMIX
* Evaluation
* Conclusion

## Introduction

![image-20220405142744439](https://user-images.githubusercontent.com/70505378/161763800-e5866dbc-0c52-4cbf-a066-d4f2f22988d0.png)

**Webtoon**은 Web과 Cartoon의 합성어로 웹 기반 만화를 뜻하며, 우리 중심에 있는 문화들 중 하나입니다. 사람들은 어제 나온 웹툰 이야기를 하기도 하고, 웹툰 안의 표현을 은유적으로 사용하기도 합니다. 

하지만, 시각적 결함을 가지고 있는 사람들에게는 이는 해당되기 어려운 일입니다. 그들은 시각적 요소로 이루어져 있는 웹툰을 제대로 즐길 수가 없습니다. 

Image description과 같은 방법이 있지만, 이는 사용자에게 웹툰의 고유 가치인 **자유로운 읽는 속도**나 **댓글 감상** 등과 같은 가치들을 제대로 보존하지 못 합니다. 

`COCOMIX`는 이러한 배경에서 등장한 **시각적 결함을 가진 사람들을 위해 댓글을 활용하여 접근성을 증진시키는 interactive webtoon reader**입니다. COCOMIX는 댓글을 기반으로 하여 사용자가 원할 시 추가적인 상세 정보를 전달할 수 있게 하고, 해당 장면과 관련된 댓글들을 바로 참조할 수 있게 합니다. 

### Related Works

**접근 가능한 이미지 표현**

시각적 결함을 가진 사람을 가진 사람들이 이미지 접근성을 높이려는 연구들은 계속해서 진행되어 왔습니다. 그 중 대표적인 것은 이미지에 해당하는 설명(alternative text description)을 제공해주는 것입니다. 이러한 설명은 사람에 의해 작성되거나, 객체 인식 기술을 활용해 작성될 수 있습니다. 

Text description을 작성하기 위해서는 가이드라인이 필요합니다. 지금까지 여러 매체의 이미지에 적용하기 위한 가이드라인들이 나왔지만, 본 논문에서 target하고 있는 '웹툰'이라는 매체에 대해서는 그 가이드라인이 부족하다고 합니다. 

**'만화'라는 매체**

만화라는 매체는 일련의 분리된 장면들로 구성되어 있습니다. 이를 panel이라고 합니다. 

이전 연구에서는 작가의 의도를 최대한 보존하면서 그 상호작용성을 높이려는 시도로 연결된 panel들을 하나의 그룹으로 묶는 시도를 했습니다. 이 그룹을 phasel이라고 합니다. 

본 논문에서는 이러한 phasel 아이디어에 기반하여 장면을 재구성하고, 비시각적 상호작용성을 높이려는 시도를 했습니다. 

**만화의 접근성을 높이려는 시도**

만화의 접근성을 높이려는 연구들 역시 진행되어 왔습니다. 

예를 들어 촉각으로 표현되는 책을 만들거나, 이미지에 대한 설명을 오디오로 출력하는 시도들이 있었습니다. 시각적 결함이 있는 사람들은 auido book에 대한 선호도가 높았다고 합니다. 

하지만, 이전 연구들에서는 '웹툰'이라는 매체를 target하여 이미지 설명을 만드려는 시도는 없었습니다. 

또한 이미지에 대한 설명은 사람이 만들어야 합니다. 이전 연구에서는 해당 책을 자주 보는 팬들의 힘을 빌려 이미지에 대한 설명을 작성하려는 시도가 있었습니다. 본 논문에서는, **댓글**을 활용하여 추가적인 비용이 들지 않으면서도 시간을 절약할 수 있는 이미지 설명 생성을 시도했습니다. 

**보조적 도구로서의 댓글의 사용**

댓글은 사용자들이 자유롭게 자신의 의견, 감정 등을 표현하는 수단입니다. 이전까지는 댓글들을 활용하여 이미지 설명을 보강하려는 시도가 이루어지지 않았습니다. 본 논문에서는 이러한 댓글의 가치에 주목하여, 필요한 정보들을 추출하여 웹툰 장면에 대한 추가적인 정보를 제공하려 했습니다. 

본 논문에서는 기존의 comment mining 기법에 기반해 웹툰의 댓글들에서 유의미한 정보를 추출하여 그 상호작용성을 높이려는 시도를 하였습니다. 





<br>

<br>

## Formative Study

![image-20220405142806760](https://user-images.githubusercontent.com/70505378/161763806-654c3596-15d0-492c-a16f-aa98f8cfebd8.png)

본 연구에서는 10명의 시각적 결함을 가진 사람들과 2개의 웹툰으로 interview와 co-reading excercise를 수행했습니다. 

### Semi-structured Interview

실험 참가자들에게 다음의 질문들을 가지고 인터뷰를 진행하였습니다. 

* 웹툰을 읽는 이유
* 웹툰에 관한 경험
* 웹툰의 인지적 접근성
* 웹툰의 접근성을 높이려는 시도
* 웹툰의 접근성을 방해하는 장애물

### Co-reading Excercise

Co-reading excercise에서는 참가자들이 image description을 듣는 과정에서 주로 어떤 정보들을 추가적으로 요구하는지에 대해 실험하였습니다. 

실험은 기본적으로 주어진 image description을 참가자들에게 들려주고, 참가자들은 해당 장면에서 추가적인 정보를 원할 경우 이야기하도록 했습니다. 

### Results

본 논문에서는 참가자들이 요구한 설명들을 크게 세 개의 카테고리로 구분했습니다. 

* 시각적 정보 요구: 실제 장면에서 묘사된 시각적 요소(그림체나 인물에 대한 묘사 등)에 대한 설명
  * 예. 주인공의 셔츠 색깔은 무엇인가요?
* 의견 요구: 해당 장면에 대한 다른 사람의 의견 설명
  * 예. 주인공은 잘 생겼나요?
* 확인 요구: 자신의 생각이 맞는지 확인
  * 예. 주인공이 아직 쳬육관에 있는것이죠?

그리고 앞서 준비한 질문들에 대한 답변은 아래와 같이 정리할 수 있었습니다. 

* 웹툰을 읽는 이유
  * 웹툰이 문화의 중심에 있다고 느끼고, 여기에 편승하기를 원함
  * 또래 간 대화를 이해하고 참여하기를 원함
* 웹툰에 관한 경험
  * 기본적으로 제공되는 이미지 설명을 이용하기 보다는, 친구나 가족을 통해 장면 묘사를 들었음
  * 웹툰이 드라마나 영화 등으로 각색된 작품들을 봤음
  * 이미지 인식 어플리케이션을 사용했음
  * 공통적인 의견은 이러한 방법들은 불편하고 제한적이며, 신뢰성이 높지 않다. 또한 댓글들을 참고하려 해도 내용 이해에 도움이 되지 않는 댓글들이 상당수여서 제대로 활용하지 못 했다. 
* 장면 설명에 대한 추가적인 요구 사항
  * 시각적 정보 요구: 62%
  * 의견 요구: 20%
  * 확인 요구: 18%
  * 웹툰을 다 읽었을 즈음에 가장 많은 질문이 나왔다. 이는 지금까지 참가자들이 이해한 내용이 맞는지 다시 한 번 확인하려는 시도로 보인다. 

### Challenges and Design Goals

Formative study의 결과, 참가자들이 크게 4가지의 어려움을 겪고 있다는 사실을 알 수 있었습니다. 

1. 반복되는 설명으로 인한 혼란
2. 긴 설명에 대한 정보처리의 어려움
3. 설명의 상세함에 대한 조절 불가
4. 내용을 이해함에 있어 도움이 되는 댓글들을 찾기 어려움

그리고 이러한 어려움들을 해결하기 위해 아래와 같이 design goals를 설정했습니다. 

1. 불필요한 정보를 최소화한 간결한 설명 제공
2. 설명의 상세함 정도를 조절할 수 있게 함
3. 사용자의 속도에 맞춰 추가 정보를 선택적으로 제공
4. 장면에 해당하는 댓글들에 대한 선택적 접근

### Guidelines for Webtoon Descriptions

앞에서 말했듯이, 본 논문에서는 webtoon description에 대한 적절한 guideline이 없어서 본 연구에서는 이를 직접 정의했습니다. 

가이드라인의 핵심 세 가지는 아래와 같습니다. 

* 제목과 같은 내용과 직접적으로 관련이 없는 정보들을 어떻게 전달할 것인가
* 웹툰 장면들을 phasel로 어떻게 나누고, description을 작성할 것인가
* 기본적으로 제공되는 description의 형식이나 상세함의 정도를 어떻게 설정할 것인가

뒤에서 Description에 대한 상세한 내용들이 나오지는 않고, 이러한 부분들을 고려하여 정의했다고 합니다. 









<br>

<br>

## Webtoon Comments Analysis

앞선 formative study에서 참가자들이 댓글을 활용하는 데 어려움을 겪는다는 사실을 알았습니다. 본 논문에서는 이를 위해 댓글 분석을 수행하고, 댓글의 사용 가능성에 대한 검토를 진행했습니다. 

댓글 분석은 5개 장르에서 2개 웹툰 씩 총 10개의 웹툰을 사용했습니다. 

![image-20220405145522644](https://user-images.githubusercontent.com/70505378/161763809-a17ca337-318d-465f-80b1-3477d8ad787d.png)

본 논문에서는 'Descriptive'의 범위에 시각적 요소에 대한 설명 뿐 아니라 상황적인 설명, 독자의 감정에 대한 설명도 포함했습니다. 이는 웹툰의 특성상 다른 매체들보다 조금 더 넓은 범위를 포함하도록 했다고 할 수 있습니다. 

댓글이 'descriptive' comment일 때 대부분의 댓글은 하나의 장면을 target하여 설명을 했으며, 드물게 여러 장면에 대한 설명을 하였습니다. 

또한 'non-descriptive' comment의 비율이 우세하고, 웹툰 시스템에서는 현재 추천순/시간순의 정렬 기능만을 제공하기 때문에 시각적 결함을 가진 사용자들이 이를 활용하기 힘들다는 것을 확인하였습니다. 



<be>

<br>

## COCOMIX

![image-20220405150221209](https://user-images.githubusercontent.com/70505378/161763814-4ef5b171-a6cf-497d-aafa-1a84d36866d5.png)

### Adaptive Description with Selective Details

2, 3번 design goal을 위해, 본 논문에서는 자체적으로 정의한 guideline과 webtoon comments를 이용해 adaptive description을 제공합니다. 

여기서 **adaptive description**이란 댓글에서 자주 언급되는 장면은 full description을 제공하고, 그렇지 않은 장면은 요약된 description을 제공하는 것을 말합니다. 그리고 이때 자주 언급되는 장면을 **key panel**이라고 칭했습니다. 

요약된 description이 제공될 경우에도, 사용자가 원할 경우 double tap을 통해 추가적인 정보를 제공받을 수 있습니다.

<br>

### Panel-anchored Comments

3, 4번 design goal을 위해, 장면과 연관된 댓글들을 분류해 제공했습니다. 

장면을 이해하는 데 도움이 되는 댓글들을 해당 장면에 배치하여 사용자가 선택적으로 정보를 제공받을 수 있도록 하였고, 이외의 댓글들도 웹툰의 마지막에서 접근 가능하도록 했습니다. 

<br>

### Computational Pipeline

![image-20220405182700740](https://user-images.githubusercontent.com/70505378/161763824-418455d4-ae5d-48d1-b007-c480e8886fa7.png)

그러면 위에서 제공하는 기능들을 어떻게 수행할 수 있는지에 대해 봐야겠죠? 위 그림은 Cocomix의 전체 pipeline 이미지입니다. 

각 부분에 대해 과정에 따라 설명해보겠습니다. 

**Data Collection and Preprocessing**

COCOMIX에서 사용하는 데이터로는 webtoon description과 webtoon comment가 있습니다. 

Webtoon comment는 아래 과정을 거쳐 사용 가능한 comment로 분류됩니다. 

1. 높은 추천 수의 댓글들을 수집
2. 부적절한 댓글 제거
   1. 3 단어 이하인 댓글 제거
   2. like-score가 10 미만인 댓글 제거
      * `like-score = number_of_likes - 2 * number_of_dislikes`

3. 전처리 수행

   1. 이모지 또는 이모티콘 제거

   2. 오탈자나 은어 수정 (using Bing Spell Check API)

   3. 구두점이나 대소문자 수정 (using Transformer-based punctuator/truecaser)

   4. 비영어 댓글 제거

   5. co-reference 문제 해결

      * ex. James have to go to school. **He** should ride a bycicle. 

      * 댓글들은 개별 문장으로 분리되어 사용되기 때문에 대명사를 대응하는 고유 명사로 교체

4. 댓글을 문장 단위로 분리 

**Scoring and abridging descriptions**

위에서 추출한 comment data를 이용해 key panel을 정의합니다. 앞서 말한 것처럼, comment data에서 자주 등장하는 요소를 가지는 장면을 key panel이라고 하고, key panel에서는 full description을 제공합니다. 

1. POS(part-of-speech) tagger를 이용해 댓글의 고유명사/명사/동사를 webtoon의 등장인물/물체/사건에 대응

2. Comment sentence의 focus score를 계산

   * CF(wk) = (# of comments that has wk) / (total # of comments)
   * PF(wk) = (# of panels that has wk) / (total # of panels)  

   ![image-20220405153111314](https://user-images.githubusercontent.com/70505378/161763819-b1be08b7-5674-47c8-8407-227f849a70d0.png)

3. Panel의 focus score를 계산
   * 해당 panel의 focus score는 sub-descriptions 들의 focus score를 모두 더한 값으로 계산
4. 상위 30% 점수의 장면을 key panel로 정의

Key panel을 제외한 panel들은 description의 길이를 **extractive summarization**(using find-tuned BERT)을 이용해 기존의 약 30~50% 수준으로 요약합니다. 

Abstract summarization 방법이 문장의 원래 구조와 상관없이 대체 어구들을 사용하여 문장을 요약하는 데 반해, 여기서 사용한 Extractive summarization은 원래 문장의 구조를 유지하면서 요약을 수행합니다. 

요약 결과에 포함되지 않는 문장들도 사용자의 요구(double-tap)에 따라 제공됩니다. 



**Extraction of descriptive comments**

문장 단위로 분리된 댓글들은 다음 순서에 따라 'descriptive comment'로 분류됩니다. 

1. Webtoon description에 대해 POS tagging을 수행하여 고유명사/명사/동사를 webtoon의 등장인물/물체/사건에 대응
2. Webtoon description과 정확한 단어 일치가 있거나, dialogue(인물 간 대화)와 세 단어 이상 연속적으로 일치하는 문장들만 추출
3. 질문 등을 제외한 Statement 형식을 띠는 문장만 추출
4. 과거/현재 시제인 문장만 추출

위 과정을 통해 최종 추출된 comment들은 'discriptive comment'로 분류되어 대응하는 panel에 대한 정보를 제공해줍니다. 



**Linking comments to relevant panels**

Descriptive comment들은 다음 과정에 의해 대응하는 panel에 연결됩니다. 

1. Comment와 description sentence들을 embedding하여 유사도를 계산
2. 할당된 comment와 해당 panel의 description 사이 겹치는 요소가 없으면 삭제
3. 할당된 comment와 다른 모든 panel 들 간의 similarity score를 계산해, 대응된 또는 이웃한 panel과의 similarity 만이 threshold 이상인 경우에 그대로 사용

**Presenting comments**

최종적으로 panel에 대응되지 않은 descriptive comment들도 webtoon의 마지막에서 참조할 수 있습니다. 또한 2개 이상의 연속된 panel에 대응하는 comment의 경우 스포일러를 방지하기 위해 마지막 panel에 대응시킵니다. 

하나의 panel에 여러 개의 comment가 대응된 경우, 이웃한 panel의 comment들과 유사성이 낮은 순으로 정렬합니다. 이는 비슷한 정보가 여러 차례 등장하는 것을 막아줍니다. 

사용자는 위로 스와이프하는 동작을 통해 해당 장면에 대응하는 댓글을 참조할 수 있습니다. 

<br>

<br>

## Evaluation

Evluation은 pipiline evaluation과 user evaluation으로 나뉩니다. 

### Pipeline Evaluation

Comment extraction과 panel linking을 평가하기 위해 10명의 시각적 결함을 가진 참가자들에게 기존의 추천수 기반 시스템과 COCOMIX 시스템을 평가하도록 요청했습니다. 

Comment extraction 평가를 위해 아래 질문을 했습니다. 

1. 댓글이 줄거리와 관련이 있는지
2. 댓글이 객관적인 묘사를 하는지
3. 댓글이 주관적인 묘사를 하는지
4. 댓글의 내용이 특정 인물을 가리키는지

그리고 panel linking을 위해 아래 질문을 했습니다. 

1. 댓글이 장면과 연관이 있는지
2. 댓글이 정확히 해당 장면에 대응하는지

결과는 아래와 같이 추천수 기반 정렬보다 COCOMIX의 댓글 제공 기능이 훨씬 더 뛰어난 평가를 받은 것을 알 수 있습니다. 

![image-20220405202531922](https://user-images.githubusercontent.com/70505378/161763827-65095748-ca79-459a-bc47-420d654ca9ac.png)







<br>

### User Evaluation

실제 사용 경험을 묻는 user evaluation에서는 아래 세 가지 질문을 했습니다. 

1. 웹툰을 읽는 중 실제로 interactive features를 어떻게 이용하는지
2. COCOMIX에 의해 제공된 댓글이 도움을 주는지
3. COCOMIX의 기능들이 사용하기 편하고 사용할 만 한지

**RQ1. Usage Pattern of Interactive Features**

네 명의 참가자들은 거의 매번 상세 정보를 확인했고, 다른 참가자들은 거의 확인하지 않았습니다. 또한 더 읽을수록 내용에 대한 이해가 뚜렷해져 상세 정보를 확인하는 빈도가 줄어드는 것을 확인했습니다. 

**RQ2. Perception of Comments Data**

모든 참가자들은 댓글을 추가적으로 참조하는 것을 선호했습니다. 댓글은 만화의 내용을 더 명확하게 해줌과 동시에, 다른 사람들과 상호작용 하는 느낌을 받았다고 이야기했습니다. 

**RQ3. Usability and Usefulness of Cocomix**

모든 참가자들은 Cocomix가 제공하는 interaction이 편리하고 쉽다고 이야기했습니다. 상호작용적 기능들은 사용자가 웹툰에 대해 충분한 control을 할 수 있다는 느낌을 주면서 만족도를 높였습니다. 

또한 Cocomix에 의해 제공되는 댓글들이 기존의 추천수 기반 댓글들보다 내용을 이해하는 데 더 도움이 된다고 이야기했습니다. 

<br>

아래는 Baseline과 Cocomix를 비교 평가한 결과표입니다. 특히나 정보의 양에 대한 control과 내용에 대한 이해 측면에서 훨씬 높은 점수를 받을 것을 볼 수 있습니다. 

![image-20220405203554336](https://user-images.githubusercontent.com/70505378/161763828-3c1ef879-7b62-4418-be79-a35b4ac6b2b0.png)







<br>

<br>

## Conclusion

이상 Cocomix의 목적성과 기능들, 그리고 그 기능들을 어떻게 구현했는지와 사용자들이 실제로 더 높게 평가한다는 것을 알 수 있었습니다. 

마지막으로 본 논문에서는 Cocomix의 적용 가능성과 한계, 그리고 앞으로의 연구 방향성에 대해 이야기합니다. 

* 확장 가능성

  * SNS 등 다른 매체로의 적용

  * 정보의 양이나 순서 등에 대한 더 정교한 설계

  * 댓글의 활용성을 확인. 하지만 동시에 충분한 양의 댓글이 확보되어야 한다는 한게점도 존재. 

  * 댓글 이외에 상호작용성을 높여줄 만한 데이터 소스 사용

* 한계점 및 연구 방향성

  * 제한된 숫자의 참가자들

  * 웹툰의 내용 뿐 아니라 그림체 등의 미적인 특성들을 제공할 방법 연구

  * 설명의 모호함이나 일관성 문제

  * double tap, swapping 외의 독자의 집중력을 떨어트리지 않을 만한 interaction 방법

<br>

<br>

_**논문에 대한 내용은 여기까지입니다. 아래부터는 개인적으로 새롭게 알고 느끼게 된 부분들을 정리하는 부분입니다.**_

<br>

# 새롭게 알게 된 것들

## Vocabulary

| Vocabulary   | meanings                                  |
| ------------ | ----------------------------------------- |
| surge        | 급등하다; 큰 파도                         |
| portmanteau  | 합성어                                    |
| elicit       | 추출하다                                  |
| distill      | 증류하다(정보를 선별하여 전달하다)        |
| imagery      | 형상                                      |
| pervasive    | 퍼지는, 만연한                            |
| render       | 세우다. 내다                              |
| juxtapose    | 나란히 하다(놓다)                         |
| infiltration | 침투                                      |
| adopt        | 입양하다, 채용하다. 채택하다, 양자로 삼다 |
| tactile      | 촉각                                      |
| braille      | 점자                                      |
| dialogue     | 대화                                      |
| onomatopoeia | 의성                                      |
| thematic     | 어간 형성 모음                            |
| viable       | 생존 가능한                               |
| recurring    | 반복되는                                  |
| alt-text     | 대체 텍스트                               |
| speculate    | 추측하다                                  |
| abridge      | 요약하다                                  |
| populate     | 채우다                                    |
| prone to     | ~하기 쉬운                                |
| syntatic     | 통사론의                                  |
| pragmatic    | 실천적인, 사실적인; 바쁜                  |
| strip        | 조각(stripped: 벗겨진)                    |
| draft        | 초안                                      |
| scalability  | 확장성                                    |
| catalyze     | 촉진하다                                  |
| peculiarity  | 특징, 특질                                |



<br>

## Domain-specific word









<br>

## Others

(기술이나 과정의 열거가 많아서 좀 빡셌다...)

<br>

<br>













