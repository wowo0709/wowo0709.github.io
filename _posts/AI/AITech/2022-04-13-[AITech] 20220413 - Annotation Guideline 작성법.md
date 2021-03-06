---
layout: single
title: "[AITech][Data Annotation] 20220413 - Annotation Guideline 작성법"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 Upstage의 '이활석' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Annotation Guideline 작성법

## 가이드라인이란?

**가이드 라인**: 좋은 데이터를 확보하기 위한 과정을 정리해놓은 문서

좋은 데이터란?

* 골고루 모여 있는 데이터
* 일정하게 라벨링된 데이터

![image-20220421175819134](https://user-images.githubusercontent.com/70505378/164431820-e237cd15-c741-4437-b7af-83116b7dc2cf.png)

가이드 라인에는 아래와 같은 요소들이 명확하게 정의되어 있어야 한다. 

* 데이터 구축의 목적
* 라벨링 대상 이미지 소개
* 기본적인 용어 정의
  * BBox, 전사, 태그 등
* Annotation 규칙
  * 작업 불가 이미지 정의
  * 작업 불가 영역 (illegibility = True) 정의
  * Bbox 작업 방식 정의
  * 최종 format

가이드 라인의 annotation 규칙을 정의할 때는 세가지 요소가 조화롭게 있어야 한다. 

* 특이 케이스

  ![image-20220421175725916](https://user-images.githubusercontent.com/70505378/164431814-677d7359-5264-4242-bcd5-d3dd11bea1d2.png)

* 단순함

  * 모든 특이 케이스를 다루는 것은 너무 많은 분량을 요구하고, 작업자들이 모두 숙지할 수 없기 때문에 라벨링 노이즈가 발생

* 명확함

  * 동일한 가이드 라인에 대해 같은 해석이 가능하도록 작성

<br>

## 데이터셋 제작 파이프라인

![image-20220421180057306](https://user-images.githubusercontent.com/70505378/164431821-b3ce8acb-d419-4154-8ae3-125ad00ec48f.png)

Raw Image 수집 시에는 crawling을 사용한다. 

![image-20220421180210014](https://user-images.githubusercontent.com/70505378/164431822-3881bfc9-7022-4ec8-b66d-b5cecea178a1.png)

* 검색어: 검색어의 집합으로 글자가 등장하는 대부분 상황에 대한 이미지들을 크롤링 할 수 있어야 한다. 
* 조건 설정
  * 크기: 이미지 크기는 되도록 큰 것으로
  * 색상: augmentation이 가능하기 때문에 조건을 걸지 않음
  * 유형: 이미지 파일. 배경이 투명한 이미지는 문제를 일으킬 수 있음.
  * 시간: 신경 안 써도 됨. 
  * 사용권: 상업적으로 사용 시 반드시 조건을 걸어서 해당 라이선스를 갖는 이미지만 모이도록 함

Crawling으로 이미지를 수집한 후에는 Filtering 작업이 필요하다. 

![image-20220421180523604](https://user-images.githubusercontent.com/70505378/164431826-c19e6100-410a-4464-9943-976e6b567aa1.png)

또는 crawling 이외에 자체적으로 이미지를 수집하는 crowd sourcing 방법이 있을 수 있습니다. 

* 수집용 가이드라인 제공
* Edge case 수집에 유리
  * 원하는 특성 명시
  * 좋은 예, 나쁜 예를 직접 사람에게 학습시킬 수 있음
* 시간, 비용이 많이 들기 때문에 일반적으로 크롤링된 이미지에 추가하는 방식으로 진행
  * 구하기 힘든 이미지
* 개인정보, 저작권 이슈에서 자유로움

![image-20220421180742878](https://user-images.githubusercontent.com/70505378/164431828-5a84ff94-31fc-420a-94e0-c47fd60989a0.png)

<br>

## 가이드라인 작성법

![image-20220421180901023](https://user-images.githubusercontent.com/70505378/164431830-182dbe08-6895-4261-b5de-219fcc18f0f5.png)

Data annotation 시 초기에 매우 많은 QnA가 있기 때문에, 처음에 아주 소량의 데이터로 우선 pilot labelling 작업을 통해 가이드라인의 완성도를 빠르게 높이는 것이 중요. 

라벨링 방식에 대해서 탐색을 많이 해야 하거나, 일반 사람들에게 익숙하지 않는 데이터에 대해 라벨링을 할 경우에는 가이드 작성의 횟수가 높아 가이드 작성/수정/교육에 대한 비용이 크기 때문에 외주보다 차라리 내부에서 인력을 고용해서 진행하는게 효율적일 수 있음.

가이드라인 제작 시에는 versioning이 매우 중요하기 때문에 각 version을 백업하고, 어느 문서를 기반으로 수정하였는지 등을 명확히 기록. 

가이드라인 구축 전 약 3%의 데이터를 직접 pilot labelling 해보면서 직접 데이터에 대해 이해하고 다양한 edge case를 확인하는 것이 중요. 이때 Annotation tools 포스팅에서 다룬 오픈소스 어노테이션 툴들을 활용할 수 있음

<br>

**기본적인 용어 정의**

| 용어          | 설명                                                         |
| ------------- | ------------------------------------------------------------ |
| HOLD          | 작업을 진행하지 않고, 이미지 전체를 제외하는 처리            |
| Points        | 글자 영역에 대한 표시 방법                                   |
| Transcription | Points 안에 존재하는 글자 시퀀스                             |
| Illegibility  | 글자 번짐, 잘림 등으로 인해 글자를 정확히 알아보기 힘들 경우 모델이 의도적으로 무시하도록 표시 <br />영역 단위의 처리로, transcription 대상이 되는 Points처럼 타이트하게 영역을 지정할 필요 없음 |
| Image_Tags    | 이미지 자체에 특이사항이 있는경우, 내용 표시                 |
| Word_Tags     | 글자 영역의 특이사항이 있는경우, 내용 표시                   |
| \<UNK\>       | 글자 영역 내에 글자가 있지만, 매우 특수하거나 실제로 인식하여 출력하기 어려운 글자의 경우 입력해주는 값 <br />인식 대상이 아닌 글자를 (예, 특수기호, 한글이나 알파벳이 아닌 글자) transcription할 때 표시 |

<br>

**Annotation 규칙**

HOLD 이미지 기준

* 이미지 내에 글자 영역이 존재하지 않는 이미지

  ![image-20220421183714863](https://user-images.githubusercontent.com/70505378/164431834-6c497984-dfe0-4f9e-99de-42e107467a28.png)

* 이미지의 모든 글자 영역 속 글자를 알아보기 어려운 경우

  ![image-20220421183756084](https://user-images.githubusercontent.com/70505378/164431839-20a27a6d-3f8d-4391-9654-cb54869da44c.png)

* 같은 글자 혹은 패턴이 5회 이상 반복되는 이미지

  ![image-20220421183805099](https://user-images.githubusercontent.com/70505378/164431842-59ae9d2d-3b1b-4d17-8495-fd128d7b06ef.png)

* 영어, 한국어가 아닌 외국어가 1/3 이상인 이미지

  ![image-20220421183813827](https://user-images.githubusercontent.com/70505378/164431845-f384bf1c-a20c-4903-a5ce-1c0e8a292d84.png)

* 개인 정보가 포함된 이미지 (단, 방송 캡쳐는 제외)

  ![image-20220421183937382](https://user-images.githubusercontent.com/70505378/164431774-91e92538-f087-401c-b74d-63b2d98b08a4.png)

* Born-digital 이미지 (화면 캡쳐, 카드뉴스, 작업물 시안 등 사람이 촬영한 이미지 이외의 처음부터 디지털 형태로 생성된 이미지)

  ![image-20220421184020620](https://user-images.githubusercontent.com/70505378/164431782-7d2cabf3-399b-4fc0-92a0-959a20f33061.png)

작업 불가 영역

`don't care`에 해당하는 영역. 학습 시 마스킹 처리하여 학습에 사용하지 않음. 

* Rule 1: 글자를 알아보기 어려울 정도로 밀도가 높거나,'글자가 일부 뭉개져서 알아보기 어려운 영역에 대해서 illegibility: True

  ![image-20220421184748404](https://user-images.githubusercontent.com/70505378/164431808-8e426a72-056c-469a-a65d-d94dd4e53716.png)

* Rule 2: 글자가 겹쳐져 있어 육안 상 글자를 정확하게 입력할 수 없다면 illegibility: True

  ![image-20220421184739131](https://user-images.githubusercontent.com/70505378/164431806-7edb8f61-6e98-4e33-ba5c-4609c88eaf4f.png)

* Rule 3: Illegibility: True인 영역은 tight하게 영역을 지정할 필요 없음

* Rule 4: 글자와 유사하지만 글자가 아닌 경우에는 Points 표기하지 않음  

작업 대상 영역

작업하는 글자 영역에 해당하는 요소들도 명확하고 구체적인 규칙을 명시해야 함

![image-20220421184932878](https://user-images.githubusercontent.com/70505378/164431812-c10880f0-7b90-44b0-8d7f-342687035b13.png)

최종 포맷

* 포맷에 대한 설명과 예시 json 파일을 첨부해서 전달
* 데이터 포맷이 변경되면 이전에 작업된 파일들도 최신 포맷으로 변경해야 데이터를 활용할 수 있음
* 데이터 포맷 고민 시 수정될 때의 이전 데이터 활용 여부도 반드시 고려

**BBox 작업 방식 정의**

* Points 영역
  * Points의 크기는 최소한 해당 글자들이 다 포함되는 영역으로 지정합니다. 박스의 타이트함은 상대적이라 Annotation 수행자의 재량에 맡기되, 느슨하게 박스를 표기하는 것은 지양합니다  

![image-20220421184141485](https://user-images.githubusercontent.com/70505378/164431789-16a4e06b-d7fd-404f-929d-af317acf5204.png)

* 구부러진 글자 영역
  * 단어가 심하게 곡선으로 배열되어 **타이트한 사각형 Points로 라벨링을 못할 경우** 짝수 개의 점들로 이루어진 polygon 형태의 Points를 지정합니다. 글자의 위아래에 점이 쌍을 이루게 하여, 점들을 기준으로 사각형 모양의 박스가 만들어지는 것이 좋습니다. **최대 점의 개수는 12개로 제한합니다 (위, 아래 각 6개). 첫 번째 글자로부터 2글자 혹은 4글자에 점을 한 개 찍습니다.**

![image-20220421184241743](https://user-images.githubusercontent.com/70505378/164431792-15b3eb4f-e4ec-451b-8193-e4cdc5e73f89.png)

![image-20220421184257051](https://user-images.githubusercontent.com/70505378/164431794-ccda9c56-b351-43c2-83f7-1cda0456b567.png)

* 진행 방향 및 그에 따른 좌표 순서

![image-20220421184328935](https://user-images.githubusercontent.com/70505378/164431799-a6f89b9e-6cc7-4ffc-8550-95b2aaee56b7.png)

* 그 외
  * 어디까지를 하나의 글자 영역으로 정의할 것인자?
    * 띄어쓰기 기준으로 분리할 것인지
    * 한 단어 내에서 글자 크기가 급격히 차이나는 경우 분리할 것인지
    * 글자가 변형되어 일반적인 표시 영역을 벗어나는 경우는 어떻게 처리할 지

![image-20220421184451367](https://user-images.githubusercontent.com/70505378/164431802-aff30a25-cae7-40a6-998f-2924a420f7f4.png)

<br>

> **Annotation guide는 절대 한 번에 완성되지 않고, 반복적인 작업을 통해 더 좋은 가이드를 만드는 것이 중요하다! 동시에 완벽한 가이드는 존재하지 않는다!**

가이드 라인에도 우선순위가 필요하다. 

Annotation 레벨에서 반드시 정확히 라벨링되어야 할 것이 무엇인지 우선순위를 아는 것이 중요하다. 

1. 읽을 수 있는 글자 영역 전부 Points 표시하기
2. Points 표시의 일관성 유지 및 transcription 정확히 하기
3. 글자는 존재하지만 육안 상 알아보기 어려운 illegibility=True 영역 (don't care) annotation
4. 각종 태그: 언어, 글자 진행 방향, 이미지 태그 및 단어 태그

<br>

Annotation 시 검수하는 과정 또한 매우 중요하다. 

1. 감독자 전수 검사
   * 한 감독자가 본인에게 할당된 작업자의 결과물 모두 시각화 하여 문제가 없는지 확인하고 문제가 있을 시에는 문제 있는 부분을 기록하여 “다른” 작업자에게 할당  
2. Peer check
   * 끝난 작업물을 다른 작업자에게 할당하여 틀린 부분을 찾아서 고치게 함
3. 다수결
   * 여러 사람이 동일한 작업을 진행하고 그 결과를 프로그래밍 적으로 하나로 합침  

아래와 같은 tip들을 참고하자. 

1. 초기에 소량씩 완성본을 받아서 품질을 확인
   * 온전한 가이드라인이 나오기 까지는 실험 후 수정의 과정이 여러 번 필요하기 때문에 처음에는 최대한 빠르게 이 iteration을 가져가는 것이 좋다  
2. 작업자 QnA 활용 가이드라인 수정
   * 작업자들의 문의 사항이 왔을 때 이 질문이 어떤 가이드라인 때문인지를 확인하여 질문이 나오지 않게 하려면 어떻게 수정해야 할지를 고민한다  
3. 추가 수정을 위한 비용과 시간이 크다면 과감한 포기
   * 어느 정도 안고 갈 라벨링 노이즈가 무엇일지 개발팀과 얘기 나눠서 정하면, 가이드라인 별 우선순위가 정해지기 때문에 진행이 훨씬 수월하다  

<br>

## Summary

* 충분한 pilot tagging을 바탕으로 가이드 제작
* 가이드라인 수정 시 versioning 필요, 기존 내용과 충돌 없도록 최소한의 변경만
* 최대한 명확하고 객관적인 표현을 사용
* 일관성 있는 데이터가 가장 잘 만들어진 데이터
* 우선순위를 알고, 필요하다면 포기하는 것도 중요

<br>

<br>

# 참고 자료

* 
