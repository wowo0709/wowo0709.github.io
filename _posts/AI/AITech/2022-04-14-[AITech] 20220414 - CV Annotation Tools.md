---
layout: single
title: "[AITech][Data Annotation] 20220414 - CV Annotation Tools"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 Upstage의 '이활석' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# CV Annotation Tools

이번 포스팅에서는 CV task에서 사용할 수 있는 Annotation tool에는 무엇이 있는지 간단히 살펴보겠습니다. 

## LabelMe

`LabelMe`는 처음에 MIT CSAIL에서 공개한 annotation tool이 서비스를 종료하고 이를 참고하여 만든 오픈소스 툴입니다. 

파이썬으로 작성되어 있어 코드를 커스터마이징하기 쉽다는 특징이 있습니다. 

Polygon, circle, rectangle, line, point 등의 annotation을 수행할 수 있습니다. 

![image-20220421105543364](https://user-images.githubusercontent.com/70505378/164361353-b126cf78-4f44-4b8f-a3e2-8043d5afded3.png)

LabelMe의 장단점은 아래와 같습니다. 

* 장점
  * 설치하기가 용이하다. 
  * python으로 작성되어 있어 추가적인 기능 추가가 가능하다. 
* 단점
  * 공동작업이 불가하다. (다수의 사용자가 사용할 수 없다)
  * object, image에 대한 속성을 부여할 수 없다.

<br>

## CVAT

다음으로는 Intel OpenVINO 팀에서 제작한 `CVAT`라는 툴이 있습니다. 

Image, video 등 일반적인 cv task에서 필요한 annotation 기능을 모두 포함하며, classification/detection/segmentation 등에 모두 활용할 수 있습니다. 

![image-20220421110106200](https://user-images.githubusercontent.com/70505378/164361355-e8777f56-33d5-4ed8-bdca-f51e3d97f372.png)

CVAT의 장단점은 아래와 같습니다. 

* 장점
  * 다양한 annotation을 지원한다. 
  * Automatic annotation 기능으로, 빠른 annotation이 가능하다. 
  * 온라인에서 바로 사용하거나, 또는 오픈소스도 제공되어 있어 on-premise로 설치하여 이용가능하다. 
  * Multi-user 기반 annotation이 가능하며 assignee, reviewer 을 할당할 수 있는 기능이 제공된다. 
* 단점
  * Model inference가 굉장히 느리다. 
  * Object, image에 대한 속성을 부여하기 까다롭다. 

<br>

## Hasty Labeling Tool

`Hasty Labeling Tool`도 cv annotation tool 중 하나입니다. 

특이하게 해당 툴에서 annotation 기능은 전체 제공 기능 중 일부이고, 데이터 제작/모델 학습/서빙/모니터링 까지 전체를 쉽게 할 수 있는 솔루션을 제공해주는 툴입니다. 

![image-20220421110812613](https://user-images.githubusercontent.com/70505378/164361352-ce486b17-660e-48be-a011-89d6aafa6999.png)

Hasty Labeling Tool의 장단점은 아래와 같습니다. 

* 장점
  * 다양한 annotation을 지원한다. 
  * Semi-automated annotation 기능을 지원한다. 
  * Cloud storage를 사용할 수 있다. (유료)
  * Multi-user 기반 annotation이 가능하며 assignee, reviewer 기능이 제공된다. 
* 단점
  * 서비스 자체가 free credit을 다 소진한 이후에는 과금을 해야 한다. 
  * Annotator가 수동으로 이미지마다 review state로 변경해주어야 한다. 
  * Hasty 플랫폼에 강하게 연결되어 있어, annotation 도구에 대한 커스터마이징이 불가능하다. 

<br>

## Summary

아래 표는 위에서 소개한 tool 들로 OCR annotation을 수행한다고 했을 때 기능들에 대한 제공 여부입니다. 

| OCR Annotation 관점 주요 특징들               | LabelMe | CVAT | Hasty Labeling Tool |
| :-------------------------------------------- | :-----: | :--: | :-----------------: |
| object(word)에 태깅이 가능한가?               |    X    |  O   |          O          |
| object(word) 별 grouping이 가능한가?          |    X    |  O   |          X          |
| 이미지에 tagging이 가능한가?                  |    X    |  X   |          O          |
| 다수의 사용자가 동시에 작업이 가능한가?       |    X    |  O   |          O          |
| 도구의 몇몇 기능을 커스터마이징 할 수 있는가? |    O    |  △   |          X          |
| Process가 직관적인가?                         |    O    |  X   |          X          |

LabelMe의 경우 본인이 간단한 프로젝트에서 글자 영역만 annotation 한다면 충분히 사용 가능하지만, 이미지 단위로 또는 영역 단위로 특정 속성을 부여할 수 없다는 단점이 있습니다. 또한 여러 사람이 동시에 공동 작업을 할 수 없다는 것도 치명적인 단점입니다. 

CVAT는 웹 기반 서비스이기 때문에 다수의 사용자가 공동 작업이 가능합니다. 하지만 CVAT 또한 tagging이 불가능하고, 작업 프로세스가 그렇게 직관적이지 않다는 불편이 존재합니다. 

Hasty Labeling Tool 또한 웹 기반 서비스로 공동 작업이 가능하지만, object 간 grouping이 불가능하고 도구 커스터마이징이 불가능하다는 점,  또한 유로로 사용해야 한다는 점 등이 단점입니다. 

<br>

CV Annotation tool이라는 키워드로 검색하면 여기서 소개한 tool 들 이외에도 다양한 tool들이 있으니 본인의 목적에 맞고 가장 편한 annotation tool을 찾는 것이 중요합니다. 

또한 대세를 이루는 annotation tool이 아직까지는 없기 때문에, 여러 기업에서는 자신들 만의 annotation tool을 만들고 이를 공개하려는 시도를 하는 중에 있습니다. 여기서 기준으로 삼은 OCR task에 fit한 OCR 전용 annotation tool들 또한 존재합니다. 

빠른 시일 내에 여러 기능을 제공하는 통합된 annotation tool이 만들어지지 않을까라는 기대를 해봅니다. 





























<br>

<br>

# 참고 자료

* 
