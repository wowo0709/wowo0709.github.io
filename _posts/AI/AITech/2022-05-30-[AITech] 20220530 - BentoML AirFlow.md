---
layout: single
title: "[AITech][Product Serving] 20220530 - BentoML & AirFlow"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 SOCAR의 '변성윤' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# BentoML & AirFlow

## BentoML

`BentoML`은 **Model Serving**에 특화된 **Serving Infrastructure**입니다. 

처음에는 Fast API로 서버를 개발하여 serving하는 것이 좋지만, 다수의 모델들을 만들고 서빙하다 보면 **반복되는 작업**이 존재합니다. 

이러한 반복되는 작업들을 추상화하여 serving 과정을 더 빠르고 간단하게 만들어주는 것이 Serving Infrasturcture의 의의입니다. 

<br>

이러한 Serving Infrasturcture로 많은 라이브러리들이 등장하고 있으며, 각 라이브러리들은 해결하려고 하는 핵심 문제가 무엇인지, 그 문제를 어떻게 해결했는지가 다르다고 할 수 있습니다. 

그 중 BentoML은 Serving에 특화된 라이브러리라고 할 수 있습니다. 

**BentoML이 해결하는 문제**

* 문제 1: Model Serving Infra의 어려움
  * Serving을 위해 다양한 라이브러리, Artifact, Asset 등 사이즈가 큰 파일을 패키징
  * Cloud Serving에 지속적인 배포를 위한 많은 작업이 필요
  * BentoML은 CLI로 이 문제의 복잡도를 낮춤 (CLI 명령어로 모두 진행 가능하도록)
* 문제 2: Online Serving의 Monitoring 및 Error Handling
  * Online Serving으로 API 형태로 생성
  * Error 처리, Logging을 추가로 구현해야 함
  * BentoML은 Python Logging Module을 사용해 Access Log, Prediction Log를 기본으로 제공
  * Config를 수정해 Logging도 커스텀할 수 있고, Prometheus 같은 Metric 수집 서버에 전송할 수 있음
* 문제 3: Online Serving 퍼포먼스 튜닝의 어려움
  * BentoML은 Adaptive Micro Batch 방식을 채택해 동시에 많은 요청이 들어와도 높은 처리량을 보여줌

<br>

현재 많이 사용되는 serving infra로는 아래와 같은 라이브러리들이 존재하며, BentoML은 2019년 출시되어 꾸준한 상승세를 보이고 있습니다. 

![image-20220603180735432](https://user-images.githubusercontent.com/70505378/171828350-1a2cd321-ae37-434e-b1b2-4d750ed7da4d.png)

**BentoML의 특징**

* 쉬운 사용성
* Online/Offline Serving 지원
* Tensorflow, PyTorch, Keras, XGBoost 등 Major 프레임워크 지원
* Docker, Kubernetes, AWS, Azure 등의 배포 환경 지원 및 가이드 제공
* Flask 대비 100배의 처리량
* 모델 저장소(Yatai) 웹 대시보드 제공
* 데이터 사이언스와 DevOps 사이의 간격을 이어주며 높은 성능의 Serving이 가능하게 함















<br>

<br>

## Apache AirFlow

`Apache AirFlow`는 **Batch Process**를 위한 라이브러리입니다. 

Batch process란 **예약된 시간에 주기적으로 실행되는 프로세스**입니다. 

* 모델을 주기적으로 학습시키는 경우 사용 (Continuous Training)
* 주기적인 Batch Serving을 하는 경우 사용
* 그 외 개발에서 필요한 배치성 작업
* 예) 매주 일요일 07:00에 새로 수집한 데이터로 모델 재학습

<br>

AirFlow의 등장 전에는 `Linux Crontab`으로 Batch process를 구축하였습니다. 

![image-20220603181529889](https://user-images.githubusercontent.com/70505378/171828351-7b31ecc0-5d18-4a5b-b5ec-e314fed20118.png)

Linux에 기본적으로 설치되어 있는 Crontab을 이용하면 간단하게 batch process를 구현할 수 있습니다. 

하지만 간단한 만큼, Linux Crontab을 사용할 때의 한계점들이 존재합니다. 

* **재실행 및 알람**
  * 파일을 실행하다 오류가 발생한 경우, 크론탭이 별도의 처리를 하지 않음
  * 예) 매주 일요일 07:00에 predict.py를 실행하다가 에러가 발생한 경우, 알람을 별도로 받지 못 함
* 실패할 경우, 자동으로 몇 번 더 실행(Retry)하고, 그래도 실패하면 실패했다는 알람을 받으면 좋음
* **과거 실행 이력 및 실행 로그**를 보기 어려움
* 여러 파일을 실행하거나, **복잡한 파이프라인**을 만들기 어려움

<br>

이를 위해 좀 더 정교한 스케쥴링 및 워크플로우 도구들이 등장하였습니다. 

![image-20220603182110140](https://user-images.githubusercontent.com/70505378/171828355-de952733-06d2-44e3-b578-3ad84e76eb9b.png)

이 중 **Apache AirFlow**는 현시점에서 스케쥴링, 워크플로우 도구의 표준이라고 할 수 있습니다. (가장 많이 사용되고 있습니다)

아래는 AirFlow의 약력입니다. 

* 에어비앤비(Airbnb)에서 개발
* 현재 릴리즈된 버전은 2.2.0으로, 업데이트 주기가 빠름
* 스케쥴링 도구로 무거울 수 있지만, 거의 모든 기능을 제공하고 확장성이 넓어 일반적으로 스케쥴링과 파이프라인 작성 도구로 많이 사용
* 특히 데이터 엔지니어링 팀에서 많이 사용

<br>

**AirFlow에서 제공하는 기능**

* 파이썬을 사용해 스케쥴링 및 파이프라인 작성

![image-20220603182540077](https://user-images.githubusercontent.com/70505378/171828360-889a60f3-5b40-4a9e-bd1d-a1a8b38dfad6.png)

* 스케쥴링 및 파이프라인 목록을 볼 수 있는 웹 UI 제공

![image-20220603182600256](https://user-images.githubusercontent.com/70505378/171828363-3d43a16e-c719-4625-ab4e-a454cd37d92e.png)

![image-20220603182609534](https://user-images.githubusercontent.com/70505378/171828366-70372cc6-4bbe-4e83-a032-686e19409d5a.png)

![image-20220603182619785](https://user-images.githubusercontent.com/70505378/171828343-0f00f045-7722-4d84-94d4-ac7cad6199a7.png)

* 실패 시 알람, 재실행 시도
* 동시 실행 worker 수
* 설정 및 변수 값 분리

















<br>

<br>

# 참고 자료

* 
