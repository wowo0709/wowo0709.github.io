---
layout: single
title: "[AITech][Product Serving] 20220523 - Fast API 1 (Fast API 개요)"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Rest API', 'Fast API']
---



<br>

_**본 포스팅은 SOCAR의 '변성윤' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Fast API 1 (Fast API 개요)

## Rest API

실제 회사에서 서비스를 제공할 때는 각 세부 서비스마다 별도의 server를 이용합니다. 

* 앱/웹 서비스를 위한 서버
* 머신러닝 서비스를 위한 서버
* 서비스 서버에서 머신러닝 서버에 예측을 요청하여 통신 (혹은 서비스 서버의 한 프로세스로 실행)

모든 서비스들을 하나의 server에서 처리하는 것을 **monolithic architecture**라고 하고, 별도의 server로 두는 것을 **microservice architecture**라고 합니다. 

<br>

서버 간 통신이 일어날 때 정해진 규약이 있어야 원활한 통신이 가능해집니다. 이를 HTTP라고 하며, 정보를 주고 받을 때 널리 사용되는 형식으로 **REST API**가 있습니다. 

* 각 요청이 어떤 동작이나 정보를 위한 것을 요청 모습 자체로 추론 가능
* 기본적인 데이터 처리: 조회 작업, 새로 추가, 수정, 삭제
  * CRUD: Create, Read, Update, Delete
* Resource: Unique한 ID를 가지는 리소스, URI
* Method: 서버에 요청을 보내기 위한 방식: GET, POST, PUT, PATCH, DELETE
  * GET : 정보를 요청하기 위해 사용(Read)
  * POST : 정보를 입력하기 위해 사용(Create)
  * PUT : 정보를 업데이트하기 위해 사용(Update)
  * PATCH : 정보를 업데이트하기 위해 사용(Update)
  * DELETE : 정보를 삭제하기 위해 사용(Delete)  

위의 HTTP method들 중 GET과 POST의 차이는 아래와 같습니다. 

* GET
  * 어떤 정보를 가져와서 조회하기 위해 사용되는 방식
  * URL에 변수(데이터)를 포함시켜 요청함
  * 데이터를 Header(헤더)에 포함하여 전송함
  * URL에 데이터가 노출되어 보안에 취약
  * 캐싱할 수 있음  
* POST
  * 데이터를 서버로 제출해 추가 또는 수정하기 위해 사용하는 방식
  * URL에 변수(데이터)를 노출하지 않고 요청
  * 데이터를 Body(바디)에 포함
  * URL에 데이터가 노출되지 않아 기본 보안은 되어 있음
  * 캐싱할 수 없음(다만 그 안에 아키텍처로 캐싱할 수 있음)  

![image-20220523141742613](https://user-images.githubusercontent.com/70505378/169752622-3148bc06-e429-4335-9442-273a9d95e1d2.png)

<br>

[Status Code]

Status code는 클라이언트의 요청에 따라 서버가 어떻게 반응하는지를 알려주는 code입니다. 

* 1xx(정보) : 요청을 받았고, 프로세스를 계속 진행함
* 2xx(성공) : 요청을 성공적으로 받았고, 실행함
* 3xx(리다이렉션) : 요청 완료를 위한 추가 작업이 필요
* 4xx(클라이언트 오류) : 요청 문법이 잘못되었거나 요청을 처리할 수 없음
* 5xx(서버 오류) 서버가 요청에 대해 실패함  

<br>

## Fast API

![image-20220523144634340](https://user-images.githubusercontent.com/70505378/169752629-de360c34-a1a6-443f-94a3-3f78d3ba9694.png)

**Fast API**는 최근 떠오르는 Python Web Framework입니다. 

Jetbrain Python Developer Servey 기준

* 2021: **FastAPI (14%)**, Flask (46%), Django (45%)
* 2020: **FastAPI (12%)**, Flask (46%), Django (43%)
* 2019: **FastAPI(없음)**, Flask(48%), Django (44%)  

Fast API의 특징으로는 아래와 같은 부분들이 있습니다. 

![image-20220523144656320](https://user-images.githubusercontent.com/70505378/169752630-3d81e402-49c2-4cd7-9c46-8b9405e1fb5a.png)

다양한 장점들에 힙입어, 그 사용률이 가파른 추세로 증가하고 있습니다. 

[FastAPI vs Flask]

* Flask보다 간결한 Router 문법

![image-20220523144800676](https://user-images.githubusercontent.com/70505378/169752632-8015e004-5b3f-4ca1-9cf5-fcb7c49ff300.png)

* Asynchronous(비동기) 지원
* Built-in API Documentation (Swagger)

![image-20220523144847145](https://user-images.githubusercontent.com/70505378/169752635-38fdb457-1730-45e4-b050-52a978c82f22.png)

* Pydantic을 이용한 Serialization 및 Validation

* 아직까지는 Flask의 유저가 더 많음
* ORM 등 Database와 관련된 라이브러리가 Flask에 비해 적음

<br>

[프로젝트 구조]

* 프로젝트의 코드가 들어갈 모듈 설정(app). 대안 : 프로젝트 이름, src 등
* \_\_main\_\_.py는 간단하게 애플리케이션을 실행할 수 있는 Entrypoint 역할 (참고)
* Entrypoint : 프로그래밍 언어에서 최상위 코드가 실행되는 시작점 또는 프로그램 진입점
* main.py 또는 app.py : FastAPI의 애플리케이션과 Router 설정
* model.py는 ML model에 대한 클래스와 함수 정의  

![image-20220523145045698](https://user-images.githubusercontent.com/70505378/169752636-69ffbbd9-a654-48ee-a175-bbfc74711183.png)

[Poetry]

* Dependency Resolver로 복잡한 의존성들의 버전 충돌을 방지
* Virtualenv를 생성해서 격리된 환경에서 빠르게 개발이 가능해짐
* 기존 파이썬 패키지 관리 도구에서 지원하지 않는 Build, Publish가 가능
* pyproject.toml을 기준으로 여러 툴들의 config를 명시적으로 관리
* 새로 만든 프로젝트라면 poetry를 사용해보고, virtualenv 등과 비교하는 것을 추천

![image-20220523145207871](https://user-images.githubusercontent.com/70505378/169752638-8151a3b0-3c79-4273-9864-bd3f6095d290.png)

[Swagger]

* REST API 설계 및 문서화할 때 사용
* Swagger가 유용한 이유
  * 다른 개발팀과 협업하는 경우
  * 구축된 프로젝트를 유지보수하는 경우
  * 클라이언트는 Swagger를 참고하여 사용 방법을 익힐 수 있음
* 기능
  * API 디자인, 빌드, 문서화, 테스팅



<br>

<br>

# 참고 자료

* 
