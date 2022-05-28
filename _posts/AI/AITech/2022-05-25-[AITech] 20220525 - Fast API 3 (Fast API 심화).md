---
layout: single
title: "[AITech][Product Serving] 20220525 - Fast API 3 (Fast API 심화)"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Event Handler', 'API Router', 'Error Handling', 'Background Task']
---



<br>

_**본 포스팅은 SOCAR의 '변성윤' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Fast API 3 (Fast API 심화)

## Event Handler

설명

* 이벤트가 발생했을 때, 그 처리를 담당하는 함수
* FastAPI에선 Application이 실행할 때, 종료될 때 특정 함수를 실행할 수 있음

코드

* `@app.on_event("startup")`
* `@app.on_event("shutdown")`

예시

* startup 할 때 머신러닝 모델 Load
* shutdown 할 때 로그 저장

![image-20220528153718912](https://user-images.githubusercontent.com/70505378/170814369-a45b5605-aca4-4cc8-94e2-073b40fbfc7f.png)





<br>

## API Router

설명

* API Router는 더 큰 애플리케이션들에서 많이 사용되는 기능
* API Endpoint를 정의
* Python Subpackage
* APIRouter는 Mini FastAPI로 여러 API를 연결해서 활용

코드

* 기존에 사용하던 @app.get, @app.post를 사용하지 않고, router 파일을 따로 설정하고 app에 import하여 사용
* `@{router명}.get`, `@{router명}.post`

예시

* user Router, order Router 2개 생성
* app에 연결 (include_router)
* 실제 활용 시에는 하나의 파일에 저장하지 않고 각각 다른 파일에 저장하여 app에서 import
  * user.py, order.py

![image-20220528154209346](https://user-images.githubusercontent.com/70505378/170814370-0b271d2b-6aaf-43b4-8b08-f428fcb001f0.png)

![image-20220528154217569](https://user-images.githubusercontent.com/70505378/170814371-b140590a-6769-4636-90b9-fed3ee9bcab2.png)

[예제 프로젝트 구조]

![image-20220528154259052](https://user-images.githubusercontent.com/70505378/170814372-50331275-3471-4f4f-bdb2-8d63b0aa945d.png)



<br>

## Error Handling

설명

* Error Handling은 웹 서버를 안정적으로 운영하기 위해 반드시 필요한 주제
* 서버에서 Error가 발생한 경우, 어떤 Error가 발생했는지 알아야 하고 요청한 클라이언트에 해당 정보를 전달해 대응할 수 있어야 함
* 서버 개발자는 모니터링 도구를 사용해 Error Log를 수집해야 함
* 발생하고 있는 오류를 빠르게 수정할 수 있도록 예외 처리를 잘 만들 필요가 있음

예시

* item_id가 1~3까지 존재, 4 이상의 숫자가 들어올 경우 Key Error 발생
* Internal Server Error

![image-20220528154537317](https://user-images.githubusercontent.com/70505378/170814373-46b40ff5-e4ff-454d-8818-ebfde87d2483.png)



* item_id가 5일 경우 Internal Server Error 500 return
* 이렇게 되면 클라이언트는 어떤 에러가 난 것인지 정보를 얻을 수 없고, 자세한 에러를 보려면 서버에 직접 접근해서 로그를 확인해야 함
* 에러 핸들링을 더 잘 하려면 에러 메시지와 에러의 이유 등을 클라이언트에 전달하도록 코드를 작성해야 함

![image-20220528154910129](https://user-images.githubusercontent.com/70505378/170814375-8af9c5d1-4d68-4247-87b2-d68635392d38.png)

**HTTPException**

* FastAPI의 **HTTPException** 클래스는 Error response를 더 쉽게 보내도록 해 줌
* HTTPException을 이용해서 클라이언트에게 더 자세한 에러 메시지 전송

![image-20220528154819748](https://user-images.githubusercontent.com/70505378/170814374-0bf127b8-876c-4f18-baa2-4468cc0a6380.png)

![image-20220528154942485](https://user-images.githubusercontent.com/70505378/170814376-b74ee351-5b12-4db1-97b3-ab59db3d0f19.png)





<br>

## Backbround Task

설명

* FastAPI는 **Stralett**이라는 비동기 프레임워크를 래핑해서 사용
* FastAPI의 기능 중 Background Tasks 기능은 오래 걸리는 작업들을 background에서 실행함
* Online Serving에서 CPU 사용이 많은 작업들을 background task로 사용하면, 클라이언트는 작업 완료를 기다리지 않고 즉시 response를 받아볼 수 있음

예시

* Background tasks를 사용하지 않은 작업들은 작업 시간만큼 응답을 기다림

![image-20220528155414819](https://user-images.githubusercontent.com/70505378/170814377-e96450b9-ad94-4107-8a2c-b258e27c3af2.png)

* Background tasks를 사용한 작업들은 기다리지 않고 바로 응답을 주기 때문에 0초 소요
  * 실제 작업은 background에서 실행

![image-20220528155454286](https://user-images.githubusercontent.com/70505378/170814379-4eeb45c0-47f8-41c8-b253-1f0dbea4fe26.png)

* 작업 결과물을 조회할 때는 Task를 어딘가에 저장해두고, GET 요청을 통해 Task가 완료됐는지 확인

![image-20220528155529906](https://user-images.githubusercontent.com/70505378/170814380-8d305248-4d86-4ce1-98e0-397aba69f3f5.png)































<br>

<br>

# 참고 자료

* 
