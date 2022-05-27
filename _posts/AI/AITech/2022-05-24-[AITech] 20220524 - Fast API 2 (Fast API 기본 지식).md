---
layout: single
title: "[AITech][Product Serving] 20220524 - Fast API 2 (Fast API 기본 지식)"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Fast API', 'Pydantic']
---



<br>

_**본 포스팅은 SOCAR의 '변성윤' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Fast API 2 (Fast API 기본 지식)

## Fast API 기본 지식

### Path Parameter & Query Parameter

[Path Parameter]

* 웹에서 GET 메서드(request header(url)에 데이터 저장)를 사용해 데이터를 전송할 수 있음
* 설명: 서버에 402라는 값을 전달하고 변수로 사용
  * 예시: `/users/402`

![image-20220527091013499](https://user-images.githubusercontent.com/70505378/170616276-4285629b-61a2-4140-8a51-c2d57a44f82f.png)

[Query Parameter]

* 웹에서 GET 메서드(request header(url)에 데이터 저장)를 사용해 데이터를 전송할 수 있음
* 설명: API 뒤에 입력 데이터를 함께 제공하는 방식으로 사용. Query String은 key, value 쌍으로 이루어지며 &로 연결해 여러 데이터를 넘길 수 있음
  * 예시: `/users?id=402`

![image-20220527091052576](https://user-images.githubusercontent.com/70505378/170616282-4b0a1ac5-4527-439d-a492-5ad80eab0e61.png)

[언제 무엇을 사용해야 할까?]

* Path parameter: 경로에 존재하는 내용이 없을 시 **404 Error** 발생
  * resource를 식별해야 하는 경우에 적합
* Query Parameter: 데이터가 없는 경우 **빈 리스트**가 나옴. => 추가적인 예외 처리 필요
  * 정렬, 필터링을 해야 하는 경우에 적합





### Optional Parameter

특정 파라미터는 선택적으로 사용하고 싶은 경우

* typing 모듈의 Optional 사용
* Optional을 사용해 해당 파라미터는 Optional임을 명시 (기본 값은 None)

![image-20220527091247528](https://user-images.githubusercontent.com/70505378/170616283-2301fcb4-21c7-4396-b1b2-e49fdf1fbfc9.png)



### Request Body

* 클라이언트 -> API 데이터 전송 시: Request Body
  * **Request Body에 데이터를 보내고 싶다면 POST 메서드를 사용해야 함**
* Body의 데이터를 설명하는 Content-Type이란 Header 필드가 존재하고, 어떤 데이터 타입인지 명시해야 함

대표적인 컨텐츠 타입

* application/x-www-form-urlencoded : BODY에 Key, Value 사용. & 구분자 사용
* text/plain : 단순 txt 파일
* multipartform-data : 데이터를 바이너리 데이터로 전송  

예제

* pydantic로 Request Body 데이터 정의 (`Item`)
* post 메서드의 함수 인자 type hinting에 생성한 `Item` class 주입
* Request body 데이터를 validation

![image-20220527092645370](https://user-images.githubusercontent.com/70505378/170616285-75b52b7a-2400-44c1-9c12-cd48bbbb208e.png)



### Response Body

* API -> 클라이언트 데이터 전송 시: Response Body
  * **Response Body에 데이터를 보내고 싶다면 POST 메서드를 사용해야 함**
* Decorator의 response_model 인자로 주입 가능

예제

* Output Data 형식을 해당 정의에 맞게 변형
* 데이터 Validation
* Response에 대한 Json Schema 추가
* 자동으로 문서화

![image-20220527092855041](https://user-images.githubusercontent.com/70505378/170616287-f5076d6f-00ea-46c7-8f54-63152ffff724.png)







### Form, File

[Form]

* Form(입력) 형태로 데이터를 받고 싶은 경우
* python-multipart 설치 필요
* 프론트도 간단히 만들기 위해 Jinja2 설치

예제

* Form 클래스 사용 시 Request의 Form Data에서 값을 가져옴

* Request 객체로 Request를 받음
* 파이썬에서 사용할 수 있는 템플릿 엔진: Jinja Template -> 프론트엔드 구성
  * templates.TemplateResponse로 해당 HTML로 데이터를 보냄

![image-20220527094200133](https://user-images.githubusercontent.com/70505378/170616288-3e5b0bcc-d3e5-4494-89cb-67fa18a2a281.png)

웹 서버 실행 시 아래 화면 출력

![image-20220527094245484](https://user-images.githubusercontent.com/70505378/170616290-72010911-cb34-47e1-971a-73c6accb465e.png)

입력 후 제출을 누르면 login 함수 실행 (POST 요청)

![image-20220527094309116](https://user-images.githubusercontent.com/70505378/170616292-39c7f3b5-84f3-4c1e-81c8-3c56d89ec369.png)

[File]

* File을 업로드하고 싶은 경우
* 마찬가지로 python-multipart 설치 필요

예제

* import UploadFile
* "/"로 접근할 때 보여줄 HTML 코드

![image-20220527094446787](https://user-images.githubusercontent.com/70505378/170616293-54be3424-1a96-453e-9acf-1aba4b81d436.png)

* HTML에서 action으로 넘김

![image-20220527094457341](https://user-images.githubusercontent.com/70505378/170616294-027adcec-8af1-4b48-a74e-783ba504cf16.png)

* 파일을 Bytes로 표현하고, 여러 파일은 List에 설정

![image-20220527094520943](https://user-images.githubusercontent.com/70505378/170616295-bf645933-bb04-4a8f-a69a-7f488a37a13c.png)

웹 서버 실행 시 아래 화면 출력

![image-20220527094543344](https://user-images.githubusercontent.com/70505378/170616297-0a454ebc-dc87-4dd4-b6b7-45bfd4d8383a.png)

<br>

<br>

## Pydantic

![image-20220527105753173](https://user-images.githubusercontent.com/70505378/170616298-ed39193c-cce5-4ae7-8490-3302b991b696.png)

* FastAPI에서 Class 사용할 때 보이던 Pydantic
* Data Validation / Settings Management 라이브러리
* Type Hint를 런타임에서 강제해 안전하게 데이터 핸들링
* 파이썬 기본 타입(String, Int 등) + List, Dict, Tuple에 대한 Validation 지원
* 기존 Validation 라이브러리보다 빠름 (Benchmark)
* Config를 효과적으로 관리하도록 도와줌
* 머신러닝 Feature Data Validation으로도 활용 가능  





### Validation

* **Machine Learning Model Input Validation**
* Online serving에서 Input 데이터를 Validation하는 case

[Validation Check Logic]

* 조건 1. **올바른 url**을 입력 받음 (url)
* 조건 2. **1-10 사이의 정수** 입력 받음 (rate)
* 조건 3. **올바른 폴더 이름**을 입력 받음 (target_dir)

[사용할 수 있는 방법]

1. 일반 Python Class를 활용한 Input Definition 및 Validation
   * Python Class로 Input Definition 및 Validation => 의미 없는 코드가 많아짐
   * 복잡한 검증 로직엔 Class Method가 복잡해지기 쉬움
   * Exception Handling을 어떻게 할지 등 커스텀하게 제어할 수 있는 있지만 메인 로직(Input을 받아서 Inference를 수행하는)에 집중하기 어려워짐  

![image-20220527110159053](https://user-images.githubusercontent.com/70505378/170616300-84c777eb-8201-4c9c-b8d7-fd6a41e2374d.png)

2. Dataclass를(python 3.7 이상 필요) 활용한 Input Definition 및 Validation
   * 인스턴스 생성 시점에서 Validation을 수행하기 쉬움
   * 여전히 Validation 로직들을 직접 작성해야 함
   * Validation 로직을 따로 작성하지 않으면, 런타임에서 type checking을 지원하지 않음  

![image-20220527110328863](https://user-images.githubusercontent.com/70505378/170616302-3eabd3e1-a1d8-4cce-b1a3-e63f3ec0fc50.png)

![image-20220527110341828](https://user-images.githubusercontent.com/70505378/170616304-ed12b046-d36c-496c-b805-7f303bdd5929.png)

3. Pydantic을 활용한 Input Definition 및 Validation  
   * 훨씬 간결해진 코드 (6라인)(vs 52라인 Python Class, vs 50라인 dataclass)
   * 주로 쓰이는 타입들(http url, db url, enum 등)에 대한 Validation이 만들어져 있음
   * 런타임에서 Type Hint에 따라서 Validation Error 발생
   * Custom Type에 대한 Validation도 쉽게 사용 가능  

![image-20220527110604992](https://user-images.githubusercontent.com/70505378/170616308-e3b25aff-4563-4dc7-8b37-f3f66bd6f62c.png)

![image-20220527110620091](https://user-images.githubusercontent.com/70505378/170616309-16c26f5a-cff4-4d82-b797-f3bcf4e9397a.png)

추가로 pydantic은 어디서 에러가 발생했는지 location, type, message 등을 알려줌

![image-20220527110724506](https://user-images.githubusercontent.com/70505378/170616312-ec70f8f0-696b-4bea-a0df-50f2e3c23c7d.png)





<br>

### Config

Pydantic에서는 Config를 체계적으로 관리할 방법을 제공

[Config 컨벤션]

* 애플리케이션은 종종 설정을 상수로 코드에 저장함
* 이것은 Twelve-Factor를 위반
* Twelve-Factor는 설정을 코드에서 엄격하게 분리하는 것을 요구함
* Twelve-Factor App은 설정을 환경 변수(envvars나 env라고도 불림)에 저장함
* 환경 변수는 코드 변경 없이 쉽게 배포 때마다 쉽게 변경할 수 있음
* The Twelve-Factor App이라는 SaaS(Software as a Service)를 만들기 위한 방법론을 정리한 규칙들에 따르면, 환경 설정은 애플리케이션 코드에서 분리되어 관리되어야 함  

참고 글: [https://12factor.net/ko/config](https://12factor.net/ko/config)

[사용할 수 있는 방법]

1. `.ini`, `.yaml` 파일 등으로 config 설정
   * yaml로 환경 설정을 관리할 경우 쉽게 환경을 설정할 수 있지만, 환경에 대한 설정을 하드코딩하는 형태
   * 때문에 변경 사항이 생길 때 유연하게 코드를 변경하기 어려움

![image-20220527111608732](https://user-images.githubusercontent.com/70505378/170616322-9ec38926-0ee8-44f6-918b-f0511e27594f.png)

2. flask-style config.py
   * Config 클래스에서 yaml, ini 파일을 불러와 python class 필드로 주입하는 과정을 구현
   * Config를 상속한 클래스에서는 config 클래스의 정보를 오버라이딩해서 사용
   * 하지만 해당 파일의 데이터가 정상적인지 체크하거나(validation), 환경 변수로부터 해당 필드를 오버라이딩(overriding)하려면 코드량이 늘어남

![image-20220527111231447](https://user-images.githubusercontent.com/70505378/170616316-85c8a7f1-e057-423d-9fcd-1798e6a33462.png)

3. pydantic base settings
   * Validation처럼 pydantic은 BaseSettings를 상속한 클래스에서 type hint로 주입된 설정 데이터를 검증할 수 있음
   * Field 클래스의 env 인자로, 환경 변수로부터 해당 필드를 오버라이딩 할 수 있음
   * yaml, ini 파일들을 추가적으로 만들지 않고, `.env` 파일들을 환경 별로 만들어두거나, 실행 환경에서 유연하게 오버라이딩 할 수 있음

![image-20220527111243820](https://user-images.githubusercontent.com/70505378/170616317-58901f79-7d38-4925-8102-f71469f4cc69.png)

environment variable overriding

![image-20220527111306596](https://user-images.githubusercontent.com/70505378/170616320-66290e54-2705-465e-9476-c0133331118c.png)

<br>

위의 세 방법 모두 틀린 방법은 아니고, 실무에서는 각 팀에서 맞는 방법을 따라가는 것이 좋습니다. 

다만 실무에서 여러 사람과 협업하는 환경에서 Human Error를 줄여주는 Pydantic의 기능들을 유용하기 때문에 기회가 되면 사용해보는 것이 좋습니다. 























<br>

<br>

# 참고 자료

* 
