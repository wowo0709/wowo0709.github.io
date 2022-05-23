---
layout: single
title: "[AITech][Product Serving] 20220518 - Serving - Cloud, CI/CD"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 SOCAR의 '변성윤' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Serving - Cloud, CI/CD

## Cloud

웹, 앱 서비스를 만드는 경우 자신의 local 컴퓨터를 사용할 수도 있고 별도의 서버 컴퓨터를 사용할 수도 있지만, 이런 경우 컴퓨터를  계속해서 끄지 않고 연결 상태를 유지해줘야 합니다. 

이에 대한 대안으로 **클라우드 서비스**가 존재합니다. (AWS, Google Cloud, Azure 등)

![image-20220523113623792](https://user-images.githubusercontent.com/70505378/169745388-611c18c6-497f-4ded-93a3-2a53c9c554b6.png)

<br>

[Cloud Service의 다양한 제품]

* Computing Service (Server)
  * 연산을 수행하는 (Computing) 서비스
  * 가상 컴퓨터, 서버
  * CPU, Memory, GPU 등을 선택할 수 있음
  * 가장 많이 사용할 제품
  * 인스턴스 생성 후, 인스턴스에 들어가서 사용 가능
  * 회사별로 월에 무료 사용량이 존재 (성능은 약 cpu 1 core, memory 2G)
* Serverless Computing
  * 앞에 나온 Computing Service와 유사하지만, 서버 관리를 클라우드 쪽에서 진행
  * **코드를 클라우드에 제출하면, 그 코드를 가지고 서버를 실행해주는 형태**
  * 요청 부하에 따라 자동으로 확장 (Auto Scaling)
  * Micro Service로 많이 활용
* Stateless Container
  * Docker를 사용한 Container 기반으로 서버를 실행하는 구조
  * Docker Image를 업로드하면 해당 이미지 기반으로 서버를 실행해주는 형태
  * 요청 부하에 따라 자동으로 확장 (Auto Scaling)
* Object Storage
  * 다양한 Object를 저장할 수 있는 저장소
  * 다양한 형태의 데이터를 저장할 수 있으며, API를 사용해 데이터에 접근할 수 있음
  * 점점 데이터 저장 비용이 저렴해지고 있음
  * 머신러닝 모델 pkl 파일, csv 파일, 실험 log 등을 저장할 수 있음
* Database (RDB)
  * Database가 필요한 경우 클라우드에서 제공하는 Database를 활용할 수 있음
  * 웹, 앱 서비스와 데이터베이스가 연결되어 있는 경우가 많으며, 대표적으로 MySQL, PosgreSQL 등을 사용할 수 있음
  * 사용자 로그 데이터를 Database에 저장할 수도 있고, object storage에 저장할 수도 있음
  * 저장된 데이터를 어떻게 사용하냐에 따라 어디에 저장할 지를 결정
* Data Warehouse
  * Database에 저장된 데이터는 데이터 분석을 메인으로 하는 저장소가 아닌 서비스에서 활용할 database
  * Database에 있는 데이터, object storage에 있는 데이터 등을 모두 모아서 Data warehouse에 저장
  * 데이터 분석에 특화된 database
* AI Platform
  * AI Research, AI Develop 과정을 더 편리하게 해주는 제품
  * MLOps 관련 서비스 제공
  * Google Cloud Platform: TPU

여러 기업에서 클라우드 서비스를 제공하고 있으며, 클라우드마다 이름은 다르지만 비슷한 형식을 띱니다. 

![image-20220523115038212](https://user-images.githubusercontent.com/70505378/169745393-6e4e4253-e70b-4860-b3ad-aacd544a9edb.png)

<br>

## CI/CD

프로젝트 진행 시 개발 프로세스는 크게 아래와 같이 나뉩니다. 

* Local (Feature)
  * 각자의 컴퓨터에서 개발
  * 환경을 통일시키기 위해 Docker 등을 사용
* Dev
  * Local에서 개발한 기능을 테스트할 수 있는 환경
  * Test 서버
* Staging
  * Production 환경에 배포하기 전에 운영하거나 보안, 성능 측정하는 환경
  * Staging 서버
* Production
  * 실제 서비스를 운영하는 환경
  * 운영 서버

![image-20220523134140262](https://user-images.githubusercontent.com/70505378/169745395-60a0180c-1d8b-467d-a078-b33ccf1921b4.png)

깃허브의 Dev, Staging, Main 브랜치에서 반복적으로 일어나는 일들을 수행할 때 이를 자동화한다면 효율이 올라갈 것입니다. 

`CI/CD`는 이렇게 지속적으로 발생하는 일들을 자동화하는 개념이라고 할 수 있습니다. 

* Continuous Integration, 지속적 통합
  * **빌드, 테스트 자동화**
  * 새롭게 작성한 코드 변경 사항이 build, test 진행한 후 test case에 통과했는지 확인
  * 지속적으로 코드 품질 관리
  * 10명의 개발자가 코드를 수정했다면 모두 CI 프로세스 진행
* Continuous Deploy/Delivery, 지속적 배포
  * **배포 자동화**
  * 작성한 코드가 항상 신뢰 가능한 상태가 되면 자동으로 배포될 수 있도록 하는 과정
  * CI 이후 CD 진행
  * dev/staging/main 브랜치에 merge될 경우 코드가 자동으로 서버에 배포

Jenkins, Travis CI, GCP cloud build, Github Action 등을 이용하여 CI/CD를 수행할 수 있습니다. 

![image-20220523134616936](https://user-images.githubusercontent.com/70505378/169745397-9659fa35-55f2-4249-b714-d85600dd09b7.png)













<br>

<br>

# 참고 자료

* 
