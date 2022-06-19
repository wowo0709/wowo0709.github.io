---
layout: single
title: "[Python] Effective Python 전체 목차"
categories: ['Language', 'Python']
toc: true
toc_sticky: true
tag: []

---

<br>

# Effective Python 전체 목차

`Effective Python 2nd Edition`을 읽으며 학습한 내용들을 정리합니다. 

* 소스 코드: [wowo0709/Effective-Python](https://github.com/wowo0709/Effective-Python)

해당 포스팅은 전체 목차를 나타내며, 원하시는 내용을 클릭하며 해당 포스팅으로 이동할 수 있습니다. 

## Contents

**1장 파이썬답게 생각하기**
- Better way 1 사용 중인 파이썬의 버전을 알아두라
- Better way 2 PEP 8 스타일 가이드를 따르라
- Better way 3 bytes와 str의 차이를 알아두라
- Better way 4 C 스타일 형식 문자열을 str.format과 쓰기보다는 f-문자열을 통한 인터폴레이션을 사용하라
- Better way 5 복잡한 식을 쓰는 대신 도우미 함수를 작성하라
- Better way 6 인덱스를 사용하는 대신 대입을 사용해 데이터를 언패킹하라
- Better way 7 range보다는 enumerate를 사용하라 066
- Better way 8 여러 이터레이터에 대해 나란히 루프를 수행하려면 zip을 사용하라
- Better way 9 for나 while 루프 뒤에 else 블록을 사용하지 말라
- Better way 10 대입식을 사용해 반복을 피하라

**2장 리스트와 딕셔너리**
- Better way 11 시퀀스를 슬라이싱하는 방법을 익혀라
- Better way 12 스트라이드와 슬라이스를 한 식에 함께 사용하지 말라
- Better way 13 슬라이싱보다는 나머지를 모두 잡아내는 언패킹을 사용하라
- Better way 14 복잡한 기준을 사용해 정렬할 때는 key 파라미터를 사용하라
- Better way 15 딕셔너리 삽입 순서에 의존할 때는 조심하라
- Better way 16 in을 사용하고 딕셔너리 키가 없을 때 KeyError를 처리하기보다는 get을 사용하라
- Better way 17 내부 상태에서 원소가 없는 경우를 처리할 때는 setdefault보다 defaultdict를 사용하라
- Better way 18 __missing__을 사용해 키에 따라 다른 디폴트 값을 생성하는 방법을 알아두라

**3장 함수**
- Better way 19 함수가 여러 값을 반환하는 경우 절대로 네 값 이상을 언패킹하지 말라
- Better way 20 None을 반환하기보다는 예외를 발생시켜라
- Better way 21 변수 영역과 클로저의 상호작용 방식을 이해하라
- Better way 22 변수 위치 인자를 사용해 시각적인 잡음을 줄여라
- Better way 23 키워드 인자로 선택적인 기능을 제공하라
- Better way 24 None과 독스트링을 사용해 동적인 디폴트 인자를 지정하라
- Better way 25 위치로만 인자를 지정하게 하거나 키워드로만 인자를 지정하게 해서 함수 호출을 명확하게 만들라
- Better way 26 functools.wrap을 사용해 함수 데코레이터를 정의하라

**4장 컴프리헨션과 제너레이터**
- Better way 27 map과 filter 대신 컴프리헨션을 사용하라
- Better way 28 컴프리헨션 내부에 제어 하위 식을 세 개 이상 사용하지 말라
- Better way 29 대입식을 사용해 컴프리헨션 안에서 반복 작업을 피하라
- Better way 30 리스트를 반환하기보다는 제너레이터를 사용하라
- Better way 31 인자에 대해 이터레이션할 때는 방어적이 돼라
- Better way 32 긴 리스트 컴프리헨션보다는 제너레이터 식을 사용하라
- Better way 33 yield from을 사용해 여러 제너레이터를 합성하라
- Better way 34 send로 제너레이터에 데이터를 주입하지 말라
- Better way 35 제너레이터 안에서 throw로 상태를 변화시키지 말라
- Better way 36 이터레이터나 제너레이터를 다룰 때는 itertools를 사용하라

**5장 클래스와 인터페이스**
- Better way 37 내장 타입을 여러 단계로 내포시키기보다는 클래스를 합성하라
- Better way 38 간단한 인터페이스의 경우 클래스 대신 함수를 받아라
- Better way 39 객체를 제너릭하게 구성하려면 @classmethod를 통한 다형성을 활용하라
- Better way 40 super로 부모 클래스를 초기화하라
- Better way 41 기능을 합성할 때는 믹스인 클래스를 사용하라
- Better way 42 비공개 애트리뷰트보다는 공개 애트리뷰트를 사용하라
- Better way 43 커스텀 컨테이너 타입은 collections.abc를 상속하라

**6장 메타클래스와 애트리뷰트**
- Better way 44 세터와 게터 메서드 대신 평범한 애트리뷰트를 사용하라
- Better way 45 애트리뷰트를 리팩터링하는 대신 @property를 사용하라
- Better way 46 재사용 가능한 @property 메서드를 만들려면 디스크립터를 사용하라
- Better way 47 지연 계산 애트리뷰트가 필요하면 __getattr__, __getattribute__, __setattr__을 사용하라
- Better way 48 __init_subclass__를 사용해 하위 클래스를 검증하라
- Better way 49 __init_subclass__를 사용해 클래스 확장을 등록하라
- Better way 50 __set_name__으로 클래스 애트리뷰트를 표시하라
- Better way 51 합성 가능한 클래스 확장이 필요하면 메타클래스보다는 클래스 데코레이터를 사용하라

**7장 동시성과 병렬성**
- Better way 52 자식 프로세스를 관리하기 위해 subprocess를 사용하라
- Better way 53 블로킹 I/O의 경우 스레드를 사용하고 병렬성을 피하라
- Better way 54 스레드에서 데이터 경합을 피하기 위해 Lock을 사용하라
- Better way 55 Queue를 사용해 스레드 사이의 작업을 조율하라
- Better way 56 언제 동시성이 필요할지 인식하는 방법을 알아두라
- Better way 57 요구에 따라 팬아웃을 진행하려면 새로운 스레드를 생성하지 말라
- Better way 58 동시성과 Queue를 사용하기 위해 코드를 어떻게 리팩터링해야 하는지 이해하라
- Better way 59 동시성을 위해 스레드가 필요한 경우에는 ThreadpoolExecutor를 사용하라
- Better way 60 I/O를 할 때는 코루틴을 사용해 동시성을 높여라
- Better way 61 스레드를 사용한 I/O를 어떻게 asyncio로 포팅할 수 있는지 알아두라
- Better way 62 asyncio로 쉽게 옮겨갈 수 있도록 스레드와 코루틴을 함께 사용하라
- Better way 63 응답성을 최대로 높이려면 asyncio 이벤트 루프를 블록하지 말라
- Better way 64 진정한 병렬성을 살리려면 concurrent.futures를 사용하라

**8장 강건성과 성능**
- Better way 65 try/except/else/finally의 각 블록을 잘 활용하라
- Better way 66 재사용 가능한 try/finally 동작을 원한다면 contextlib과 with 문을 사용하라
- Better way 67 지역 시간에는 time보다는 datetime을 사용하라
- Better way 68 copyreg를 사용해 pickle을 더 신뢰성 있게 만들라
- Better way 69 정확도가 매우 중요한 경우에는 decimal을 사용하라
- Better way 70 최적화하기 전에 프로파일링을 하라
- Better way 71 생산자-소비자 큐로 deque를 사용하라
- Better way 72 정렬된 시퀀스를 검색할 때는 bisect를 사용하라
- Better way 73 우선순위 큐로 heapq를 사용하는 방법을 알아두라
- Better way 74 bytes를 복사하지 않고 다루려면 memoryview와 bytearray를 사용하라

**9장 테스트와 디버깅**
- Better way 75 디버깅 출력에는 repr 문자열을 사용하라
- Better way 76 TestCase 하위 클래스를 사용해 프로그램에서 연관된 행동 방식을 검증하라
- Better way 77 setUp, tearDown, setUpModule, tearDownModule을 사용해 각각의 테스트를 격리하라
- Better way 78 목을 사용해 의존 관계가 복잡한 코드를 테스트하라
- Better way 79 의존 관계를 캡슐화해 모킹과 테스트를 쉽게 만들라
- Better way 80 pdb를 사용해 대화형으로 디버깅하라
- Better way 81 프로그램이 메모리를 사용하는 방식과 메모리 누수를 이해하기 위해 tracemalloc을 사용하라

**10장 협업**
- Better way 82 커뮤니티에서 만든 모듈을 어디서 찾을 수 있는지 알아두라
- Better way 83 가상 환경을 사용해 의존 관계를 격리하고 반복 생성할 수 있게 하라
- Better way 84 모든 함수, 클래스, 모듈에 독스트링을 작성하라
- Better way 85 패키지를 사용해 모듈을 체계화하고 안정적인 API를 제공하라
- Better way 86 배포 환경을 설정하기 위해 모듈 영역의 코드를 사용하라
- Better way 87 호출자를 API로부터 보호하기 위해 최상위 Exception을 정의하라
- Better way 88 순환 의존성을 깨는 방법을 알아두라
- Better way 89 리팩터링과 마이그레이션 방법을 알려주기 위해 warning을 사용하라
- Better way 90 typing과 정적 분석을 통해 버그를 없애라
