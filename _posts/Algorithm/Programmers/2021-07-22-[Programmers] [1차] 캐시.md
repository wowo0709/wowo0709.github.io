---
layout: single
title: "[Programmers] [1차] 캐시"
categories: ['Algorithm', 'SequentialSearch', 'Queue']
---

# [1차] 캐시

### 문제 설명

---

- ## 캐시

  지도개발팀에서 근무하는 제이지는 지도에서 도시 이름을 검색하면 해당 도시와 관련된 맛집 게시물들을 데이터베이스에서 읽어 보여주는 서비스를 개발하고 있다.
  이 프로그램의 테스팅 업무를 담당하고 있는 어피치는 서비스를 오픈하기 전 각 로직에 대한 성능 측정을 수행하였는데, 제이지가 작성한 부분 중 데이터베이스에서 게시물을 가져오는 부분의 실행시간이 너무 오래 걸린다는 것을 알게 되었다.
  어피치는 제이지에게 해당 로직을 개선하라고 닦달하기 시작하였고, 제이지는 DB 캐시를 적용하여 성능 개선을 시도하고 있지만 캐시 크기를 얼마로 해야 효율적인지 몰라 난감한 상황이다.
  
  어피치에게 시달리는 제이지를 도와, DB 캐시를 적용할 때 캐시 크기에 따른 실행시간 측정 프로그램을 작성하시오.
  
  ### 입력 형식
  
  - 캐시 크기(`cacheSize`)와 도시이름 배열(`cities`)을 입력받는다.
  - `cacheSize`는 정수이며, 범위는 0 ≦ `cacheSize` ≦ 30 이다.
  - `cities`는 도시 이름으로 이뤄진 문자열 배열로, 최대 도시 수는 100,000개이다.
  - 각 도시 이름은 공백, 숫자, 특수문자 등이 없는 영문자로 구성되며, 대소문자 구분을 하지 않는다. 도시 이름은 최대 20자로 이루어져 있다.
  
  ### 출력 형식
  
  - 입력된 도시이름 배열을 순서대로 처리할 때, "총 실행시간"을 출력한다.
  
  ### 조건
  
  - 캐시 교체 알고리즘은 `LRU`(Least Recently Used)를 사용한다.
  - `cache hit`일 경우 실행시간은 `1`이다.
  - `cache miss`일 경우 실행시간은 `5`이다.
  
  ### 입출력 예제
  
  | 캐시크기(cacheSize) | 도시이름(cities)                                             | 실행시간 |
  | ------------------- | ------------------------------------------------------------ | -------- |
  | 3                   | ["Jeju", "Pangyo", "Seoul", "NewYork", "LA", "Jeju", "Pangyo", "Seoul", "NewYork", "LA"] | 50       |
  | 3                   | ["Jeju", "Pangyo", "Seoul", "Jeju", "Pangyo", "Seoul", "Jeju", "Pangyo", "Seoul"] | 21       |
  | 2                   | ["Jeju", "Pangyo", "Seoul", "NewYork", "LA", "SanFrancisco", "Seoul", "Rome", "Paris", "Jeju", "NewYork", "Rome"] | 60       |
  | 5                   | ["Jeju", "Pangyo", "Seoul", "NewYork", "LA", "SanFrancisco", "Seoul", "Rome", "Paris", "Jeju", "NewYork", "Rome"] | 52       |
  | 2                   | ["Jeju", "Pangyo", "NewYork", "newyork"]                     | 16       |
  | 0                   | ["Jeju", "Pangyo", "Seoul", "NewYork", "LA"]                 | 25       |
  
  [해설 보러가기](http://tech.kakao.com/2017/09/27/kakao-blind-recruitment-round-1/)



### 문제 풀이

---

이 문제에서 사용하는 캐시 교체 알고리즘은 LRU죠. 

LRU란 가장 사용한 지 오래된 블록을 쫓아내는 것을 말합니다. 



따라서 가장 오래된 블록이 무엇인지 알아야 하고, 저는 다음과 같이 추적했습니다. 

**리스트에서 가장 오래된 블록이 가장 앞(인덱스 0)에 오도록 한다. **

이렇게 하면 Cache miss가 일어났을 때 자연스럽게 가장 최근 블록은 맨 뒤에 append 해주고, 용량이 초과했을 경우 맨 앞의 블록을 pop해주면 됩니다. 

또한 Cache hit가 일어났을 때는 리스트에 있던 기존의 hit된 블록은 제거하고, 같은 값을 가지는 블록을 맨 뒤에 append 해주면 됩니다.  

```python
def solution(cacheSize, cities):
    cache = []
    answer = 0
    for city in cities:
        city = city.lower()
        if city in cache: # cache hit
            cache.remove(city)
            cache.append(city)
            answer += 1
        else:             # cache miss
            cache.append(city) 
            answer += 5
            if len(cache) > cacheSize: cache.pop(0)

    return answer
```

위와 같은 코드를 실행시키면 된다.





그럼 안녕!
