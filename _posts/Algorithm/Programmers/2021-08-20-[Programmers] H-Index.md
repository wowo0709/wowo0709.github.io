---
layout: single
title: "[Programmers] H-Index"
categories: ['Algorithm', 'Programmers']
---

# H-Index

### 문제 설명

##### 문제 설명

H-Index는 과학자의 생산성과 영향력을 나타내는 지표입니다. 어느 과학자의 H-Index를 나타내는 값인 h를 구하려고 합니다. 위키백과[1](https://programmers.co.kr/learn/courses/30/lessons/42747#fn1)에 따르면, H-Index는 다음과 같이 구합니다.

어떤 과학자가 발표한 논문 `n`편 중, `h`번 이상 인용된 논문이 `h`편 이상이고 나머지 논문이 h번 이하 인용되었다면 `h`의 최댓값이 이 과학자의 H-Index입니다.

어떤 과학자가 발표한 논문의 인용 횟수를 담은 배열 citations가 매개변수로 주어질 때, 이 과학자의 H-Index를 return 하도록 solution 함수를 작성해주세요.

##### 제한사항

* 과학자가 발표한 논문의 수는 1편 이상 1,000편 이하입니다.
* 논문별 인용 횟수는 0회 이상 10,000회 이하입니다.

##### 입출력 예

| citations       | return |
| --------------- | ------ |
| [3, 0, 6, 1, 5] | 3      |

##### 입출력 예 설명

이 과학자가 발표한 논문의 수는 5편이고, 그중 3편의 논문은 3회 이상 인용되었습니다. 그리고 나머지 2편의 논문은 3회 이하 인용되었기 때문에 이 과학자의 H-Index는 3입니다.

※ 공지 - 2019년 2월 28일 테스트 케이스가 추가되었습니다.

------

1. https://en.wikipedia.org/wiki/H-index "위키백과" [↩](https://programmers.co.kr/learn/courses/30/lessons/42747#fnref1)

<br>



### 문제 풀이

---

#### \# 정렬

<br>

h번 이상 인용된 논문이 h개 이상인 h의 최댓값을 찾습니다. 

이는 h번 이상 인용된 논문이 h개 이하가 되는 순간 return하면 된다는 뜻입니다. 왜 **미만**이 아니라 **이하**가 되는 순간이냐면, 인용 횟수가 h번인 논문의 개수가 h개가 되는 순간 h가 더 커지면 만족하는 논문의 개수는 무조건 h개 미만이되기 때문입니다. 

> *h*-index (*f*) = ![{\displaystyle \max\{i\in \mathbb {N} :f(i)\geq i\}}](https://wikimedia.org/api/rest_v1/media/math/render/svg/d94958e3aa230763f408edfe59576e00a3688583)

그런데 이 때 어떤 값을 리턴해야 할까요?

```python
def solution(citations):
    citations.sort()
    N = len(citations)
    for i in range(N):
        if N-i <= citations[i]: 
            return ?
    return ?
```

`?`에 들어갈 값은 무엇일까요?

첫번째 `?`에서는 **N-i**를 반환합니다. 예를 들어보죠. 

* [0, 1, 3, 5] ➡ N-i = 4-2 = 2
* [0, 1, 3, 5, 5] ➡ N-i = 5-2 = 3
* [0, 1, 3, 5, 5, 5] ➡ N-i = 6-3 = 3
* [0, 1, 3, 5, 5, 5, 5] ➡ N-i = 7-3 = 4
* [0, 1, 3, 5, 5, 5, 5, 5] ➡ N-i = 8-3 = 5

N-i 라는 숫자는 **현재 논문의 인용수 이상의 인용수를 갖는 논문의 수**를 나타내죠. 이 수가 citations[i], 즉 현재 논문의 인용수 이하의 수라면 그 논문의 수를 반환합니다. 이 논문의 수를 반환하면 N-i 번 이상 인용된 논문의 수가 N-i개 이상이라는 것은 자명합니다. 

<br>

그렇다면 두번째 `?`에는 무엇이 들어갈까요?

이 경우는 모든 논문의 인용수가 0인 경우이므로, **0**을 반환합니다. 모두 0일 때만 for문 안의 조건인 **h번 이상 인용된 논문이 h개 이하인 경우**에 걸리지 않을 수 있죠. 

<br>

그럼 포스팅 여기서 마치겠습니다. 
