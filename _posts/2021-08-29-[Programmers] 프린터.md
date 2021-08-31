---
layout: single
title: "[Programmers] 프린터"
---



# 프린터

### 문제 설명

##### 문제 설명

일반적인 프린터는 인쇄 요청이 들어온 순서대로 인쇄합니다. 그렇기 때문에 중요한 문서가 나중에 인쇄될 수 있습니다. 이런 문제를 보완하기 위해 중요도가 높은 문서를 먼저 인쇄하는 프린터를 개발했습니다. 이 새롭게 개발한 프린터는 아래와 같은 방식으로 인쇄 작업을 수행합니다.

```
1. 인쇄 대기목록의 가장 앞에 있는 문서(J)를 대기목록에서 꺼냅니다.
2. 나머지 인쇄 대기목록에서 J보다 중요도가 높은 문서가 한 개라도 존재하면 J를 대기목록의 가장 마지막에 넣습니다.
3. 그렇지 않으면 J를 인쇄합니다.
```

예를 들어, 4개의 문서(A, B, C, D)가 순서대로 인쇄 대기목록에 있고 중요도가 2 1 3 2 라면 C D A B 순으로 인쇄하게 됩니다.

내가 인쇄를 요청한 문서가 몇 번째로 인쇄되는지 알고 싶습니다. 위의 예에서 C는 1번째로, A는 3번째로 인쇄됩니다.

현재 대기목록에 있는 문서의 중요도가 순서대로 담긴 배열 priorities와 내가 인쇄를 요청한 문서가 현재 대기목록의 어떤 위치에 있는지를 알려주는 location이 매개변수로 주어질 때, 내가 인쇄를 요청한 문서가 몇 번째로 인쇄되는지 return 하도록 solution 함수를 작성해주세요.

##### 제한사항

- 현재 대기목록에는 1개 이상 100개 이하의 문서가 있습니다.
- 인쇄 작업의 중요도는 1~9로 표현하며 숫자가 클수록 중요하다는 뜻입니다.
- location은 0 이상 (현재 대기목록에 있는 작업 수 - 1) 이하의 값을 가지며 대기목록의 가장 앞에 있으면 0, 두 번째에 있으면 1로 표현합니다.

##### 입출력 예

| priorities         | location | return |
| ------------------ | -------- | ------ |
| [2, 1, 3, 2]       | 2        | 1      |
| [1, 1, 9, 1, 1, 1] | 0        | 5      |

##### 입출력 예 설명

예제 #1

문제에 나온 예와 같습니다.

예제 #2

6개의 문서(A, B, C, D, E, F)가 인쇄 대기목록에 있고 중요도가 1 1 9 1 1 1 이므로 C D E F A B 순으로 인쇄합니다.

[출처](http://www.csc.kth.se/contest/nwerc/2006/problems/nwerc06.pdf)

<br>

### 문제 풀이

---

#### \# 큐

<br>

큐의 개념을 활용하는 어렵지 않은 문제이므로, 이번 포스팅에서는 문제 해설보다 문법에 대해 살펴보겠습니다. 

우선 풀이를 보시죠. 

* 1번 풀이

```python
def solution(docs, loc):
    ans = 0
    n = len(docs)
    while True:
        ans += 1
        maxIdx = 0
        for i in range(1,n-ans+1):
            if docs[maxIdx] < docs[i]: maxIdx = i
        if maxIdx == loc: 
            return ans
        docs = docs[maxIdx+1:] + docs[:maxIdx]
        loc = loc-maxIdx+n-ans if loc < maxIdx else loc-maxIdx-1
```

문서에서 pop이 일어날 때마다 문서 배열을 나타내는 docs와 원하는 문서의 위치를 나타내는 loc 을 갱신해줍니다. 

만약 loc과 maxIdx가 같다면, 답을 출력합니다. 

매번 pop과 append를 반복하는 것은 비효율적이라고 생각했기 때문에 이렇게 풀이하였습니다. 

<br>

* 2번 풀이

```python
def solution(priorities, location):
    queue =  [(i,p) for i,p in enumerate(priorities)]
    answer = 0
    while True:
        cur = queue.pop(0)
        if any(cur[1] < q[1] for q in queue):
            queue.append(cur)
        else:
            answer += 1
            if cur[0] == location:
                return answer
```

제 풀이는 아니고, 해당 문제의 `다른 사람의 풀이`란에 있는 풀이들 중 가장 많은 좋아요를 받은 풀이입니다. 

먼저 해당 풀이에서 **초기 위치와 문서 우선순위를 튜플로 저장**한 것이 눈에 띕니다. 제가 매번 loc을 갱신하는 것에 반해, 초기위치를 이런 식으로 묶어서 저장해두면 위치가 바뀌어도 초기 위치를 찾는데는 문제가 없겠죠. 

두 번째로는 **any**를 사용한 부분이 눈에 띄네요. 이를 한 번 정리해보도록 합시다. 

<br>

✋ **파이썬의 any와 all 함수**

**all( ) 함수**: 인자로 받은 **Iterable**의 **모든 요소가 참(True)이면 참(True)을 반환**하는 함수

* 구현 코드

```python
def all(iterable):
    for element in iterable:
        if not element:
            return False
    return True
```

* 예시

```python
nums = [1,2,3,4,5]
print(all(0 < num for num in nums))
print(all(1 < num for num in nums))

out:
    True
    False
```

<br>

**any( ) 함수**: 인자로 받은 **Iterable**의 요소 중 **하나라도 참(True)이면 참(True)을 반환**하는 함수

* 구현 코드

```python
def any(iterable):
    for element in iterable:
        if element:
            return True
    return False
```

* 예시

```python
nums = [1,2,3,4,5]
print(any(0 < num for num in nums))
print(any(1 < num for num in nums))
print(any(5 < num for num in nums))

out:
    True
    True
    False
```

<br>

적절히 사용하면 아주 유용한 함수가 될 것 같네요!

위 내용은 아래 블로그를 참조하였으며, 더 많은 예시를 보고 싶으신 분은 아래 블로그로 가서 보시면 좋을 것 같습니다. 

> https://blockdmask.tistory.com/430