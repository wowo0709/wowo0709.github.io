---
layout: single
title: "[Programmers] N으로 표현"
categories: ['Algorithm', 'Programmers']
toc: true
toc_sticky: true
tag: ['동적계획법']
---



<br>

## 문제 설명

### 문제 설명

n명이 입국심사를 위해 줄을 서서 기다리고 있습니다. 각 입국심사대에 있는 심사관마다 심사하는데 걸리는 시간은 다릅니다.

처음에 모든 심사대는 비어있습니다. 한 심사대에서는 동시에 한 명만 심사를 할 수 있습니다. 가장 앞에 서 있는 사람은 비어 있는 심사대로 가서 심사를 받을 수 있습니다. 하지만 더 빨리 끝나는 심사대가 있으면 기다렸다가 그곳으로 가서 심사를 받을 수도 있습니다.

모든 사람이 심사를 받는데 걸리는 시간을 최소로 하고 싶습니다.

입국심사를 기다리는 사람 수 n, 각 심사관이 한 명을 심사하는데 걸리는 시간이 담긴 배열 times가 매개변수로 주어질 때, 모든 사람이 심사를 받는데 걸리는 시간의 최솟값을 return 하도록 solution 함수를 작성해주세요.

### 제한사항

* 입국심사를 기다리는 사람은 1명 이상 1,000,000,000명 이하입니다.
* 각 심사관이 한 명을 심사하는데 걸리는 시간은 1분 이상 1,000,000,000분 이하입니다.
* 심사관은 1명 이상 100,000명 이하입니다.

### 입출력 예

| n    | times   | return |
| ---- | ------- | ------ |
| 6    | [7, 10] | 28     |

### 입출력 예 설명

가장 첫 두 사람은 바로 심사를 받으러 갑니다.

7분이 되었을 때, 첫 번째 심사대가 비고 3번째 사람이 심사를 받습니다.

10분이 되었을 때, 두 번째 심사대가 비고 4번째 사람이 심사를 받습니다.

14분이 되었을 때, 첫 번째 심사대가 비고 5번째 사람이 심사를 받습니다.

20분이 되었을 때, 두 번째 심사대가 비지만 6번째 사람이 그곳에서 심사를 받지 않고 1분을 더 기다린 후에 첫 번째 심사대에서 심사를 받으면 28분에 모든 사람의 심사가 끝납니다.

[출처](http://hsin.hr/coci/archive/2012_2013/contest3_tasks.pdf)

<br>

## 문제 풀이

### \# 이분 탐색

<br>

문제를 읽으면 입력의 크기가 아\~\~\~주 크다는 것을 알 수 있습니다. 

그래서 `이분탐색`으로 풀어야 하는 것이겠죠. **이분탐색**은 탐색의 속도가 `O(logn)`이니까요. 

<br>

각설하고, 제가 문제를 푼 과정을 보도록 하겠습니다. 

### 1번 풀이(시간 초과)

처음에는 아래와 같이 풀려고 했습니다. 

![image-20211110151408826](C:\Users\wjsdu\AppData\Roaming\Typora\typora-user-images\image-20211110151408826.png)

[심시시간, 종료시간] 을 원소로 갖는 리스트를 하나 생성하고 이를 오름차순 정렬합니다. 

그리고 인원 수 `n` 만큼 반복문을 돌면서 맨 앞 원소(종료 시간이 가장 빠른 원소)의 종료 시간을 늘려주고, 다시 리스트에 오름차순 정렬되도록 삽입합니다. 이 때 이분탐색 삽입 함수인 `bisect.insort`를 사용합니다. 

```python
def solution(n, times):
    from collections import deque
    times = sorted(list([span,end] for span,end in zip(times,times))) # 심시시간, 종료시간
    print(times)
    from bisect import insort_left
    for _ in range(n-1):
        # fastest_time = times.popleft()
        # insort_left(times, (fastest_time[0],fastest_time[1]+fastest_time[0]),key=lambda t:t(t[1],t[0]))
        times[0] = (times[0][0],times[0][1]+times[0][0])
        times.sort(key=lambda t:(t[1],t[0]))
        
    return times[0][1]
```

그런데 위 코드를 보면, `insort` 함수를 사용하지 않고 리스트의 `sort` 메서드를 사용하였습니다. 

리스트를 정렬할 때 먼저 **종료 시간을 기준**으로 정렬한 뒤 종료시간이 같다면 **심시시간을 기준**으로 정렬해야 하는데, `insort` 함수는 `key = lambda t:(t[1],t[0])` 같이 정렬 기준 key에 두 개의 기준을 넘겨주는 것이 불가능했습니다. 

그래서, 어쨌든 `sort` 메서드의 시간 복잡도 또한 `O(logn)`이기 때문에 대신 `osrt`를 사용하였는데... 결과는 시간 초과!!!😢

<br>

위 코드의 시간 복잡도는 `O(nlogt)`가 됩니다. (n은 인원 수, t는 심사대 수)

`n`의 크기가 매우 크기 때문에, 우리는 시간 복잡도에서 `n`이라는 항을 `log(n)`과 같이  로그 시간복잡도로 탐색해야 할 것 같습니다. 

<br>

<br>

### 2번 풀이(6번, 9번 TC 오답)

그래서 아래와 같이 풀이합니다. 

![image-20211110151652963](C:\Users\wjsdu\AppData\Roaming\Typora\typora-user-images\image-20211110151652963.png)

**각 인원이 가장 빨리 마칠 수 있는 심사대를 계산하는 것이 아니라, 해당 시간 동안 몇 명이나 심사할 수 있는지를 계산하는 것입니다.**

약간의 발상의 전환이죠. 

<br>

```python
def solution(n, times):
    lo, hi = 1, max(times)*n
    ans = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        possible_n = sum(map(lambda t:mid//t, times))
        if possible_n > n: hi = mid - 1
        elif possible_n < n: lo = mid + 1
        else: # possible_n == n
            ans = mid
            hi = mid - 1

    return ans
```

그렇게 되면 시간 복잡도는 `O(t * log(max(t)*n))` 이 됩니다. 

`n`을 로그 복잡도의 시간으로 탐색하는데 성공하였고, 시간 초과를 피할 수 있습니다!!!

그리고 주의해야 할 것. `possible_n == n`인 값을 찾아도 탐색을 계속해서 이어가야 합니다. 이 조건을 만족하는 `mid` 값은 여러 개일 수 있고, 이 값들 중 최소값이 우리가 찾는 값이기 때문이죠. 

<br>

그런데 왜 틀렸냐구요..?

그건 아래 설명에서 알 수 있습니다. 

<br>

<br>

### 3번 풀이 (정답)

`Time complexity` 좋고, `lower bound algorithm` 좋습니다. 

그런데 습관적으로 이 이분탐색 알고리즘을 사용하다 보니, 문제에서 요구하는 중요한 것을 놓쳤습니다. 

정답 코드는 아래와 같아야 합니다. 

```python
def solution(n, times):
    lo, hi = 1, max(times)*n
    ans = 0

    while lo <= hi:
        mid = (lo + hi) // 2
        possible_n = sum(map(lambda t:mid//t, times))
        if possible_n >= n: # !!!!!
            ans = mid
            hi = mid - 1
        elif possible_n < n: 
            lo = mid + 1

    return ans
```

차이를 아시겠나요?

보통의 이분탐색 알고리즘에서는 값이 같을 때, 값보다 클 때, 값보다 작을 때의 3가지 경우로 나누어 범위를 좁힙니다. 

하지만 이 문제에서는 심사 가능한 사람의 수를 가리키는 변수 `possible_n`이 `n`보다 클 때도 정답이 될 수 있습니다. 

따라서 `possible_n >= n`일 때 정답 `ans`를 갱신해줍니다. 

<br>

<br>

### 정리

프로그래머스의 **3단계 이분탐색** 문제입니다. 

확실히, 3단계 문제들은 2단계 문제들보다 한 단계 더 나아간 사고를 요구하는 것 같습니다. 

이분 탐색 문제를 풀 때, <span style="color:red">**무엇을 이분 탐색 대상으로 삼을 수 있는지, 시간 복잡도는 어떻게 변하는지**</span>를 주의깊게 고려해야 할 것 같습니다. 

<br>

그리고 `lower bound algorithm`은 기본이므로 언제든지 사용할 수 있도록 익혀두도록 합시다!!!

```python
while start <= end:
    mid = (start + end) // 2

    if data[mid] == target:
        return mid # 함수를 끝내버린다.
    elif data[mid] < target:
        start = mid + 1
    else:
        end = mid -1

    return None
```

















