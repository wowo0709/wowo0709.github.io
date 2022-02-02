---
layout: single
title: "[Programmers] 단어 변환"
categories: ['Algorithm', 'DFSBFS']
toc: true
toc_sticky: true
---



<br>

## 문제 설명

### 문제 설명

두 개의 단어 begin, target과 단어의 집합 words가 있습니다. 아래와 같은 규칙을 이용하여 begin에서 target으로 변환하는 가장 짧은 변환 과정을 찾으려고 합니다.

```
1. 한 번에 한 개의 알파벳만 바꿀 수 있습니다.
2. words에 있는 단어로만 변환할 수 있습니다.
```

예를 들어 begin이 "hit", target가 "cog", words가 ["hot","dot","dog","lot","log","cog"]라면 "hit" -> "hot" -> "dot" -> "dog" -> "cog"와 같이 4단계를 거쳐 변환할 수 있습니다.

두 개의 단어 begin, target과 단어의 집합 words가 매개변수로 주어질 때, 최소 몇 단계의 과정을 거쳐 begin을 target으로 변환할 수 있는지 return 하도록 solution 함수를 작성해주세요.

### 제한사항

- 각 단어는 알파벳 소문자로만 이루어져 있습니다.
- 각 단어의 길이는 3 이상 10 이하이며 모든 단어의 길이는 같습니다.
- words에는 3개 이상 50개 이하의 단어가 있으며 중복되는 단어는 없습니다.
- begin과 target은 같지 않습니다.
- 변환할 수 없는 경우에는 0를 return 합니다.

### 입출력 예

| begin | target | words                                      | return |
| ----- | ------ | ------------------------------------------ | ------ |
| "hit" | "cog"  | ["hot", "dot", "dog", "lot", "log", "cog"] | 4      |
| "hit" | "cog"  | ["hot", "dot", "dog", "lot", "log"]        | 0      |

#### 입출력 예 설명

예제 #1
문제에 나온 예와 같습니다.

예제 #2
target인 "cog"는 words 안에 없기 때문에 변환할 수 없습니다.

<br>

## 문제 풀이

### \# DFS/BFS



<br>

### 전체 코드

전체코드입니다. 

```python
def solution(begin, target, words):
    '''
    한 번에 한 개의 알파벳만 변환
    word에 있는 단어로만 변환 가능, 모든 단어의 길이 동일, 중복 없음
    최소 변환 횟수 반환 -> BFS
    변환 불가 시 0 반환
    => 시간 복잡도 = (단어의 길이) * 26
    '''
    from string import ascii_lowercase
    from collections import deque
    deq = deque([(begin,0)]) # (word,변환횟수)...
    visited = {word:0 for word in [begin]+words}
    while deq:
        convert,n = deq.popleft()
        if visited[convert] == 1: continue
        visited[convert] = 1
        if convert == target: return n
        for i in range(len(convert)):
            for alp in ascii_lowercase:
                word = convert[:i]+alp+convert[i+1:]
                if word in words:
                    deq.append((word,n+1))          
    return 0
```

<br>

