---
layout: single
title: "[Programmers] N-Queen"
categories: ['Algorithm', 'Backtracking']
toc: true
toc_sticky: true
---



<br>

## 문제 설명

### 문제 설명

가로, 세로 길이가 n인 정사각형으로된 체스판이 있습니다. 체스판 위의 n개의 퀸이 서로를 공격할 수 없도록 배치하고 싶습니다.

예를 들어서 n이 4인경우 다음과 같이 퀸을 배치하면 n개의 퀸은 서로를 한번에 공격 할 수 없습니다.

![Imgur](https://i.imgur.com/lt2zdK6.png)
![Imgur](https://i.imgur.com/5c5EUrq.png)

체스판의 가로 세로의 세로의 길이 n이 매개변수로 주어질 때, n개의 퀸이 조건에 만족 하도록 배치할 수 있는 방법의 수를 return하는 solution함수를 완성해주세요.

### 제한사항

- 퀸(Queen)은 가로, 세로, 대각선으로 이동할 수 있습니다.
- n은 12이하의 자연수 입니다.

### 입출력 예

| n    | result |
| ---- | ------ |
| 4    | 2      |

#### 입출력 예 설명

입출력 예 #1
문제의 예시와 같습니다.

<br>

## 문제 풀이

### \# 백트래킹



<br>

### 풀이 과정

백트래킹 알고리즘을 사용하는 유명한 문제입니다. 

백트래킹의 기본은 재귀로, `base case`와 `general case`를 잘 세워주면 답이 보일 수 있습니다. 

여기서는 N개의 퀸을 배치할 수 있는 경우의 `i == N`이 **base case**이고, 행의 각 열에 대해 `앞서 배치된 퀸들과 서로 공격할 수 없는지 확인하는 것`이 **general case**입니다. 더불어 **general case**에서는 어떤 경우가 조건을 만족하는 경우, **1. 조건을 업데이트(추가)하고 2. 다음 재귀를 호출하고 3. 조건을 다시 업데이트(제외)**하는 일반적인 과정을 따릅니다. 

<br>

### 전체 코드

전체코드입니다. 

```python
def solve_nqueen(N, i, cnt, locs):
        if i == N:
            return cnt+1
        for j in range(N):
            for x,y in locs:
                if j == y or abs(i-x) == abs(j-y):
                    break
            else:
                locs.append((i,j))
                cnt = solve_nqueen(N,i+1,cnt,locs)
                locs.pop()
        return cnt

def solution(n):
    return solve_nqueen(n,0,0,[])
```

<br>

