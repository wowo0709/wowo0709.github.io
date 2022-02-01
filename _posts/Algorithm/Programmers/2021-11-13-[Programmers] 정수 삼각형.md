---
layout: single
title: "[Programmers] 정수 삼각형"
categories: ['Algorithm', 'Programmers']
toc: true
toc_sticky: true
tag: ['동적계획법']
---



<br>

## 문제 설명

### 문제 설명

![스크린샷 2018-09-14 오후 5.44.19.png](https://grepp-programmers.s3.amazonaws.com/files/production/97ec02cc39/296a0863-a418-431d-9e8c-e57f7a9722ac.png)

위와 같은 삼각형의 꼭대기에서 바닥까지 이어지는 경로 중, 거쳐간 숫자의 합이 가장 큰 경우를 찾아보려고 합니다. 아래 칸으로 이동할 때는 대각선 방향으로 한 칸 오른쪽 또는 왼쪽으로만 이동 가능합니다. 예를 들어 3에서는 그 아래칸의 8 또는 1로만 이동이 가능합니다.

삼각형의 정보가 담긴 배열 triangle이 매개변수로 주어질 때, 거쳐간 숫자의 최댓값을 return 하도록 solution 함수를 완성하세요.

### 제한사항

- 삼각형의 높이는 1 이상 500 이하입니다.
- 삼각형을 이루고 있는 숫자는 0 이상 9,999 이하의 정수입니다.

### 입출력 예

| triangle                                                | result |
| ------------------------------------------------------- | ------ |
| [[7], [3, 8], [8, 1, 0], [2, 7, 4, 4], [4, 5, 2, 6, 5]] | 30     |

[출처](http://stats.ioinformatics.org/countries/SWE)

<br>

## 문제 풀이

### \# 동적계획법

<br>

### 풀이 과정

`동적계획법`의 가장 기초적이자 중요한 문제입니다. 

코드를 보면 직관적인 이해가 가능해서 자세한 코드 설명은 생략합니다. 

<br>

### 전체 코드

전체 코드입니다. 

```python
def solution(triangle):
    for i in range(1, len(triangle)):
        for j in range(i+1):
            if j == 0: triangle[i][j] = triangle[i][j] + triangle[i-1][j]
            elif j == i: triangle[i][j] = triangle[i][j] + triangle[i-1][j-1]
            else: triangle[i][j] = triangle[i][j] + max(triangle[i-1][j], triangle[i-1][j-1])
                
    return max(triangle[-1])
```

<br>

### 정리

항상 `동적계획법` 문제를 풀 때면 다음 두 가지를 생각합니다. 

**1. dp 배열의 인덱스와 값에는 어떤 값(어떤 기준)을 사용할지**

**2. dp 배열의 값은 어떻게 구할 수 있는지(어떤 점화식을 세울 수 있는지)**

위 두가지를 고민하여 빠르게 잡아낼 수 있는 것이 동적 계획법 문제를 잘 풀이할 수 있는 방법이라고 생각합니다. 









