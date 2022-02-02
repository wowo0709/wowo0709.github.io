---
layout: single
title: "[Programmers] 하노이의 탑"
categories: ['Algorithm', 'Implementation']
toc: true
toc_sticky: true
tag: ['Recursive']
---



<br>

## 문제 설명

### 문제 설명

하노이 탑(Tower of Hanoi)은 퍼즐의 일종입니다. 세 개의 기둥과 이 기동에 꽂을 수 있는 크기가 다양한 원판들이 있고, 퍼즐을 시작하기 전에는 한 기둥에 원판들이 작은 것이 위에 있도록 순서대로 쌓여 있습니다. 게임의 목적은 다음 두 가지 조건을 만족시키면서, 한 기둥에 꽂힌 원판들을 그 순서 그대로 다른 기둥으로 옮겨서 다시 쌓는 것입니다.

1. 한 번에 하나의 원판만 옮길 수 있습니다.
2. 큰 원판이 작은 원판 위에 있어서는 안됩니다.

하노이 탑의 세 개의 기둥을 왼쪽 부터 1번, 2번, 3번이라고 하겠습니다. 1번에는 n개의 원판이 있고 이 n개의 원판을 3번 원판으로 최소 횟수로 옮기려고 합니다.

1번 기둥에 있는 원판의 개수 n이 매개변수로 주어질 때, n개의 원판을 3번 원판으로 최소로 옮기는 방법을 return하는 solution를 완성해주세요.

### 제한사항

* n은 15이하의 자연수 입니다.

### 입출력 예

| n    | result                  |
| ---- | ----------------------- |
| 2    | [ [1,2], [1,3], [2,3] ] |

#### 입출력 예 설명

입출력 예 #1
다음과 같이 옮길 수 있습니다.

![Imgur](https://i.imgur.com/SWEqD08.png)
![Imgur](https://i.imgur.com/mrmOzV2.png)
![Imgur](https://i.imgur.com/Ent83gA.png)
![Imgur](https://i.imgur.com/osJFfhF.png)

<br>

## 문제 풀이

### \# 재귀



<br>

### 풀이 과정

재귀 함수 문제 중 가장 유명한 문제 중 하나인 하노이의 탑 문제입니다. 

재귀함수는 때로는 직관을 필요로 합니다. 

하노이의 탑을 1에서 3으로 옮긴다는 것은, 

1. 가장 아래층을 제외한 N-1개의 층을 2로 먼저 옮기고
2. 가장 아래층을 1에서 3으로 옮기고
3. 2로 옮겼던 N-1개 층을 3으로 옮기는

과정을 따릅니다. 

여기서 코드의 일반화를 위해 1은 start, 2는 by, 3은 end로 간주하고 코드를 작성합니다. 

<br>

그리고 함수에서 by 위치의 값은, **중간에 해당 값을 가지고 있기 위함이지 꼭 거친다는 의미는 아닙니다.** 즉, start와 end 위치의 값이 맞으면 됩니다. 

<br>

### 전체 코드

전체코드입니다. 

```python
def solve_hanoi(n,start,by,end,result):
    if n == 1: return result + [[start,end]]
    
    result = solve_hanoi(n-1,start,end,by,result)
    result = solve_hanoi(1,start,by,end,result)
    result = solve_hanoi(n-1,by,start,end,result)
    
    return result

def solution(n):
    return solve_hanoi(n,1,2,3,[])
```

<br>

