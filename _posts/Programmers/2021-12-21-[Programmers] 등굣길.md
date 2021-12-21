---
layout: single
title: "[Programmers] 등굣길"
categories: ['Algorithm', 'Programmers']
toc: true
toc_sticky: true
tag: ['BFS','동적계획법']
---



<br>

## 문제 설명

### 문제 설명

계속되는 폭우로 일부 지역이 물에 잠겼습니다. 물에 잠기지 않은 지역을 통해 학교를 가려고 합니다. 집에서 학교까지 가는 길은 m x n 크기의 격자모양으로 나타낼 수 있습니다.

아래 그림은 m = 4, n = 3 인 경우입니다.

![image0.png](https://grepp-programmers.s3.amazonaws.com/files/ybm/056f54e618/f167a3bc-e140-4fa8-a8f8-326a99e0f567.png)

가장 왼쪽 위, 즉 집이 있는 곳의 좌표는 (1, 1)로 나타내고 가장 오른쪽 아래, 즉 학교가 있는 곳의 좌표는 (m, n)으로 나타냅니다.

격자의 크기 m, n과 물이 잠긴 지역의 좌표를 담은 2차원 배열 puddles이 매개변수로 주어집니다. **오른쪽과 아래쪽으로만 움직여** 집에서 학교까지 갈 수 있는 최단경로의 개수를 1,000,000,007로 나눈 나머지를 return 하도록 solution 함수를 작성해주세요.

### 제한사항

- 격자의 크기 m, n은 1 이상 100 이하인 자연수입니다.
  - m과 n이 모두 1인 경우는 입력으로 주어지지 않습니다.
- 물에 잠긴 지역은 0개 이상 10개 이하입니다.
- 집과 학교가 물에 잠긴 경우는 입력으로 주어지지 않습니다.

### 입출력 예

| m    | n    | puddles  | return |
| ---- | ---- | -------- | ------ |
| 4    | 3    | [[2, 2]] | 4      |

#### 입출력 예 설명

![image1.png](https://grepp-programmers.s3.amazonaws.com/files/ybm/32c67958d5/729216f3-f305-4ad1-b3b0-04c2ba0b379a.png)

<br>

## 문제 풀이

### \# BFS \# 동적계획법



<br>

### 전체 코드

👍 **1번 풀이**: BFS (이동 방향 제한 없이 이동 가능)

```python
def solution(m, n, puddles):
    from collections import deque
    div = 1e+09 + 7
    puddles = [(p[1]-1,p[0]-1) for p in puddles]
    visited = [[[0,float('inf'),1] for _ in range(m)] for _ in range(n)] # [방문여부,최단거리,최단경로개수]...
    q = deque([[(-1,-1),(0,0),0]]) # [(fromi,fromj),(toi,toj),d]...
    while q:
        (from_i,from_j),(cur_i,cur_j),d = q.popleft()
        if (cur_i,cur_j) in puddles: continue
        if visited[cur_i][cur_j][0]:
            v,min_d,min_d_cnt = visited[cur_i][cur_j]
            if d < min_d: 
                visited[cur_i][cur_j] = [1,d,1]
            elif d == min_d:
                visited[cur_i][cur_j] = [1,min_d,(min_d_cnt+visited[from_i][from_j][2]) % div]
            continue
        visited[cur_i][cur_j] = [1,d,visited[from_i][from_j][2] % div]
        for di,dj in [(0,1),(1,0)]:
            if cur_i+di<n and cur_j+dj<m:
                q.append([(cur_i,cur_j),(cur_i+di,cur_j+dj),d+1])
        
    return visited[n-1][m-1][2] % div if visited[n-1][m-1][1] != float('inf') else 0
```

<br>

👍 **2번 풀이**: 학창시절 때 배웠던 경로찾기 (우/하 방향만 이동 가능)

```python
def solution(m,n,puddles):
    grid = [[0]*(m+1) for i in range(n+1)] # grid[i][j] = 최단경로 개수
    if puddles != [[]]:                    
        for a, b in puddles:
            grid[b][a] = -1                
    grid[1][1] = 1
    for i in range(1,n+1):
        for j in range(1,m+1):
            if i == j == 1: continue
            if grid[i][j] == -1:
                grid[i][j] = 0
                continue
            grid[i][j] = (grid[i][j-1] + grid[i-1][j])%1000000007   # [a,b] = [a-1,b] + [a,b-1] 공식

    return grid[n][m]
```



<br>

