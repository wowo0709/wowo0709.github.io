---
layout: single
title: "[Programmers] 빛의 경로 사이클"
categories: ['Algorithm', 'Bruteforce', 'Implementation']
toc: true
toc_sticky: true
---



<br>

## 문제 설명

### 문제 설명

각 칸마다 S, L, 또는 R가 써져 있는 격자가 있습니다. 당신은 이 격자에서 빛을 쏘고자 합니다. 이 격자의 각 칸에는 다음과 같은 특이한 성질이 있습니다.

* 빛이 "S"가 써진 칸에 도달한 경우, 직진합니다.
* 빛이 "L"이 써진 칸에 도달한 경우, 좌회전을 합니다.
* 빛이 "R"이 써진 칸에 도달한 경우, 우회전을 합니다.
* 빛이 격자의 끝을 넘어갈 경우, 반대쪽 끝으로 다시 돌아옵니다. 예를 들어, 빛이 1행에서 행이 줄어드는 방향으로 이동할 경우, 같은 열의 반대쪽 끝 행으로 다시 돌아옵니다.

당신은 이 격자 내에서 빛이 이동할 수 있는 경로 사이클이 몇 개 있고, 각 사이클의 길이가 얼마인지 알고 싶습니다. 경로 사이클이란, 빛이 이동하는 순환 경로를 의미합니다.

예를 들어, 다음 그림은 격자 `["SL","LR"]`에서 1행 1열에서 2행 1열 방향으로 빛을 쏠 경우, 해당 빛이 이동하는 경로 사이클을 표현한 것입니다.

![ex1.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/f3c02c50-f82e-45d0-b633-ad3ecadba316/ex1.png)

이 격자에는 길이가 16인 사이클 1개가 있으며, 다른 사이클은 존재하지 않습니다.

격자의 정보를 나타내는 1차원 문자열 배열 `grid`가 매개변수로 주어집니다. 주어진 격자를 통해 만들어지는 빛의 경로 사이클의 모든 길이들을 배열에 담아 오름차순으로 정렬하여 return 하도록 solution 함수를 완성해주세요.

### 제한사항

* 1 ≤ `grid`의 길이 ≤ 500
* 1 ≤ `grid`의 각 문자열의 길이 ≤ 500
  * `grid`의 모든 문자열의 길이는 서로 같습니다.
* `grid`의 모든 문자열은 `'L', 'R', 'S'`로 이루어져 있습니다.

### 입출력 예

| grid          | result      |
| ------------- | ----------- |
| `["SL","LR"]` | `[16]`      |
| `["S"]`       | `[1,1,1,1]` |
| `["R","R"]`   | `[4,4]`     |

### 입출력 예 설명

**입출력 예 #1**

* 문제 예시와 같습니다.
* 길이가 16인 사이클이 하나 존재하므로(다른 사이클은 없습니다.), `[16]`을 return 해야 합니다.

**입출력 예 #2**

* 주어진 격자를 통해 만들 수 있는 사이클들은 다음 그림과 같습니다.

![ex2.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/88a2717d-14ab-4297-af06-00baab718080/ex2.png)

* 4개의 사이클의 길이가 모두 1이므로, `[1,1,1,1]`을 return 해야 합니다.

**입출력 예 #3**

* 주어진 격자를 통해 만들 수 있는 사이클들은 다음 그림과 같습니다.

![ex3.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/076dbe07-2b33-414e-b6db-1e73ae2055f3/ex3.png)

* 2개의 사이클의 길이가 모두 4이므로, `[4,4]`를 return 해야 합니다.

<br>

## 문제 풀이

### \# 탐색

<br>

### 풀이 과정

각 `'S', 'R', 'L'` 노드를 만나면 갈 수 있는 방향이 정해져 있습니다. 

하나의 사이클은 빛이 시작 노드로 돌아와 출발 방향과 같은 방향으로 이동할 시 종료됩니다. 

<br>

우리가 선택할 수 있는 것은 `어느 노드에서 시작할 것인지`와 `시작 노드에서 어느 방향으로 출발할 것인지`입니다. 

각 노드에 대해 상/하/좌/우 방향으로 출발해보면서, 그 경로가 이미 찾은 사이클에 포함되어 있는 경로라면 탐색을 하지 않고 포함되어 있지 않다면 탐색을 진행합니다. 

<br>

### 전체 코드

전체 코드입니다. 

```python
def solution(grid):
    N, M = len(grid), len(grid[0]) # 세로, 가로
    ans = []
    visited = {(i, j): [] for i in range(N) for j in range(M)}
    
    for cur in visited.keys():
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]: # 상하좌우
            cnt = 0
            ### 사이클에 포함된 경로가 나오기 전까지 이동
            while (di, dj) not in visited[tuple(cur)]:
                visited[tuple(cur)].append((di,dj))
                cur = [cur[0]+di, cur[1]+dj]
                cnt += 1
                ### 위치 조정 (격자를 넘은 경우)
                if cur[0] < 0: cur[0] += N
                if cur[0] >= N: cur[0] -= N
                if cur[1] < 0: cur[1] += M
                if cur[1] >= M: cur[1] -= M
                ### 노드에 따라 이동 경로 조정
                if grid[cur[0]][cur[1]] == 'L': di,dj = (-dj,-di) if di == 0 else (dj,di)
                elif grid[cur[0]][cur[1]] == 'R': di,dj = (dj,di) if di == 0 else (-dj,-di)
                else: pass # grid[cur[0]][cur[1]] == 'S'   
            # 사이클 종료 시
            if cnt: ans.append(cnt)

    return sorted(ans)
```

<br>

### 정리

이 문제처럼 4방위로 움직일 수 있는 문제에서 저는 항상 dx, dy의 값이 헷갈렸습니다. 

그래서 이제부터는 직관적으로 이해하기 쉽게 di(행의 위치), dj(열의 위치)로 변수명을 지으려구요!







