---
layout: single
title: "[Programmers] [카카오 인턴] 경주로 건설"
categories: ['Algorithm', 'DFSBFS', 'ShortestPath']
toc: true
toc_sticky: true
tag: ['Dijkstra']
---



<br>

## 문제 설명

### 문제 설명

![kakao_road1.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/384b9e2a-4eb5-460d-bce2-d12359b03b14/kakao_road1.png)

건설회사의 설계사인 `죠르디`는 고객사로부터 자동차 경주로 건설에 필요한 견적을 의뢰받았습니다.
제공된 경주로 설계 도면에 따르면 경주로 부지는 `N x N` 크기의 정사각형 격자 형태이며 각 격자는 `1 x 1` 크기입니다.
설계 도면에는 각 격자의 칸은 `0` 또는 `1` 로 채워져 있으며, `0`은 칸이 비어 있음을 `1`은 해당 칸이 벽으로 채워져 있음을 나타냅니다.
경주로의 출발점은 (0, 0) 칸(좌측 상단)이며, 도착점은 (N-1, N-1) 칸(우측 하단)입니다. 죠르디는 출발점인 (0, 0) 칸에서 출발한 자동차가 도착점인 (N-1, N-1) 칸까지 무사히 도달할 수 있게 중간에 끊기지 않도록 경주로를 건설해야 합니다.
경주로는 상, 하, 좌, 우로 인접한 두 빈 칸을 연결하여 건설할 수 있으며, 벽이 있는 칸에는 경주로를 건설할 수 없습니다.
이때, 인접한 두 빈 칸을 상하 또는 좌우로 연결한 경주로를 `직선 도로` 라고 합니다.
또한 두 `직선 도로`가 서로 직각으로 만나는 지점을 `코너` 라고 부릅니다.
건설 비용을 계산해 보니 `직선 도로` 하나를 만들 때는 100원이 소요되며, `코너`를 하나 만들 때는 500원이 추가로 듭니다.
죠르디는 견적서 작성을 위해 경주로를 건설하는 데 필요한 최소 비용을 계산해야 합니다.

예를 들어, 아래 그림은 `직선 도로` 6개와 `코너` 4개로 구성된 임의의 경주로 예시이며, 건설 비용은 6 x 100 + 4 x 500 = 2600원 입니다.

![kakao_road2.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/0e0911e8-f88e-44fe-8bdc-6856a56df8e0/kakao_road2.png)

또 다른 예로, 아래 그림은 `직선 도로` 4개와 `코너` 1개로 구성된 경주로이며, 건설 비용은 4 x 100 + 1 x 500 = 900원 입니다.

![kakao_road3.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/3f5d9c5e-d7d9-4248-b111-140a0847e741/kakao_road3.png)

------

도면의 상태(0은 비어 있음, 1은 벽)을 나타내는 2차원 배열 board가 매개변수로 주어질 때, 경주로를 건설하는데 필요한 최소 비용을 return 하도록 solution 함수를 완성해주세요.

### **[제한사항]**

- board는 2차원 정사각 배열로 배열의 크기는 3 이상 25 이하입니다.
- board 배열의 각 원소의 값은 0 또는 1 입니다.
  - 도면의 가장 왼쪽 상단 좌표는 (0, 0)이며, 가장 우측 하단 좌표는 (N-1, N-1) 입니다.
  - 원소의 값 0은 칸이 비어 있어 도로 연결이 가능함을 1은 칸이 벽으로 채워져 있어 도로 연결이 불가능함을 나타냅니다.
- board는 항상 출발점에서 도착점까지 경주로를 건설할 수 있는 형태로 주어집니다.
- 출발점과 도착점 칸의 원소의 값은 항상 0으로 주어집니다.

### **입출력 예**

| board                                                        | result |
| ------------------------------------------------------------ | ------ |
| [[0,0,0],[0,0,0],[0,0,0]]                                    | 900    |
| [[0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0],[0,0,0,0,0,1,0,0],[0,0,0,0,1,0,0,0],[0,0,0,1,0,0,0,1],[0,0,1,0,0,0,1,0],[0,1,0,0,0,1,0,0],[1,0,0,0,0,0,0,0]] | 3800   |
| [[0,0,1,0],[0,0,0,0],[0,1,0,1],[1,0,0,0]]                    | 2100   |
| [[0,0,0,0,0,0],[0,1,1,1,1,0],[0,0,1,0,0,0],[1,0,0,1,0,1],[0,1,0,0,0,1],[0,0,0,0,0,0]] | 3200   |

#### **입출력 예에 대한 설명**

**입출력 예 #1**

본문의 예시와 같습니다.

**입출력 예 #2**

![kakao_road4.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/ccc72e9c-2e22-4a09-a94b-ff057b081a70/kakao_road4.png)

위와 같이 경주로를 건설하면 `직선 도로` 18개, `코너` 4개로 총 3800원이 듭니다.

**입출력 예 #3**

![kakao_road5.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/422e86e0-a7d7-4a09-9b42-2b6218a9b5f0/kakao_road5.png)

위와 같이 경주로를 건설하면 `직선 도로` 6개, `코너` 3개로 총 2100원이 듭니다.

**입출력 예 #4**

![kakao_road6.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/4fe42f47-2592-4cb8-91fb-31d6a6da8639/kakao_road6.png)

붉은색 경로와 같이 경주로를 건설하면 `직선 도로` 12개, `코너` 4개로 총 3200원이 듭니다.
만약, 파란색 경로와 같이 경주로를 건설한다면 `직선 도로` 10개, `코너` 5개로 총 3500원이 들며, 더 많은 비용이 듭니다.

------

※ 공지 - 2021년 8월 30일 테스트케이스가 추가되었습니다.

<br>

## 문제 풀이

### \# BFS \# 다익스트라



<br>

### 풀이 과정

이 문제...!! 쉽지 않았습니다. 

처음 문제를 봤을 때는, 목적지까지 갈 수 있는 경로가 여러 개 있고, 최소 비용을 찾기 위해서는 각 경로 간의 비용을 비교해야 하기 때문에, dfs(backtracking) 방법을 사용했습니다. 그러나, board의 최대 크기가 25인 상태에서 탐색해야 할 경우의 수는 기하 급수적으로 늘어날 수 있습니다. 따라서, 이 문제에는 적합하지 않습니다. 

두 번째로, 시간 초과를 극복하기 위해 bfs와 dynamic programming을 사용하려 했고, 조금 더 효율적이고 빠른 방법을 제공하는 dijkstra 방법을 사용했습니다. 이 문제의 도로의 형태에 따른 비용을 **가중치**라고 본다면, 다익스트라 알고리즘을 사용할 수 있습니다. 하지만 이 방법도 대부분의 TC를 맞히긴 했지만, **24, 25번 문제를 통과하지 못했습니다.**

문제가 요구하는 방향은 이게 맞는거 같은데, 도대체 뭐가 문제일까요??

2번째 풀이의 문제점은 아래에서 확인할 수 있습니다. 

> ![image-20220129205325041](https://user-images.githubusercontent.com/70505378/151660504-a63bd5d0-3ab0-4870-989d-fda27929c5c0.png)
>
> 예를 들어, 위와 같은 일부분의 board를 생각해봅시다. 
>
> **27 VS 29**인 블록에서 당연히 **27**이 선택이 되는데, 이로 인해 그 아래 블록인 **33 VS 30**인 블록에서 최솟값인 **30**을 찾지 못하고 **33**을 찾고 맙니다. 
>
> 참조: https://programmers.co.kr/questions/21325 

위와 같은 경우를 방지하기 위해, 우리는 해당 블록까지의 최소 비용에 **방향**이라는 정보를 추가해주어야 합니다. 

따라서, 세번째 풀이에서는 해당 블록에 오기 위해 상/하/좌/우 어디서 왔는지에 따라 최솟값을 따로 저장하여 위와 같은 경우를 방지합니다. 





<br>

### 전체 코드

😂 **1번 풀이: 시간초과**

Backtracking을 사용했지만 시간 초과가 발생한 코드입니다. 

```python
def solution(board):
    N = len(board)
    didj = [(-1,0),(1,0),(0,-1),(0,1)] # 상하좌우
    board[0][0] = 1
    ans = []
    
    def backtracking(cur, direction, cost):
        if cur == (N-1,N-1):
            ans.append(cost)
            return
        for di,dj in _didj:
            new_i, new_j = cur[0]+di, cur[1]+dj
            if 0 <= new_i < N and 0 <= new_j < N and not board[new_i][new_j]:
                board[new_i][new_j] = 1
                if direction == None: _cost = 100 
                elif direction == (di,dj): _cost = cost + 100
                else: _cost = cost + 600
                backtracking((new_i,new_j), (di,dj), _cost)
                board[new_i][new_j] = 0
        return
                
    backtracking((0,0), None, 0)
    return min(ans)
```

😂 **2번 풀이: 24, 25번 TC 실패**

다익스트라 알고리즘을 사용했지만 이후 방향에 따른 최솟값 역전 문제를 고려하지 못 해 실패한 코드입니다. 

```python
def solution(board):
    N = len(board)
    didj = [(-1,0),(1,0),(0,-1),(0,1)] # 상하좌우
    # dijkstra
    from heapq import heappush, heappop
    costs = [[float('inf') for _ in range(N)] for _ in range(N)]
    costs[0][0] = 0
    heap = []
    heappush(heap, (0,(0,0), None)) # cost, cur, direction
    while heap:
        cost, cur, direction = heappop(heap)
        if costs[cur[0]][cur[1]] < cost:
            continue
        for di, dj in didj:
            new_i, new_j = cur[0]+di, cur[1]+dj
            if direction == None: new_cost = 100
            elif direction == (di,dj): new_cost = cost + 100
            else: new_cost = cost + 600
            if 0 <= new_i < N and 0 <= new_j < N and not board[new_i][new_j]\
                                                    and costs[new_i][new_j] >= new_cost:
                heappush(heap,(new_cost, (new_i,new_j), (di,dj)))
                costs[new_i][new_j] = new_cost
    return costs[N-1][N-1]
```

😁 **3번 풀이: 성공**

마찬가지로 다익스트라 알고리즘을 사용했으며, 이동 방향에 따른 최소 비용을 따로 저장해 최종 성공한 코드입니다. 

```python
def solution(board):
    N = len(board)
    didj = [(-1,0),(1,0),(0,-1),(0,1)] # 상하좌우
    # dijkstra
    from heapq import heappush, heappop
    costs = [[{k:v for (k,v) in zip(didj,[float('inf')]*4)} for _ in range(N)] for _ in range(N)] # i, j, direction
    for k in didj:
        costs[0][0][k] = 0
    heap = []
    heappush(heap, (0,(0,0), None)) # cost, cur, direction
    while heap:
        cost, cur, direction = heappop(heap)
        if direction and costs[cur[0]][cur[1]][direction] < cost:
            continue
        for di, dj in didj:
            new_i, new_j = cur[0]+di, cur[1]+dj
            if direction == None: new_cost = 100
            elif direction == (di,dj): new_cost = cost + 100
            else: new_cost = cost + 600
            if 0 <= new_i < N and 0 <= new_j < N and not board[new_i][new_j]\
                                                    and costs[new_i][new_j][(di,dj)] >= new_cost:
                heappush(heap,(new_cost, (new_i,new_j), (di,dj)))
                costs[new_i][new_j][(di,dj)] = new_cost
    return min(costs[N-1][N-1].values())
```







<br>

