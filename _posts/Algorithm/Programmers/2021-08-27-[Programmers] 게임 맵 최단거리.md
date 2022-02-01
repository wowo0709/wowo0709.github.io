---
layout: single
title: "[Programmers] 게임 맵 최단거리"
categories: ['Algorithm', 'Programmers']
---



# 게임 맵 최단거리

### 문제 설명

##### 문제 설명

ROR 게임은 두 팀으로 나누어서 진행하며, 상대 팀 진영을 먼저 파괴하면 이기는 게임입니다. 따라서, 각 팀은 상대 팀 진영에 최대한 빨리 도착하는 것이 유리합니다.

지금부터 당신은 한 팀의 팀원이 되어 게임을 진행하려고 합니다. 다음은 5 x 5 크기의 맵에, 당신의 캐릭터가 (행: 1, 열: 1) 위치에 있고, 상대 팀 진영은 (행: 5, 열: 5) 위치에 있는 경우의 예시입니다.

![최단거리1_sxuruo.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/dc3a1b49-13d3-4047-b6f8-6cc40b2702a7/%E1%84%8E%E1%85%AC%E1%84%83%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A5%E1%84%85%E1%85%B51_sxuruo.png)

위 그림에서 검은색 부분은 벽으로 막혀있어 갈 수 없는 길이며, 흰색 부분은 갈 수 있는 길입니다. 캐릭터가 움직일 때는 동, 서, 남, 북 방향으로 한 칸씩 이동하며, 게임 맵을 벗어난 길은 갈 수 없습니다.
아래 예시는 캐릭터가 상대 팀 진영으로 가는 두 가지 방법을 나타내고 있습니다.

- 첫 번째 방법은 11개의 칸을 지나서 상대 팀 진영에 도착했습니다.

![최단거리2_hnjd3b.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/9d909e5a-ca95-4088-9df9-d84cb804b2b0/%E1%84%8E%E1%85%AC%E1%84%83%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A5%E1%84%85%E1%85%B52_hnjd3b.png)

- 두 번째 방법은 15개의 칸을 지나서 상대팀 진영에 도착했습니다.

![최단거리3_ntxygd.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/4b7cd629-a3c2-4e02-b748-a707211131de/%E1%84%8E%E1%85%AC%E1%84%83%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A5%E1%84%85%E1%85%B53_ntxygd.png)

위 예시에서는 첫 번째 방법보다 더 빠르게 상대팀 진영에 도착하는 방법은 없으므로, 이 방법이 상대 팀 진영으로 가는 가장 빠른 방법입니다.

만약, 상대 팀이 자신의 팀 진영 주위에 벽을 세워두었다면 상대 팀 진영에 도착하지 못할 수도 있습니다. 예를 들어, 다음과 같은 경우에 당신의 캐릭터는 상대 팀 진영에 도착할 수 없습니다.

![최단거리4_of9xfg.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/d963b4bd-12e5-45da-9ca7-549e453d58a9/%E1%84%8E%E1%85%AC%E1%84%83%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A5%E1%84%85%E1%85%B54_of9xfg.png)

게임 맵의 상태 maps가 매개변수로 주어질 때, 캐릭터가 상대 팀 진영에 도착하기 위해서 지나가야 하는 칸의 개수의 **최솟값**을 return 하도록 solution 함수를 완성해주세요. 단, 상대 팀 진영에 도착할 수 없을 때는 -1을 return 해주세요.

##### 제한사항

- maps는 n x m 크기의 게임 맵의 상태가 들어있는 2차원 배열로, n과 m은 각각 1 이상 100 이하의 자연수입니다.
  - n과 m은 서로 같을 수도, 다를 수도 있지만, n과 m이 모두 1인 경우는 입력으로 주어지지 않습니다.
- maps는 0과 1로만 이루어져 있으며, 0은 벽이 있는 자리, 1은 벽이 없는 자리를 나타냅니다.
- 처음에 캐릭터는 게임 맵의 좌측 상단인 (1, 1) 위치에 있으며, 상대방 진영은 게임 맵의 우측 하단인 (n, m) 위치에 있습니다.

------

##### 입출력 예

| maps                                                         | answer |
| ------------------------------------------------------------ | ------ |
| [[1,0,1,1,1],[1,0,1,0,1],[1,0,1,1,1],[1,1,1,0,1],[0,0,0,0,1]] | 11     |
| [[1,0,1,1,1],[1,0,1,0,1],[1,0,1,1,1],[1,1,1,0,0],[0,0,0,0,1]] | -1     |

##### 입출력 예 설명

입출력 예 #1
주어진 데이터는 다음과 같습니다.

![최단거리6_lgjvrb.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/6db71f7f-58d3-4623-9fab-7cd99fa863a5/%E1%84%8E%E1%85%AC%E1%84%83%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A5%E1%84%85%E1%85%B56_lgjvrb.png)

캐릭터가 적 팀의 진영까지 이동하는 가장 빠른 길은 다음 그림과 같습니다.

![최단거리2_hnjd3b (1).png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/d223d017-b3e2-4772-9045-a565133d45ff/%E1%84%8E%E1%85%AC%E1%84%83%E1%85%A1%E1%86%AB%E1%84%80%E1%85%A5%E1%84%85%E1%85%B52_hnjd3b%20%281%29.png)

따라서 총 11칸을 캐릭터가 지나갔으므로 11을 return 하면 됩니다.

입출력 예 #2
문제의 예시와 같으며, 상대 팀 진영에 도달할 방법이 없습니다. 따라서 -1을 return 합니다.

<br>

### 문제 풀이

---

#### \# 최단거리 탐색 \# BFS

<br>

**탐색 알고리즘**에는 DFS와 BFS가 있다. 

**DFS**는 스택 또는 재귀를 활용하고, **BFS**는 큐(덱 또는 힙)를 활용한다. 

<br>

물론 임의의 탐색 문제에 있어 두 방식 모두로 풀릴 수 있다. 

하지만, 문제 유형에 따라 어느 탐색 알고리즘을 사용하느냐에 의해 그 효율성이 결정될 수 있다. 

<br>

**DFS**는 보통 모든 경우의 수를 찾아야 할 때 사용하며, 그 예로 **백트래킹**이 있다. 

**BFS**는 보통 최단거리를 찾아야 할 때 사용하며, 그 예로 **다익스트라, 벨만-포드** 등이 있다. 

<br>

<br>

각설하고, 이 문제에서는 BFS를 사용해야 한다. DFS를 사용하면 효율성 테스트를 통과하지 못하니 말이다. 

앞의 일련의 설명들에 충분히 공감했다면, 이 문제를 왜 BFS로 풀어야 하는지 감을 잡을 수 있을 것이다. 

<br>

바로, **최단 거리 찾기** 문제이기 때문이다. 

DFS를 사용하면 들어갈 수 있는 가장 깊은 경로까지 들어 간 다음 다른 곳을 탐색하기 때문에, 초반 경로에 따라 효율성이 정말 안 좋을 수 있다. 

반면 BFS는 특정 지점까지 가는 데는 DFS보다 느릴지 몰라도, 목표 지점까지 꾸준하게 나아간다. 

```python
def solution(map):
    from collections import deque
    search = deque([(0,0,1)])
    n, m = len(map), len(map[0])
    dxdy = [(0,1),(0,-1),(-1,0),(1,0)]
    visited = dict()
    visited[(0,0)] = 1
    while search:
        x, y, d = search.popleft()
        for dx, dy in dxdy:
            nx, ny = x + dx, y + dy
            # 종료 조건문은 for 문 안에!
            if (ny,nx) == (n-1,m-1): 
                return d + 1
            if -1<nx<m and -1<ny<n and map[ny][nx] and (nx,ny) not in visited:
                # visited에는 새로 방문하는 지점을 추가!
                visited[(nx,ny)] = 1
                search.append((nx,ny,d+1))
    return -1
```

코드는 전형적이다. 

**덱을 사용**하고, **visited** 배열(여기서는 딕셔너리)을 만들고, 조건에 맞으면 덱에 추가한다. 

<br>

코드에서 볼 수 있는 몇가지 특징을 정리하자. 

* visited 배열의 자료구조로 딕셔너리를 사용한다. 이는 방문 지점 조건을 검사할 때 시간 복잡도를 **O(1)**으로 하기 위함이다. 
* 종료 조건문은 for문 안에 넣는다. 이는 아래와 같이 하지 말자는 것이다. 

```python
    while search:
        x, y, d = search.popleft()
        # 종료 조건을 밖으로 뺌
        if (y,x) == (n-1,m-1): 
                return d
        for dx, dy in dxdy:
            nx, ny = x + dx, y + dy
            if -1<nx<m and -1<ny<n and map[ny][nx] and (nx,ny) not in visited:
                # visited에는 새로 방문하는 지점을 추가!
                visited[(nx,ny)] = 1
                search.append((nx,ny,d+1))
    return -1
```

얼마나 시간이 더 걸린다고 단정 짓지는 못하지만, 다음 후보지가 많은 경우에 후보지를 모두 추가한 후 그것을 다시 꺼낼 때(탐색할 차례가 올 때) 종료 조건을 검사하는 것은 비효율적일 수 있다. 

* visited에는 새로 방문하는 지점을 추가한다. 역시 다음과 같이 하지 말자는 소리다. 

```python
	visited = dict()
    # visited[(0,0)] = 1
    while search:
        x, y, d = search.popleft()
        for dx, dy in dxdy:
            nx, ny = x + dx, y + dy
            if (ny,nx) == (n-1,m-1): 
                return d + 1
            if -1<nx<m and -1<ny<n and map[ny][nx] and (nx,ny) not in visited:
                # visited에 현재 출발 지점을 추가
                visited[(x,y)] = 1
                search.append((nx,ny,d+1))
    return -1
```

실제로 위와 같이 코드를 작성해도 같은 로직이다. 

하지만 이는 새롭게 방문하는 지점이 쓸데없이 다른 경로에 한 번 더 추가되는 결과를 낳게 된다. 

<br>

<br>

<span style="color:red"> 🔥관습적으로 **'탐색에는 DFS나 BFS!'**를 외치지 말고, 그 핵심을 이해하여 **문제 유형에 맞는 탐색**을 선택하도록 하자! 🔥</span>

