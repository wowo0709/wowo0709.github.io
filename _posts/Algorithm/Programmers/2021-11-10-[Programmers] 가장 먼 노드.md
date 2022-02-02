---
layout: single
title: "[Programmers] 가장 먼 노드"
categories: ['Algorithm', 'Graph', 'ShortestPath']
toc: true
toc_sticky: true
tag: ['Dijkstra', 'BellmanFord', 'FloydWarshall']
---



<br>

## 문제 설명

### 문제 설명

n개의 노드가 있는 그래프가 있습니다. 각 노드는 1부터 n까지 번호가 적혀있습니다. 1번 노드에서 가장 멀리 떨어진 노드의 갯수를 구하려고 합니다. 가장 멀리 떨어진 노드란 최단경로로 이동했을 때 간선의 개수가 가장 많은 노드들을 의미합니다.

노드의 개수 n, 간선에 대한 정보가 담긴 2차원 배열 vertex가 매개변수로 주어질 때, 1번 노드로부터 가장 멀리 떨어진 노드가 몇 개인지를 return 하도록 solution 함수를 작성해주세요.

### 제한사항

* 노드의 개수 n은 2 이상 20,000 이하입니다.
* 간선은 양방향이며 총 1개 이상 50,000개 이하의 간선이 있습니다.
* vertex 배열 각 행 [a, b]는 a번 노드와 b번 노드 사이에 간선이 있다는 의미입니다.

### 입출력 예

| n    | vertex                                                   | return |
| ---- | -------------------------------------------------------- | ------ |
| 6    | [[3, 6], [4, 3], [3, 2], [1, 3], [1, 2], [2, 4], [5, 2]] | 3      |

### 입출력 예 설명

예제의 그래프를 표현하면 아래 그림과 같고, 1번 노드에서 가장 멀리 떨어진 노드는 4,5,6번 노드입니다.

![image.png](https://grepp-programmers.s3.amazonaws.com/files/ybm/fadbae38bb/dec85ab5-0273-47b3-ba73-fc0b5f6be28a.png)<br>

<br>

## 문제 풀이

### \# 그래프 \# 최단거리

<br>

우리가 자주 사용하는 `최단 거리 알고리즘`에는 3가지가 있죠. 

* `다익스트라 알고리즘`: (음수 가중치가 없을 때) 한 정점에서 다른 모든 정점까지의 최단거리를 계산. 
* `벨만-포드 알고리즘`: (음수 가중치가 있을 때) 한 정점에서 다른 모든 정점까지의 최단거리를 계산. 
* `플로이드-워셜 알고리즘`: 모든 정점에서 모든 정점까지의 최단거리를 계산. 그래프 배열이 필요없다. 

이 문제는 **다익스트라 알고리즘**으로 간단히 풀 수 있습니다. 

<br>

코드를 보기 전에, 다익스트라 알고리즘을 구현하기 위한 과정을 정리해봅시다. 

> 1. 그래프와 거리 배열을 생성한다. 이 때 그래프 배열은 주로 딕셔너리 형태로, 거리 배열은 리스트 형태로 정의한다. 
>
>    1-1. 그래프는 양방향인지, 단방향인지, 가중치가 있는지에 따라 적절히 정의. 
>
>    1-2. 거리 배열은 INF로 초기화하고 초기 정점 거리 값은 0으로 초기화. 
>
> 2. heap을 생성한다. heap은 (최단거리, 정점 번호)를 원소로 갖는다. 
>
>    2-1. heap에 (0, 초기 정점)을 push하여 초기화해준다. 
>
> 3. 힙이 빌 때까지 그래프를 탐색하며 최단거리를 갱신한다. 
>
>    3-1. heap의 원소를 pop. (현재 탐색 중인 최단거리,정점)
>
>    3-2. 현재 탐색 중인 최단거리가 기존의 최단거리보다 크면 continue
>
>    3-3. 아니라면, 현재 탐색 중인 노드의 인접한 모든 노드에 대해 최단 거리 계산. 계산된 최단거리가 기존의 최단거리보다 작다면 최단거리를 갱신하고 heap에 push. 

<br>

위 과정에 따라 솔루션 코드를 작성하면 다음과 같다. 

```python
def solution(n, edge):
    # 1. 그래프와 거리 배열을 생성한다.
    graph = {node+1: [] for node in range(n)}
    for i, j in edge:
        graph[i].append(j)
        graph[j].append(i)
    dists = [float('inf') for _ in range(n+1)]
    dists[0], dists[1] = -1, 0
    # 2. heap을 생성하고 초기화한다. 
    from heapq import heappush, heappop
    heap = []
    heappush(heap, (0,1))
    # 3. 힙이 빌 때까지 그래프를 탐색하며 최단거리를 갱신한다. 
    while heap:
        dist, node = heappop(heap)
        if dists[node] < dist: 
            continue
        for adj_node in graph[node]:
            if dist+1 < dists[adj_node]:
                dists[adj_node] = dist+1
                heappush(heap, (dist+1, adj_node))

    return dists.count(max(dists))
```

<br>

<br>

✋ **참고하세요!!!**

보통 최단거리 알고리즘에서 **다익스트라 알고리즘**이 제일 많이 사용되기는 하지만, 다른 알고리즘들도 종종 사용되니 이번 기회에 익숙해지세요!

* [다익스트라 알고리즘](https://justkode.kr/algorithm/python-dijkstra)

```python
import heapq  # 우선순위 큐 구현을 위함

... 
graph = collections.defaultdict(list) 

def dijkstra(graph, start):
  distances = {node: float('inf') for node in graph}  # start로 부터의 거리 값을 저장하기 위함
  distances[start] = 0  # 시작 값은 0이어야 함
  queue = []
  heapq.heappush(queue, [distances[start], start])  # 시작 노드부터 탐색 시작 하기 위함.

  while queue:  # queue에 남아 있는 노드가 없으면 끝
    current_distance, current_destination = heapq.heappop(queue)  # 탐색 할 노드, 거리를 가져옴.

    if distances[current_destination] < current_distance:  # 기존에 있는 거리보다 길다면, 볼 필요도 없음
      continue
    
    for new_destination, new_distance in graph[current_destination].items():
      distance = current_distance + new_distance  # 해당 노드를 거쳐 갈 때 거리
      if distance < distances[new_destination]:  # 알고 있는 거리 보다 작으면 갱신
        distances[new_destination] = distance
        heapq.heappush(queue, [distance, new_destination])  # 다음 인접 거리를 계산 하기 위해 큐에 삽입
    
  return distances
```

* [벨만-포드 알고리즘](https://cotak.tistory.com/90)

```python
... 
graph = collections.defaultdict(list) 

for u, v, w in inputs: # 양방향 그래프라고 가정 
    graph[u].append([v, w]) 
    graph[v].append([u, w]) 
    
def bellman_ford(s): 
    dist = [INF] * (V+1) # V는 노드의 개수 
    dist[s] = 0 # 시작 노드의 거리는 0으로 설정 
    # 최단경로 갱신
    for _ in range(V-1): # (정점개수 - 1) 번 반복
        for u in range(1, V+1): # 모든 정점 탐색
            for v, c in graph[u]: # 거리 계산 및 갱신 (모든 간선 탐색)
                if dist[v] > dist[u] + c: 
                    dist[v] = dist[u] + c 
    # 음수 사이클이 존재하는지 체크 
    for u in range(1, V+1): 
        for v, c in graph[u]: 
            if dist[v] > dist[u] + c: 
                return False 
            
    return dist
```

* [플로이드-워셜 알고리즘](https://it-garden.tistory.com/247)

```python
def floyd_warshall():
    dist = [[float('inf')]*n for _ in range(n)]
    # 최단 경로 배열 초기화
    for i in range(n):
        for j in range(n):
            dist[i][j] = a[i][j]
    # 모든 정점까지의 거리 계산
    for k in range(n):
        for i in range(n):
            for j in range(n):
                # k를 거쳤을 때의 경로가 더 적은 경로
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    
    return dist

n = 4 # 정점 수
a = [[0,2,float('inf'),4], [2,0,float('inf'),5], [3,float('inf'),0,float('inf')], [float('inf'),2,1,0]] # a에서 b까지의 거리 정보

dist = floyd_warshall()

for i in range(n):
    for j in range(n):
        print(dist[i][j], end=' ')
    print()
```

















