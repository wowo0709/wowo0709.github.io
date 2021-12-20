---
layout: single
title: "[Programmers] 섬 연결하기"
categories: ['Algorithm', 'Programmers']
toc: true
toc_sticky: true
tag: ['BFS','유니온파인드']
---



<br>

## 문제 설명

### 문제 설명

n개의 섬 사이에 다리를 건설하는 비용(costs)이 주어질 때, 최소의 비용으로 모든 섬이 서로 통행 가능하도록 만들 때 필요한 최소 비용을 return 하도록 solution을 완성하세요.

다리를 여러 번 건너더라도, 도달할 수만 있으면 통행 가능하다고 봅니다. 예를 들어 A 섬과 B 섬 사이에 다리가 있고, B 섬과 C 섬 사이에 다리가 있으면 A 섬과 C 섬은 서로 통행 가능합니다.

### **제한사항**

- 섬의 개수 n은 1 이상 100 이하입니다.
- costs의 길이는 `((n-1) * n) / 2`이하입니다.
- 임의의 i에 대해, costs[i][0] 와 costs[i] [1]에는 다리가 연결되는 두 섬의 번호가 들어있고, costs[i] [2]에는 이 두 섬을 연결하는 다리를 건설할 때 드는 비용입니다.
- 같은 연결은 두 번 주어지지 않습니다. 또한 순서가 바뀌더라도 같은 연결로 봅니다. 즉 0과 1 사이를 연결하는 비용이 주어졌을 때, 1과 0의 비용이 주어지지 않습니다.
- 모든 섬 사이의 다리 건설 비용이 주어지지 않습니다. 이 경우, 두 섬 사이의 건설이 불가능한 것으로 봅니다.
- 연결할 수 없는 섬은 주어지지 않습니다.

### **입출력 예**

| n    | costs                                     | return |
| ---- | ----------------------------------------- | ------ |
| 4    | [[0,1,1],[0,2,2],[1,2,5],[1,3,1],[2,3,8]] | 4      |

#### **입출력 예 설명**

costs를 그림으로 표현하면 다음과 같으며, 이때 초록색 경로로 연결하는 것이 가장 적은 비용으로 모두를 통행할 수 있도록 만드는 방법입니다.

<br>

## 문제 풀이

### \# BFS \# 유니온 파인드



<br>

### 전체 코드

👍 **1번 풀이**: BFS

```python
def solution(n, costs):
    def bfs(u,v):
        from collections import deque
        visited = [0 for _ in range(n)]
        q = deque([u])
        while q:
            cur_node = q.popleft()
            if visited[cur_node]: continue
            visited[cur_node] = 1
            if cur_node == v: return 1
            for adj_node in graph[cur_node]:
                q.append(adj_node)
        return 0

    costs.sort(key=lambda x:x[2])
    graph = {i:[] for i in range(n)}
    ans = 0
    for u,v,cost in costs:
        if bfs(u,v): 
            continue
        graph[u].append(v)
        graph[v].append(u)
        ans += cost
            
    return ans
```

<br>

👍 **2번 풀이**: 유니온 파인드

```python
def solution(n, costs):
    # 부모 노드 탐색
    def find(x):
        if parent[x] < 0:
            return x
        p = find(parent[x])
        parent[x] = p
        return p
    # 두 트리를 합침
    def union(x,y):
        x = find(x)
        y = find(y)
        # 이미 같은 트리에 속한 노드일 경우
        if x == y: return 0
        # 두 노드가 속한 트리 중 높이가 낮은 트리를 높은 트리에 합침
        if parent[x] < parent[y]:
            parent[x] += parent[y]
            parent[y] = x
        else:
            parent[y] += parent[x]
            parent[x] = y
        return 1
    
    costs.sort(key=lambda x:x[2])
    # 음수이면 최상위 노드. 최상위 노드인 경우 절댓값은 집합의 크기를 나타냄. 
    parent = [-1]*n
    ans = 0
    for u,v,cost in costs:
        if union(u,v) == 0: continue
        else: ans += cost
        if -parent[find(u)] == n: break
            
    return ans
```



<br>

