---
layout: single
title: "[Programmers] 전력망을 둘로 나누기"
categories: ['Algorithm', 'Bruteforce', 'DFSBFS', 'UnionFind']
toc: true
toc_sticky: true
---



<br>

## 문제 설명

### 문제 설명

n개의 송전탑이 전선을 통해 하나의 [트리](https://en.wikipedia.org/wiki/Tree_(data_structure)) 형태로 연결되어 있습니다. 당신은 이 전선들 중 하나를 끊어서 현재의 전력망 네트워크를 2개로 분할하려고 합니다. 이때, 두 전력망이 갖게 되는 송전탑의 개수를 최대한 비슷하게 맞추고자 합니다.

송전탑의 개수 n, 그리고 전선 정보 wires가 매개변수로 주어집니다. 전선들 중 하나를 끊어서 송전탑 개수가 가능한 비슷하도록 두 전력망으로 나누었을 때, 두 전력망이 가지고 있는 송전탑 개수의 차이(절대값)를 return 하도록 solution 함수를 완성해주세요.

### 제한사항

- n은 2 이상 100 이하인 자연수입니다.
- wires는 길이가 `n-1`인 정수형 2차원 배열입니다.
  - wires의 각 원소는 [v1, v2] 2개의 자연수로 이루어져 있으며, 이는 전력망의 v1번 송전탑과 v2번 송전탑이 전선으로 연결되어 있다는 것을 의미합니다.
  - 1 ≤ v1 < v2 ≤ n 입니다.
  - 전력망 네트워크가 하나의 트리 형태가 아닌 경우는 입력으로 주어지지 않습니다.

### 입출력 예

| n    | wires                                               | result |
| ---- | --------------------------------------------------- | ------ |
| 9    | `[[1,3],[2,3],[3,4],[4,5],[4,6],[4,7],[7,8],[7,9]]` | 3      |
| 4    | `[[1,2],[2,3],[3,4]]`                               | 0      |
| 7    | `[[1,2],[2,7],[3,7],[3,4],[4,5],[6,7]]`             | 1      |

#### 입출력 예 설명

입출력 예 #1

- 다음 그림은 주어진 입력을 해결하는 방법 중 하나를 나타낸 것입니다.
- ![ex1.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/5b8a0dcd-cba0-47ca-b5e3-d3bafc81f9d6/ex1.png)
- 4번과 7번을 연결하는 전선을 끊으면 두 전력망은 각 6개와 3개의 송전탑을 가지며, 이보다 더 비슷한 개수로 전력망을 나눌 수 없습니다.
- 또 다른 방법으로는 3번과 4번을 연결하는 전선을 끊어도 최선의 정답을 도출할 수 있습니다.

입출력 예 #2

- 다음 그림은 주어진 입력을 해결하는 방법을 나타낸 것입니다.
- ![ex2.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/b28865e1-a18e-429d-ae7a-14e77e801539/ex2.png)
- 2번과 3번을 연결하는 전선을 끊으면 두 전력망이 모두 2개의 송전탑을 가지게 되며, 이 방법이 최선입니다.

입출력 예 #3

- 다음 그림은 주어진 입력을 해결하는 방법을 나타낸 것입니다.
- ![ex3.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/0a7f21af-1e07-4015-8ad3-c06155c613b3/ex3.png)
- 3번과 7번을 연결하는 전선을 끊으면 두 전력망이 각각 4개와 3개의 송전탑을 가지게 되며, 이 방법이 최선입니다.

<br>

## 문제 풀이

### \# BruteForce \# BFS \# UnionFind



<br>

### 풀이 과정

전선을 하나씩 끊어가면서, 둘로 나눠진 두 전력망의 크기가 가장 비슷해질 때를 찾는 문제입니다. 

아마 알고리즘을 좀 풀어보신 분들이라면 문제 유형 자체는 익숙하지 않을까 싶습니다. 

결국 모든 경우의 수를 따져야 하므로 **BruteFoce** 알고리즘을 사용하고, 하나의 전력망의 크기를 구할 때는 **BFS(또는 UnionFind)** 알고리즘을 사용하면 됩니다. 이렇게 그래프에서 인접한 노드의 개수를 세야 할 때는 bfs나 union find 알고리즘이 주로 사용되죠. (시간 상으로는 union find가 조금 더 빠른 것 같습니다)

여기서 짚고 넘어갈 것은 **어떻게 전선을 하나씩 제외할 것인가**의 문제인데, 저는 입력의 크기가 크지 않아서 그냥 매번 그래프를 새로 만드는 것으로 코드를 짰지만 이 부분에 대해 더 고민해보시면 좋을 듯 합니다. 



<br>

### 전체 코드



```python
def solution(n, wires):
    from collections import deque
    def count_adj_nodes(node):
        cnt = 1
        q = deque([node])
        visited = [0 for _ in range(n)]
        visited[node] = 1
        while q:
            node = q.popleft()
            for adj_node in graph[node]:
                if not visited[adj_node]:
                    q.append(adj_node)
                    visited[adj_node] = 1
                    cnt += 1
        return cnt
    
    ans = n 
    for except_wire in range(len(wires)):
        graph = {i:[] for i in range(n)}
        for i,(u,v) in enumerate(wires):
            if i == except_wire: continue
            graph[u-1].append(v-1)
            graph[v-1].append(u-1)
        cnt = count_adj_nodes(0)
        ans = min(ans, abs(n-2*cnt))
    return ans
```







<br>

