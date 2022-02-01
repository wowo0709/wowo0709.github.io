---
layout: single
title: "[Programmers] 네트워크"
categories: ['Algorithm', 'Programmers']
toc: true
toc_sticky: true
tag: ['DFS/BFS']
---



<br>

## 문제 설명

### 문제 설명

네트워크란 컴퓨터 상호 간에 정보를 교환할 수 있도록 연결된 형태를 의미합니다. 예를 들어, 컴퓨터 A와 컴퓨터 B가 직접적으로 연결되어있고, 컴퓨터 B와 컴퓨터 C가 직접적으로 연결되어 있을 때 컴퓨터 A와 컴퓨터 C도 간접적으로 연결되어 정보를 교환할 수 있습니다. 따라서 컴퓨터 A, B, C는 모두 같은 네트워크 상에 있다고 할 수 있습니다.

컴퓨터의 개수 n, 연결에 대한 정보가 담긴 2차원 배열 computers가 매개변수로 주어질 때, 네트워크의 개수를 return 하도록 solution 함수를 작성하시오.

### 제한사항

- 컴퓨터의 개수 n은 1 이상 200 이하인 자연수입니다.
- 각 컴퓨터는 0부터 `n-1`인 정수로 표현합니다.
- i번 컴퓨터와 j번 컴퓨터가 연결되어 있으면 computers[i][j]를 1로 표현합니다.
- computer[i][i]는 항상 1입니다.

### 입출력 예

| n    | computers                         | return |
| ---- | --------------------------------- | ------ |
| 3    | [[1, 1, 0], [1, 1, 0], [0, 0, 1]] | 2      |
| 3    | [[1, 1, 0], [1, 1, 1], [0, 1, 1]] | 1      |

### 입출력 예 설명

예제 #1
아래와 같이 2개의 네트워크가 있습니다.
<img src="https://grepp-programmers.s3.amazonaws.com/files/ybm/5b61d6ca97/cc1e7816-b6d7-4649-98e0-e95ea2007fd7.png" alt="image0.png" style="zoom:67%;" />

예제 #2
아래와 같이 1개의 네트워크가 있습니다.
<img src="https://grepp-programmers.s3.amazonaws.com/files/ybm/7554746da2/edb61632-59f4-4799-9154-de9ca98c9e55.png" alt="image1.png" style="zoom:67%;" />

<br>

## 문제 풀이

### \# 동적계획법

<br>

### 풀이 과정

주어진 그래프에서 몇 개의 집합(subset)이 있는지 탐색하는 문제입니다. 

문제를 보자마자 DFS 또는 BFS를 사용하면 되겠다는 생각이 듭니다. 저는 DFS를 사용했습니다. 

바로 코드를 보시죠. 

<br>

### 전체 코드

전체 코드입니다. 

```python
def solution(n, computers):
    visited = [0 for _ in range(n)]
    ans = 0
    for start_node in range(n):
        if visited[start_node]: continue
        s = [start_node]
        while s:
            cur_node = s.pop()
            for i in range(n):
                if i == cur_node or not computers[cur_node][i] or visited[i]:
                    continue
                s.append(i)
                visited[i] = 1
        ans += 1
        
    return ans
```

`visited`는 해당 노드 탐색의 완료 여부를 나타내기 위한 리스트입니다. 이것으로 같은 노드를 2번 이상 방문하지 않습니다. 

탐색 시작 노드는 **0 ~ n-1** 번 노드 중 어떤 노드든 될 수 있습니다. 다만, 앞서서 이미 탐색을 진행했었다면 다른 노드와의 네트워크를 이루고 있는 것이기 때문에 **continue**합니다. 

아직 방문하지 않았다면, 해당 시작 노드와 네트워크를 이루고 있는 노드들을 모두 탐색합니다. 

<br>

### 정리

DFS/BFS를 활용하면 되는 어렵지 않은 문제였습니다. 

다만, DFS와 BFS를 사용하는 경우를 어느정도 구분할 줄 아는 것이 중요합니다. 

그저 "`DFS`는 `스택/재귀`, `BFS`는 `큐`를 사용하자!" 의 수준에 머무르면 안됩니다. 

<br>

그래서 간단히 정리를 하자면, 

* **BFS**: 너비 우선 탐색. 미로 탐색 문제와 같이 최단 경로를 탐색하는 문제에서 주로 사용. 
* **DFS**: 깊이 우선 탐색. 탐색에서 가중치가 존재하거나 이동 과정에서 제약이 있는 경우, 또는 모든 노드를 방문해봐야 하는 경우. (탐색 시간은 더 걸리지만 가중치에 대한 변수를 지속해서 관리할 수 있다는 장점이 있음)

이렇게 알아두고 그래프 탐색 문제를 풀 때 어느 탐색 기법이 더 효율적인지 고민하고 문제를 푸는 것이 좋습니다. 







