---
layout: single
title: "[Programmers] 양과 늑대"
categories: ['Algorithm', 'Programmers']
toc: true
toc_sticky: true
tag: ['Tree','Graph', 'DFS']
---



<br>

## 문제 설명

### 문제 설명

2진 트리 모양 초원의 각 노드에 늑대와 양이 한 마리씩 놓여 있습니다. 이 초원의 루트 노드에서 출발하여 각 노드를 돌아다니며 양을 모으려 합니다. 각 노드를 방문할 때 마다 해당 노드에 있던 양과 늑대가 당신을 따라오게 됩니다. 이때, 늑대는 양을 잡아먹을 기회를 노리고 있으며, 당신이 모은 양의 수보다 늑대의 수가 같거나 더 많아지면 바로 모든 양을 잡아먹어 버립니다. 당신은 중간에 양이 늑대에게 잡아먹히지 않도록 하면서 최대한 많은 수의 양을 모아서 다시 루트 노드로 돌아오려 합니다.

![03_2022_공채문제_양과늑대_01.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/ed7118a9-a99b-4f3a-9779-a94816529e78/03_2022_%E1%84%80%E1%85%A9%E1%86%BC%E1%84%8E%E1%85%A2%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A6_%E1%84%8B%E1%85%A3%E1%86%BC%E1%84%80%E1%85%AA%E1%84%82%E1%85%B3%E1%86%A8%E1%84%83%E1%85%A2_01.png)

예를 들어, 위 그림의 경우(루트 노드에는 항상 양이 있습니다) 0번 노드(루트 노드)에서 출발하면 양을 한마리 모을 수 있습니다. 다음으로 1번 노드로 이동하면 당신이 모은 양은 두 마리가 됩니다. 이때, 바로 4번 노드로 이동하면 늑대 한 마리가 당신을 따라오게 됩니다. 아직은 양 2마리, 늑대 1마리로 양이 잡아먹히지 않지만, 이후에 갈 수 있는 아직 방문하지 않은 모든 노드(2, 3, 6, 8번)에는 늑대가 있습니다. 이어서 늑대가 있는 노드로 이동한다면(예를 들어 바로 6번 노드로 이동한다면) 양 2마리, 늑대 2마리가 되어 양이 모두 잡아먹힙니다. 여기서는 0번, 1번 노드를 방문하여 양을 2마리 모은 후, 8번 노드로 이동한 후(양 2마리 늑대 1마리) 이어서 7번, 9번 노드를 방문하면 양 4마리 늑대 1마리가 됩니다. 이제 4번, 6번 노드로 이동하면 양 4마리, 늑대 3마리가 되며, 이제 5번 노드로 이동할 수 있게 됩니다. 따라서 양을 최대 5마리 모을 수 있습니다.

각 노드에 있는 양 또는 늑대에 대한 정보가 담긴 배열 `info`, 2진 트리의 각 노드들의 연결 관계를 담은 2차원 배열 `edges`가 매개변수로 주어질 때, 문제에 제시된 조건에 따라 각 노드를 방문하면서 모을 수 있는 양은 최대 몇 마리인지 return 하도록 solution 함수를 완성해주세요.

### 제한사항

- 2 ≤ `info`의 길이 ≤ 17
  - `info`의 원소는 0 또는 1 입니다.
  - info[i]는 i번 노드에 있는 양 또는 늑대를 나타냅니다.
  - 0은 양, 1은 늑대를 의미합니다.
  - info[0]의 값은 항상 0입니다. 즉, 0번 노드(루트 노드)에는 항상 양이 있습니다.
- `edges`의 세로(행) 길이 = `info`의 길이 - 1
  - `edges`의 가로(열) 길이 = 2
  - `edges`의 각 행은 [부모 노드 번호, 자식 노드 번호] 형태로, 서로 연결된 두 노드를 나타냅니다.
  - 동일한 간선에 대한 정보가 중복해서 주어지지 않습니다.
  - 항상 하나의 이진 트리 형태로 입력이 주어지며, 잘못된 데이터가 주어지는 경우는 없습니다.
  - 0번 노드는 항상 루트 노드입니다.

### 입출력 예

| info                      | edges                                                        | result |
| ------------------------- | ------------------------------------------------------------ | ------ |
| [0,0,1,1,1,0,1,0,1,0,1,1] | [[0,1],[1,2],[1,4],[0,8],[8,7],[9,10],[9,11],[4,3],[6,5],[4,6],[8,9]] | 5      |
| [0,1,0,1,1,0,1,0,0,1,0]   | [[0,1],[0,2],[1,3],[1,4],[2,5],[2,6],[3,7],[4,8],[6,9],[9,10]] | 5      |

#### 입출력 예 설명

**입출력 예 #1**

문제의 예시와 같습니다.

**입출력 예 #2**

주어진 입력은 다음 그림과 같습니다.

![03_2022_공채문제_양과늑대_02.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/32656ee0-814e-4dd9-93a3-abed1ce31ec1/03_2022_%E1%84%80%E1%85%A9%E1%86%BC%E1%84%8E%E1%85%A2%E1%84%86%E1%85%AE%E1%86%AB%E1%84%8C%E1%85%A6_%E1%84%8B%E1%85%A3%E1%86%BC%E1%84%80%E1%85%AA%E1%84%82%E1%85%B3%E1%86%A8%E1%84%83%E1%85%A2_02.png)

0번 - 2번 - 5번 - 1번 - 4번 - 8번 - 3번 - 7번 노드 순으로 이동하면 양 5마리 늑대 3마리가 됩니다. 여기서 6번, 9번 노드로 이동하면 양 5마리, 늑대 5마리가 되어 양이 모두 잡아먹히게 됩니다. 따라서 늑대에게 잡아먹히지 않도록 하면서 최대로 모을 수 있는 양은 5마리입니다.

------

##### 제한시간 안내

- 정확성 테스트 : 10초

<br>

## 문제 풀이

### \# DFS

<br>

### 풀이 과정

어렵네요... 고민 많이 한 문제입니다. 

처음에는 bfs로 트리(그래프)를 탐색해나가며 **최적값(양이 제일 많은 값)을 전파**하도록 하려 했습니다. 여기서 전파 과정은 다음과 같습니다. 

* 이미 방문한 이웃 노드인 경우
  * 현재 노드의 최적값이 이웃 노드의 최적값보다 큰 경우
    * 이웃 노드에 현재 노드의 최적값을 복사, 큐에 append
  * 현재 노드의 최적값이 이웃 노드의 최적값 이하인 경우
    * continue
* 아직 방문하지 않은 이웃 노드인 경우
  * 이웃 노드의 info 값을 참조하여 양의 수가 늑대의 수보다 많은 경우의 이웃 노드일 경우
    * 이웃 노드에 현재 노드의 최적값을 복사하고 info 값에 맞춰 +1, 큐에 append
  * 양의 수가 늑대의 수 이하인 경우의 이웃 노드일 경우
    * continue

그런데 위와 같은 로직은, 항상 **값의 복사**가 이루어지기 때문에 부모 노드가 자식 노드를 2개 가지고 있는 경우 값이 더해지는 것이 아니라 **덮어써지게** 됩니다. 즉, 더해야 할 경우가 있고 복사해야 할 경우가 있는데 위 로직에서는 이를 잡아내지 못 하는 것이죠. 

해당 로직만 추가하는 것도 방법이겠으나... 너무 복잡한 것 같아 시도하지는 않았습니다. 

<br>

그래서 두 번째로 시도한 방법은, 값의 전파를 시키지 않고 **현재 방문한 노드들에서 다음으로 방문할 수 있는 노드들을 추가하며 그래프를 늘리면서 최적값을 탐색**하는 것입니다. 이 방법에서는 값의 전파가 일어나지 않고 새로 추가된 자식 노드의 값만 갱신되기 때문에 양방향 graph가 아닌 `childs` 방향 그래프를 사용합니다. 

해당 탐색을 하기 위해서 dfs를 활용했으며(다른 분은 [Bruteforce + bfs](https://wadekang.tistory.com/10)로 풀이하신 분도 있습니다), 노드 수의 제한이 17 개로 작기 때문에 시간초과는 걱정하지 않아도 됩니다. 

또한, 그래프가 만들어진 각 경우에서의 최적값을 저장하기 위해 `cnts`라는 딕셔너리를 만들었습니다. 여기서 cnts의 키 값은 **방문한 노드의 순서**가 됩니다. 

<br>

### 전체 코드

😂 **1번 풀이: 실패**

첫번째로 시도한 bfs를 활용한 풀이입니다. 

```python
def solution(info, edges):
    N = len(info)
    graph = {i:[] for i in range(N)}
    for u,v in edges:
        graph[u].append(v)
        graph[v].append(u)
    # bfs
    from collections import deque
    q = deque([0])
    cnts = [{'sheep':0, 'wolf':0} for _ in range(N)]
    cnts[0]['sheep'] = 1
    visited = [0 for _ in range(N)]
    while q:
        node = q.popleft()
        if cnts[node]['wolf'] >= cnts[node]['sheep']:
            continue
        visited[node] = 1
        for adj_node in graph[node]:
            # 자식 노드에서 부모 노드로 전파할 때 초기화되는 문제...
            if visited[adj_node]:
                if cnts[node]['sheep'] > cnts[adj_node]['sheep']:
                    cnts[adj_node] = cnts[node].copy()
                    q.append(adj_node)
            else:
                if info[adj_node] == 0 and cnts[node]['sheep']+1 > cnts[node]['wolf'] or\
                        info[adj_node] == 1 and cnts[node]['sheep'] > cnts[node]['wolf']+1:
                    cnts[adj_node] = cnts[node].copy()
                    if info[adj_node] == 0: cnts[adj_node]['sheep'] += 1
                    else: cnts[adj_node]['wolf'] += 1
                    q.append(adj_node)

    return cnts[0]['sheep']
```



😁 **2번 풀이: 성공**

두번째로 시도한 dfs를 활용한 풀이입니다. 

```python
def solution(info, edges):
    N = len(info)
    childs = [[] for _ in range(N)]
    for p, c in edges:
        childs[p].append(c)
    s = [(0,)] # 루트 노드 시작
    cnts = {(0,): [1,0]} # 양, 늑대
    # dfs
    while s:
        visited_nodes = s.pop()
        for visited_node in visited_nodes:
            for child_node in childs[visited_node]:
                if child_node in visited_nodes:
                    continue
                new_visited_nodes = (*visited_nodes,child_node)
                if info[child_node] == 0: 
                    if (cnts[visited_nodes][0]+1) - cnts[visited_nodes][1] > 0 and visited_nodes not in s:
                        cnts[new_visited_nodes] = [cnts[visited_nodes][0]+1,cnts[visited_nodes][1]]
                        s.append(new_visited_nodes)
                else:
                    if cnts[visited_nodes][0] - (cnts[visited_nodes][1]+1) > 0 and visited_nodes not in s:
                        cnts[new_visited_nodes] = [cnts[visited_nodes][0],cnts[visited_nodes][1]+1]
                        s.append(new_visited_nodes)
    return max([sheep for (sheep,wolf) in cnts.values()])
```





<br>

