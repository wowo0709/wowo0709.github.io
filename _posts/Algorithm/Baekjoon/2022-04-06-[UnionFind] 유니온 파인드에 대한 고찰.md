---
layout: single
title: "[UnionFind] '유니온 파인드'에 대한 고찰"
categories: ['Graph', 'Tree', 'UnionFind']
toc: true
toc_sticky: true
tag: ['union', 'find']
---

_'유니온 파인드'에 대해 새로운 부분을 발견할 때마다 업데이트합니다._

# '유니온 파인드'에 대한 고찰

**유니온 파인드**

![image-20220406151313490](https://user-images.githubusercontent.com/70505378/161908002-3e70efbc-7158-4f87-a1e8-52277a4c0f0c.png)

`유니온 파인드`는 두 집합이 연결되어 있는지 여부를 확인하고 연결할 수 있는 알고리즘이다. 

집합의 연결성을 탐색하기 위해 그래프에서 **BFS**가 사용된다면, 트리에서는 **유니온 파인드**가 흔하게 사용된다. (물론 교차해서 사용도 가능하다)

유니온 파인드에서는 말 그대로 **union**과 **find** 연산이 핵심 연산이다. 순서 상 find 연산이 앞서서 일어난다. 

<br>

* `parent`: 자신의 루트 노드를 가리킴. 자신이 루트 노드라면 -(트리의 노드 수) 값을 가짐. 

연산에 앞서서 union-find에서 사용되는 변수인 `parent` 리스트에 대해 알아보겠습니다. parent의 값은 처음에는 전부 -1로 초기화되며, 이는 본인의 루트 노드가 본인(즉, 모든 노드가 떨어져 있는 상태)이라는 뜻입니다. 

```python
parent = [-1 for _ in range(N+1)]
```

 `parent[x] = y`는 x의 루트 노드가 y라는 것이며, y가 루트 노드인 경우 `parent[y] = -(트리의 노드 수)` 값을 가집니다. 

Union 과정에서 집합들이 합쳐지면 값들이 갱신됩니다. 

<br>

* `find`: 인자로 임의의 노드 x를 전달하면 x의 루트 노드(최상위 부모 노드)를 반환한다. 

  ```python
  # find root node
  def find(x):
      if parent[x] < 0: # 루트 노드라면 본인을 return
          return x
      p = find(parent[x])
      parent[x] = p
      return p
  ```

<br>

* `union`: 인자로 임의의 두 노드 x, y를 전달하면 x, y 노드가 같은 집합 내에 있는지 검사하고, 아니라면 두 집합을 합친다. 

  * 일반적으로 연산의 효율성을 증대하기 위해 **더 작은 트리를 더 큰 트리에 합칩니다.**

  ```python
  # merge two tree
  def union(x,y):
      # 루트 노드 탐색
      x = find(x)
      y = find(y)
      # 이미 같은 집합에 있는 경우
      if x == y: return False
      # 작은 트리를 큰 트리에 합침
      if parent[x] < parent[y]:
          parent[x] += parent[y]
          parent[y] = x
      else:
          parent[y] += parent[x]
          parent[x] = y
      return True
  ```





















<br>













