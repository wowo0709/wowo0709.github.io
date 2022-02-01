---
layout: single
title: "[Programmers] 순위"
categories: ['Algorithm', 'Programmers']
toc: true
toc_sticky: true
tag: ['그래프', '완전탐색']
---



<br>

## 문제 설명

### 문제 설명

n명의 권투선수가 권투 대회에 참여했고 각각 1번부터 n번까지 번호를 받았습니다. 권투 경기는 1대1 방식으로 진행이 되고, 만약 A 선수가 B 선수보다 실력이 좋다면 A 선수는 B 선수를 항상 이깁니다. 심판은 주어진 경기 결과를 가지고 선수들의 순위를 매기려 합니다. 하지만 몇몇 경기 결과를 분실하여 정확하게 순위를 매길 수 없습니다.

선수의 수 n, 경기 결과를 담은 2차원 배열 results가 매개변수로 주어질 때 정확하게 순위를 매길 수 있는 선수의 수를 return 하도록 solution 함수를 작성해주세요.

### 제한사항

* 선수의 수는 1명 이상 100명 이하입니다.
* 경기 결과는 1개 이상 4,500개 이하입니다.
* results 배열 각 행 [A, B]는 A 선수가 B 선수를 이겼다는 의미입니다.
* 모든 경기 결과에는 모순이 없습니다.

### 입출력 예

| n    | results                                  | return |
| ---- | ---------------------------------------- | ------ |
| 5    | [[4, 3], [4, 2], [3, 2], [1, 2], [2, 5]] | 2      |

### 입출력 예 설명

2번 선수는 [1, 3, 4] 선수에게 패배했고 5번 선수에게 승리했기 때문에 4위입니다.
5번 선수는 4위인 2번 선수에게 패배했기 때문에 5위입니다.

[출처](http://contest.usaco.org/JAN08.htm)

<br>

## 문제 풀이

### \# 그래프 \# 완전탐색

<br>

### 풀이 과정

* **우선 첫번째로, 입력의 크기가 매우 작습니다.**
  * 이로부터 완전 탐색 `for i in range(n)`을 고려해볼 수 있습니다. 
* **다음으로, 순위가 정해질 조건입니다.**
  * `이긴 사람 수 + 진 사람 수 = n-1`이면 순위를 정할 수 있습니다. 
  * 1이 2에게 졌다면 1한테 진 사람들은 2한테 무조건 집니다. 
  * 3이 4에게 이겼다면 3이 진 사람들은 4한테 무조건 이깁니다. 

<br>

### 전체 코드

전체 코드입니다. 

```python
def solution(n, results):
    ### 순위가 정해질 조건 = 이긴 사람 수 + 진 사람 수 = n-1
    # win_graph = 1: {2,3,4} -> 1이 2,3,4에게 이김
    # lose_graph = 4: {2,3} -> 4가 2,3에게 짐
    win_graph = {i:set() for i in range(1,n+1)}
    lose_graph = {i:set() for i in range(1,n+1)}
    for win, lose in results:
        win_graph[win].add(lose)
        lose_graph[lose].add(win)
    # 1이 2에게 졌다면 1한테 진 사람들은 2한테 무조건 짐
    # 3이 4에게 이겼다면 3이 진 사람들은 4한테 무조건 이김
    for i in range(1,n+1):
        for loser in win_graph[i]:
            lose_graph[loser].update(lose_graph[i])
        for winner in lose_graph[i]:
            win_graph[winner].update(win_graph[i])
    # 순위가 정해지는지 검사
    ans = 0
    for i in range(1,n+1):
        if len(win_graph[i])+len(lose_graph[i]) == n-1: ans += 1
    
    return ans
```

<br>

### 정리

문제 유형이 `그래프`이길래 dfs/bfs를 활용해보려다 오히려 많이 돌아가버린 문제입니다...

힌트에 집착하지 않고 문제를 풀 수 있는 본질에 집중해야 할 것 같습니다. 

**+ 아, 그리고 입력의 크기가 중요한 힌트가 될 수 있다는 사실을 항상 인지합시다!!**







