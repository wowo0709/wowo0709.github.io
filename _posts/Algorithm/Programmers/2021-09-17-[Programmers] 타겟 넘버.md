---
layout: single
title: "[Programmers] 타겟 넘버"
categories: ['Algorithm', 'DFSBFS']
toc: true
toc_sticky: true
tag: ['Product']
---



<br>

## 문제 설명

### 문제 설명

n개의 음이 아닌 정수가 있습니다. 이 수를 적절히 더하거나 빼서 타겟 넘버를 만들려고 합니다. 예를 들어 [1, 1, 1, 1, 1]로 숫자 3을 만들려면 다음 다섯 방법을 쓸 수 있습니다.

```
-1+1+1+1+1 = 3
+1-1+1+1+1 = 3
+1+1-1+1+1 = 3
+1+1+1-1+1 = 3
+1+1+1+1-1 = 3
```

사용할 수 있는 숫자가 담긴 배열 numbers, 타겟 넘버 target이 매개변수로 주어질 때 숫자를 적절히 더하고 빼서 타겟 넘버를 만드는 방법의 수를 return 하도록 solution 함수를 작성해주세요.

### 제한사항

* 주어지는 숫자의 개수는 2개 이상 20개 이하입니다.
* 각 숫자는 1 이상 50 이하인 자연수입니다.
* 타겟 넘버는 1 이상 1000 이하인 자연수입니다.

### 입출력 예

| numbers         | target | return |
| --------------- | ------ | ------ |
| [1, 1, 1, 1, 1] | 3      | 5      |

### 입출력 예 설명

문제에 나온 예와 같습니다.

<br>

## 문제 풀이

### \# 탐색 \# dfs/bfs

<br>

### 👍 1번 풀이

재귀를 통한 dfs를 사용합니다. 

```python
def solution(numbers, target):
    N = len(numbers)

    def dfs(i,v,ans):
        if i == N:
            return ans+1 if v == target else ans
        ans = dfs(i+1,v+numbers[i],ans)
        ans = dfs(i+1,v-numbers[i],ans)
        return ans

    return dfs(0,0,0)
```

 <br>

### 👍 2번 풀이

product 모듈을 사용합니다. 

```python
def solution(numbers, target):
    from itertools import product
    l = [(x, -x) for x in numbers]
    s = list(map(sum, product(*l)))
    return s.count(target)
```



