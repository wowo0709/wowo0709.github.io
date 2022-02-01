---
layout: single
title: "[Programmers] 행렬 테두리 회전하기"
categories: ['Algorithm', 'Programmers']
toc: true
toc_sticky: true
tag: ['행렬', '회전']
---



<br>

## 문제 설명

### 문제 설명

rows x columns 크기인 행렬이 있습니다. 행렬에는 1부터 rows x columns까지의 숫자가 한 줄씩 순서대로 적혀있습니다. 이 행렬에서 직사각형 모양의 범위를 여러 번 선택해, 테두리 부분에 있는 숫자들을 시계방향으로 회전시키려 합니다. 각 회전은 (x1, y1, x2, y2)인 정수 4개로 표현하며, 그 의미는 다음과 같습니다.

* x1 행 y1 열부터 x2 행 y2 열까지의 영역에 해당하는 직사각형에서 테두리에 있는 숫자들을 한 칸씩 시계방향으로 회전합니다.

다음은 6 x 6 크기 행렬의 예시입니다.

<img src="https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/ybm/4c3c0fab-11f4-43b6-b290-6f4017e9379f/grid_example.png" alt="grid_example.png" style="zoom:67%;" />

이 행렬에 (2, 2, 5, 4) 회전을 적용하면, 아래 그림과 같이 2행 2열부터 5행 4열까지 영역의 테두리가 시계방향으로 회전합니다. 이때, 중앙의 15와 21이 있는 영역은 회전하지 않는 것을 주의하세요.

<img src="https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/ybm/962df137-5c71-4091-ad9f-8e322910c1ab/rotation_example.png" alt="rotation_example.png" style="zoom:67%;" />

행렬의 세로 길이(행 개수) rows, 가로 길이(열 개수) columns, 그리고 회전들의 목록 queries가 주어질 때, 각 회전들을 배열에 적용한 뒤, 그 회전에 의해 위치가 바뀐 숫자들 중 **가장 작은 숫자들을 순서대로 배열에 담아** return 하도록 solution 함수를 완성해주세요.

### 제한사항

* rows는 2 이상 100 이하인 자연수입니다.
* columns는 2 이상 100 이하인 자연수입니다.
* 처음에 행렬에는 가로 방향으로 숫자가 1부터 하나씩 증가하면서 적혀있습니다.
    * 즉, 아무 회전도 하지 않았을 때, i 행 j 열에 있는 숫자는 ((i-1) x columns + j)입니다.
* queries의 행의 개수(회전의 개수)는 1 이상 10,000 이하입니다.
* queries의 각 행은 4개의 정수 [x1, y1, x2, y2]입니다.
    * x1 행 y1 열부터 x2 행 y2 열까지 영역의 테두리를 시계방향으로 회전한다는 뜻입니다.
    * 1 ≤ x1 < x2 ≤ rows, 1 ≤ y1 < y2 ≤ columns입니다.
    * 모든 회전은 순서대로 이루어집니다.
    * 예를 들어, 두 번째 회전에 대한 답은 첫 번째 회전을 실행한 다음, 그 상태에서 두 번째 회전을 실행했을 때 이동한 숫자 중 최솟값을 구하면 됩니다.

---

### 입출력 예시

| rows  | columns | queries                                     | result         |
| ----- | ------- | ------------------------------------------- | -------------- |
| `6`   | `6`     | `[[2,2,5,4],[3,3,6,6],[5,1,6,3]]`           | `[8, 10, 25]`  |
| `3`   | `3`     | `[[1,1,2,2],[1,2,2,3],[2,1,3,2],[2,2,3,3]]` | `[1, 1, 5, 3]` |
| `100` | `97`    | `[[1,1,100,97]]`                            | `[1]`          |

### 입출력 예 설명

입출력 예 #1

* 회전을 수행하는 과정을 그림으로 표현하면 다음과 같습니다.
* ![example1.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/ybm/8c8cdd84-d0ec-4b9d-bdf7-f100d0098c5e/example1.png)

입출력 예 #2

* 회전을 수행하는 과정을 그림으로 표현하면 다음과 같습니다.
* ![example2.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/ybm/e3fce2bf-9da9-41e4-926a-5d19b4f31188/example2.png)

입출력 예 #3

* 이 예시에서는 행렬의 테두리에 위치한 모든 칸들이 움직입니다. 따라서, 행렬의 테두리에 있는 수 중 가장 작은 숫자인 1이 바로 답이 됩니다.

<br>

## 문제 풀이

### \# 행렬 \# 회전

<br>

사실 키워드로 **행렬**을 선택하긴 했지만, 행렬에 대한 이해도와는 상관이 없는 문제입니다. 

다만, **회전**이라는 점에서 알아두면 좋은 알고리즘일 듯 합니다. 

```python
def solution(rows, columns, queries):
    ans = []
    matrix = [list(range(1+row*columns,columns+1+row*columns)) for row in range(rows)]

    for query in queries:
        nums = []
        x1,y1,x2,y2 = map(lambda x:x-1,query)
        tmp = matrix[x1][y1]
        for j in range(y1,y2): # 우로 이동
            nums.append(tmp)
            _tmp = matrix[x1][j+1]
            matrix[x1][j+1] = tmp
            tmp = _tmp
        for i in range(x1,x2): # 하로 이동
            nums.append(tmp)
            _tmp = matrix[i+1][y2]
            matrix[i+1][y2] = tmp
            tmp = _tmp
        for j in range(y2,y1,-1): # 좌로 이동
            nums.append(tmp)
            _tmp = matrix[x2][j-1]
            matrix[x2][j-1] = tmp
            tmp = _tmp
        for i in range(x2,x1,-1): # 상으로 이동
            nums.append(tmp)
            _tmp = matrix[i-1][y1]
            matrix[i-1][y1] = tmp
            tmp = _tmp
        ans.append(min(nums))
    
    return ans
```







<br>
