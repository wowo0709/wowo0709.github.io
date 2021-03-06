---
layout: single
title: "[Programmers] 가장 큰 정사각형 찾기"
categories: ['Algorithm', 'Bruteforce', 'DynamicProgramming']
---

# 가장 큰 정사각형 찾기

### 문제 설명

---

- 1와 0로 채워진 표(board)가 있습니다. 표 1칸은 1 x 1 의 정사각형으로 이루어져 있습니다. 표에서 1로 이루어진 가장 큰 정사각형을 찾아 넓이를 return 하는 solution 함수를 완성해 주세요. (단, 정사각형이란 축에 평행한 정사각형을 말합니다.)

  예를 들어
  
  |  1   |  2   |  3   |  4   |
  | :--: | :--: | :--: | :--: |
  |  0   |  1   |  1   |  1   |
  |  1   |  1   |  1   |  1   |
  |  1   |  1   |  1   |  1   |
  |  0   |  0   |  1   |  0   |

  가 있다면 가장 큰 정사각형은

  |  1   |  2   |  3   |  4   |
  | :--: | :--: | :--: | :--: |
  |  0   | `1`  | `1`  | `1`  |
  |  1   | `1`  | `1`  | `1`  |
  |  1   | `1`  | `1`  | `1`  |
  |  0   |  0   |  1   |  0   |
  
  가 되며 넓이는 9가 되므로 9를 반환해 주면 됩니다.
  
  ##### 제한사항
  
  - 표(board)는 2차원 배열로 주어집니다.
  - 표(board)의 행(row)의 크기 : 1,000 이하의 자연수
  - 표(board)의 열(column)의 크기 : 1,000 이하의 자연수
  - 표(board)의 값은 1또는 0으로만 이루어져 있습니다.

  ------

  ##### 입출력 예

  | board                                     | answer |
  | ----------------------------------------- | ------ |
  | [[0,1,1,1],[1,1,1,1],[1,1,1,1],[0,0,1,0]] | 9      |
  | [[0,0,1,1],[1,1,1,1]]                     | 4      |
  
  ##### 입출력 예 설명
  
  입출력 예 #1
  위의 예시와 같습니다.
  
  입출력 예 #2
  | 0 | 0 | `1` | `1` |
  | 1 | 1 | `1` | `1` |
  로 가장 큰 정사각형의 넓이는 4가 되므로 4를 return합니다.



### 문제 풀이

---

```python
def solution(board):

    for i in range(1, len(board)):
        for j in range(1,len(board[0])):
            if board[i][j] >= 1:
                board[i][j] = min(board[i - 1][j],board[i][j - 1],board[i - 1][j - 1]) + 1

    return max([n for arr in board for n in arr])**2
```

* 가장 큰 정사각형을 구하는 방법에 대해 이해하고 있으면 좋을 것 같습니다. 
* 히스토그램 모형에서 가장 큰 직사각형의 넓이를 구하는 문제가 있습니다. 범용적으로 고려해 볼 때, **주어진 영역에서의 특정 모양의 최대 넓이를 구하는 문제**는 **for문 안에서 최대 넓이가 될 조건과 점화식을 함께 고려**하여 순차적으로 접근하면 문제를 수월하게 풀이할 수 있을 것입니다.  



**[참고] inline 이중 for문**

이차원 배열 원소의 순차 접근이 필요할 때, 리스트 내에서 이중 for문을 사용할 수 있습니다. 

* 풀어쓴 이중 for 문

```python
for arr in board:
    for n in arr:
        print(n)
```

* inline 이중 for 문

```python
[n for arr in board for n in arr]
```

inline for문 배치 순서는 바깥 for문일수록 앞에 옵니다. 





그럼 안녕!
