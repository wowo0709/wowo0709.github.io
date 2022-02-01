---
layout: single
title: "[Programmers] 자물쇠와 열쇠"
categories: ['Algorithm', 'Programmers']
toc: true
toc_sticky: true
tag: ['완전탐색', '해쉬맵']
---



<br>

## 문제 설명

### 문제 설명

고고학자인 **"튜브"**는 고대 유적지에서 보물과 유적이 가득할 것으로 추정되는 비밀의 문을 발견하였습니다. 그런데 문을 열려고 살펴보니 특이한 형태의 **자물쇠**로 잠겨 있었고 문 앞에는 특이한 형태의 **열쇠**와 함께 자물쇠를 푸는 방법에 대해 다음과 같이 설명해 주는 종이가 발견되었습니다.

잠겨있는 자물쇠는 격자 한 칸의 크기가 **`1 x 1`**인 **`N x N`** 크기의 정사각 격자 형태이고 특이한 모양의 열쇠는 **`M x M`** 크기인 정사각 격자 형태로 되어 있습니다.

자물쇠에는 홈이 파여 있고 열쇠 또한 홈과 돌기 부분이 있습니다. 열쇠는 회전과 이동이 가능하며 열쇠의 돌기 부분을 자물쇠의 홈 부분에 딱 맞게 채우면 자물쇠가 열리게 되는 구조입니다. 자물쇠 영역을 벗어난 부분에 있는 열쇠의 홈과 돌기는 자물쇠를 여는 데 영향을 주지 않지만, 자물쇠 영역 내에서는 열쇠의 돌기 부분과 자물쇠의 홈 부분이 정확히 일치해야 하며 열쇠의 돌기와 자물쇠의 돌기가 만나서는 안됩니다. 또한 자물쇠의 모든 홈을 채워 비어있는 곳이 없어야 자물쇠를 열 수 있습니다.

열쇠를 나타내는 2차원 배열 key와 자물쇠를 나타내는 2차원 배열 lock이 매개변수로 주어질 때, 열쇠로 자물쇠를 열수 있으면 true를, 열 수 없으면 false를 return 하도록 solution 함수를 완성해주세요.

### 제한사항

* key는 M x M(3 ≤ M ≤ 20, M은 자연수)크기 2차원 배열입니다.
* lock은 N x N(3 ≤ N ≤ 20, N은 자연수)크기 2차원 배열입니다.
* M은 항상 N 이하입니다.
* key와 lock의 원소는 0 또는 1로 이루어져 있습니다.
  * 0은 홈 부분, 1은 돌기 부분을 나타냅니다.

------

### 입출력 예

| key                               | lock                              | result |
| --------------------------------- | --------------------------------- | ------ |
| [[0, 0, 0], [1, 0, 0], [0, 1, 1]] | [[1, 1, 1], [1, 1, 0], [1, 0, 1]] | true   |

### 입출력 예에 대한 설명

![자물쇠.jpg](https://grepp-programmers.s3.amazonaws.com/files/production/469703690b/79f2f473-5d13-47b9-96e0-a10e17b7d49a.jpg)

key를 시계 방향으로 90도 회전하고, 오른쪽으로 한 칸, 아래로 한 칸 이동하면 lock의 홈 부분을 정확히 모두 채울 수 있습니다.

------

혼자 풀기가 막막하다면, 풀이 강의를 들어보세요 [(클릭)](https://programmers.co.kr/learn/courses/10336?utm_source=programmers&utm_medium=test_course10336&utm_campaign=course_10336)

<br>

## 문제 풀이

### \# 완전탐색 \# 해쉬맵

<br>

### 풀이 과정

* **우선 첫번째로, 입력의 크기가 매우 작습니다.**
  * 이로부터 완전 탐색을 고려해볼 수 있습니다. 
* **다음으로, 배열의 값들 중 일부 값들만 조건에 만족하는 지 확인하면 됩니다.**
  * 이로부터 해쉬맵을 사용할 수 있습니다. 
  * 그리고 _완전 탐색_ 혹은 _그 값이 있는지 검사_ 할 때 해쉬맵을 자주 사용합니다. 

* **이 문제에서 변하는 것은 키의 좌표이며, 변하는 방법은 회전과 이동 밖에 없습니다.**
  * 회전, 상하 이동, 좌우 이동의 경우를 탐색합니다. 


이번 문제는 자료구조/알고리즘틱한 문제라기 보다는 구현 실력을 테스트하는 문제 같습니다. 

문제를 보고 막막할 수 있지만 보통 이렇게 입력이 작고 상황이 명확한 경우에서는 조금만 고민하면 탐색의 범위를 찾을 수 있습니다. 

결국, **회전 시 좌표의 변환**과 **이동 시 좌표의 범위**를 구하는 것이 중요한 문제입니다. 

<br>

### 전체 코드

전체 코드입니다. 

```python
def solution(key, lock):
    M, N = len(key), len(lock)
    # 자물쇠: 홈, 열쇠: 돌기 부분만 딕셔너리에 저장
    key_dict = {(i,j):1 for i in range(M) for j in range(len(key[i])) if key[i][j] == 1}
    lock_dict = {(i,j):1 for i in range(N) for j in range(len(lock[i])) if lock[i][j] == 0}
    # 회전, 상하이동, 좌우이동 완전 탐색
    rotated_key = key_dict.keys()
    for rotate in range(4):
        rotated_key = [[j,M-1-i] for (i,j) in rotated_key]
        for ver_move in range(-M+1,N):
            ver_moved_key = [[i+ver_move,j] for [i,j] in rotated_key]
            for hor_move in range(-M+1,N):
                hor_moved_key = [[i,j+hor_move] for [i,j] in ver_moved_key]
                complete = 0
                for key in hor_moved_key:
                    if key[0]<0 or key[0]>=N or key[1]<0 or key[1]>=N: continue
                    if tuple(key) in lock_dict: complete += 1
                    else: break
                if complete == len(lock_dict): return True
    return False
```

<br>

### 정리

이런 문제는 겁 먹지 않고 문제를 푸는 데 필요한 것들을 차근히 하나씩 구해내는 것이 중요한 것 같습니다. 자신감을 갖기!

덧붙이자면... 위 코드의 다섯 번째 줄에서 코드를 처음에 `lock_dict = {(i,j):1 for i in range(M) for j in range(len(lock[i])) if lock[i][j] == 1}`이라고 쓰는 바람에 이걸 찾는데 30분이 넘게 걸렸답니다...하하😂 (애꿎은 탐색 알고리즘 부분만 계속 들여다봤네요)

<span style="color:red">**문제를 풀다보면 주어진 조건이나 사소한 변수/값 등을 종종 놓치는 경우가 있는데, 그러지 않도록 집중합시다!!!**</span>







