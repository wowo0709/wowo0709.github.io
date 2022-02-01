---
layout: single
title: "[Programmers] [1차] 프렌즈4블록"
categories: ['Algorithm', 'Programmers']
---

# [1차] 프렌즈4블록

### 문제 설명

---

## 프렌즈4블록

블라인드 공채를 통과한 신입 사원 라이언은 신규 게임 개발 업무를 맡게 되었다. 이번에 출시할 게임 제목은 "프렌즈4블록".
같은 모양의 카카오프렌즈 블록이 2×2 형태로 4개가 붙어있을 경우 사라지면서 점수를 얻는 게임이다.

![board map](http://t1.kakaocdn.net/welcome2018/pang1.png)
만약 판이 위와 같이 주어질 경우, 라이언이 2×2로 배치된 7개 블록과 콘이 2×2로 배치된 4개 블록이 지워진다. 같은 블록은 여러 2×2에 포함될 수 있으며, 지워지는 조건에 만족하는 2×2 모양이 여러 개 있다면 한꺼번에 지워진다.

![board map](http://t1.kakaocdn.net/welcome2018/pang2.png)

블록이 지워진 후에 위에 있는 블록이 아래로 떨어져 빈 공간을 채우게 된다.

![board map](http://t1.kakaocdn.net/welcome2018/pang3.png)

만약 빈 공간을 채운 후에 다시 2×2 형태로 같은 모양의 블록이 모이면 다시 지워지고 떨어지고를 반복하게 된다.
![board map](http://t1.kakaocdn.net/welcome2018/pang4.png)

위 초기 배치를 문자로 표시하면 아래와 같다.

```
TTTANT
RRFACC
RRRFCC
TRRRAA
TTMMMF
TMMTTJ
```

각 문자는 라이언(R), 무지(M), 어피치(A), 프로도(F), 네오(N), 튜브(T), 제이지(J), 콘(C)을 의미한다

입력으로 블록의 첫 배치가 주어졌을 때, 지워지는 블록은 모두 몇 개인지 판단하는 프로그램을 제작하라.

### 입력 형식

- 입력으로 판의 높이 `m`, 폭 `n`과 판의 배치 정보 `board`가 들어온다.
- 2 ≦ `n`, `m` ≦ 30
- `board`는 길이 `n`인 문자열 `m`개의 배열로 주어진다. 블록을 나타내는 문자는 대문자 A에서 Z가 사용된다.

### 출력 형식

입력으로 주어진 판 정보를 가지고 몇 개의 블록이 지워질지 출력하라.

### 입출력 예제

| m    | n    | board                                                        | answer |
| ---- | ---- | ------------------------------------------------------------ | ------ |
| 4    | 5    | ["CCBDE", "AAADE", "AAABF", "CCBBF"]                         | 14     |
| 6    | 6    | ["TTTANT", "RRFACC", "RRRFCC", "TRRRAA", "TTMMMF", "TMMTTJ"] | 15     |

### 예제에 대한 설명

- 입출력 예제 1의 경우, 첫 번째에는 A 블록 6개가 지워지고, 두 번째에는 B 블록 4개와 C 블록 4개가 지워져, 모두 14개의 블록이 지워진다.
- 입출력 예제 2는 본문 설명에 있는 그림을 옮긴 것이다. 11개와 4개의 블록이 차례로 지워지며, 모두 15개의 블록이 지워진다.

[해설 보러가기](http://tech.kakao.com/2017/09/27/kakao-blind-recruitment-round-1/)



### 문제 풀이

---

문제 풀이에 앞서, [가장 큰 정사각형 찾기](https://programmers.co.kr/learn/courses/30/lessons/12905) 문제를 참고하고 오면 좋을 것 같다. 



이 문제에서는 크게 2가지를 고려해야 한다. 

	1. 2x2 크기의 정사각형을 어떻게 찾고, 그 위치를 어떻게 저장할 것인가?
 	2. 저장된 위치의 블록들을 어떻게 없앨 것인가?



주어진 게임에서 2x2의 정사각 블록이 사라지면, 위에 있던 블록들은 아래로 내려온다. 그런데 배열에서는 원소가 사라지면 나머지 원소들은 왼쪽으로 당겨지기 때문에 이를 고려하여 먼저 board를 적절히 바꿔준다. 

newboard를 초기화 하는데, 이는 매개변수로 주어진 board를 오른쪽으로 90도 회전시킨 모양이다. 이것으로 블록이 사라지면 위에 있던 블록들이 아래로 내려오는 환경을 만들 수 있게 된다. 

```python
newboard = [[board[j][i] for j in range(m)][::-1] for i in range(n)]
```



다음으로 2x2 정사각형 블록을 찾고, 저장하는 단계이다. 위의 '가장 큰 정사가형 찾기' 문제를 참고하고 오라는 것은 도형을 찾는 메커니즘이 같기 때문이다. 사라질 블록들의 위치는 popList라는 이름의 set 자료구조를 이용해, 겹치는 경우에도 중복되어 저장되지 않고 하나만 저장되도록 하였다. 

도형을 찾는 과정에서는 try-except의 예외 처리문을 사용하였는데, 이는 블록들이 사라짐에 따라 해당 인덱스를 가지는 원소가 없을 수도 있기 때문이다. 

```python
while True: 
    popList = set()
    for i in range(1,n):
        for j in range(1,m):
            try:
                if newboard[i][j] == newboard[i-1][j] == newboard[i][j-1] == newboard[i-1][j-1]:
                    popList = popList.union({(i,j),(i-1,j),(i,j-1),(i-1,j-1)})
            except:
                continue
```



마지막으로 저장된 인덱스의 블록들을 삭제하거나 더 이상 없다면 리턴하는 부분이다. 

삭제하기에 앞서서 popList를 리스트로 바꾸고 정렬을 수행한다. 정렬을 수행하는 이유는 원소를 삭제할 때 뒷쪽의 원소를 먼저 삭제하기 위함이다. 

4x4 크기의 이차원 배열의 0번째 일차원 배열에서, 0번째와 1번째 원소를 삭제해야 한다고 생각해보자. 

앞의 원소부터 삭제 시, 0번째 원소를 삭제하고 나면 1번째 원소는 0번째 원소가 된다. 이후에 1번째 원소를 삭제하게 된다면 원래 삭제하려 했던 1번째 원소(현재 0번째 원소)가 아닌 원래의 2번째 원소(현재 1번째 원소)를 삭제하게 되는 것이다. 따라서 정렬을 수행하여 뒷쪽의 원소부터 삭제해야 한다.   

만약 더 이상 삭제할 블록들이 없다면 리턴을 하는데, 최초 board의 크기에서 현재 newboard의 크기를 빼서 구해준다. 

```python
popList = sorted(list(popList),key=lambda x:x[1],reverse=True)
if popList:
    for i,j in popList: del newboard[i][j]
else:
    return m*n - sum(list(map(len,newboard)))
```





아래의 코드가 전체 코드이다. 

```python
def solution(m, n, board):
    newboard = [[board[j][i] for j in range(m)][::-1] for i in range(n)]
    while True: 
        popList = set()
        for i in range(1,n):
            for j in range(1,m):
                try:
                    if newboard[i][j] == newboard[i-1][j] == newboard[i][j-1] == newboard[i-1][j-1]:
                        popList = popList.union({(i,j),(i-1,j),(i,j-1),(i-1,j-1)})
                except:
                    continue
        
        popList = sorted(list(popList),key=lambda x:x[1],reverse=True)
        if popList:
            for i,j in popList: del newboard[i][j]
        else:
            return m*n - sum(list(map(len,newboard)))
```



그럼 안녕!
