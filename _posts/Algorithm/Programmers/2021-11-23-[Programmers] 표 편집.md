---
layout: single
title: "[Programmers] [1차] 셔틀버스"
categories: ['Algorithm', 'Bruteforce', 'Implementation']
toc: true
toc_sticky: true
tag: ['DoublyLinkedList']
---



<br>

## 문제 설명

### 문제 설명

**[본 문제는 정확성과 효율성 테스트 각각 점수가 있는 문제입니다.]**

업무용 소프트웨어를 개발하는 니니즈웍스의 인턴인 앙몬드는 명령어 기반으로 표의 행을 선택, 삭제, 복구하는 프로그램을 작성하는 과제를 맡았습니다. 세부 요구 사항은 다음과 같습니다

![table_1.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/d8e89054-53ba-4222-a485-dc56893f45e4/table_1.png)

위 그림에서 파란색으로 칠해진 칸은 현재 **선택된 행**을 나타냅니다. 단, 한 번에 한 행만 선택할 수 있으며, 표의 범위(0행 ~ 마지막 행)를 벗어날 수 없습니다. 이때, 다음과 같은 명령어를 이용하여 표를 편집합니다.

* `"U X"`: 현재 선택된 행에서 X칸 위에 있는 행을 선택합니다.
* `"D X"`: 현재 선택된 행에서 X칸 아래에 있는 행을 선택합니다.
* `"C"` : 현재 선택된 행을 삭제한 후, 바로 아래 행을 선택합니다. 단, 삭제된 행이 가장 마지막 행인 경우 바로 윗 행을 선택합니다.
* `"Z"` : 가장 최근에 삭제된 행을 원래대로 복구합니다. **단, 현재 선택된 행은 바뀌지 않습니다.**

예를 들어 위 표에서 `"D 2"`를 수행할 경우 아래 그림의 왼쪽처럼 4행이 선택되며, `"C"`를 수행하면 선택된 행을 삭제하고, 바로 아래 행이었던 "네오"가 적힌 행을 선택합니다(4행이 삭제되면서 아래 있던 행들이 하나씩 밀려 올라오고, 수정된 표에서 다시 4행을 선택하는 것과 동일합니다).

![table_2.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/453bbb71-df69-4be2-a223-67361878202c/table_2.png)

다음으로 `"U 3"`을 수행한 다음 `"C"`를 수행한 후의 표 상태는 아래 그림과 같습니다.

![table_3.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/61261fa2-701d-4db5-9aa2-a56dd85a3dbf/table_3.png)

다음으로 `"D 4"`를 수행한 다음 `"C"`를 수행한 후의 표 상태는 아래 그림과 같습니다. 5행이 표의 마지막 행 이므로, 이 경우 바로 윗 행을 선택하는 점에 주의합니다.

![table_4.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/b1a63278-be97-4e3a-a653-5a6aa0f477ba/table_4.png)

다음으로 `"U 2"`를 수행하면 현재 선택된 행은 2행이 됩니다.

![table_5.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/b1189eff-e4ee-4119-bb55-a1f06e388c29/table_5.png)

위 상태에서 `"Z"`를 수행할 경우 가장 최근에 제거된 `"라이언"`이 적힌 행이 원래대로 복구됩니다.

![table_6.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/0a386d19-0391-46a7-8086-9f36db31940d/table_6.png)

다시한번 `"Z"`를 수행하면 그 다음으로 최근에 제거된 `"콘"`이 적힌 행이 원래대로 복구됩니다. 이때, 현재 선택된 행은 바뀌지 않는 점에 주의하세요.
![table_7.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/8900360f-bf0b-449b-a508-98918a14ef1d/table_7.png)

이때, 최종 표의 상태와 처음 주어진 표의 상태를 비교하여 삭제되지 않은 행은 `"O"`, 삭제된 행은 `"X"`로 표시하면 다음과 같습니다.

![table_8.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/87a31aeb-50fb-4c0d-9f6b-8427632b582e/table_8.png)

처음 표의 행 개수를 나타내는 정수 n, 처음에 선택된 행의 위치를 나타내는 정수 k, 수행한 명령어들이 담긴 문자열 배열 cmd가 매개변수로 주어질 때, 모든 명령어를 수행한 후 표의 상태와 처음 주어진 표의 상태를 비교하여 삭제되지 않은 행은 O, 삭제된 행은 X로 표시하여 문자열 형태로 return 하도록 solution 함수를 완성해주세요.

### 제한사항

* 5 ≤ `n` ≤ 1,000,000

* 0 ≤ `k` < `n`

* 1 ≤

   

  ```
  cmd
  ```

  의 원소 개수 ≤ 200,000

  * `cmd`의 각 원소는 `"U X"`, `"D X"`, `"C"`, `"Z"` 중 하나입니다.
  * X는 1 이상 300,000 이하인 자연수이며 0으로 시작하지 않습니다.
  * X가 나타내는 자연수에 ',' 는 주어지지 않습니다. 예를 들어 123,456의 경우 123456으로 주어집니다.
  * `cmd`에 등장하는 모든 X들의 값을 합친 결과가 1,000,000 이하인 경우만 입력으로 주어집니다.
  * 표의 모든 행을 제거하여, 행이 하나도 남지 않는 경우는 입력으로 주어지지 않습니다.
  * 본문에서 각 행이 제거되고 복구되는 과정을 보다 자연스럽게 보이기 위해 `"이름"` 열을 사용하였으나, `"이름"`열의 내용이 실제 문제를 푸는 과정에 필요하지는 않습니다. `"이름"`열에는 서로 다른 이름들이 중복없이 채워져 있다고 가정하고 문제를 해결해 주세요.

* 표의 범위를 벗어나는 이동은 입력으로 주어지지 않습니다.

* 원래대로 복구할 행이 없을 때(즉, 삭제된 행이 없을 때) "Z"가 명령어로 주어지는 경우는 없습니다.

* 정답은 표의 0행부터 n - 1행까지에 해당되는 O, X를 순서대로 이어붙인 문자열 형태로 return 해주세요.

#### 정확성 테스트 케이스 제한 사항

* 5 ≤ `n` ≤ 1,000
* 1 ≤ `cmd`의 원소 개수 ≤ 1,000

#### 효율성 테스트 케이스 제한 사항

* 주어진 조건 외 추가 제한사항 없습니다.

### 입출력 예

| n    | k    | cmd                                                       | result       |
| ---- | ---- | --------------------------------------------------------- | ------------ |
| 8    | 2    | `["D 2","C","U 3","C","D 4","C","U 2","Z","Z"]`           | `"OOOOXOOO"` |
| 8    | 2    | `["D 2","C","U 3","C","D 4","C","U 2","Z","Z","U 1","C"]` | `"OOXOXOOO"` |

#### 입출력 예 설명

**입출력 예 #1**

문제의 예시와 같습니다.

**입출력 예 #2**

다음은 9번째 명령어까지 수행한 후의 표 상태이며, 이는 입출력 예 #1과 같습니다.

![table_7.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/8900360f-bf0b-449b-a508-98918a14ef1d/table_7.png)

10번째 명령어 `"U 1"`을 수행하면 `"어피치"`가 적힌 2행이 선택되며, 마지막 명령어 `"C"`를 수행하면 선택된 행을 삭제하고, 바로 아래 행이었던 `"제이지"`가 적힌 행을 선택합니다.

![table_9.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/c9798574-4aa9-4029-901f-21f83fe43164/table_9.png)

따라서 처음 주어진 표의 상태와 최종 표의 상태를 비교하면 다음과 같습니다.

![table_10.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/e7ba17b2-9461-4e92-8356-81cc90adb2ec/table_10.png)

### 제한시간 안내

* 정확성 테스트 : 10초
* 효율성 테스트 : 언어별로 작성된 정답 코드의 실행 시간의 적정 배수

<br>

## 문제 풀이

### \# 이중 연결 리스트

<br>

### 풀이 과정

이야...이중 링크드 리스트 자료구조 문제라니...

풀다가 효율성 검사 6~10번이 도저히 통과가 안돼서(시간 초과) 검색해서 다른 분들 코드를 참고했는데, 안 했으면 푸는데 얼마나 걸렸을지 짐작도 잘 안가네요...ㅎㅎ

<br>

저는 알고리즘 문제에서 `이중 연결 리스트`를 사용해보는게 처음이라, 자료구조를 떠올리는데 애를 먹었지만 문제를 곱씹어보면 분명 힌트는 있었던 것 같아요. 

이 문제에서는 **어떻게 탐색할 것이냐**가 핵심입니다. 

좀 더 구체적으로 말하면, 행이 삭제/복구되면서 명령어에 따른 현재 행의 위치(앞으로는 이를 `cur`라고 함)는 계속해서 바뀝니다. 위로 이동하든, 아래로 이동하든, 행을 삭제하든, 복구하든 중간에 행이 없으면 다음 위치의 `cur`의 값이 무엇일지 행이 없는 만큼 계산해주어야 합니다. 그래야 올바른 위치를 찾을 수 있죠. 

<br>

하지만, `cur`의 다음 위치를 **계산**하는 순간 이 문제를 제 시간 안에 풀지 못합니다. 

즉, 이 문제는 이 <span style="color:red">**`cur`의 다음 위치를 탐색하는 시간 복잡도를 O(1)으로 만드는 것**</span>이 핵심입니다. 

**파이썬**에서 O(1)의 탐색 시간이라 하면 **딕셔너리 자료구조**가 떠오르고, 딕셔너리를 이용해 **이중 연결 리스트**를 구현할 수 있습니다. 

<br>

이제 풀이 과정에 대해 간단히 서술하자면, 다음과 같습니다. 

* rows라는 딕셔너리(이중 연결 리스트)를 생성합니다. 
  * rows[row] = [prev, next]
  * prev와 next는 현재 row행의 이전/다음 행으로, 행이 삭제되고 복구되면서 업데이트 됩니다. 
* 위로 이동할 때는 prev를, 아래로 이동할 때는 next를 참조합니다. 
* 행을 삭제, 복구할 때는 삭제/복구되는 행이 첫번째행/마지막행/나머지(중간)행 인지에 따라 prev와 next를 적절히 업데이트합니다. 

이렇게 **이중 연결 리스트**를 사용함으로써 `cur`의 다음 위치 탐색의 시간 복잡도를 O(1)으로 줄일 수 있고,  문제를 제시간에 풀 수 있습니다.  

<br>

### 전체 코드

전체 코드입니다. 

* **1차 풀이 (시간초과)**

```python
def solution(n, k, cmds):
    # 다음 커서 위치를 계산하는 과정에서 오랜 시간이 걸림('X' 상태인 행도 모두 탐색)
    def move_cur(n, cur, move):
        for i in range(n):
            while True:
                cur = cur-1 if move=='up' else cur+1
                if rows[cur]: break
        return cur

    cur, rows = k, {i:1 for i in range(n)}
    trash = []
    for cmd in cmds:
        if cmd[0] == 'U':
            cur = move_cur(int(cmd.split()[1]), cur, 'up')
        elif cmd[0] == 'D':
            cur = move_cur(int(cmd.split()[1]), cur, 'down')
        elif cmd[0] == 'C':
            rows[cur] = 0
            trash.append(cur)
            if cur == len(rows) - 1:
                cur = move_cur(1, cur, 'up')
            else:
                cur = move_cur(1, cur, 'down')
        elif cmd[0] == 'Z':
            rows[trash.pop()] = 1
    
    ans = ['X']*n
    for row in rows:
        if rows[row]: ans[row] = 'O'
    return ''.join(ans)
```

<br>

* **2차 풀이 (성공)**

```python
def solution(n, k, cmds):
    # 이중 연결 리스트: rows[row]: [prev, next]
    cur, rows = k, {i:[i-1,i+1] for i in range(n)}
    trash = []
    ans = ['O']*n
    for cmd in cmds:
        if cmd[0] == 'U':
            for i in range(int(cmd.split()[1])):
                cur = rows[cur][0]
        elif cmd[0] == 'D':
            for i in range(int(cmd.split()[1])):
                cur = rows[cur][1]
        elif cmd[0] == 'C':
            trash.append([cur,rows[cur]])
            ans[cur] = 'X'
            if rows[cur][0] < 0: # 첫 행
                rows[rows[cur][1]][0] = rows[cur][0]
                cur = rows[cur][1]
            elif rows[cur][1] > n-1: # 마지막 행
                rows[rows[cur][0]][1] = rows[cur][1]
                cur = rows[cur][0]
            else: 
                rows[rows[cur][0]][1] = rows[cur][1]
                rows[rows[cur][1]][0] = rows[cur][0]
                cur = rows[cur][1]
        elif cmd[0] == 'Z':
            z,(z_prev,z_next) = trash.pop()
            ans[z] = 'O'
            if rows[z][0] < 0: # 첫 행
                rows[rows[z][1]][0] = z
            elif rows[z][1] > n-1: # 마지막 행
                rows[rows[z][0]][1] = z
            else: 
                rows[rows[z][0]][1] = z
                rows[rows[z][1]][0] = z

    return ''.join(ans)
```



<br>

### 정리

분명 익숙지 않은 자료구조를 사용하는 문제지만, 그렇다고 아주 어려운 문제라고는 할 수 없을 것 같습니다. 

* <span style="color:red">**다음 위치의 탐색이 필요한 문제에서 `이중 연결 리스트`를 사용할 수 있다는 것**</span>
* <span style="color:red">**문제를 풀기 위한 핵심이 무엇이고, 그것을 구하기 위해 시간을 줄이는 방법에 무엇이 있을 지 파악하는 습관**</span>

을 기르자는 교훈을 얻으면서 문제를 마우리할 수 있을 것 같습니다. 

<br>

**+ [힙으로 푸는 풀이](https://sangsangss.tistory.com/94)**
