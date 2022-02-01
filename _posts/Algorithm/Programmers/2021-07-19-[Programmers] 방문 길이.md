---
layout: single
title: "[Programmers] 쿼드 압축 후 개수 세기"
categories: ['Algorithm', 'Programmers']
---

# 쿼드 압축 후 개수 세기

### 문제 설명

---

- 게임 캐릭터를 4가지 명령어를 통해 움직이려 합니다. 명령어는 다음과 같습니다.

  - U: 위쪽으로 한 칸 가기
  - D: 아래쪽으로 한 칸 가기
  - R: 오른쪽으로 한 칸 가기
  - L: 왼쪽으로 한 칸 가기

  캐릭터는 좌표평면의 (0, 0) 위치에서 시작합니다. 좌표평면의 경계는 왼쪽 위(-5, 5), 왼쪽 아래(-5, -5), 오른쪽 위(5, 5), 오른쪽 아래(5, -5)로 이루어져 있습니다.

  ![방문길이1_qpp9l3.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/ace0e7bc-9092-4b95-9bfb-3a55a2aa780e/%E1%84%87%E1%85%A1%E1%86%BC%E1%84%86%E1%85%AE%E1%86%AB%E1%84%80%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B51_qpp9l3.png)

  예를 들어, "ULURRDLLU"로 명령했다면

  ![방문길이2_lezmdo.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/668c7458-e184-472d-9d32-f5d2acca759a/%E1%84%87%E1%85%A1%E1%86%BC%E1%84%86%E1%85%AE%E1%86%AB%E1%84%80%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B52_lezmdo.png)

  - 1번 명령어부터 7번 명령어까지 다음과 같이 움직입니다.

  ![방문길이3_sootjd.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/08558e36-d667-4160-bfec-b754c78a7d85/%E1%84%87%E1%85%A1%E1%86%BC%E1%84%86%E1%85%AE%E1%86%AB%E1%84%80%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B53_sootjd.png)

  - 8번 명령어부터 9번 명령어까지 다음과 같이 움직입니다.

  ![방문길이4_hlpiej.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/a52af28e-5835-438b-9f40-5467ebf9bf03/%E1%84%87%E1%85%A1%E1%86%BC%E1%84%86%E1%85%AE%E1%86%AB%E1%84%80%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B54_hlpiej.png)

  이때, 우리는 게임 캐릭터가 지나간 길 중 **캐릭터가 처음 걸어본 길의 길이**를 구하려고 합니다. 예를 들어 위의 예시에서 게임 캐릭터가 움직인 길이는 9이지만, 캐릭터가 처음 걸어본 길의 길이는 7이 됩니다. (8, 9번 명령어에서 움직인 길은 2, 3번 명령어에서 이미 거쳐 간 길입니다)

  단, 좌표평면의 경계를 넘어가는 명령어는 무시합니다.

  예를 들어, "LULLLLLLU"로 명령했다면

  ![방문길이5_nitjwj.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/f631f005-f8de-4392-a76c-a9ef64b6de08/%E1%84%87%E1%85%A1%E1%86%BC%E1%84%86%E1%85%AE%E1%86%AB%E1%84%80%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B55_nitjwj.png)

  - 1번 명령어부터 6번 명령어대로 움직인 후, 7, 8번 명령어는 무시합니다. 다시 9번 명령어대로 움직입니다.

  ![방문길이6_nzhumd.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/35e62f0a-43c6-4142-bec6-6d28fbc57216/%E1%84%87%E1%85%A1%E1%86%BC%E1%84%86%E1%85%AE%E1%86%AB%E1%84%80%E1%85%B5%E1%86%AF%E1%84%8B%E1%85%B56_nzhumd.png)

  이때 캐릭터가 처음 걸어본 길의 길이는 7이 됩니다.

  명령어가 매개변수 dirs로 주어질 때, 게임 캐릭터가 처음 걸어본 길의 길이를 구하여 return 하는 solution 함수를 완성해 주세요.

  ##### 제한사항

  - dirs는 string형으로 주어지며, 'U', 'D', 'R', 'L' 이외에 문자는 주어지지 않습니다.
  - dirs의 길이는 500 이하의 자연수입니다.

  ##### 입출력 예

  | dirs        | answer |
  | ----------- | ------ |
  | "ULURRDLLU" | 7      |
  | "LULLLLLLU" | 7      |

  ##### 입출력 예 설명

  입출력 예 #1
  문제의 예시와 같습니다.

  입출력 예 #2
  문제의 예시와 같습니다.



### 문제 풀이

---

이 문제에서는 두 가지를 고려해주면 됩니다. 

1. 움직일 수 없는 경우에는 위치를 그대로 유지
2. 움직일 수 있어도 이미 방문한 길일 경우 방문 길이에 포함 X



(내 풀이)

아래 코드에서는 조건1을 조건문으로, 조건2를 set 자료구조로 풀이하였다. 

무방향이기 때문에 경로를 추가할 경우 이동 방향과 상관없이 하 -> 상, 좌 -> 우 로 추가해준다. 

```python
def solution(dirs):
    paths = set() # 중복 제거
    x,y = 0,0
    for dir in dirs:
        if dir == 'U' and y < 5:
            paths.add(((x, y), (x, y+1)))
            y += 1
            
        elif dir  == 'D' and y > -5:
            paths.add(((x, y-1), (x, y)))
            y -= 1
            
        elif dir  == 'R' and x < 5:
            paths.add(((x, y), (x+1, y)))
            x += 1
            
        elif dir  == 'L' and x > -5:
            paths.add(((x-1, y), (x, y)))
            x -= 1
    return len(paths)
```



(다른 풀이)

아래 코드는 해당 프로그래머스 문제에서 '다른 사람의 풀이'를 누르면 있는 코드들 중 '최성우'님의 코드이다. 

조건을 풀이한 방식은 같지만, 딕셔너리로 매핑을 한 뒤 새로운 위치를 먼저 구하고, 그 새로운 위치를 조건문의 조건으로 비교하니 코드가 한결 간결해졌다. 

해당 코드에서는 무방향인 것을 두 가지 이동 겅로를 모두 추가하여, return 시에 길이를 2로 나누도록 하였다. 

```python
def solution(dirs):
    s = set()
    d = {'U': (0,1), 'D': (0, -1), 'R': (1, 0), 'L': (-1, 0)}
    x, y = 0, 0
    for i in dirs:
        nx, ny = x + d[i][0], y + d[i][1]
        if -5 <= nx <= 5 and -5 <= ny <= 5:
            s.add((x,y,nx,ny))
            s.add((nx,ny,x,y))
            x, y = nx, ny
    return len(s)//2
```



* 반복을 없애야 하는 경우에는 set을 이용하자. 
* 문제에 '위치'와 같은 제한 조건이 있을 경우 새로운 위치를 먼저 구하고 조건문으로 비교하면 코드가 한결 간결해질 수 있다. 또한 위치를 이동시킬 경우 딕셔너리를 이용한 매핑을 활용하자. 



그럼 안녕!
