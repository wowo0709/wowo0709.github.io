# 조이스틱

##### 문제 설명

조이스틱으로 알파벳 이름을 완성하세요. 맨 처음엔 A로만 이루어져 있습니다.
ex) 완성해야 하는 이름이 세 글자면 AAA, 네 글자면 AAAA

조이스틱을 각 방향으로 움직이면 아래와 같습니다.

```
▲ - 다음 알파벳
▼ - 이전 알파벳 (A에서 아래쪽으로 이동하면 Z로)
◀ - 커서를 왼쪽으로 이동 (첫 번째 위치에서 왼쪽으로 이동하면 마지막 문자에 커서)
▶ - 커서를 오른쪽으로 이동
```

예를 들어 아래의 방법으로 "JAZ"를 만들 수 있습니다.

```
- 첫 번째 위치에서 조이스틱을 위로 9번 조작하여 J를 완성합니다.
- 조이스틱을 왼쪽으로 1번 조작하여 커서를 마지막 문자 위치로 이동시킵니다.
- 마지막 위치에서 조이스틱을 아래로 1번 조작하여 Z를 완성합니다.
따라서 11번 이동시켜 "JAZ"를 만들 수 있고, 이때가 최소 이동입니다.
```

만들고자 하는 이름 name이 매개변수로 주어질 때, 이름에 대해 조이스틱 조작 횟수의 최솟값을 return 하도록 solution 함수를 만드세요.

##### 제한 사항

* name은 알파벳 대문자로만 이루어져 있습니다.
* name의 길이는 1 이상 20 이하입니다.

##### 입출력 예

| name     | return |
| -------- | ------ |
| "JEROEN" | 56     |
| "JAN"    | 23     |

[출처](https://commissies.ch.tudelft.nl/chipcie/archief/2010/nwerc/nwerc2010.pdf)

※ 공지 - 2019년 2월 28일 테스트케이스가 추가되었습니다.

<br>



### 문제 풀이

---

조금 애를 먹은 문제입니다. 

조이스틱의 상하 이동 로직은 어렵지 않지만, 좌우 이동 로직을 짜면서 정말 이게 최선인지, 문제 의도가 이게 맞는 지 등과 같은 의구심이 계속 들더라구요...ㅎㅎ

어쨌든 최종적으로 완성한 코드는 아래와 같습니다. 

<br>

저는 파라미터로 주어진 name을 'A...A' 인 문자열로 바꾸도록 코드를 구성했습니다. 

* move_vertical(문자1, 문자2): 상하 이동 로직

    조이스틱이 상하로 이동할 때는 name의 글자가 'A'가 아닐 때입니다. 두 글자 사이의 최소 거리를 리턴합니다. 

* move_horizontal(문자열 길이, 문자열, 현재 위치): 좌우 이동 로직

    조이스틱이 좌우로 이동할 때는 name의 다른 글자 중 'A'가 아닌 글자가 있을 때 입니다. 현재 위치에서 가까운 위치부터 양쪽으로  'A'가 아닌 글자를 탐색합니다. 이는 함수 호출 시 마다 가장 가까운 다음 위치로 이동하는 **탐욕 알고리즘**이며, 발견하지 못했다면 그 의미로 0과 현재 위치 그대로를 리턴합니다. 

* solution(문자열)

    상하 이동 함수와 좌우 이동 함수를 호출하며 총 이동 횟수만을 계산합니다. 좌우 이동 함수가 0을 리턴하면 문자열 변환이 완료되었다는 뜻이므로 결과를 리턴합니다. 

```python
def move_vertical(s1,s2):
    dist = abs(ord(s1) - ord(s2) )
    return dist if dist <= 13 else 26-dist

def move_horizontal(N, s, cur):
    for dx in range(1,N//2+1):
        next = cur + dx if cur + dx < N else cur + dx - N
        if s[next] != 'A': return dx, next
        next = cur - dx if cur - dx >= 0 else N + cur - dx
        if s[next] != 'A': return dx, next
    return 0,cur

def solution(name):
    answer = 0
    N, cur = len(name), 0
    while True:
        # 상하이동
        answer += move_vertical(name[cur], 'A')   
        name = name[:cur] + 'A' + name[cur+1:]
        # 좌우이동
        moves_x,cur = move_horizontal(N, name, cur)
        answer += moves_x
        # 변환 완료
        if moves_x == 0: return answer
```

