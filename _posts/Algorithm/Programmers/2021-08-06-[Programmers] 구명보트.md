---
layout: single
title: "[Programmers] 구명보트"
categories: ['Algorithm', 'Queue', 'TwoPointer', 'Greedy']
---

# 구명보트

##### 문제 설명

무인도에 갇힌 사람들을 구명보트를 이용하여 구출하려고 합니다. 구명보트는 작아서 한 번에 최대 **2명**씩 밖에 탈 수 없고, 무게 제한도 있습니다.

예를 들어, 사람들의 몸무게가 [70kg, 50kg, 80kg, 50kg]이고 구명보트의 무게 제한이 100kg이라면 2번째 사람과 4번째 사람은 같이 탈 수 있지만 1번째 사람과 3번째 사람의 무게의 합은 150kg이므로 구명보트의 무게 제한을 초과하여 같이 탈 수 없습니다.

구명보트를 최대한 적게 사용하여 모든 사람을 구출하려고 합니다.

사람들의 몸무게를 담은 배열 people과 구명보트의 무게 제한 limit가 매개변수로 주어질 때, 모든 사람을 구출하기 위해 필요한 구명보트 개수의 최솟값을 return 하도록 solution 함수를 작성해주세요.

##### 제한사항

* 무인도에 갇힌 사람은 1명 이상 50,000명 이하입니다.
* 각 사람의 몸무게는 40kg 이상 240kg 이하입니다.
* 구명보트의 무게 제한은 40kg 이상 240kg 이하입니다.
* 구명보트의 무게 제한은 항상 사람들의 몸무게 중 최댓값보다 크게 주어지므로 사람들을 구출할 수 없는 경우는 없습니다.

##### 입출력 예

| people           | limit | return |
| ---------------- | ----- | ------ |
| [70, 50, 80, 50] | 100   | 3      |
| [70, 80, 50]     | 100   | 3      |

<br>



### 문제 풀이

---

먼저 문제를 보고 바로 떠오른 풀이. 아마 대부분의 사람들이 이렇게 생각하지 않았나 싶다. 

그런데 문제는, 사람들을 빼내는 과정에서 pop(0)을 쓰거나 인덱스 슬라이싱 등을 사용하면 효율성 검사에서 시간초과가 난다는 것이다. 

pop을 여러번 반복해야 할 때 가장 효율적인 것은 deque 클래스를 사용하는 것이니, 정렬된 people 리스트를 dq로 만들어 알고리즘을 진행한다. 

```python
def solution(people, limit):
    from collections import deque
    dq = deque(sorted(people))
    ans = 0
    while dq:
        if len(dq) > 1 and dq[-1] + dq[0] <= limit: dq.popleft()
        dq.pop()
        ans += 1
    return ans
```

<br>

그런데...정말 **pop을 해야 할까?**

사람들을 제외해나가야 하기 때문에 맨 처음에 pop이 떠오른건 자연스럽지만, pop을 하지 않을 수 있다면 시간을 줄일 수 있을 것이다. 

그래서 **투 포인터** 개념을 활용한다. 

앞뒤로 pop을 하는 것 대신, 포인터만 좁혀주어 사람을 제외하는 효과를 보는 것이다.  

```python
def solution(people, limit) :
    answer = 0
    people.sort()

    a = 0
    b = len(people) - 1
    while a < b :
        if people[b] + people[a] <= limit :
            a += 1
            answer += 1
        b -= 1
    return len(people) - answer
```

알고리즘을 구상하고, 그것을 구현할 때 <span style="color:rgb(255,10,10)">**'정말 이 메서드를 써야만 하는가? 정말 이 리스트를 따로 만들어야만 하는가?'**</span> 등의 생각을 한 번 더 해봐야 할 것이다. 

그래야 더 효율적이고 간결한 코드가 나온다. 반성하자!

<br>

<br>

<br>

