---
layout: single
title: "[Programmers] 다리를 지나는 트럭"
categories: ['Algorithm', 'Programmers']
---

# 다리를 지나는 트럭

### 문제 설명

---

##### 문제 설명

트럭 여러 대가 강을 가로지르는 일차선 다리를 정해진 순으로 건너려 합니다. 모든 트럭이 다리를 건너려면 최소 몇 초가 걸리는지 알아내야 합니다. 다리에는 트럭이 최대 bridge_length대 올라갈 수 있으며, 다리는 weight 이하까지의 무게를 견딜 수 있습니다. 단, 다리에 완전히 오르지 않은 트럭의 무게는 무시합니다.

예를 들어, 트럭 2대가 올라갈 수 있고 무게를 10kg까지 견디는 다리가 있습니다. 무게가 [7, 4, 5, 6]kg인 트럭이 순서대로 최단 시간 안에 다리를 건너려면 다음과 같이 건너야 합니다.

| 경과 시간 | 다리를 지난 트럭 | 다리를 건너는 트럭 | 대기 트럭 |
| --------- | ---------------- | ------------------ | --------- |
| 0         | []               | []                 | [7,4,5,6] |
| 1~2       | []               | [7]                | [4,5,6]   |
| 3         | [7]              | [4]                | [5,6]     |
| 4         | [7]              | [4,5]              | [6]       |
| 5         | [7,4]            | [5]                | [6]       |
| 6~7       | [7,4,5]          | [6]                | []        |
| 8         | [7,4,5,6]        | []                 | []        |

따라서, 모든 트럭이 다리를 지나려면 최소 8초가 걸립니다.

solution 함수의 매개변수로 다리에 올라갈 수 있는 트럭 수 bridge_length, 다리가 견딜 수 있는 무게 weight, 트럭 별 무게 truck_weights가 주어집니다. 이때 모든 트럭이 다리를 건너려면 최소 몇 초가 걸리는지 return 하도록 solution 함수를 완성하세요.

##### 제한 조건

* bridge_length는 1 이상 10,000 이하입니다.
* weight는 1 이상 10,000 이하입니다.
* truck_weights의 길이는 1 이상 10,000 이하입니다.
* 모든 트럭의 무게는 1 이상 weight 이하입니다.

##### 입출력 예

| bridge_length | weight | truck_weights                   | return |
| ------------- | ------ | ------------------------------- | ------ |
| 2             | 10     | [7,4,5,6]                       | 8      |
| 100           | 100    | [10]                            | 101    |
| 100           | 100    | [10,10,10,10,10,10,10,10,10,10] | 110    |

[출처](http://icpckorea.org/2016/ONLINE/problem.pdf)

※ 공지 - 2020년 4월 06일 테스트케이스가 추가되었습니다.

<br>



### 문제 풀이

---

#### \# 큐 \# 덱

<br>

* 슬라이싱 사용 시

```python
# 슬라이싱 사용 시
def solution(bridge_length, max_weight, truck_weights):
    bridge = [0] * bridge_length
    order, bridge_weight, time = 0, 0, 0 # 차의 순서, 다리 무게, 소요 시간
    while True:
        bridge_weight -= bridge[0]
        bridge = bridge[1:] # 여기!!!
        if order < len(truck_weights) and bridge_weight + truck_weights[order] <= max_weight:
            bridge_weight += truck_weights[order]
            bridge.append(truck_weights[order])
            order += 1
        else:
            bridge.append(0)
        time += 1
        if bridge_weight == 0: return time

'''
정확성  테스트
테스트 1 〉	통과 (13.61ms, 10.1MB)
테스트 2 〉	통과 (1053.79ms, 10.2MB)
테스트 3 〉	통과 (0.05ms, 10.2MB)
테스트 4 〉	통과 (205.25ms, 10.1MB)
테스트 5 〉	통과 (5535.53ms, 10.1MB)
테스트 6 〉	통과 (988.74ms, 10.2MB)
테스트 7 〉	통과 (5.00ms, 10.1MB)
테스트 8 〉	통과 (0.20ms, 10.1MB)
테스트 9 〉	통과 (4.53ms, 10.1MB)
테스트 10 〉	통과 (0.24ms, 10.1MB)
테스트 11 〉	통과 (0.01ms, 10.1MB)
테스트 12 〉	통과 (0.24ms, 10.1MB)
테스트 13 〉	통과 (2.49ms, 10.2MB)
테스트 14 〉	통과 (0.06ms, 10.1MB)
채점 결과
정확성: 100.0
합계: 100.0 / 100.0
'''
```

<br>

* deque 모듈 사용 시

```python
# deque 사용 시
def solution(bridge_length, max_weight, truck_weights):
    from collections import deque
    bridge = deque([0] * bridge_length)
    order, bridge_weight, time = 0, 0, 0 # 차의 순서, 다리 무게, 소요 시간
    while True:
        bridge_weight -= bridge[0]
        bridge.popleft() # 여기!!!
        if order < len(truck_weights) and bridge_weight + truck_weights[order] <= max_weight:
            bridge_weight += truck_weights[order]
            bridge.append(truck_weights[order])
            order += 1
        else:
            bridge.append(0)
        time += 1
        if bridge_weight == 0: return time

'''
정확성  테스트
테스트 1 〉	통과 (0.64ms, 10.1MB)
테스트 2 〉	통과 (9.70ms, 10.3MB)
테스트 3 〉	통과 (0.02ms, 10.2MB)
테스트 4 〉	통과 (8.42ms, 10.1MB)
테스트 5 〉	통과 (79.56ms, 10.1MB)
테스트 6 〉	통과 (22.35ms, 10.2MB)
테스트 7 〉	통과 (0.52ms, 10.1MB)
테스트 8 〉	통과 (0.10ms, 10.2MB)
테스트 9 〉	통과 (2.81ms, 10.2MB)
테스트 10 〉	통과 (0.12ms, 10.1MB)
테스트 11 〉	통과 (0.01ms, 10.1MB)
테스트 12 〉	통과 (0.19ms, 10.2MB)
테스트 13 〉	통과 (0.74ms, 10.1MB)
테스트 14 〉	통과 (0.03ms, 10.2MB)
채점 결과
정확성: 100.0
합계: 100.0 / 100.0
'''
```

<br>

#### <span style="color:red">🔥 우리 모두 **deque**를 사용합시다!</span>

<br>

**PS. 아 그리고 합계가 필요할 때 sum 함수를 많이 사용하는데, 호출 횟수가 너무 많아질 때는 합계를 나타내는 변수를 만들고 간단한 산술 연산만을 하며 합계 변수를 들고다니는 것이 더 효율적일 수 있습니다!**
