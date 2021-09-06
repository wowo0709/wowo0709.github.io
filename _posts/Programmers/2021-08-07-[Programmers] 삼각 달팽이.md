---
layout: single
title: "[Programmers] 삼각 달팽이"
categories: ['Algorithm', 'Programmers']
---

# 삼각 달팽이

##### 문제 설명

정수 n이 매개변수로 주어집니다. 다음 그림과 같이 밑변의 길이와 높이가 n인 삼각형에서 맨 위 꼭짓점부터 반시계 방향으로 달팽이 채우기를 진행한 후, 첫 행부터 마지막 행까지 모두 순서대로 합친 새로운 배열을 return 하도록 solution 함수를 완성해주세요.

![examples.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/e1e53b93-dcdf-446f-b47f-e8ec1292a5e0/examples.png)

------

##### 제한사항

* n은 1 이상 1,000 이하입니다.

------

##### 입출력 예

| n    | result                                                    |
| ---- | --------------------------------------------------------- |
| 4    | `[1,2,9,3,10,8,4,5,6,7]`                                  |
| 5    | `[1,2,12,3,13,11,4,14,15,10,5,6,7,8,9]`                   |
| 6    | `[1,2,15,3,16,14,4,17,21,13,5,18,19,20,12,6,7,8,9,10,11]` |

------

##### 입출력 예 설명

입출력 예 #1

* 문제 예시와 같습니다.

입출력 예 #2

* 문제 예시와 같습니다.

입출력 예 #3

* 문제 예시와 같습니다.

<br>



### 문제 풀이

---

그림에 현혹되면 안된다! 숫자가 채워지는 방향은 세 방향이고, 각 방향은 **아래, 오른쪽, 좌상향 대각선 방향**이다. 

따라서 숫자를 저장할 리스트를 snail이라는 리스트로 만들고, snail의 i번 째 행은 i+1개의 원소만 가진다. (i는 0부터)

방향이 바뀔 때마다 채워지는 숫자가 하나씩 적어진다. 따라서 바깥 for문으로 방향을 나타내는 dir 변수를, 안쪽 for문의 범위를 range(dir, n)으로 하여 방향이 바뀔 때마다 채우는 숫자가 하나씩 적어지도록 한다.

그리고 for문 안에서는 dir의 값에 따라 세 방향 중 하나의 방향으로 숫자들을 채운다.  

```python
def solution(n):
    snail = [[0 for col in range(row+1)] for row in range(n)]
    i, j = -1, 0  # 행, 열
    num = 1
    for dir in range(n):
        for _ in range(dir, n):
            # 0: 왼쪽 아래, 1: 오른쪽, 2: 왼쪽 위
            if dir % 3 == 0: i += 1
            elif dir % 3 == 1: j += 1
            elif dir % 3 == 2: i, j = i-1, j-1
      
            snail[i][j] = num
            num += 1

    ans = []
    for i in range(n):
        for j in range(i+1):
            ans.append(snail[i][j])
    return ans
```

위 코드처럼 짜는 것이 가장 일반화된 풀이일 것이다. 

그런데 여기서는 알고리즘보다, **출력 형태**에 주목해보려 한다. 이차원 리스트에 있는 원소들을 모두 풀어 순서대로 출력해야 하는데, 모듈을 불러와 이를 간단하게 표현해보자. 

<br>

**1. from functools import reduce**

```python
def solution(n):
    from functools import reduce
    snail = [[0 for col in range(row+1)] for row in range(n)]
    i, j = -1, 0  # 행, 열
    num = 1
    for dir in range(n):
        for _ in range(dir, n):
            # 0: 왼쪽 아래, 1: 오른쪽, 2: 왼쪽 위
            if dir % 3 == 0: i += 1
            elif dir % 3 == 1: j += 1
            elif dir % 3 == 2: i, j = i-1, j-1
      
            snail[i][j] = num
            num += 1

    return reduce(lambda x, y: x + y, snail, [])
```

우선 **reduce** 메서드를 사용하는 방법이 있다. reduce는 iterable에 대해, 누적값이 필요할 때 사용할 수 있다. 

위의 reduce 문에서는 이차원 리스트의 원소인 각각의 (일차원) 리스트에 + 연산을 적용해 모두 이어붙인 값을 얻을 수 있다. 

<br>

그런데 reduce 메서드는 다양한 연산을 일괄적으로 적용시켜 누적값을 얻어낼 수 있는 함수이기 때문에, 단순히 이차원 리스트의 각 원소를 얻으려는 반복 작업에는 적합하지 않다고 할 수 있다. 실제로 이중 for문을 사용했을 때보다 시간이 5배~10배 정도 오래 걸렸다. 

<br>

**2. from itertools import chain**

따라서 반복 작업에 특화되어 있는 itertools 모듈에서 chain 클래스를 사용해보도록 하자. 

```python
def solution(n):
    from itertools import chain
    snail = [[0 for col in range(row+1)] for row in range(n)]
    i, j = -1, 0  # 행, 열
    num = 1
    for dir in range(n):
        for _ in range(dir, n):
            # 0: 왼쪽 아래, 1: 오른쪽, 2: 왼쪽 위
            if dir % 3 == 0: i += 1
            elif dir % 3 == 1: j += 1
            elif dir % 3 == 2: i, j = i-1, j-1
      
            snail[i][j] = num
            num += 1

    return list(chain.from_iterable(snail))
```

chain 모듈의 **from_iterable** 메서드는 인자로 iterable을 전달하면 각 원소를 풀어서 하나의 리스트로 만들어준다. 리스트의 차원을 줄여준다고 생각할 수 있겠다. 

chain 모듈의 사용 예시는 아래와 같다. 

```python
# 각각의 원소를 전달 시 chain 사용
chain('ABC', 'DEF')
# >> 'A' 'B' 'C' 'D' 'E' 'F'

# 원소가 들어있는 iterable 전달 시 chain.from_iterable 사용
chain.from_iterable(['ABC', 'DEF'])
# >> 'A' 'B' 'C' 'D' 'E' 'F'
```



<br>

유용한 클래스와 메서드가 많은 파이썬의 **itertools**와 **functools** 모듈은 다음 문서에서 더 자세히 확인할 수 있다. 

* [itertools - 효율적인 looping을 위한 iterator를 만드는 함수](https://docs.python.org/ko/3.8/library/itertools.html)

* [functools - 고차 함수와 callable 객체에 대한 연산](https://docs.python.org/ko/3.8/library/functools.html)

