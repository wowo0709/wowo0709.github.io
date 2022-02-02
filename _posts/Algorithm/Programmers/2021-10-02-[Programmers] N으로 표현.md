---
layout: single
title: "[Programmers] N으로 표현"
categories: ['Algorithm', 'DynamicProgramming']
toc: true
toc_sticky: true
tag: ['Product']
---



<br>

## 문제 설명

### 문제 설명

아래와 같이 5와 사칙연산만으로 12를 표현할 수 있습니다.

12 = 5 + 5 + (5 / 5) + (5 / 5)
12 = 55 / 5 + 5 / 5
12 = (55 + 5) / 5

5를 사용한 횟수는 각각 6,5,4 입니다. 그리고 이중 가장 작은 경우는 4입니다.
이처럼 숫자 N과 number가 주어질 때, N과 사칙연산만 사용해서 표현 할 수 있는 방법 중 N 사용횟수의 최솟값을 return 하도록 solution 함수를 작성하세요.

### 제한사항

* N은 1 이상 9 이하입니다.
* number는 1 이상 32,000 이하입니다.
* 수식에는 괄호와 사칙연산만 가능하며 나누기 연산에서 나머지는 무시합니다.
* 최솟값이 8보다 크면 -1을 return 합니다.

### 입출력 예

| N    | number | return |
| ---- | ------ | ------ |
| 5    | 12     | 4      |
| 2    | 11     | 3      |

### 입출력 예 설명

예제 #1
문제에 나온 예와 같습니다.

예제 #2
`11 = 22 / 2`와 같이 2를 3번만 사용하여 표현할 수 있습니다.

[출처](https://www.oi.edu.pl/old/php/show.php?ac=e181413&module=show&file=zadania/oi6/monocyfr)

※ 공지 - 2020년 9월 3일 테스트케이스가 추가되었습니다.

<br>

## 문제 풀이

### \# 동적계획법

<br>

동적계획법 문제는 항상 인덱스와 값에 무엇을 넣을 것인지가 가장 중요하죠. 

이 문제를 풀기에 가장 알맞은 `dp배열`로 **인덱스에 N의 개수**를, **값에 인덱스 개수의 N을 사용해서 만들 수 있는 숫자들**을 설정합니다. 

<br>

  그러면 `i`개의 `N` 을 사용해서 만들 수 있는 숫자들은 어떻게 구할 수 있을까요?

문제를 보면, 새로운 숫자를 만들어낼 때 가능한 연산의 개수는 5개입니다. (사칙 연산 + 숫자 붙이기)

따라서 `dp[i]` 는 `i개의 N을 붙인 숫자 + (dp[n]의 숫자와 dp[i-n]의 숫자를 사칙연산한 숫자들(i//2 <= n <= i-1))`로 구할 수 있습니다. 

```python
# dp[N의 개수] = {만들 수 있는 숫자들}
def solution(N, number):
    dp = [set() for _ in range(9)]
    dp[0].add(0)
    from itertools import product
    for i in range(1,9):
        dp[i].add(int("{}".format(N)*i))
        for n in range(i//2,i):
            for a,b in product(dp[n],dp[i-n]):
                dp[i].add(a+b)
                dp[i].add(a-b)
                dp[i].add(b-a)
                dp[i].add(a*b)
                if b != 0: dp[i].add(a//b)
                if a != 0: dp[i].add(b//a)
        if number in dp[i]:
            return i

    return -1
```















<br>
