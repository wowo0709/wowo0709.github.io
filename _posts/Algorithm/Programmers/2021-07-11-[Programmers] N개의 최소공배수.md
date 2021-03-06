---
layout: single
title: "[Programmers] N개의 최소공배수"
categories: ['Algorithm', 'Math']
tag: ['GCD', 'LCM']
---

# N개의 최소공배수

### 문제 설명

---

두 수의 최소공배수(Least Common Multiple)란 입력된 두 수의 배수 중 공통이 되는 가장 작은 숫자를 의미합니다. 예를 들어 2와 7의 최소공배수는 14가 됩니다. 정의를 확장해서, n개의 수의 최소공배수는 n 개의 수들의 배수 중 공통이 되는 가장 작은 숫자가 됩니다. n개의 숫자를 담은 배열 arr이 입력되었을 때 이 수들의 최소공배수를 반환하는 함수, solution을 완성해 주세요.

##### 제한 사항

- arr은 길이 1이상, 15이하인 배열입니다.
- arr의 원소는 100 이하인 자연수입니다.

##### 입출력 예

| arr        | result |
| ---------- | ------ |
| [2,6,8,14] | 168    |
| [1,2,3]    | 6      |



### 문제 풀이

---

**두 수의 최소공배수는 두 수의 곱을 두 수의 최대공약수로 나눔**으로써 구할 수 있습니다. 

math 라이브러리의 gcd 함수를 import 하면 간단히 구할 수 있다. 

```python
from math import gcd
def solution(arr):
    ans = arr[0]
    for nbr in arr:
        ans = ans*nbr // gcd(ans,nbr)

    return ans
```

위와 같은 코드를 실행시키면 된다. 



그럼 안녕!
