---
layout: single
title: "[Programmers] 소수 찾기"
categories: ['Algorithm', 'Implementation']
tag: ['Permutations', 'Set', 'Prime']
---

# 소수 찾기

##### 문제 설명

한자리 숫자가 적힌 종이 조각이 흩어져있습니다. 흩어진 종이 조각을 붙여 소수를 몇 개 만들 수 있는지 알아내려 합니다.

각 종이 조각에 적힌 숫자가 적힌 문자열 numbers가 주어졌을 때, 종이 조각으로 만들 수 있는 소수가 몇 개인지 return 하도록 solution 함수를 완성해주세요.

##### 제한사항

* numbers는 길이 1 이상 7 이하인 문자열입니다.
* numbers는 0~9까지 숫자만으로 이루어져 있습니다.
* "013"은 0, 1, 3 숫자가 적힌 종이 조각이 흩어져있다는 의미입니다.

##### 입출력 예

| numbers | return |
| ------- | ------ |
| "17"    | 3      |
| "011"   | 2      |

##### 입출력 예 설명

예제 #1
[1, 7]으로는 소수 [7, 17, 71]를 만들 수 있습니다.

예제 #2
[0, 1, 1]으로는 소수 [11, 101]를 만들 수 있습니다.

* 11과 011은 같은 숫자로 취급합니다.

[출처](http://2009.nwerc.eu/results/nwerc09.pdf)

<br>



### 문제 풀이

---

**키워드**

### \# 순열 \# 집합 \# 소수 구하기

<br>

그렇게 어려운 문제는 아닙니다. 위의 키워드들을 떠올리고 차례로 이용하면 어렵지 않게 풀 수 있습니다. 

<br>

**1. 조합이 가능한 수 모두 찾기**

먼저 문제의 조건에 맞게 각 숫자들로 조합이 가능한 수를 모두 찾습니다. 이 때 순열을 생성해주는 **itertools.permutations**를 사용합니다. 

```python
    from itertools import permutations
    splited_nums = list(numbers)
    made_nums = set()
    for n in range(1,len(splited_nums)+1):
        made_nums |= set(map(int, map(''.join,permutations(splited_nums,n))))
```

splited_nums에 각 흩어진 숫자 조각들을 저장합니다. 

made_nums에는 그 숫자 조각들로 조합이 가능한 숫자들을 저장할 것입니다. 이때 중복이 될 수 있으므로 자료구조로는 **set**을 사용합니다. 

<br>

**2. 소수의 개수 찾기**

made_nums에 있는 수들 중 소수의 개수를 찾습니다. 

소수 구하기 알고리즘으로는 가장 효율적인 **에라토스테네스의 체** 알고리즘을 사용합니다. 

```python
    maxnum = max(made_nums)
    is_prime = [False, False] + [True]*(maxnum-1)
    for i in range(2,int(maxnum**0.5)+1):
        if not is_prime[i]: continue
        for j in range(2*i, maxnum+1, i):
            is_prime[j] = False

    return sum([True for num in made_nums if is_prime[num]])
```

<br>

전체 코드는 다음과 같습니다. 

```python
def solution(numbers):
    from itertools import permutations
    splited_nums = list(numbers)
    made_nums = set()
    for n in range(1,len(splited_nums)+1):
        made_nums |= set(map(int, map(''.join,permutations(splited_nums,n))))

    maxnum = max(made_nums)
    is_prime = [False, False] + [True]*(maxnum-1)
    for i in range(2,int(maxnum**0.5)+1):
        if not is_prime[i]: continue
        for j in range(2*i, maxnum+1, i):
            is_prime[j] = False

    return sum([True for num in made_nums if is_prime[num]])
```

<br>

<br>

### 조금 다른 풀이

아래 풀이는 소수를 찾는 과정에서 is_prime이라는 리스트를 따로 만들지 않고, 바로 made_nums 집합에서 제외함으로써 조금 더 간결해진 코드입니다. 전체적인 알고리즘은 위 풀이와 같습니다. 

다만, 위 풀이에는 있는 **continue 문**이 없기 때문에 모든 경우를 검사해 실행 시간은 3~4배 가량 더 소요되는 것을 확인할 수 있었습니다. 

```python
def solution(numbers):
    from itertools import permutations
    splited_nums = list(numbers)
    made_nums = set()
    for n in range(1,len(splited_nums)+1):
        made_nums |= set(map(int, map(''.join,permutations(splited_nums,n))))

    # 소수 리스트를 따로 만들지 않고 소수가 아니면 바로 집합에서 제외
    # continue 문이 없어서 시간은 더 걸림
    maxnum = max(made_nums)
    for i in range(2,int(maxnum**0.5)+1):
        made_nums -= set(range(2*i, maxnum+1,i))

    # 0과 1 제외
    made_nums -= set([0,1])
    return len(made_nums)
```

<br>

### set 연산

set에서 사용하는 **합집합, 교집합, 차집합**에 대해 알고 넘어갑시다. 

```python
>>> s1 = set([1, 2, 3, 4, 5, 6])
>>> s2 = set([4, 5, 6, 7, 8, 9])
```

<br>

**1. 합집합**

합집합 연산에는 **' | '** 기호 또는 union 메서드를 사용합니다.  

```python
>>> s1 | s2
{1, 2, 3, 4, 5, 6, 7, 8, 9}
>>> s1.union(s2)
{1, 2, 3, 4, 5, 6, 7, 8, 9}
```

<br>

**2. 교집합**

교집합 연산에는 ' & ' 기호 또는 intersection 메서드를 사용합니다. 

```python
>>> s1 & s2
{4, 5, 6}
>>> s1.intersection(s2)
{4, 5, 6}
```

<br>

**3. 차집합**

```python
>>> s1 - s2
{1, 2, 3}
>>> s2 - s1
{8, 9, 7}
>>> s1.difference(s2)
{1, 2, 3}
>>> s2.difference(s1)
{8, 9, 7}
```

<br>

출처: [점프 투 파이썬 - 집합 자료형](https://wikidocs.net/1015)

