---
layout: single
title: "[AITech] 20220118 - Python data structure"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 강의 복습 내용

### Python data structure

**Stack**

* 나중에 넣은 데이터를 먼저 반환하도록 설계된 메모리 구조(LIFO)
* 리스트를 사용하여 스택 구조 표현 가능
  * push는 append(), pop은 pop() 사용

**Queue**

* 먼저 넣은 데이터를 먼저 반환하도록 설계된 메모리 구조(FIFO)
* 리스트를 사용하여 큐 구조 표현 가능
  * put은 append(), get은 pop(0) 사용

**Tuple**

* 값의 변경이 불가능한 리스트
  * 선언 시 "( )"로 선언
* **set, dict 등** 원소(키)로 hashable element를 요구하는 자료 구조에 사용 가능

**Set**

* 값을 순서없이 저장, 중복 불허하는 자료형

```python
s1 = set([1,2,3,4,5])
s2 = set([3,4,5,6,7])
s1 = s1.union(s2) # s1 |= s2 (합집합)
# {1,2,3,4,5,6,7}
s1 = s1.intersection(s2) # s1 &= s2 (교집합)
# {3,4,5}
s1 = s1.difference(s2) # s1 -= s2 (차집합)
# {1,2}
```

**Dict**

* 데이터를 저장할 때 구분 지을 수 있는 값을 함께 저장(key-value)
* key 값을 활용하여, value 값을 관리
* 다른 언어에서는 **Hash Table**이라는 용어를 사용

**Collections**

* Python Built-in 확장 자료 구조(모듈)
* 편의성, 실행 효율 등을 사용자에게 제공

- `deque`
  - rotate, reverse 등 Linked List의 특성을 지원
  - 효율적 메모리 구조로 처리 속도 향상

```python
# deque - Stack과 Queue를 구현하는데 List보다 효율적이고 빠른 저장 방식을 제공
# using collections module
from collections import deque
import time

start_time = time.time()
deque_list = deque()
# deque
for i in range(10**6):
    deque_list.append(i)
for i in range(10**6):
    deque_list.popleft()
print(time.time() - start_time, "seconds")
'''0.17937803268432617 seconds'''
# using list
import time

start_time = time.time()
just_list = []
for i in range(10**6):
    just_list.append(i)
for i in range(10**6):
    just_list.pop(0)
print(time.time() - start_time, "seconds")
'''102.59535932540894 seconds'''
```

* `OrderedDict`
  * 데이터를 입력한 순서대로 dict를 반환
  * 일반 dict도 python 3.6부터는 입력한 순서를 보장하여 출력
* `defaultdict`
  * Dict type의 값에 기본 값을 지정, 신규값 생성 시 사용하는 방법

```python
from collections import defaultdict
d = defaultdict(lambda: 0) # 함수 형태로 사용
print(d["first"])
# 0
```

* `Counter`
  * Sequence type의 data element들의 갯수를 dict 형태로 반환

```python
from collections import Counter

# sequence -> dict
c = Counter('gallahad')
print(c)
'''
Counter({'a': 3, 'l': 2, 'g': 1, 'h': 1, 'd': 1})
'''

# dict -> list
c = Counter({'red':4, 'blue':2})
print(c)
print(c.elements())
print(list(c.elements()))
'''
Counter({'red': 4, 'blue': 2})
<itertools.chain object at 0x000001EDC28B65E0>
['red', 'red', 'red', 'red', 'blue', 'blue']
'''

# set의 연산들을 지원
c = Counter(a=4, b=2, c=0, d=-2)
d = Counter(a=1, b=2, c=3)
print(c+d)
print(c&d)
print(c|d)
c.subtract(d)
print(c)
'''
Counter({'a': 5, 'b': 4, 'c': 3})
Counter({'b': 2, 'a': 1})
Counter({'a': 4, 'c': 3, 'b': 2})
Counter({'a': 3, 'b': 0, 'd': -2, 'c': -3})
'''

# 많은 순서대로 반환
Counter('Hello My name is Bread').most_common()
'''
[(' ', 4),
 ('e', 3),
 ('l', 2),
 ('a', 2),
 ('H', 1),
 ('o', 1),
 ('M', 1),
 ('y', 1),
 ('n', 1),
 ('m', 1),
 ('i', 1),
 ('s', 1),
 ('B', 1),
 ('r', 1),
 ('d', 1)]
'''
```

* `namedtuple`
  * Tuple 형태로 Data 구조체를 저장
  * 저장된는 data의 varaible을 사전에 지정해서 저장

```python
from collections import namedtuple
Point = namedtuple('Point', ['x','y'])
p = Point(11, y=22)
print(p[0] + p[1])
# 33

x, y = p
print(x, y)
print(p.x, p.y)
print(Point(11, 22))
'''
11 22
11 22
Point(x=11, y=22)
'''
```



<br>
