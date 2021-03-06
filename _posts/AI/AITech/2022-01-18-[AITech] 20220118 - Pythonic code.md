---
layout: single
title: "[AITech] 20220118 - Pythonic code"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['split&join', 'list comprehension', 'enumerate&zip', 'lambda&map&reduce', 'generator', 'asterisk']
---



<br>

## 강의 복습 내용

### Pythonic code

* 파이썬 스타일의 코딩 기법
* 파이썬 특유의 문법을 활용하여 효율적으로 코드를 표현

#### **split&join**

```python
# split
items = 'zero one two three'.split()
print(items)
# ['zero', 'one', 'two', 'three']

# join
colors = ['red','blue','green','yellow']
print('&'.join(colors))
# red&blue&green&yellow
```



#### **list comprehension**

* list를 사용하여 간단히 다른 list를 만드는 기법

```python
result = [i for i in range(10)]
result2 = [i for i in range(10) if i%2 == 0]
print(result, result2)
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] [0, 2, 4, 6, 8]

# nested loop
word_1 = "Hello"
word_2 = "World"
result3 = [i+j for i in word_1 for j in word_2]
print(result3)
# ['HW', 'Ho', 'Hr', 'Hl', 'Hd', 'eW', 'eo', 'er', 'el', 'ed', 'lW', 'lo', 'lr', 'll', 'ld', 'lW', 'lo', 'lr', 'll', 'ld', 'oW', 'oo', 'or', 'ol', 'od']

# conditional 
case_1 = ["A","B","C"]
case_2 = ["D","E","A"]
result4 = [i+j for i in case_1 for j in case_2 if not i==j]
result5 = [i+j if not i==j else i for i in case_1 for j in case_2]
print(result4, result5)
'''
['AD', 'AE', 'BD', 'BE', 'BA', 'CD', 'CE', 'CA'] 
['AD', 'AE', 'A', 'BD', 'BE', 'BA', 'CD', 'CE', 'CA']
'''
# 2-d list
result6 = [i+j for i in case_1 for j in case_2] # 1-d list -> case1이 바깥 loop, case2가 안쪽 loop
result7 = [[i+j for i in case_1] for j in case_2] # 2-d list -> case1이 안쪽 loop, case2가 바깥 loop
print(result6, result7)
'''
['AD', 'AE', 'AA', 'BD', 'BE', 'BA', 'CD', 'CE', 'CA'] 
[['AD', 'BD', 'CD'], ['AE', 'BE', 'CE'], ['AA', 'BA', 'CA']]
'''
```



#### **enumerate&zip**

```python
# enumerate
for i,v in enumerate("ABC"):
    print(i,v)
'''
0 A
1 B
2 C
'''
my_str = "ABCD"
my_dict = {v:i for i,v in enumerate(my_str)}
print(my_dict)
# {'A': 0, 'B': 1, 'C': 2, 'D': 3}

# zip
alist = ['a1','a2','a3']
blist = ['b1','b2','b3']
for a,b in zip(alist, blist):
    print(a,b)
'''
a1 b1
a2 b2
a3 b3
'''
[sum(x) for x in zip((1,2,3),(10,20,30),(100,200,300))]
# [111, 222, 333]
```



#### **lambda&map&reduce**

코드의 직관성이 떨어져서 lambda나 reduce는 python3에서 사용을 권장하지는 않음!

그러나 여전히 많은 곳에서 사용 중

* lambda

```python
f = lambda x,y: x+y
print(f(1,4))
# print((lambda x: x+1)(5))
# 5
'''
PEP8에서는 lambda의 사용을 권장하지는 않음
- 어려운 문법
- 테스트의 어려움
- 문서화 docstring 지원 미비
- 코드 해석의 어려움
- 이름이 존재하지 않는 함수의 출현
- 그래도 많이 쓴다...
'''
```

* map
  * python3부터는 map generator를 생성하기 때문에 list로 사용하려면 `list(map(...))`와 같이 사용

```python
# map
ex = [1,2,3,4,5]
f = lambda x, y: x+y
print(list(map(f,ex,ex)))
# [2, 4, 6, 8, 10]
list(
    map(
    lambda x: x**2 if x%2 == 0
    else x,
    ex)
)
# [1, 4, 3, 16, 5]
```

* reduce

```python
# reduce
from functools import reduce
print(reduce(lambda x, y: x+y, [1,2,3,4,5],10)) # reduce(function,sequence,initializer)
# 25
```







#### **generator**

* 내부적 구현으로 `__iter__`와 `__next__`가 사용
* iter()와 next() 함수로 iterable 객체를 iterator object로 사용

```python
cities = ["Seoul", "Busan", "Jeju"]

iter_obj = iter(cities)

print(next(iter_obj))
print(next(iter_obj))
print(next(iter_obj))
print(next(iter_obj))
'''
Seoul
Busan
Jeju
---------------------------------------------------------------------------
StopIteration                             Traceback (most recent call last)
<ipython-input-48-8d293490d776> in <module>
      6 print(next(iter_obj))
      7 print(next(iter_obj))
----> 8 print(next(iter_obj))

StopIteration: 
'''
```

* `generator`
  * iterable object를 특수한 형태로 사용
  * element가 사용되는 시점에 값을 메모리에 반환
    * yeild를 사용해 한 번에 하나의 element만 반환
  * 메모리를 훨씬 더 절약
    * 파일 데이터 등의 대용량 데이터 처리 시 사용

```python
def general_list(value):
    result = []
    for i in range(value):
        result.append(i)
    return result

result = general_list(50)
print(result,sys.getsizeof(result))
'''
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49] 520
'''

def generator_list(value):
    result = []
    for i in range(value):
        yield i
        
result = generator_list(50)
print(result, sys.getsizeof(result))
'''
<generator object generator_list at 0x000001EDC28B9C10> 112
'''
# 값을 불러오려면 for문 사용
result = list(generator_list(50))
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49]
for a in generator_list(50):
    print(a,end=' ')
# 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 45 46 47 48 49 
```



#### **asterisk**

* 함수에 입력되는 arguments의 다양한 형태
  1. Keyword arguments
  2. Default arguments
  3. Variable-length arguments
* Keyword arguments

```python
def print_something(my_name, your_name):
    print("Hello {0}, My name is {1}".format(your_name, my_name))
    
print_something("Sungchul","TEAMLAB")
print_something(your_name="TEAMLAB",my_name="Sungchul")

'''
Hello TEAMLAB, My name is Sungchul
Hello TEAMLAB, My name is Sungchul
'''
```

* Default arguments

```python
def print_something_2(my_name, your_name="TEAMLAB"):
    print("Hello {0}, My name is {1}".format(your_name, my_name))
    
print_something_2("Sungchul","TEAMLAB")
print_something_2("Sungchul")

'''
Hello TEAMLAB, My name is Sungchul
Hello TEAMLAB, My name is Sungchul
'''
```

* Variable-length parameter
  * **개수가 정해지지 않은 변수**를 함수의 parameter로 사용하는 법
  * **Asterisk(*)** 기호를 사용하여 함수의 parameter를 표시함
  * 입력된 값을 **tuple type**으로 사용
  * 가변 인자는 오직 한 개만 맨 마지막 parameter로 사용 가능

```python
# 가변 인자 사용하기
def asterisk_test(*args):
    x, y, z = args
    print(x, y, z, sum(args))
    
asterisk_test(3,4,5)

# 3 4 5 12
```



* Keyword varaiable-length parameter
  * **Asterisk(*) 두 개**를 사용하여 함수의 parameter를 표시
  * 입력된 값은 **dict type**으로 사용
  * 가변 인자는 오직 한 개만 기존 가변 인자 다음에 사용

```python
# 키워드 가변인자 사용하기
def kwargs_test(one, two, *args, **kwargs):
    print(one+two+sum(args))
    print(kwargs)
    
kwargs_test(3,4,5,6,7,8,9,first=3,second=4,third=5)

'''
42
{'first': 3, 'second': 4, 'third': 5}
'''
```

**가변 인자 및 키워드 가변 인자는 머신러닝에서 매우 많이 사용되므로 잘 알아둘 것!!**

* Unpacking
  * tuple, dict 등 자료형에 들어가 있는 값을 unpacking
  * 함수의 입력값, zip 등에 유용하게 사용 가능

```python
# 언패킹
def asterisk_test(a, *args):
    print(a, args)
    print(type(args))
    
asterisk_test(1,*(2,3,4,5,6))
'''
1 (2, 3, 4, 5, 6)
<class 'tuple'>
'''
def asterisk_test(a, args):
    print(a, *args)
    print(type(args))
    
asterisk_test(1,(2,3,4,5,6))
'''
1 2 3 4 5 6
<class 'tuple'>
'''

# zip에 사용하기
for data in zip(*([1,2],[3,4],[5,6])):
    print(data)
'''
(1, 3, 5)
(2, 4, 6)
'''
```



<br>
