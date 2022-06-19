---
layout: single
title: "[Python] Effective Python CH 1. 파이썬답게 생각하기 - 2"
categories: ['Language', 'Python']
toc: true
toc_sticky: true
tag: []

---



# CH 1. 파이썬답게 생각하기 - 2

`Effective Python 2nd Edition`을 읽으며 학습한 내용들을 정리합니다. 

* 목차 선택하기: [Effective Python 전체 목차](https://wowo0709.github.io/language/python/Python-Effective-Python-%EC%A0%84%EC%B2%B4-%EB%AA%A9%EC%B0%A8/)
* 소스 코드: [wowo0709/Effective-Python](https://github.com/wowo0709/Effective-Python)

해당 포스팅은 [Chapter 1. 파이썬답게 생각하기]의 두번째 포스팅입니다. 

## Bettery way 6. 인덱스를 사용하는 대신 대입을 사용해 데이터를 언패킹하라

- 파이썬은 한 문장 안에서 여러 값을 대입할 수 있는 언패킹이라는 특별한 문법을 제공한다. 
- 파이썬 언패킹은 일반화돼 있으므로 모든 이터러블에 적용할 수 있다. 이터러블이 여러 계층으로 내포된 경우에도 언패킹을 적용할 수 있다. 
- 인텍스를 사용해 시퀀스 내부에 접근하는 대신 언패킹을 사용해 코드를 더 명확하게 만들라. 

**예시 1**


```python
# 인덱싱
item = ('Peanut butter', 'Jelly')
first = item[0]
second = item[1]
print(first, 'and', second)

# 언패킹
item = ('Peanut butter', 'Jelly')
first, second = item  # Unpacking
print(first, 'and', second)
```

    Peanut butter and Jelly
    Peanut butter and Jelly

**예시 2**


```python
# 인덱싱
def bubble_sort(a):
	for _ in range(len(a)):
		for i in range(1, len(a)):
			if a[i] < a[i-1]:
				temp = a[i]
				a[i] = a[i-1]
				a[i-1] = temp

names = ['pretzels', 'carrots', 'arugula', 'bacon']
bubble_sort(names)
print(names)

# 언패킹
def bubble_sort(a):
	for _ in range(len(a)):
		for i in range(1, len(a)):
			if a[i] < a[i-1]:
				a[i-1], a[i] = a[i], a[i-1]  # Swap

names = ['pretzels', 'carrots', 'arugula', 'bacon']
bubble_sort(names)
print(names)
```

    ['arugula', 'bacon', 'carrots', 'pretzels']
    ['arugula', 'bacon', 'carrots', 'pretzels']


**예시 3**


```python
# 인덱싱
snacks = [('bacon', 350), ('donut', 240), ('muffin', 190)]
for i in range(len(snacks)):
	item = snacks[i]
	name = item[0]
	calories = item[1]
	print(f'#{i+1}: {name} has {calories} calories')

# 언패킹
for rank, (name, calories) in enumerate(snacks, 1):
	print(f'#{rank}: {name} has {calories} calories')
```

    #1: bacon has 350 calories
    #2: donut has 240 calories
    #3: muffin has 190 calories
    #1: bacon has 350 calories
    #2: donut has 240 calories
    #3: muffin has 190 calories

<br>

## Bettery way 7. range보다는 enumerate를 사용하라

- enumerate를 사용하면 이터레이터에 대해 루프를 돌면서 이터레이터에서 가져오는 원소의 인덱스까지 얻는 코드를 간결하게 작성할 수 있다. 
- range에 대해 루프를 돌면서 시퀀스의 원소를 가져오기보다는 enumerate를 사용하라. 
- enumerate의 두 번째 파라미터로 어디부터 수를 세기 시작할 지 지정할 수 있다. 

range를 사용할 때는 보통 아래와 같이 사용하며, `len`과 함께 사용한다. 

이러한 코드는 list의 길이를 알아야 하고, 인덱스를 사용해 배열 원소에 접근해야 한다. 


```python
flavor_list = ['vanilla', 'chocolate', 'pecan', 'strawberry']

for i in range(len(flavor_list)):
    flavor = flavor_list[i]
    print(f'{i + 1}: {flavor}')
```

    1: vanilla
    2: chocolate
    3: pecan
    4: strawberry


대신에, enumerate 내장 함수를 사용할 수 있다. 

enumerate는 이터레이터를 `lazy generator`로 감싼다. lazy generator에서는 `next()` 또는 `for 문`을 통해 직접적인 참조가 이루어질 때 값을 메모리에 올리기 때문에 효율적이며, 가독성도 더 좋다. 


```python
# enumerate 예시
it = enumerate(flavor_list)
print(next(it))
print(next(it))
```

    (0, 'vanilla')
    (1, 'chocolate')

enumerate가 넘겨주는 각 쌍을 for문에서 간결하게 언패킹 할 수 있다. 

또한 enumerate의 두 번째 파라미터로 어디부터 수를 세기 시작할 지 지정할 수 있다. 


```python
for i, flavor in enumerate(flavor_list, 1):
    print(f'{i}: {flavor}')
```

    1: vanilla
    2: chocolate
    3: pecan
    4: strawberry

<br>

## Better way 8. 여러 iterator에 대해 나란히 loop를 수행하려면 zip을 사용하라

- zip 내장 함수를 이용하여 여러 이터레이터를 나란히 이터레이션 할 수 있다. 
- 입력 이터레이터의 길이가 서로 다르면 zip은 아무 경고 없이 가장 짧은 이터레이터 길이까지만 튜플을 내놓고 더 긴 이터레이터의 나머지 원소를 무시한다. 
- 가장 짧은 이터레이터에 맞춰 길이를 제한하지 않고 길이가 서로 다른 이터레이터에 대해 루프를 수행하려면 itertools 내장 모듈의 `zip_longest` 함수를 사용하라. 

`zip` 이터레이터는 각 이터레이터의 다음 값이 들어있는 튜플을 반환한다. 이 튜플을 for 문에서 바로 언패킹할 수 있다. 

zip은 입력으로 주어진 이터레이터 중 어느 하나가 끝날 때까지 튜플을 내놓는다. 즉, zip의 출력은 가장 짧은 입력 이터레이터의 길이와 같다. 


```python
names = ['Cecilia', 'Lise', 'Marie']
counts = [len(n) for n in names]

names.append('Rosalind')
for name, count in zip(names, counts):
    print(name)
```

    Cecilia
    Lise
    Marie


긴 이터레이터의 뒷부분을 이용해야 한다면, itertools 내장 모듈에 들어있는 `zip_longest` 함수를 사용하라. 

zip_longest는 존재하지 않는 값을 자신에게 전달된 `fillvalue`로 대신한다. 디폴트 fillvalue는 None이다. 


```python
import itertools

for name, count in itertools.zip_longest(names, counts, fillvalue=-1):
    print(f'{name}: {count}')
```

    Cecilia: 7
    Lise: 4
    Marie: 5
    Rosalind: -1

<br>

## Better way 9. for나 while loop 뒤에 else 블록을 사용하지 말라

- 파이썬은 for나 while 루프에 속한 블록 바로 뒤에 else 블록을 허용하는 특별한 문법을 제공한다. 
- 루프 뒤에 오는 else 블록은 루프가 정상종료 되었을 때(break를 만나지 않았을 때)에만 실행된다. 
- 동작이 직관적이지 않고 혼동을 야기할 수 있으므로, 루프 뒤에 else 블록을 사용하지 말라. 

파이썬에서 if-else, try-except-else-finally 문 등을 배운 프로그래머는 for-else 문에서 else 부분을 **루프가 정상적으로 완료되지 않으면 이 블록을 실행하라**라는 뜻으로 가정하기 쉽다. 

하지만 실제 for-else에서 else 문은 **for 문이 끝까지 정상 종료했을 때** 실행된다. 


```python
for i in range(3):
    print('Loop', i)
else:
    print('Else block 1')

for i in range(3):
    print('Loop', i)
    if i == 1:
        break
else:
    print('Else block 2')
```

    Loop 0
    Loop 1
    Loop 2
    Else block 1
    Loop 0
    Loop 1


for 문에서 empty sequence가 주어졌을 때도 else 문은 바로 실행된다. 

또한 while 루프의 조건이 처음부터 False인 경우에도 else 블록이 바로 실행된다. 


```python
for x in []:
    print('Never runs')
else:
    print('For Else block!')

while False:
    print('Never runs')
else:
    print('While Else block!')
```

    For Else block!
    While Else block!


이런 식으로 동작하는 이유는 루프를 사용해 검색을 수행할 경우, 루프 바로 뒤에 있는 else 블록이 그와 같이 동작해야 유용하기 때문이다. 

예를 들어, 두 수가 서로소인지 검사하는 코드를 작성한다고 하자. 


```python
a = 4
b = 9

for i in range(2, min(a, b) + 1):
    print('Testing', i)
    if a % i == 0 and b % i == 0:
        print('Not coprime')
        break
else:
    print('Coprime')
```

    Testing 2
    Testing 3
    Testing 4
    Coprime


대신에, 계산을 수행하는 도우미 함수를 작성하는 것이 좋다. 

for-else 문을 대체하기 위해 2가지 방식으로 작성할 수 있다. 


```python
# 방식 1
def coprime(a, b):
    for i in range(2, min(a, b) + 1):
        if a % i == 0 and b % i == 0:
            return False
    return True

assert coprime(4, 9)
assert not coprime(3, 6)
```


```python
# 방식 2
def coprime_alternate(a, b):
    is_coprime = True
    for i in range(2, min(a, b) + 1):
        if a % i == 0 and b % i == 0:
            is_coprime = False
            break
    return is_coprime

assert coprime_alternate(4, 9)
assert not coprime_alternate(3, 6)
```

뭐가 되었든, for-else 문을 사용하는 것보다 훨씬 명확해 보인다. 

for-else 문을 사용하여 얻을 수 있는 표현력보다는 이 코드를 이해하려는 사람들(자신 포함)이 느끼게 될 부담감이 더 크다. 

파이썬에서 루프와 같은 간단한 구성 요소는 그 자체로 의미가 명확해야 한다. 따라서 절대로 루프 뒤에 else 블록을 사용하지 말아야 한다. 

<br>

## Bettery way 10. 대입식을 사용해 반복을 피하라

대입식은 영어로 assignment expression이며, **왈러스 연산자 (walrus operator)** 라고도 부른다. 이 대입식은 파이썬 언어에서 고질적인 코드 중복 문제를 해결하고자 파이썬 3.8에서 새롭게 도입된 구문이다. 

일반 대입문은 `a = b` 라고 쓰며 `a equal b`라고 읽는다. 왈러스 연산자는 `a := b`라고 쓰며 `a walrus b`라고 읽는다. (왈러스라는 이름은 `:=`이 바다코끼리(walrus)의 눈과 엄니처럼 보이기 때문에 붙여졌다).

왈러스 연산자을 사용하면 코드의 길이도 짧아지지만, **변수가 특정 부분에서만 의미있다는 것**을 명확히 드러낼 수 있어 더 얽기 쉽다.


```python
# 변수 및 함수 정의
fresh_fruit = {
    'apple': 10,
    'banana': 8,
    'lemon': 5,
}

def make_lemonade(count):
    print(f'Making {count} lemons into lemonade')

def make_cider(count):
    print(f'Making cider with {count} apples')

def out_of_stock():
    print('Out of stock!')

def slice_bananas(count):
    print(f'Slicing {count} bananas')
    return count * 4

def make_smoothies(count):
    print(f'Making a smoothies with {count} banana slices')
```


```python
# 대입문
count = fresh_fruit.get('apple', 0)
if count >= 4:
    make_cider(count)
else:
    out_of_stock()
```

    Making cider with 10 apples



```python
# 왈러스 연산자
if (count := fresh_fruit.get('apple', 0)) >= 4:
    make_cider(count)
else:
    out_of_stock()
```

    Making cider with 10 apples

<br>

왈러스 연산자는 `switch-case` 문의 훌륭한 대안이 되기도 한다. 

아래 두 코드를 비교해보자. 


```python
# 다중 if-else를 활용한 switch-case 문
count = fresh_fruit.get('banana', 0)
if count >= 2:
    pieces = slice_bananas(count)
    to_enjoy = make_smoothies(pieces)
else:
    count = fresh_fruit.get('apple', 0)
    if count >= 4:
        to_enjoy = make_cider(count)
    else:
        count = fresh_fruit.get('lemon', 0)
        if count:
            to_enjoy = make_lemonade(count)
        else:
            to_enjoy = 'Nothing'
```

    Slicing 8 bananas
    Making a smoothies with 32 banana slices



```python
# 왈러스 연산자를 활용한 switch-case 문
if (count := fresh_fruit.get('banana', 0)) >= 2:
    pieces = slice_bananas(count)
    to_enjoy = make_smoothies(pieces)
elif (count := fresh_fruit.get('apple', 0)) >= 4:
    to_enjoy = make_cider(count)
elif count := fresh_fruit.get('lemon', 0):
    to_enjoy = make_lemonade(count)
else:
    to_enjoy = 'Nothing'
```

    Slicing 8 bananas
    Making a smoothies with 32 banana slices

<br>

또는 `do-while` 문을 대체할 수도 있다. 

많은 경우에 파이썬에서는 while문이 아래와 같이 작성된다. 


```python
# 변수 및 함수 정의
FRUIT_TO_PICK = [
    {'apple': 1, 'banana': 3},
    {'lemon': 2, 'lime': 5},
    {'orange': 3, 'melon': 2},
]

def pick_fruit():
    if FRUIT_TO_PICK:
        return FRUIT_TO_PICK.pop(0)
    else:
        return []

def make_juice(fruit, count):
    return [(fruit, count)]
```


```python
# 파이썬에서의 while 구문
bottles = []
fresh_fruit = pick_fruit()
while fresh_fruit:
    for fruit, count in fresh_fruit.items():
        batch = make_juice(fruit, count)
        bottles.extend(batch)
    fresh_fruit = pick_fruit()

print(bottles)
```

    [('apple', 1), ('banana', 3), ('lemon', 2), ('lime', 5), ('orange', 3), ('melon', 2)]


위 코드는 `fresh_fruit = pick_fruit()` 호출을 두 번하므로 반복적이다. 

이 상황에서 코드 재사용을 향상시키기 위해 **무한 루프-중간에서 끝내기(loop-and-a-half)** 관용어를 사용할 수 있다. 


```python
FRUIT_TO_PICK = [
    {'apple': 1, 'banana': 3},
    {'lemon': 2, 'lime': 5},
    {'orange': 3, 'melon': 2},
]

bottles = []
while True:                     # Loop
    fresh_fruit = pick_fruit()
    if not fresh_fruit:         # And a half
        break
    for fruit, count in fresh_fruit.items():
        batch = make_juice(fruit, count)
        bottles.extend(batch)

print(bottles)
```

    [('apple', 1), ('banana', 3), ('lemon', 2), ('lime', 5), ('orange', 3), ('melon', 2)]


하지만 위 코드는 while 루프를 맹목적인 무한 루프로 만들기 때문에 while 루프의 유용성이 떨어지며, 루프의 흐름 제어가 모두 break 문에 달려있기 때문에 권장되지 않는다. 

대신에 **왈러스 연산자**를 사용하면 더 짧고 읽기 쉽게 작성할 수 있다. 


```python
FRUIT_TO_PICK = [
    {'apple': 1, 'banana': 3},
    {'lemon': 2, 'lime': 5},
    {'orange': 3, 'melon': 2},
]

bottles = []
while fresh_fruit := pick_fruit(): # walrus operator
    for fruit, count in fresh_fruit.items():
        batch = make_juice(fruit, count)
        bottles.extend(batch)

print(bottles)
```

    [('apple', 1), ('banana', 3), ('lemon', 2), ('lime', 5), ('orange', 3), ('melon', 2)]

