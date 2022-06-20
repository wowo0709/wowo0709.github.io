---
layout: single
title: "[Python] Effective Python CH 2. 리스트와 딕셔너리 - 1"
categories: ['Language', 'Python']
toc: true
toc_sticky: true
tag: []

---



# CH 2. 리스트와 딕셔너리 - 1

`Effective Python 2nd Edition`을 읽으며 학습한 내용들을 정리합니다. 

* 목차 선택하기: [Effective Python 전체 목차](https://wowo0709.github.io/language/python/Python-Effective-Python-%EC%A0%84%EC%B2%B4-%EB%AA%A9%EC%B0%A8/)
* 소스 코드: [wowo0709/Effective-Python](https://github.com/wowo0709/Effective-Python)

해당 포스팅은 [Chapter 2. 리스트와 딕셔너리]의 첫번째 포스팅입니다. 

## Better way 11. 시퀀스를 슬라이싱하는 방법을 익혀라

- 슬라이싱 할 때는 간결하게 하라. 시작 인덱스에 0을 넣거나, 끝 인덱스에 시퀀스 길이를 넣지 말라. 
- 슬라이싱은 범위를 넘어가는 시작 인덱스나 끝 인덱스도 허용한다. 따라서 시퀀스의 시작이나 끝에서 길이를 제한하는 슬라이스를 쉽게 표현할 수 있다. 
  - 그러나 범위를 넘어가는 인덱스를 직접 참조할 때는 예외가 발생한다. 
- 리스트 슬라이싱은 기본적으로 **새로운 리스트를 생성**한다(`b = a[3:5]`). 
- 그러나 **리스트 슬라이스에 대입할 때**는 원래 시퀀스에서 슬라이스가 가리키는 부분을 대입 연산자 오른쪽에 있는 **시퀀스의 복사본**으로 대치한다(`b[3:5] = a`).
  - 이때 슬라이스의 길이와 대치되는 시퀀스의 길이가 달라도 된다. 

기본적으로 아래와 같은 슬라이싱 형태는 자주 사용한다. 


```python
a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

print(a[3:5])
assert a[:5] == a[0:5]
assert a[5:] == a[5:len(a)]
assert a[:7] == a[:-1]
```

    ['d', 'e']


슬라이싱할 때 리스트의 인덱스를 넘어가는 시작과 끝 인덱스는 무시된다. 


```python
print(a[:20])
print(a[-20:])
```

    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']


반면 같은 인덱스에 직접 접근하면 예외가 발생한다. 


```python
print(a[20])
```


    ---------------------------------------------------------------------------
    
    IndexError                                Traceback (most recent call last)
    
    ~\AppData\Local\Temp/ipykernel_13300/2853071919.py in <module>
    ----> 1 print(a[20])


    IndexError: list index out of range

<br>

슬라이싱한 결과는 완전히 **새로운 리스트**이며, 슬라이싱은 새로운 리스트 객체를 생성하는 연산이다. (따라서 시간 복잡도는 O(n)이다)

**대입에 슬라이스**를 사용하면 원본 리스트에서 지정한 범위에 들어있는 원소를 변경하는데, 이때 **슬라이싱 범위와 대입되는 리스트의 길이는 같을 필요가 없다.**


```python
# 슬라이싱 범위보다 대입되는 리스트의 길이가 짧을 때
print('Before ', a)
a[2:7] = [99, 22, 14]
print('After  ', a)
```

    Before  ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    After   ['a', 'b', 99, 22, 14, 'h']



```python
# 슬라이싱 범위보다 대입되는 리스트의 길이가 길 때
print('Before ', a)
a[2:3] = [47, 11]
print('After  ', a)
```

    Before  ['a', 'b', 99, 22, 14, 'h']
    After   ['a', 'b', 47, 11, 22, 14, 'h']

<br>

슬라이싱에서 시작과 끝 인덱스를 모두 생략하면 원래 리스트를 복사한 새 리스트를 얻는다. 


```python
b = a[:]
assert b == a and b is not a # 내용물은 같지만 다른 객체
```

시작과 끝 인덱스가 없는 슬라이스에 대입하면 (새 리스트를 만들어내는 대신), 슬라이싱 범위의 리스트 값을 **대입하는 리스트의 복사본**으로 덮어 쓴다. 


```python
b = a
print('Before a', a)
print('Before b', b)
a[:] = [101, 102, 103]
assert a is b             # Still the same list object
print('After a ', a)      # Now has different contents
print('After b ', b)      # Same list, so same contents as a
```

    Before a ['a', 'b', 47, 11, 22, 14, 'h']
    Before b ['a', 'b', 47, 11, 22, 14, 'h']
    After a  [101, 102, 103]
    After b  [101, 102, 103]



```python
a = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
b = [1, 2, 3, 4, 5]
b[3:5] = a
print(a, b)

assert b[3:] == a and b[3:] is not a
```

    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h'] [1, 2, 3, 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

<br>

## Better way 12. 스트라이드와 슬라이스를 한 식에 함께 사용하지 말라

- 슬라이스에 시작, 끝, 증가값을 함께 지정하면 코드의 의미를 혼동하기 쉽다. 
- 시작이나 끝 인덱스가 없는 슬라이스를 만들 때는 양수 증가값을 사용하라. 가급적 음수 증가값은 피하라. 
- 한 슬라이스 내에서 시작, 끝, 증가값을 모두 써야 하는 경우, 두 번 대입을 사용(한 번은 스트라이딩, 한 번은 슬라이싱)하거나 itertools 내장 모듈의 `islice`를 사용하라.

앞서 살펴본 기본 슬라이싱 외에, 파이썬에서는 `리스트[시작:끝:증가값]`으로 일정한 간격을 두고 슬라이싱을 할 수 있는 특별한 구문을 제공한다. 이를 **스트라이드(stride)**라고 한다. 


```python
x = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']

x[::2]      # ['a', 'c', 'e', 'g']
x[::-2]     # ['h', 'f', 'd', 'b']
x[2::2]     # ['c', 'e', 'g']
x[-2::-2]   # ['g', 'e', 'c', 'a']
x[-2:2:-2]  # ['g', 'e']
x[2:2:-2]   # []
```




    []

<br>

하지만 슬라이싱 구문에 스트라이딩까지 들어가면 코드 밀도가 너무 높아서 읽기 어렵다. 게다가 증가값에 따라 시작값과 끝값이 어떤 역할을 하는지 불분명하다. 특히 증가값이 음수인 경우는 더 그렇다. 

따라서 시작이나 끝 인덱스와 함께 증가값을 사용해야 한다면, **스트라이딩한 결과를 변수에 대입한 다음 슬라이싱하라.**


```python
y = x[::2]   # ['a', 'c', 'e', 'g']
z = y[1:-1]  # ['c', 'e']
print(x)
print(y)
print(z)
```

    ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    ['a', 'c', 'e', 'g']
    ['c', 'e']

<br>

## Better way 13. 슬라이싱보다는 나머지를 모두 잡아내는 언패킹을 사용하라

- 언패킹 대입에 별표 식을 사용하면 언패킹 패턴에서 대입되지 않는 모든 부분을 리스트에 잡아낼 수 있다. 
- 별표 식은 언패킹 패턴의 어떤 위치에든 놓을 수 있다. 별표 식에 대입된 결과는 항상 리스트가 되며, 이 리스트에는 별표 식이 받은 값이 0개 또는 그 이상 들어간다. 
- 리스트를 서로 겹치지 않게 여러 조각으로 나눌 경우, 슬라이싱과 인덱승을 사용하기보다는 나머지를 모두 잡아내는 언패킹을 사용해야 실수할 여지가 훨씬 줄어든다. 

기본 언패킹의 한계점은 언패킹할 시퀀스의 길이를 미리 알고 있어야 한다는 것이다. 


```python
car_ages = [0, 9, 4, 8, 7, 20, 19, 1, 6, 15]
car_ages_descending = sorted(car_ages, reverse=True)
oldest, second_oldest = car_ages_descending
```


    ---------------------------------------------------------------------------
    
    ValueError                                Traceback (most recent call last)
    
    ~\AppData\Local\Temp/ipykernel_13300/2451885761.py in <module>
          1 car_ages = [0, 9, 4, 8, 7, 20, 19, 1, 6, 15]
          2 car_ages_descending = sorted(car_ages, reverse=True)
    ----> 3 oldest, second_oldest = car_ages_descending


    ValueError: too many values to unpack (expected 2)


위와 같은 코드에서 나머지 원소들을 가져오고 싶은 경우, 인덱스와 슬라이싱을 사용할 수 있다. 


```python
oldest = car_ages_descending[0]
second_oldest = car_ages_descending[1]
others = car_ages_descending[2:]
print(oldest, second_oldest, others)
```

    20 19 [15, 9, 8, 7, 6, 4, 1, 0]

그러나 위 코드는 시각적 잡음이 많다. 실제로 이런 식으로 시퀀스의 원소를 여러 하위 집합으로 나누면 1 차이 나는 인덱스로 인한 오류를 만들어 내기 쉽다. 

<br>

이를 위해 파이썬에서는 **별표 식(starred expression)** 을 사용해 모든 값을 담는 언패킹을 할 수 있게 지원한다. 이 구문을 사용하면 언패킹 패턴의 다른 부분에 들어가지 못하는 모든 값을 별이 붙은 부분에 다 담을 수 있다. 

다음은 별표 식을 사용하여 위 코드와 동일하게 동작하는 코드이다. 


```python
oldest, second_oldest, *others = car_ages_descending
print(oldest, second_oldest, others)
```

    20 19 [15, 9, 8, 7, 6, 4, 1, 0]


뿐만 아니라 별표 식을 다른 위치에도 사용할 수 있다. 


```python
oldest, *others, youngest = car_ages_descending
print(oldest, youngest, others)

*others, second_youngest, youngest = car_ages_descending
print(youngest, second_youngest, others)
```

    20 0 [19, 15, 9, 8, 7, 6, 4, 1]
    0 1 [20, 19, 15, 9, 8, 7, 6, 4]

<br>

별표 식이 포함된 언패킹 대입을 처리하려면 **필수인 부분이 적어도 하나는 있어야** 하고, 한 수준의 언패킹 패턴에 **별표 식을 두 개 이상 사용할 수 없다.**


```python
*others = car_ages_descending
```


      File "C:\Users\user\AppData\Local\Temp/ipykernel_13300/2422727027.py", line 1
        *others = car_ages_descending
        ^
    SyntaxError: starred assignment target must be in a list or tuple




```python
first, *middle, *second_middle, last = [1, 2, 3, 4]
```


      File "C:\Users\user\AppData\Local\Temp/ipykernel_13300/1187532336.py", line 1
        first, *middle, *second_middle, last = [1, 2, 3, 4]
        ^
    SyntaxError: multiple starred expressions in assignment



아래의 방식이 권장되지는 않지만, 아래와 같이 다른 수준에서는 별표 식을 각각 사용할 수 있다. 


```python
car_inventory = {
	'Downtown': ('Silver Shadow', 'Pinto', 'DMC'),
	'Airport': ('Skyline', 'Viper', 'Gremlin', 'Nova'),
}

((loc1, (best1, *rest1)),
 (loc2, (best2, *rest2))) = car_inventory.items()

print(f'Best at {loc1} is {best1}, {len(rest1)} others')
print(f'Best at {loc2} is {best2}, {len(rest2)} others')
```

    Best at Downtown is Silver Shadow, 2 others
    Best at Airport is Skyline, 3 others

<br>

별표 식은 항상 **list** 인스턴스가 된다. 언패킹하는 시퀀스에 남는 원소가 없으면 별표 식 부분은 빈 리스트가 된다. 

이러한 특징은 원소가 최소 N개 들어있다는 사실을 미리 아는 시퀀스를 처리할 때 유용하다. 

다만, 별표식을 사용하지 않아도 되는 짧은 시퀀스에 대해 별표식을 사용하는 것은 권장되지 않는다. 


```python
short_list = [1, 2]
first, second, *rest = short_list
print(first, second, rest)
```

    1 2 []

<br>

## Bettery way 14. 복잡한 기준을 사용해 정렬할 때는 key 파라미터를 사용하라

- 리스트 타입에 들어있는 sort 메서드를 사용하면 원소 타입이 문자열, 정수, 튜플 등과 같은 내장 타입인 경우 자연스러운 순서로 리스트의 원소를 정렬할 수 있다. 
- 원소 타입에 특별 메서드를 통해 자연스러운 순서가 정의돼 있지 않으면 sort 메서드를 쓸 수 없다. 하지만 원소 타입에 순서 특별 메서드르르 정의하는 경우는 드물다. 
- sort 메서드의 key 파라미터를 사용하면 리스트의 각 원소 대신 비교에 사용할 객체를 반환하는 도우미 함수를 제공할 수 있다. 
- key 함수에서 튜플을 반환하면 여러 정렬 기준을 하나로 엮을 수 있다. 단항 부호 반전 연산자를 사용하면 부호를 바꿀 수 있는 타입이 정렬 기준인 경우 정렬 순서를 반대로 바꿀 수 있다. 
- 부호를 바꿀 수 없는 타입의 경우, 여러 정렬 기준을 조합하려면 각 정렬 기준마다 reverse 값으로 정렬 순서를 지정하면서 sort 메서드를 여러 번 사용해야 한다. 이때 정렬 기준의 우선순위가 높아지는 순서로 sort를 호출한다. 

list 내장 타입에는 리스트의 원소를 여러 기준에 따라 정렬할 수 있는 sort 메서드가 들어있다. 

sort 메서드는 자연스럽게 순서를 정할 수 있는 내장 타입들(문자열, 정수, 부동소수점 수 등)에 대해 잘 작동한다. 하지만 우리는 실제로 **정렬에 사용하고 싶은 애트리뷰트가 객체에 들어있는 경우**가 더 많다. 

이런 상황을 지원하기 위해 sort에는 `key`라는 파라미터가 있으며, key는 함수여야 한다. key 함수가 반환하는 값은 원소 대신 정렬 기준으로 사용할, 비교 가능한(즉, 자연스러운 순서가 정의된) 값이어야만 한다. 


```python
# 클래스 정의 및 인스턴스 생성
class Tool:
    def __init__(self, name, weight):
        self.name = name
        self.weight = weight

    def __repr__(self):
        return f'Tool({self.name!r}, {self.weight})'

tools = [
    Tool('level', 3.5),
    Tool('hammer', 1.25),
    Tool('screwdriver', 0.5),
    Tool('chisel', 0.25),
]
```


```python
# key에 정렬 기준 지정
print('Unsorted:', tools)
tools.sort(key=lambda x: x.name)
print('\nSorted:  ', tools)
```

    Unsorted: [Tool('level', 3.5), Tool('hammer', 1.25), Tool('screwdriver', 0.5), Tool('chisel', 0.25)]
    
    Sorted:   [Tool('chisel', 0.25), Tool('hammer', 1.25), Tool('level', 3.5), Tool('screwdriver', 0.5)]

<br>

단순 애트리뷰트 뿐 아니라 추가적인 메서드 호출이나 식이 들어갈 수도 있다. 


```python
places = ['home', 'work', 'New York', 'Paris']
places.sort()
print('Case sensitive:  ', places)
places.sort(key=lambda x: x.lower()) # 소문자로 변경 후 정렬
print('Case insensitive:', places)
```

    Case sensitive:   ['New York', 'Paris', 'home', 'work']
    Case insensitive: ['home', 'New York', 'Paris', 'work']

<br>

때로는 여러 기준을 사용해 정렬해야 할 수도 있다. 

이런 경우에는 정렬에 사용할 애트리뷰트들을 tuple로 제공할 수 있다. tuple에서 앞에 올수록 우선순위가 높은 정렬 기준이다. 


```python
power_tools = [
    Tool('drill', 4),
    Tool('circular saw', 5),
    Tool('jackhammer', 40),
    Tool('sander', 4),
]

power_tools.sort(key=lambda x: (x.weight, x.name))
print(power_tools)
```

    [Tool('drill', 4), Tool('sander', 4), Tool('circular saw', 5), Tool('jackhammer', 40)]

<br>

sort 메서드에 `reverse` 파라미터를 넘기면 tuple에 들어있는 두 기준의 정렬 순서가 똑같이 영향을 받는다. 


```python
power_tools.sort(key=lambda x: (x.weight, x.name),
                 reverse=True)  # 모든 비교 기준을 내림차순으로 만든다. 
print(power_tools)
```

숫자 값의 경우 단항(unary) 부호 반전(-) 연산자를 사용해 정렬 방향을 혼합할 수 있다. 

부호 반전 연산자는 반환되는 튜플에 들어가는 값 중 하나의 부호를 반대로 만들기 때문에, 결과적으로 나머지 값의 정렬 순서는 그대로 둔 채로 반전된 값의 정렬 순서를 반대로 만든다. 


```python
power_tools.sort(key=lambda x: (-x.weight, x.name))
print(power_tools)
```

    [Tool('jackhammer', 40), Tool('circular saw', 5), Tool('drill', 4), Tool('sander', 4)]


아래 코드는 위 코드와 동일하게 동작한다. 


```python
power_tools.sort(key=lambda x: x.name)                 # 2 순위
power_tools.sort(key=lambda x: x.weight, reverse=True) # 1 순위
print(power_tools)
```

    [Tool('jackhammer', 40), Tool('circular saw', 5), Tool('drill', 4), Tool('sander', 4)]

