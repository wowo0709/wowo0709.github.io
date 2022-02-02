---
layout: single
title: "[AITech] 20220118 - Python object-oriented programming"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Charateristics', 'Decoration']
---



<br>

## 강의 복습 내용

### Python object-oriented programming

만들어 놓은 코드를 **재사용**하고 싶다!!

* 객체: 실생활에서 일종의 물건
  * **속성**과 **행동**을 가짐
* OOP는 이러한 객체 개념을 프로그램으로 표현
  * **속성은 변수, 행동은 함수**로 표현
* OOP는 설계도에 해당하는 **클래스**와 실제 구현체인 **인스턴스**로 나뉨

```python
class SoccerPlayer(object):
    def __init__(self, name, position, back_number): # initializing function
        self.name = name
        self.position = position
        self.back_number = back_number
        
    def __str__(self): # majic function(mangling)
        return "Hello, My name is %s. I play in %s in center "% \
        (self.name, self.position)
    
jinhyun = SoccerPlayer("Jinhyun", "MF", 10)
print(jinhyun)
# Hello, My name is Jinhyun. I play in MF in center 
```

#### **OOP characteristics**

* Inheritance
* Polymorphism

```python
class Animal:
    def __init__(self, name):
        self.name = name
        
    def talk(self):
        return "None"
        
class Cat(Animal): # 상속
    def talk(self): # 재정의
        return "Meow!"
    
class Dog(Animal): # 상속
    def talk(self): # 재정의
        return "Woof! Woof!"
    
    
animals = [Animal('Missy'), Cat('Ruby'), Dog('Lassie')]

for animal in animals:
    print(animal.name + ":" + animal.talk())
    
'''
Missy:None
Ruby:Meow!
Lassie:Woof! Woof!
'''
```



* Visibility
  * 객체의 정보를 볼 수 있는 레벨을 조절하는 것

```python
class Inventory(object):
    def __init__(self):
        self.__items = [] # private 변수로 선언, 타객체가 접근하지 못 함
        
    @property # property decorator: 숨겨진 변수를 반환하게 해 줌
    def items(self):
        return self.__items
    	# return deepcopy(self.__items) -> 주로 복사체를 리턴해 줌
        
...
items = my_inventory.items # property decorator로 함수를 변수처럼 호출
```

#### **Decorate**

* first-class objects
  * 일등함수 또는 일급객체
  * 변수나 데이터 구조에 할당이 가능한 객체
  * 파라미터로 전달이 가능 + 리턴 값으로 사용 가능
  * **파이썬의 함수는 모두 일급함수**

```python
def formula(method, argument_list):
    return [method(value) for value in argument_list]
```

* Inner function
  * 함수 안의 또 다른 함수
  * **closures: inner function을 return 값으로 반환**
    * 같은 이름으로 다양한 함수 사용 가능

```python
def print_msg(msg):
    def printer():
        print(msg)
    return printer

another = print_msg("Hello, Python")
another()
# Hello, Python
```

* Decorator function
  * 복잡한 클로저 함수를 간단하게!

```python
def star(func): # func = percent
    def inner(*args, **kwargs):
        print("*"*30)
        func(*args, **kwargs) 
        print(func)
        print("*"*30)
    return inner

def percent(func): # func = printer
    def inner(*args, **kwargs): # *args = msg
        print("%"*30)
        func(*args, **kwargs) 
        print(func)
        print("%"*30)
    return inner

@star
@percent
def printer(msg):
    print(msg)
printer("Hello")

'''
******************************
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Hello
<function printer at 0x000001EDC28B69D0>
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
<function percent.<locals>.inner at 0x000001EDBF8BD1F0>
******************************
'''
```

* Decorator function with argument
  * 데코레이터 함수에 인자를 전달하려면 wrapper 함수가 필요하다. 

```python
def generate_power(exponent): # 2
    print(exponent)
    def wrapper(f): # f = raise_two
        print(f)
        def inner(*args): # *args = n = 7
            print(args)
            result = f(*args)
            return exponent**result
        return inner
    return wrapper

@generate_power(2)
def raise_two(n):
    return n**2

print(raise_two(7))

'''
2
<function raise_two at 0x000001EDBFA39790>
(7,)
562949953421312
'''
```

**이해하기 좀 어렵지만 데코레이터 함수 호출 순서는 `(데코레이터 인자 -> 다음 데코레이터 함수)... -> 원 함수 -> 원 함수 인자` 순이라 생각하면 될 듯..!!**

<br>. 
