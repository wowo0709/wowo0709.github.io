---
layout: single
title: "[AITech] 20220107 - Python Basics"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 강의 복습 내용

### 1. File System & Terminal Basic

**컴퓨터 OS**

* `Operating System(OS, 운영체제)`: 프로그램이 동작할 수 있는 구동 환경. SW 프로그램과 HW를 연결하는 매개체. 
* 프로그램은 OS에 의존적이다. 
  * 파이썬은 OS에 의존적이지 않다. (인터프리터 언어)

**파일 시스템**

* OS에서 파일을 저장하는 **트리 구조** 저장 체계
* 파일의 기본 체계 - 파일 VS 디렉토리
  * 디렉토리: 폴더 또는 디렉토리라 불리며 파일과 다른 디렉토리를 포함할 수 있음
  * 파일: 컴퓨터에서 정보를 저장하는 논리적인 단위, 파일은 파일 명과 확장자로 식별됨
* 경로 - 컴퓨터 파일의 고유한 위치
  * 절대 경로: 루트 디렉토리부터 파일위치까지의 경로
  * 상대 경로: 현재 있는 디렉토리부터 타깃 파일까지의 경로
  * 윈도우: '\\' 사용/ 리눅스, 맥: '/' 사용

**터미널**

* `터미널`: 마우스가 아닌 키보드로 명령을 입력하여 프로그램을 실행(Command Line Interface, CLI)

  * Windows: cmd, windows terminal, cmder, ubuntu 등
  * Mac, Linux: Terminal

* 기본 명령어

  * 각 터미널에서는 프로그램을 작동하는 shell이 존재
  * 각 shell마다 다른 명령어를 사용

  | 윈도우 cmd창 명령어 | shell 명령어 | 설명                                                         |
  | ------------------- | ------------ | ------------------------------------------------------------ |
  | cd                  | cd           | 현재 디렉터리 이름을 보여주거나 바꿉니다. (change directory) |
  | cls                 | clear        | cmd 화면에 표시된 것을 모두 지웁니다.                        |
  | dopy                | cp           | 하나 이상의 파일을 다른 위치로 복사합니다.                   |
  | del                 | rm           | 하나 이상의 파일을 지웁니다.                                 |
  | dir                 | ls           | 디렉터리에 있는 파일과 하위 디렉터리 목록을 보여줍니다.      |

<br>

### 2. Python Overview

**Python의 시작**

* 플랫폼 독립적(인터프리터 언어)

  * 플랫폼 = OS

    * 윈도우, 리눅스, 안드로이드 등 프로그램이 실행되는 운영 체제를 **플랫폼**이라 함

  * 독립적인 = 상관 없는

    * OS에 상관없이 한 번 프로그램을 작성하면 어디서든 사용 가능

  * 인터프리터 = 통역기를 사용하는 언어

    * 소스 코드를 바로 실행할 수 있게 지원하는 프로그램 실행 방법

  * 컴파일러 VS 인터프리터

    |           | 컴파일러 언어                                                | 인터프리터 언어                                              |
    | --------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
    | 작동방식  | 소스코드를 기계어로 먼저 번역. <br />해당 플랫폼에 최적화되어 프로그램을 실행 | 별도의 번역과정 없이 소스코드를 실행 시점에 해석하여<br />컴퓨터가 처리할 수 있도록 함 |
    | 장점/단점 | 실행 속도가 빠름/ 한 번에 많은 메모리가 필요                 | 간단히 작성, 메모리가 적게 필요/ 실행 속도가 느림            |
    | 주요 언어 | C, Java, C++, C#                                             | 파이썬, 스칼라, 자바스크립트                                 |

* 객체 지향 동적 타이핑 언어

  * 객체 지향적 언어: **실행 순서가 아닌 단위 모듈(객체) 중심**으로 프로그램을 작성
  * 동적 타이핑 언어: 프로그램이 **실행하는 시점**에 프로그램이 사용해야 할 **데이터에 대한 타입을 결정**

* 처음 C언어로 구현

**Python의 특징(Why Python?)**

* 쉽고 간단하며 다양하다
* 이해하기 쉬운 문법
* 다양한 라이브러리
* 이미 널리 쓰이는 어디서든 쓸 수 있는 언어

> *Life is short. You need Python.*

<br>

### 3. Jupyter & Colab

**한 번 합쳐보자! Python Shell + 코드 편집도구**

**Jupyter 개요**

* IPython 커널을 기반으로 한 대화형 파이썬 shell
* 일반적인 터미널 shell + 웹 기반 데이터 분석 Notebook 제공
* 미디어, 텍스트, 코드, 수식 등을 하나의 문서로 표현 가능
* 사실 상의 데이터 분석 Interactive Shell의 표준
* **Ju**lia + **Pyt**hon + **R**
* http://localhost:8888/tree 주소로 jupyter가 실행

**Colab 개요**

* 구글이 개발한 클라우드 기반의 jupyter notebook
* 구글 드라이브 + GCP + jupyter 등이 합쳐져서 사용자가 손쉽게 접근
* 초반 여러가지 모듈 설치의 장점을 가짐
* 구글 드라이브 파일을 업로드하여 사용가능한 장점을 가짐
* VSCode 등과 연결해서 사용 가능
* V100 이상의 GPU를 무료로 쓸 수 있다는 장점을 가짐

**주요 단축 키**

| Actions                 | Colab        | Jupyter      |
| ----------------------- | ------------ | ------------ |
| 키보드 단축키           | Ctrl/Cmd M H | H            |
| 위에 셀 삽입            | Ctrl/Cmd M A | A            |
| 아래에 셀 삽입          | Ctrl/Cmd M B | B            |
| 셀 삭제                 | Ctrl/Cmd M D | DD           |
| 실행 인터럽트           | Ctrl/Cmd M I | II           |
| 코드 셀로 변환          | Ctrl/Cmd M Y | Y            |
| 텍스트 셀로 변환        | Ctrl/Cmd M M | M            |
| 커서를 기준으로 셀 분할 | Ctrl/Cmd M - | Ctrl Shift - |
| 아레 셀과 병합          |              | Shift M      |



<br>

### 4. Python: Variable & List

**변수**

`변수`는 데이터(값)를 저장하기 위한 **메모리 공간**의 프로그래밍 상 이름이다. 

변수는 **메모리 주소**를 가지고 있고 변수에 들어가는 **값**은 **메모리 주소**에 할당된다. 

* 변수 - 프로그램에서 사용하기 위한 특정한 값을 저장하는 공간
  * 선언되는 순간 **메모리 특정 영역에 물리적인 공간**이 할당
  * 변수에는 값이 할당되고 해당 값은 메모리에 저장

**데이터 형 변환**

* `int()` 함수 사용 시 소수점 이하 내림
* 문자열과 정수/실수 간 변환도 가능
  * 단, int() 사용 시 문자열은 정수여야 함(float() 사용 시 문자열은 정수/실수 모두 가능)

**리스트**

* 동시에 여러 타입의 데이터 저장 가능

```python
a = [5,4,3,2,1]
b = [1,2,3,4,5]
b = a # b가 a의 주소를 가리킴
print(b)
# [5,4,3,2,1]
a.sort() # a 정렬
print(b) # b는 a와 같은 주소를 가리킴
# [1,2,3,4,5]
b = [6,7,8,9,10] # 새로운 주소 할당
print(a,b)
# [1,2,3,4,5] [6,7,8,9,10]
```

* "="는 메모리 주소에 해당 값을 할당(연결)한다는 의미이다. 
* 리스트 복사
  * 일차원 리스트 복사
    1. 슬라이싱: a = b[:]
    2. 리스트 함수: a = list(b)
    3. copy 함수: a = b.copy()
  * 이차원 리스트 복사
    * from copy import deepcopy

<br>

### 5. Function and Console I/O

**function**

* parameter VS argument
  * parameter: 함수의 입력 값 인터페이스
  * argument: 실제 parameter에 대입된 값(호출 시 전달하는 값)

**console in/out**

* print formatting

```python
# 1. % string
print("I ate %d apples. I was sick for %s days."%(number, day))
# %5d: 전체 5칸 할당, 오른쪽 정렬. %8.2f: 전체 8칸 할당, 소수점 이하 2자리 출력. 

# 2. str.format()
print("My name is {0} and {1} years old.".format(name, age))
# {0:5d}, {1:8.2f}와 같이 포맷팅. 

# 3. f-string(요즘 대세!!!)
name = "Youngwoo"
age = 39

print(f"Hello, {name}. You are {age}.")
print(f'{name:20}')
print(f'{name:>20}')
print(f'{name:*<20}')
print(f'{name:*>20}')
print(f'{name:^20}')
print(f'{name:*^20}')
'''
Hello, Youngwoo. You are 39.
Youngwoo            
            Youngwoo
Youngwoo************
************Youngwoo
      Youngwoo      
******Youngwoo******
'''
number = 3.1415926
print(f'{number:.2f}')
'''
3.14
'''
```

<br>

### 6. Conditionals and Loops

**조건 판단 방법**

* x > < == != >= <= y
  * 값을 비교
* x is/ is not y
  * 주소를 비교
* 숫자형의 경우 수학에서의 참/거짓과 동일
  * 0: 거짓
  * 그 외: 참
* 문자열도 동일
  * "": 거짓
  * "abc": 참

**삼항 연산자**

`is_even = True if value % 2 == 0 else False`

**반복의 제어 - else**

break에 걸리지 않고 정상적으로 종료했을 경우에만 else문 실행

```python
# for-else
for i in range(10):
    print(i,)
else:
    print("EOP")
    
# while-else
i = 0
while i < 10:
    print(i,)
    i += 1
else:
    print("EOP")
```

**debugging**

* 코드의 오류를 발견하여 수정하는 과정

* **문법적 에러**를 찾기 위한 에러 메시지 분석

  * 들여쓰기, 오탈자, 대소문자 구분 등
  * 에러 발생 시 인터프리터가 알려줌

* **논리적 에러**를 찾기 위한 테스트도 중요

  * 뜻대로 실행이 안 되는 코드
  * 중간 중간 print문을 찍어서 확인하는 습관

  ```python
  def addition(x, y):
      return x+y
  
  def multiplication(x, y):
      return x*y
  
  def divided_by_2(x):
      return x/2
  
  # __name__ 코드 부분은 '다른 모듈에서 import 시 실행되지 않고', '해당 모듈을 직접 컴파일할 때만 실행됨'
  if __name__ == '__main__':
      print(addition(10,5))
      print(multiplication(10,5))
      print(divided_by_2(50))
      
  '''
  ex) __name__ = "__main__"
  - 있을 경우
  >>> import trapezium_area_test
  >>>
  - 없을 경우
  >>> import trapezium_area_test
  addition: 15
  multiplication: 50
  divided_by_2: 25.0
  '''
  ```

<br>

### 7. String and Advances function concept

**함수 호출 방식 개요**

* 값에 의한 호출(Call by Value)
  * 함수에 인자를 넘길 때 값만 넘김. 
  * 함수 내에 인자값을 변경해도 호출자에게 영향을 주지 않음. 
* 참조에 의한 호출(Call by Reference)
  * 함수에 인자를 넘길 때 메모리 주소(변수)를 넘김. 
  * 함수 내에 인자 값 변경 시 호출자의 값도 변경됨. 
* 객체 참조에 의한 호출(Call by Object Reference)

```python
# 값만 변경
def swap_value(x,y):
    temp = x
    x = y
    y = temp
    return x,y
    
# 메모리가 가지는 값을 변경
def swap_reference(list_ex, offset_x, offset_y):
    temp = list_ex[offset_x]
    list_ex[offset_x] = list_ex[offset_y]
    list_ex[offset_y] = temp
```

**변수의 범위**

* 지역 변수: 함수 내에서만 사용
* 전역 변수: 프로그램 전체에서 사용

```python
def test(t):
    t = 20
    print("In Function: ", t)

x = 10
print("Before: ", x)
test(x)
print("After: ", x)
'''
Before:  10
In Function:  20
After:  10
'''
```

* 전역변수는 함수에서 사용 가능
* But, 함수 내에 전역 변수와 같은 이름의 변수를 선언하면 새로운 지역 변수가 생김

```python
# 전역 변수
def f():
    global s
    s = "I love London!"
    print(s)
    
s = "I love Paris!"
f()
print(s)
'''
I love London!
I love London!
'''

# 지역 변수
def f():
    s = "I love London!"
    print(s)
    
s = "I love Paris!"
f()
print(s)
'''
I love London!
I love Paris!
'''
```

**function type hints**

* 사용자에게 인터페이스를 명확히 알려준다. 
* 함수의 문서화 시 parameter에 대한 정보를 명확히 알 수 있다. 
* mypy 또는 IDE, linter 등을 통해 코드의 발생 가능한 오류를 사전에 확인
* 시스템 전체적인 안정성을 확보한다. 

```python
def type_hint_example(name: str) -> str:
    return f"Hello, {name}"
```

**docstring**

* 파이썬 함수에 대한 상세스펙을 사전에 작성 -> 함수 사용자의 이행도 UP
* 세 개의 따옴표로 docstring 영역 표시(함수명 아래)

```python
def add_binary(a, b):
    '''
    Returns the sum of two decimal numbers in binary digits. 
    
            Parameters:
                    a (int): A decimal integer
                    b (int): Another decimal integer
                
            Returns:
                    binary_sum (str): Binary string of the sum of a and b
    '''
    binary_sum = bin(a+b)[2:]
    return binary_sum

print(add_binary.__doc__)
'''
    Returns the sum of two decimal numbers in binary digits. 
    
            Parameters:
                    a (int): A decimal integer
                    b (int): Another decimal integer
                
            Returns:
                    binary_sum (str): Binary string of the sum of a and b
'''
```

**파이썬 코딩 컨벤션**

* **"flake8" 모듈**로 체크 - flake8 <파일명>
  * conda install -c anaconda flake8
* 최근에는 **"black" 모듈**을 활용하여 pep8 like 수준을 준수
  * black codename.py 명령을 사용

<br>

<br>
