---
layout: single
title: "[Python] Effective Python CH 1. 파이썬답게 생각하기 - 1"
categories: ['Language', 'Python']
toc: true
toc_sticky: true
tag: []

---



# CH 1. 파이썬답게 생각하기 - 1

`Effective Python 2nd Edition`을 읽으며 학습한 내용들을 정리합니다. 

* 목차 선택하기: [Effective Python 전체 목차](https://wowo0709.github.io/language/python/Python-Effective-Python-%EC%A0%84%EC%B2%B4-%EB%AA%A9%EC%B0%A8/)
* 소스 코드: [wowo0709/Effective-Python](https://github.com/wowo0709/Effective-Python)

해당 포스팅은 [Chapter 1. 파이썬답게 생각하기]의 첫번째 포스팅입니다. 

## Better way 1. 사용 중인 파이썬의 버전을 알아두라. 

- 파이썬 3는 파이썬 최신 버전이며 현재 가장 잘 지원되고 있다. 따라서 프로젝트에서 파이썬 3를 써야 한다. 
- 현재 사용 중인 파이썬의 버전이 내가 원하는 버전인지 확인하라. 

**커맨드 라인**


```python
!python --version
```

    Python 3.9.7


**`sys` 내장 모듈 사용**


```python
import sys
print(sys.version_info)
print(sys.version)
```

    sys.version_info(major=3, minor=9, micro=7, releaselevel='final', serial=0)
    3.9.7 (default, Sep 16 2021, 16:59:28) [MSC v.1916 64 bit (AMD64)]

<br>

## Better way 2. PEP 8 스타일 가이드를 따르라

- PEP(Python Enhancement Proposal) : 파이썬 코드를 어떤 식으로 작성할 지 알려주는 스타일 가이드

**공백**

- 들여쓰기는 탭 대신 스페이스(4칸) 사용
- 라인 길이는 79개 문자 이하
- 하나의 식을 다음 줄에 이어서 쓸 경우에는 일반적인 들여쓰기보다 4칸 더(총 8칸) 들여쓴다. 
- 파일 안에서 각 함수와 클래스 사이에는 빈 줄 2 줄 사용
- 클래스 안에서 각 메서드 사이에는 빈 줄 1줄 사용
- 변수 대입 시 `=` 전후에는 공백 1 칸만 사용
- 콜론(`:`) 사용 시 전에는 공백 없이, 후에는 공백 1칸 사용

**명명 규약**

- 함수, 변수, 애트리뷰트(속성)는 `lowercase_underscore`
- 보호되어야 하는 애트리뷰트는 `_leading_underscore`
- 비공개(private) 애트리뷰트는 앞에 밑줄 2개 사용 `__leading_underscore`
- 클래스는 각 단어의 첫 글자를 대문자로 `CapitalizedWord`
- 모듈 수준의 상수는 모든 글자를 대문자로 하고 글자 사이를 밑줄로 연결 `ALL_CAPS`
- 인스턴스 메서드는 첫번째 인자로 `self` 사용
- 클래스 메서드는 첫번째 인자로 `cls` 사용

**식과 문**

- 긍정문의 부정(`if not a is b`) 대신 부정문의 긍정(`if a is not b`)을 사용
- 빈 객체를 검사할 때는 길이(`if len(컨테이너) == 0`) 대신 False로 취급된다는 사실을 이용(`if not 컨테이너`)해라
- 마찬가지로 비어 있지 않은 객체를 검사할 때에도 길이(`if len(컨테이너) > 0`) 대신 True로 취급된다는 사실을 이용(`if 컨테이너`)해라
- 한 줄 짜리 if/for/while/except 문을 사용하지 마라. 명확성을 위해 각 부분을 여러 줄에 나눠 작성해라. 
- 식을 한 줄 안에 다 쓸 수 없는 경우, 식을 괄호로 묶고 줄바꿈과 들여쓰기를 이용해 읽기 쉽게 하라
- 여러 줄에 걸쳐 식을 쓸 때는 `\` 문자보다는 괄호(`()`)를 사용하라

**임포트**

- import 문은 항상 파일 맨 앞에 위치
- 모듈을 임포트 할 때는 항상 절대 경로(from bar import foo)를 사용하라
- 반드시 상대 경로로 임포트해야 하는 경우 `from . import foo` 처럼 명시적인 구문을 사용
- 임포트를 적을 때는 표준 라이브러리 모듈, 서드 파티 모듈, 직접 만든 모듈 순서로 섹션을 나눈다. 각 섹션에서는 알파벳 순서로 임포트 문을 적는다. 

<br>

## Better way 3. bytes와 str의 차이를 알아두라

- bytes에는 8비트 값의 시퀀스가 들어 있고, str에는 유니코드 코드 포인트의 시퀀스가 들어 있다. 
- bytes와 str 인스턴스를 (<, >, ==, +, % 와 같은) 연산자에 섞어서 사용할 수 없다. 
- 이진 데이터를 파일에서 읽고 쓸 때는 항상 이진 모드('rb', 'wb')로 파일을 열어라
  - 유니코드 데이터를 파일에서 읽거나 파일에 쓰고 싶을 때는 시스템 디폴트 인코딩에 주의하라. 인코딩을 명시하기 위해서 open의 encoding 파라미터를 지정하라. 

**bytes VS str**

- `bytes`에서는 부호가 없는 8바이트 데이터가 그대로 들어간다. (종종 ASCII code)
- `str`에서는 사람이 사용하는 문자인 유니코드 code point가 들어가 있다. 


```python
# bytes
a = b'h\x65llo'
print(list(a))
print(a)
```

    [104, 101, 108, 108, 111]
    b'hello'



```python
a = 'a\u0300 propos'
print(list(a))
print(a)
```

    ['a', '̀', ' ', 'p', 'r', 'o', 'p', 'o', 's']
    à propos

<br>

**두 자료형은 호환되지 않는다.**

- str 인스턴스에는 직접 대응하는 이진 인코딩이 없고, bytes에는 직접 대응하는 텍스트 인코딩이 없다. 
  - `문자열.encode`: 유니코드 데이터 -> 이진 데이터
  - `바이트.decode`: 이진 데이터 -> 유니코드 데이터


```python
s = 'abcd'
s_bytes = s.encode()
print(type(s_bytes), s_bytes)

s_str = s_bytes.decode()
print(type(s_str), s_str)

print(s == s_str)
```

    <class 'bytes'> b'abcd'
    <class 'str'> abcd
    True


* 두 자료형 간 연산이 허용되지 않는다. 


```python
print(b'one' + b'two')
print('one' + 'two')
print(b'one' + 'two')
```

    b'onetwo'
    onetwo



    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    ~\AppData\Local\Temp/ipykernel_16088/115322005.py in <module>
          1 print(b'one' + b'two')
          2 print('one' + 'two')
    ----> 3 print(b'one' + 'two')


    TypeError: can't concat str to bytes



```python
print(b'one' < b'two')
print('one' < 'two')
print(b'one' < 'two')
```

    True
    True



    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    ~\AppData\Local\Temp/ipykernel_16088/3862181309.py in <module>
          1 print(b'one' < b'two')
          2 print('one' < 'two')
    ----> 3 print(b'one' < 'two')


    TypeError: '<' not supported between instances of 'bytes' and 'str'


* 내부에 동일한 문자들이 들어있더라도 str과 bytes 인스턴스를 비교하면 항상 False가 나온다. 


```python
print(b'foo' == 'foo')
```

    False

<br>

**파일 입출력 시 주의하라**

- 텍스트 쓰기/읽기 모드('w'/'r')와 이진 쓰기/읽기 모드('wb'/'rb')는 구분된다. 


```python
with open('data.bin', 'w') as f:
    f.write(b'\xf1\xf2\xf3\xf4\xf5')
```


    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    ~\AppData\Local\Temp/ipykernel_16088/1307206295.py in <module>
          1 with open('data.bin', 'w') as f:
    ----> 2     f.write(b'\xf1\xf2\xf3\xf4\xf5')


    TypeError: write() argument must be str, not bytes



```python
with open('data.bin', 'wb') as f:
    f.write(b'\xf1\xf2\xf3\xf4\xf5')
```


```python
with open('data.bin', 'r') as f:
    data = f.read()
```


    ---------------------------------------------------------------------------
    
    UnicodeDecodeError                        Traceback (most recent call last)
    
    ~\AppData\Local\Temp/ipykernel_16088/800754018.py in <module>
          1 with open('data.bin', 'r') as f:
    ----> 2     data = f.read()


    UnicodeDecodeError: 'cp949' codec can't decode byte 0xf5 in position 4: incomplete multibyte sequence



```python
with open('data.bin', 'rb') as f:
    data = f.read()

print(data == b'\xf1\xf2\xf3\xf4\xf5')
```

    True

<br>

## Bettery way 4. C 스타일 형식 문자열을 str.format과 쓰기보다는 f-문자열을 통한 인터폴레이션을 사용하라

- % 연산자를 사용하는 C 스타일 문자열은 여러가지 단점과 번잡성이라는 문제가 있다. 
- str.format 메서드는 유용한 추가 기능들을 제공하지만, 여전히 C 스타일 문자열의 문제점을 그대로 가지고 있다. 
- f-문자열은 값을 문자열 안에 넣는 새로운 구문으로, 간결하면서도 기존 문제점을 해결하고 기능은 그대로 제공한다. 

**`%`를 사용하는 C 스타일 formatting**

- C에서 시작된 포맷팅 방식으로, 익숙하게 대부분의 언어들에서 통용된다는 것이 장점
- tuple과 dictionary를 사용할 수 있으며, 4가지 불편함 존재.
- tuple style: % 앞에는 문자열 형식, 뒤에는 튜플을 사용함.

  1. tuple 내 변수들의 순서를 마음대로 바꿀 수 없다. 
  2. Formatting 시 변수의 값을 살짝 바꿔주고 싶을 때, 복잡하고 가독성이 떨어진다. 
  3. 같은 값을 여러 번 사용하고 싶다면 tuple에서 같은 값을 여러 번 반복해야 한다. 

- dictionary style: % 앞에는 문자열 형식, 뒤에는 딕셔너리를 사용함. 1번과 3번 불편함을 해소할 수 있음. 

  4. Dictionary 사용 시 문장이 길어지고, 번잡스러워진다. 


```python
# tuple style 예시
key = 'my_var'
value = 1.234
formatted = '%-10s = %.2f' % (key, value)
print(formatted)
```

    my_var     = 1.23



```python
# dictionary style 예시
name = 'Max'

template = '%s loves food. See %s cook.'
before = template % (name, name)   # Tuple

template = '%(name)s loves food. See %(name)s cook.'
after = template % {'name': name}  # Dictionary

assert before == after
```


```python
# 값을 바꿀 때 가독성이 매우 떨어짐
pantry = [
    ('avocados', 1.25),
    ('bananas', 2.5),
    ('cherries', 15),
]

for i, (item, count) in enumerate(pantry):
    before = '#%d: %-10s = %d' % (
        i + 1,
        item.title(),
        round(count))

    after = '#%(loop)d: %(item)-10s = %(count)d' % {
        'loop': i + 1,
        'item': item.title(),
        'count': round(count),
    }

    assert before == after
```

<br>

**내장함수 `format`과 `str.format`**

- 오래된 스타일의 C 스타일 문자열보다 더 표현력이 좋은 고급 문자열 formatting 기능
- 하지만 여전히 2번째 문제점(formatting 시 값의 변경이 필요한 경우 코드가 복잡해지는 문제)을 갖고 있음


```python
# format 예시
a = 1234.5678
formatted = format(a, ',.2f')
print(formatted)

b = 'my string'
formatted = format(b, '^20s')
print('*', formatted, '*')
```

    1,234.57
    *      my string       *



```python
# str.format 예시
key = 'my_var'
value = 1.234

formatted = '{:<10} = {:.2f}'.format(key, value) # help('FORMATTING')
print(formatted)

formatted = '{1} = {0}'.format(key, value)
print(formatted)
```

    my_var     = 1.23
    1.234 = my_var

<br>

**Interpolation을 통한 formatting**

- 짧게 `f-문자열`이라고 부름
- C-style formatting과 format 내장 함수를 사용하는 방법은 f-문자열의 동작과 유용성을 이해하는 데 도움을 주는 역사적인 유물로 간주해야 한다!
- formatting 시 파이썬 영역에서 사용할 수 있는 모든 이름을 자유롭게 참조할 수 있도록 허용함으로써 간결함을 제공


```python
# f-string 예시
key = 'my_var'
value = 1.234

formatted = f'{key:<10} = {value:.2f}'
print(formatted)
```

    'my_var    ' = 1.23

<br>

**C-style VS format 함수 VS f-string 비교**


```python
f_string = f'{key:<10} = {value:.2f}'

c_tuple  = '%-10s = %.2f' % (key, value)

str_args = '{:<10} = {:.2f}'.format(key, value)

str_kw   = '{key:<10} = {value:.2f}'.format(key=key, value=value)

c_dict   = '%(key)-10s = %(value).2f' % {'key': key, 'value': value}
```


```python
for i, (item, count) in enumerate(pantry):
    old_style = '#%d: %-10s = %d' % (
        i + 1,
        item.title(),
        round(count))

    new_style = '#{}: {:<10s} = {}'.format(
        i + 1,
        item.title(),
        round(count))

    f_string = f'#{i+1}: {item.title():<10s} = {round(count)}'

    assert old_style == new_style == f_string
```

<br>

## Better way 5. 복잡한 식을 쓰는 대신 도우미 함수를 작성하라

- 파이썬 문법을 사용하면 아주 복잡하고 읽기 어려운 한 줄짜리 식을 작성할 수 있다. 
- 복잡한 식은 도우미 함수로 작성하라. 특히 같은 로직을 2번 이상 반복해 사용할 때는 도우미 함수를 꼭 사용해라. 
- boolean 연산자 or나 and를 사용하는 것보다 if/else 식을 쓰는 것이 가독성이 더 좋다. 


```python
from urllib.parse import parse_qs

my_values = parse_qs('red=5&blue=0&green=',
                     keep_blank_values=True)
print(repr(my_values))
```

    {'red': ['5'], 'blue': ['0'], 'green': ['']}

<br>

파라미터가 없거나 비어 있을 경우 0이 디폴트 값으로 대입되도록 해보자. 


```python
# or 연산자 사용
red = my_values.get('red', [''])[0] or 0
green = my_values.get('green', [''])[0] or 0
opacity = my_values.get('opacity', [''])[0] or 0

print(f'Red:     {red}')
print(f'Green:   {green}')
print(f'Opacity: {opacity}')
```

    Red:     5
    Green:   0
    Opacity: 0



```python
# if/else 문 사용
red_str = my_values.get('red', [''])
red = int(red_str[0]) if red_str[0] else 0
green_str = my_values.get('green', [''])
green = int(green_str[0]) if green_str[0] else 0
opacity_str = my_values.get('opacity', [''])
opacity = int(opacity_str[0]) if opacity_str[0] else 0

print(f'Red:     {red}')
print(f'Green:   {green}')
print(f'Opacity: {opacity}')
```

    Red:     5
    Green:   0
    Opacity: 0



```python
# 도우미 함수 작성
def get_first_int(values, key, default=0):
    found = values.get(key, [''])
    if found[0]:
        return int(found[0])
    return default

red = get_first_int(my_values, 'red')
green = get_first_int(my_values, 'green')
opacity = get_first_int(my_values, 'opacity')

print(f'Red:     {red}')
print(f'Green:   {green}')
print(f'Opacity: {opacity}')
```

    Red:     5
    Green:   0
    Opacity: 0

