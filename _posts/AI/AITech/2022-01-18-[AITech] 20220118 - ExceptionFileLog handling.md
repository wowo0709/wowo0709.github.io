---
layout: single
title: "[AITech] 20220118 - Exception/File/Log handling"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

## 강의 복습 내용

### Exception/File/Log handling

#### Exception Handling

`Exception`은 **예상이 가능한 예외**와 **예상이 불가능한 예외**로 구분할 수 있다. 

* 예상 가능한 예외
  * 발생 여부를 사전에 인지할 수 있는 예외
  * 사용자의 잘못된 입력, 파일 호출 시 파일 없음 등
  * 개발자가 반드시 명시적으로 정의해야 함
* 예상 불가능한 예외
  * 인터프리터 과정에서 발생하는 예외
  * 리스트의 범위를 넘어가는 값 호출 등

* try~except

```python
for i in range(10):
    try:
    	print(10 / i)
    except ZeroDivisionError as e:
        print(e)
        print("Not divided by 0")
```

* try\~except\~else
  * Exception이 발생하지 않은 경우에만 else문 실행

```python
for i in range(10):
    try:
        result = 10 / i
    except ZeroDivisionError:
        print("Not divided by 0")
    else:
        print(10 / i)
```

* try\~except\~finally
  * 예외 발생과 상관없이 무조건 finally 문 실행

```python
try:
    for i in range(1, 10):
        result = 10 // i
        print(result)
except ZeroDivisionError:
    print("Not divided by 0")
finally:
    print("종료되었습니다.")
```

* raise
  * 필요에 따라 강제 Exception 발생

```python
while True:
    value = input("변환할 정수 값을 입력해주세요")
    for digit in value:
        if digit not in "0123456789":
            raise ValueError("숫자값을 입력하지않으셨습니다")

print("정수값으로 변환된 숫자 -", int(value))
```

* assert
  * 특정 조건에 만족하지 않을 경우 예외 발생

```python
def get_binary_nmubmer(decimal_number):
    assert isinstance(decimal_number, int)
    return bin(decimal_number)

print(get_binary_nmubmer(10))
```

<br>

#### File Handling

* 기본적인 파일 종류로 text 파일과 binary 파일로 나눔
  * Binary 파일
    * 컴퓨터만 이해할 수 있는 형태인 이진법 형식으로 저장된 파일
    * 일반적으로 메모장으로 열면 내용이 깨져 보임
    * 엑셀파일, 워드파일 등
  * Text 파일
    * 인간도 이해할 수 있는 형태인 문자열 형식으로 저장된 파일
    * 메모장으로 열면 내용 확인 가능
    * 메모장에 저장된 파일, HTML 파일, 파이썬 코드 파일 등
* 컴퓨터는 text 파일을 처리하기 위해 binary 파일로 변환시킴
* 모든 text 파일도 실제로는 binary 파일

* 파일 읽기

```python
# 파일 읽기
f = open("i_have_a_dream.txt", "r" )
contents = f.read()
print(contents)
f.close()
# or
with open("i_have_a_dream.txt", "r") as my_file:
    contents = my_file.read()
    print (type(contents), contents)
    
# 한 줄씩 읽기
with open("i_have_a_dream.txt", "r") as my_file:
    content_list = my_file.readlines() #파일 전체를 list로 반환
    print(type(content_list)) #Type 확인
    print(content_list) #리스트 값 출력
```

* 파일 쓰기

```python
f = open("count_log.txt", 'w', encoding="utf8") # mode='a'로 지정 시 기존 파일 내용에 추가
for i in range(1, 11):
    data = "%d번째 줄입니다.\n" % i
    f.write(data)
f.close()
```

* 파이썬의 directory 다루기
  * 최근에는 pathlib 모듈을 사용하여 path를 객체로 다움

```python
# os
import os
if not os.path.isdir("log"):
    os.mkdir("log")
```

* Pickle로 객체 저장하기(영속화)

```python
import pickle
f = open("list.pickle", "wb")
test = [1, 2, 3, 4, 5]
pickle.dump(test, f)
f.close()

f = open("list.pickle", "rb")
test_pickle = pickle.load(f)
print(test_pickle)
f.close()
```

<br>

#### Logging Handling

**로그 남기기**

* 프로그램이 실행되는 동안 일어나는 정보를 기록으로 남기기
* 유저의 접근, 프로그램의 Exception, 특정 함수의 사용 등
* Console 화면에 출력, 파일에 남기기, DB에 남기기 등

**logging level**

* DEBUG > INFO > WARNING > ERROR > CRITICAL

| Level    | 개요                                                         | 예시                                                         |
| -------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| debug    | 개발 시 처리 기록을 남겨야 하는 로그 정보를 남김             | - 다음 함수로 A를 호출함<br />- 변수 A를 무엇으로 변경함     |
| info     | 처리가 진행되는 동안의 정보를 알림                           | - 서버 시작/종료<br />- 사용자 A가 프로그램에 접속함         |
| warning  | 사용자가 잘못 입력한 정보나 처리는 가능하나 원래 개발 시 의도치 않는 정보가 들어왔을 때 알림 | - str 입력을 기대했으나 int 입력<br />- 함수에 argument로 이차원 리스트를 기대했으나 일차원 리스트가 들어옴 |
| error    | 잘못된 처리로 인해 에러가 났으나, 프로그램은 동작할 수 있음을 알림 | - 파일에 기록을 해야 하는데 파일이 없음<br />- 외부 서비스와 연결 불가 |
| critical | 잘못된 처리로 데이터 손실이나 더이상 프로그램이 동작할 수 없음을 알림 | - 잘못된 접근으로 해당 파일이 삭제됨<br />- 사용자의 의한 강제 종료 |

```python
import logging
logger = logging.getLogger("main")

stream_handler = logging.FileHandler(
    "my_log", mode='w', encoding="utf8")
logger.addHandler(stream_handler)
```

실제 프로그램을 실행할 땐 '데이터 파일 위치, 파일 저장 장소, Operation Type' 등 여러 설정이 필요. 이러한 정보를 설정해주는 방법으로 **파일에 저장해두는 configparser**와 **실행 시점에 입력하는 argparser**가 있다. 

**Configparser**

* 프로그램의 실행 설정을 file에 저장
* Section, Key, Value 값의 형태로 설정된 설정 파일을 사용
* 설정 파일을 Dict Type으로 호출 후 사용

```python
import configparser
config = configparser.ConfigParser()
config.sections()

config.read('example.cfg')
config.sections()

for key in config['SectionOne']:
    print(key)
    
config['SectionOne']["status"]
```

* example.cfg 파일

![image-20220118210926186](https://user-images.githubusercontent.com/70505378/150050520-3350e6a0-62a5-41c4-85da-3bec37616016.png)

**Argparser**

* Console 창에서 프로그램 실행 시 Setting 정보를 저장
* 거의 모든 Console 기반 Python 프로그램 기본으로 제공
* 특수 모듈도 많이 존재하지만(TF), 일반적으로 argparse를 사용
* Command-Line Option 이라고 부름

```python
def main(): 
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example') 
    parser.add_argument('--batch-size', type=int, default=64, metavar='N', help='input batch size for training (default: 
    64)') 
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N', help='input batch size for testing 
    (default: 1000)') 
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)') 
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR', help='learning rate (default: 0.01)') 
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)') 
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training') 
    parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)’) 
    parser.add_argument('--save-model', action='store_true', default=False, help='For Saving the current Model') 
                        
    args = parser.parse_args() 
                        
                        
if __name__ == '__main__': 
    main()
```

**Logging formatter**

* Log의 결과값의 format을 지정

```python
formatter = logging.Formatter('%(asctime)s %(levelname)s %(process)d %(message)s')

'''
2018-01-18 22:47:04,385 ERROR 4410 ERROR occurred
2018-01-18 22:47:22,458 ERROR 4439 ERROR occurred
2018-01-18 22:47:22,458 INFO 4439 HERE WE ARE
2018-01-18 22:47:24,680 ERROR 4443 ERROR occurred
2018-01-18 22:47:24,681 INFO 4443 HERE WE ARE
2018-01-18 22:47:24,970 ERROR 4445 ERROR occurred
2018-01-18 22:47:24,970 INFO 4445 HERE WE ARE
'''
```

**Log config file**

```python
logging.config.fileConfig('logging.conf')
logger = logging.getLogger()
```

* logging.conf

![image-20220118211324946](https://user-images.githubusercontent.com/70505378/150050522-e2ef6dd9-5461-4e68-9b73-b8da89f0e301.png)

<br>
