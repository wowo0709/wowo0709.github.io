---
layout: single
title: "[AITech] 20220118 - Python data handling"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['CSV', 'HTML', 'XML', 'JSON']
---



<br>

## 강의 복습 내용

### Python data handling

#### CSV

* CSV, 필드를 쉼표(,)로 구분한 텍스트 파일
* 엑셀 양식의 데이터를 프로그램에 상관없이 쓰기 위한 데이터 형식이라고 생각하면 쉬움

```python
import csv
reader = csv.reader(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_ALL)
```

| Attribute      | Default       | Meaning                                             |
| -------------- | ------------- | --------------------------------------------------- |
| delimeter      | ,             | 글자를 나누는 기준                                  |
| lineterminator | \\r\\n        | 줄 바꿈 기준                                        |
| quotechar      | "             | 문자열을 둘러싸는 신호 문자                         |
| quoting        | QUOTE_MINIMAL | 데이터 나누는 기준이 quotechar에 의해 둘러싸인 레발 |



#### HTML

* World Wide Web(WWW), 줄여서 웹이라고 부름
* 우리가 늘 쓰는 인터넷 공간의 정식 명칭
* HTML 예시

```html
<!doctype html>
<html>
    <head>
        <title>Hello HTML</title>
    </head>
    <body>
        <p>Hello World!</p>
    </body>
</html>
```

* 파싱에는 **정규표현식 모듈 re**를 사용하거나 **BeautifulSoup** 모듈을 사용할 수 있음



#### XML

* 데이터의 구조와 의미를 설명하는 TAG(MarkUp)를 사용하여 표시하는 언어
* TAG와 TAG 사이에 값이 표시되고, 구조적인 정보를 표현할 수 있음
* XML은 컴퓨터 간에 정보를 주고받기 매우 유용한 저장 방식

* XML 예시

```xml
<?xml version="1.0"?> 
<고양이> 
    <이름>나비</이름> 
    <품종>샴</품종> 
    <나이>6</나이> 
    <중성화>예</중성화> 
    <발톱 제거>아니요</발톱 제거>
	<등록 번호>Izz138bod</등록 번호>
    <소유자>이강주</소유자>
</고양이>
```

* XML도 HTML과 같이 구조적 markup 언어
* 따라서 **정규 표현식** 또는 **BeautifulSoup**을 이용해 파싱 가능

```python
import urllib.request
from bs4 import BeautifulSoup

with open("US08621662-20140107.XML", "r", encoding="utf8") as patent_xml:
    xml = patent_xml.read() # File을 String으로 읽어오기

soup = BeautifulSoup(xml, "lxml") #lxml parser 호출

#invention-title tag 찾기
invention_title_tag = soup.find("invention-title")
print (invention_title_tag.get_text())
```



#### JSON

* JavaScript Object Notation
* 원래 웹 언어인 JavaScript의 데이터 객체 표현 방식
* 간결성으로 기계/인간이 모두 이해하기 편함
* 데이터 용량이 적고, Code로의 전환이 쉬움
* 이로 인해 XML의 대체제로 많이 활용

```json
{"employees":[ 
    {"name":"Shyam", 
    "email":"shyamjaiswal@gmail.com"}, 
    {"name":"Bob", 
    "email":"bob32@gmail.com"}, 
    {"name":"Jai", 
    "email":"jai87@gmail.com"} ]
} 
```

* JSON 파일의 구조를 확인 -> 읽어온 후 -> Dict type처럼 처리
  * json 모듈 사용

```python
# 읽기
import json

with open("json_example.json", "r", encoding="utf8") as f:
    contents = f.read()
    json_data = json.loads(contents)
    print(json_data["employees"])

# 쓰기
import json

dict_data = {'Name': 'Zara', 'Age': 7, 'Class': 'First'}

with open("data.json", "w") as f:
    json.dump(dict_data, f)
```



<br>

<br>
