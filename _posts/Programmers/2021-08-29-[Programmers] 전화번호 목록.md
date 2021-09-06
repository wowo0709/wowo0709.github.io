---
layout: single
title: "[Programmers] 전화번호 목록"
categories: ['Algorithm', 'Programmers']
---



# 전화번호 목록

### 문제 설명

##### 문제 설명

전화번호부에 적힌 전화번호 중, 한 번호가 다른 번호의 접두어인 경우가 있는지 확인하려 합니다.
전화번호가 다음과 같을 경우, 구조대 전화번호는 영석이의 전화번호의 접두사입니다.

- 구조대 : 119
- 박준영 : 97 674 223
- 지영석 : 11 9552 4421

전화번호부에 적힌 전화번호를 담은 배열 phone_book 이 solution 함수의 매개변수로 주어질 때, 어떤 번호가 다른 번호의 접두어인 경우가 있으면 false를 그렇지 않으면 true를 return 하도록 solution 함수를 작성해주세요.

##### 제한 사항

- phone_book의 길이는 1 이상 1,000,000 이하입니다.
  - 각 전화번호의 길이는 1 이상 20 이하입니다.
  - 같은 전화번호가 중복해서 들어있지 않습니다.

##### 입출력 예제

| phone_book                        | return |
| --------------------------------- | ------ |
| ["119", "97674223", "1195524421"] | false  |
| ["123","456","789"]               | true   |
| ["12","123","1235","567","88"]    | false  |

##### 입출력 예 설명

입출력 예 #1
앞에서 설명한 예와 같습니다.

입출력 예 #2
한 번호가 다른 번호의 접두사인 경우가 없으므로, 답은 true입니다.

입출력 예 #3
첫 번째 전화번호, “12”가 두 번째 전화번호 “123”의 접두사입니다. 따라서 답은 false입니다.

------

**알림**

2021년 3월 4일, 테스트 케이스가 변경되었습니다. 이로 인해 이전에 통과하던 코드가 더 이상 통과하지 않을 수 있습니다.

[출처](https://ncpc.idi.ntnu.no/ncpc2007/ncpc2007problems.pdf)

<br>

### 문제 풀이

---

#### \# 해쉬 맵

<br>

**특정 요소가 있는 지 검사**하고 싶을 때, 해쉬 맵을 활용하면 좋은 경우들이 많습니다. 

이 문제도 그것에 초점을 맞춘 문제입니다. 



```python
def solution(phone_book):
    phone_dic = dict()
    for phone_number in sorted(phone_book, key=lambda x: len(x)):
        if any(phone_number[:i] in phone_dic for i in range(len(phone_number))):
            return False
        else:
            phone_dic[phone_number] = 1
    return True
```

먼저 파라미터로 전달받은 phone_book 리스트를 길이 순서로 정렬해주어야 합니다. 정렬을 하지 않아 길이가 긴 번호가 먼저 나온다면, 뒤에 나오는 짧은 번호를 포함하는데도 검사를 못하는 경우가 발생합니다. 

정렬을 하고 나면 앞에서부터 각 번호들에 대해 phone_dic 딕셔너리에 접두사가 있는 지 탐색하고, 없으면 번호를 딕셔너리에 추가합니다. 

여기서 탐색에 소요되는 시간은 **O(1)**입니다. 

<br>

#### [추가] 다른 풀이

해쉬맵을 사용한 풀이는 아니지만, 다른 분의 풀이 중 기발한 풀이가 있어 첨부합니다. 

```python
def solution(phoneBook):
    phoneBook = sorted(phoneBook)

    for p1, p2 in zip(phoneBook, phoneBook[1:]):
        if p2.startswith(p1):
            return False
    return True
```

파라미터로 넘어온 phoneBook을 아스키코드 순으로 정렬하면(자동으로 길이순으로도 정렬) 인접한 것만 검사함으로써 같은 결과를 얻을 수 있습니다. 