---
layout: single
title: "[Programmers] JadenCase 문자열 만들기"
categories: ['Algorithm', 'String']
---

# JadenCase 문자열 만들기

### 문제 설명

---

JadenCase란 모든 단어의 첫 문자가 대문자이고, 그 외의 알파벳은 소문자인 문자열입니다. 문자열 s가 주어졌을 때, s를 JadenCase로 바꾼 문자열을 리턴하는 함수, solution을 완성해주세요.

##### 제한 조건

- s는 길이 1 이상인 문자열입니다.
- s는 알파벳과 공백문자(" ")로 이루어져 있습니다.
- 첫 문자가 영문이 아닐때에는 이어지는 영문은 소문자로 씁니다. ( 첫번째 입출력 예 참고 )

##### 입출력 예

| s                       |         return          |
| ----------------------- | :---------------------: |
| "3people unFollowed me" | "3people Unfollowed Me" |
| "for the last week"     |   "For The Last Week"   |



### 문제 풀이

---

각 인덱스의 원소에 대해, 단어 머리의 원소일 경우 대문자로, 아닐 경우 소문자로 바꿉니다. 

```python
def solution(s):
    ans = ""
    for i in range(len(s)):
        if i == 0 or s[i-1] == " ": ans += s[i].upper()
        else: ans += s[i].lower()

    return ans
```

위와 같은 코드를 실행시키면 된다. 



그럼 안녕!
