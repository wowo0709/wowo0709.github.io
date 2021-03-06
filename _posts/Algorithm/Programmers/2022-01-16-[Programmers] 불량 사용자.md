---
layout: single
title: "[Programmers] 불량 사용자"
categories: ['Algorithm', 'Bruteforce']
toc: true
toc_sticky: true
tag: ['Product']
---



<br>

## 문제 설명

### 문제 설명

개발팀 내에서 이벤트 개발을 담당하고 있는 "무지"는 최근 진행된 카카오이모티콘 이벤트에 비정상적인 방법으로 당첨을 시도한 응모자들을 발견하였습니다. 이런 응모자들을 따로 모아 `불량 사용자`라는 이름으로 목록을 만들어서 당첨 처리 시 제외하도록 이벤트 당첨자 담당자인 "프로도" 에게 전달하려고 합니다. 이 때 개인정보 보호을 위해 사용자 아이디 중 일부 문자를 '\*' 문자로 가려서 전달했습니다. 가리고자 하는 문자 하나에 '\*' 문자 하나를 사용하였고 아이디 당 최소 하나 이상의 '\*' 문자를 사용하였습니다.
"무지"와 "프로도"는 불량 사용자 목록에 매핑된 응모자 아이디를 `제재 아이디` 라고 부르기로 하였습니다.

예를 들어, 이벤트에 응모한 전체 사용자 아이디 목록이 다음과 같다면

| 응모자 아이디 |
| ------------- |
| frodo         |
| fradi         |
| crodo         |
| abc123        |
| frodoc        |

다음과 같이 불량 사용자 아이디 목록이 전달된 경우,

| 불량 사용자 |
| ----------- |
| fr*d*       |
| abc1**      |

불량 사용자에 매핑되어 당첨에서 제외되어야 야 할 제재 아이디 목록은 다음과 같이 두 가지 경우가 있을 수 있습니다.

| 제재 아이디 |
| ----------- |
| frodo       |
| abc123      |

| 제재 아이디 |
| ----------- |
| fradi       |
| abc123      |

이벤트 응모자 아이디 목록이 담긴 배열 user_id와 불량 사용자 아이디 목록이 담긴 배열 banned_id가 매개변수로 주어질 때, 당첨에서 제외되어야 할 제재 아이디 목록은 몇가지 경우의 수가 가능한 지 return 하도록 solution 함수를 완성해주세요.

### **[제한사항]**

* user_id 배열의 크기는 1 이상 8 이하입니다.
* user_id 배열 각 원소들의 값은 길이가 1 이상 8 이하인 문자열입니다.
  * 응모한 사용자 아이디들은 서로 중복되지 않습니다.
  * 응모한 사용자 아이디는 알파벳 소문자와 숫자로만으로 구성되어 있습니다.
* banned_id 배열의 크기는 1 이상 user_id 배열의 크기 이하입니다.
* banned_id 배열 각 원소들의 값은 길이가 1 이상 8 이하인 문자열입니다.
  * 불량 사용자 아이디는 알파벳 소문자와 숫자, 가리기 위한 문자 '*' 로만 이루어져 있습니다.
  * 불량 사용자 아이디는 '\*' 문자를 하나 이상 포함하고 있습니다.
  * 불량 사용자 아이디 하나는 응모자 아이디 중 하나에 해당하고 같은 응모자 아이디가 중복해서 제재 아이디 목록에 들어가는 경우는 없습니다.
* 제재 아이디 목록들을 구했을 때 아이디들이 나열된 순서와 관계없이 아이디 목록의 내용이 동일하다면 같은 것으로 처리하여 하나로 세면 됩니다.

### **[입출력 예]**

| user_id                                           | banned_id                                | result |
| ------------------------------------------------- | ---------------------------------------- | ------ |
| `["frodo", "fradi", "crodo", "abc123", "frodoc"]` | `["fr*d*", "abc1**"]`                    | 2      |
| `["frodo", "fradi", "crodo", "abc123", "frodoc"]` | `["*rodo", "*rodo", "******"]`           | 2      |
| `["frodo", "fradi", "crodo", "abc123", "frodoc"]` | `["fr*d*", "*rodo", "******", "******"]` | 3      |

#### **입출력 예에 대한 설명**

**입출력 예 #1**

문제 설명과 같습니다.

**입출력 예 #2**

다음과 같이 두 가지 경우가 있습니다.

| 제재 아이디 |
| ----------- |
| frodo       |
| crodo       |
| abc123      |

| 제재 아이디 |
| ----------- |
| frodo       |
| crodo       |
| frodoc      |

**입출력 예 #3**

다음과 같이 세 가지 경우가 있습니다.

| 제재 아이디 |
| ----------- |
| frodo       |
| crodo       |
| abc123      |
| frodoc      |

| 제재 아이디 |
| ----------- |
| fradi       |
| crodo       |
| abc123      |
| frodoc      |

| 제재 아이디 |
| ----------- |
| fradi       |
| frodo       |
| abc123      |
| frodoc      |

<br>

## 문제 풀이

### \# 완전 탐색 \# 중복 순열



<br>

### 풀이 과정

문제를 읽으며, 우선 **입력의 크기가 매우 작다**라는 것을 캐치해야 합니다. 입력의 크기가 최대 8로 매우 작기 때문에, 자연스럽게 **완전 탐색**을 해 볼 수 있겠다는 생각을 가질 수 있게 되죠. 

완전 탐색을 한다면, 시간 복잡도를 계산해봐야 합니다. 이 문제의 경우 

1. banned_ids의 각 원소에 대해 (최대 8)
2. user_ids의 각 원소의 (최대 8)
3. 각 글자를 비교한다. (최대 8)

와 같이 탐색할 수 있습니다. 

위 탐색 과정으로 우선 각 `banned_id`에 어떤 `user_id`가 해당될 수 있는지를 구할 수 있고, **최대 비교 횟수는 512번**으로 그렇게 크지 않다는 것을 알 수 있죠. 

그런데 이 연산 이후 실제 답을 구할 때의 연산 횟수까지 고려해봐야 합니다. 

<br>

다음으로는, 위에서 얻은 **각 banned_id에 해당할 수 있는 user_id의 경우의 수 리스트**에서 **최종 아이디 목록 경우의 수**를 구합니다. 

이는 **중복 순열**로 구할 수 있고, 여기서 **최대 8^8 번**의 비교 횟수가 발생합니다. 8^8은 16,777,216으로, 약 10^7 정도 됩니다. 

사실 이 정도 연산 횟수면, 시간 초과가 날 지 나지 않을 지 판단하기 약간 애매합니다. 그러니, 우선 위의 방법으로 시도해보고 시간 초과가 난다면 다른 방법을 고안해보는 것도 괜찮습니다. 

<br>

결과적으로는 이 문제는 10^7 번의 연산 횟수 정도는 허용을 하며, 시간 초과가 발생하지 않습니다. 

<br>

### 전체 코드

```python
def solution(user_ids, banned_ids):
    case_list = [[] for _ in range(len(banned_ids))] # 각 banned_id에 해당할 수 있는 user_id 리스트
    for n,banned_id in enumerate(banned_ids):
        for user_id in user_ids:
            if len(banned_id) == len(user_id):
                for i in range(len(banned_id)):
                    if banned_id[i] != '*' and banned_id[i] != user_id[i]:
                        break
                else:
                    case_list[n].append(user_id)

    from itertools import product as PI
    cases = []
    for case_in_list in PI(*case_list): # 최종 아이디 목록 구하기
        case_in_set = set(case_in_list)
        if len(case_in_list) == len(case_in_set) and case_in_set not in cases:
            cases.append(case_in_set)

    return len(cases)
```



<br>

### 추가

> 중복 순열 함수 `product`를 사용할 때, 인자로 리스트들을 가지고 있는 리스트가 아닌 개별 리스트들을 전달해야 합니다. 
>
> 즉, product([[1,2,3],[4,5,6]])이 아니라, product([1,2,3],[4,5,6])과 같이 전달해야 합니다. 
>
> 그래서 위 코드의 14번째 줄에서 `PI(*case_list)`와 같이 전달한 것입니다. 
>
> <br>
>
> 또 추가로, **set과 dict는 원소(키)로 set과 list(unhashable type)를 가질 수 없습니다.**
