---
layout: single
title: "[Programmers] 후보키"
categories: ['Algorithm', 'Implementation', 'HashMap']
tag: ['Combinations', 'Counter']
---

# 후보키

### 문제 설명

---

프렌즈대학교 컴퓨터공학과 조교인 제이지는 네오 학과장님의 지시로, 학생들의 인적사항을 정리하는 업무를 담당하게 되었다.

그의 학부 시절 프로그래밍 경험을 되살려, 모든 인적사항을 데이터베이스에 넣기로 하였고, 이를 위해 정리를 하던 중에 후보키(Candidate Key)에 대한 고민이 필요하게 되었다.

후보키에 대한 내용이 잘 기억나지 않던 제이지는, 정확한 내용을 파악하기 위해 데이터베이스 관련 서적을 확인하여 아래와 같은 내용을 확인하였다.

* 관계 데이터베이스에서 릴레이션(Relation)의 튜플(Tuple)을 유일하게 식별할 수 있는 속성(Attribute) 또는 속성의 집합 중, 다음 두 성질을 만족하는 것을 후보 키(Candidate Key)라고 한다.
    * 유일성(uniqueness) : 릴레이션에 있는 모든 튜플에 대해 유일하게 식별되어야 한다.
    * 최소성(minimality) : 유일성을 가진 키를 구성하는 속성(Attribute) 중 하나라도 제외하는 경우 유일성이 깨지는 것을 의미한다. 즉, 릴레이션의 모든 튜플을 유일하게 식별하는 데 꼭 필요한 속성들로만 구성되어야 한다.

제이지를 위해, 아래와 같은 학생들의 인적사항이 주어졌을 때, 후보 키의 최대 개수를 구하라.

![cand_key1.png](https://grepp-programmers.s3.amazonaws.com/files/production/f1a3a40ede/005eb91e-58e5-4109-9567-deb5e94462e3.jpg)

위의 예를 설명하면, 학생의 인적사항 릴레이션에서 모든 학생은 각자 유일한 "학번"을 가지고 있다. 따라서 "학번"은 릴레이션의 후보 키가 될 수 있다.
그다음 "이름"에 대해서는 같은 이름("apeach")을 사용하는 학생이 있기 때문에, "이름"은 후보 키가 될 수 없다. 그러나, 만약 ["이름", "전공"]을 함께 사용한다면 릴레이션의 모든 튜플을 유일하게 식별 가능하므로 후보 키가 될 수 있게 된다.
물론 ["이름", "전공", "학년"]을 함께 사용해도 릴레이션의 모든 튜플을 유일하게 식별할 수 있지만, 최소성을 만족하지 못하기 때문에 후보 키가 될 수 없다.
따라서, 위의 학생 인적사항의 후보키는 "학번", ["이름", "전공"] 두 개가 된다.

릴레이션을 나타내는 문자열 배열 relation이 매개변수로 주어질 때, 이 릴레이션에서 후보 키의 개수를 return 하도록 solution 함수를 완성하라.

##### 제한사항

* relation은 2차원 문자열 배열이다.
* relation의 컬럼(column)의 길이는 `1` 이상 `8` 이하이며, 각각의 컬럼은 릴레이션의 속성을 나타낸다.
* relation의 로우(row)의 길이는 `1` 이상 `20` 이하이며, 각각의 로우는 릴레이션의 튜플을 나타낸다.
* relation의 모든 문자열의 길이는 `1` 이상 `8` 이하이며, 알파벳 소문자와 숫자로만 이루어져 있다.
* relation의 모든 튜플은 유일하게 식별 가능하다.(즉, 중복되는 튜플은 없다.)

##### 입출력 예

| relation                                                     | result |
| ------------------------------------------------------------ | ------ |
| `[["100","ryan","music","2"],["200","apeach","math","2"],["300","tube","computer","3"],["400","con","computer","4"],["500","muzi","music","3"],["600","apeach","music","2"]]` | 2      |

##### 입출력 예 설명

입출력 예 #1
문제에 주어진 릴레이션과 같으며, 후보 키는 2개이다.

<br>



### 문제 풀이

---

#### \# 조합

<br>

위 문제의 핵심 요소들을 간략하게 정리해봅시다. 

1. 학생들의 인적사항 정리

2. 후보 키: 유일성, 최소성을 만족해야 하고 N개의 피처를 포함

3. 유일성: 같은 속성 조합 값을 갖는 원소가 없는 것

4. 최소성: 후보 키의 속성 중 하나를 제거하면 더 이상 후보키가 되지 않는 것

5. 크기는 (8, 20) 이하

위의 5가지 정보가 되겠습니다. 

<br>

**1. 피쳐 딕셔너리 만들기**

우선 저는 relations 이차원 리스트를 피쳐 단위로 나눈 딕셔너리를 생성하기로 했습니다.  

딕셔너리의 key는 임의로 0부터 증가하는 숫자 문자열로 하였고, value는 각 피쳐의 원소들로 하였습니다. 

예제 문제의 경우 피쳐 딕셔너리는 다음과 같이 생성됩니다. 

```python
M = len(relations[0]) # 피쳐 개수
features = dict() # 피쳐 딕셔너리
for key,col in enumerate(range(M)):
    features[str(key)] = [row[col] for row in relations]
    
out: # features
  {'0': ['100', '200', '300', '400', '500', '600'], 
   '1': ['ryan', 'apeach', 'tube', 'con', 'muzi', 'apeach'], 
   '2': ['music', 'math', 'computer', 'computer', 'music', 'music'], 
   '3': ['2', '2', '3', '4', '3', '2']}
```

<br>

**2. 모든 키의 조합 검사**

후보키는 1~M개의 피쳐의 조합으로 생성됩니다. 물론 유일성과 최소성 검사를 해야하지만, 일단을 모든 피쳐의 조합을 구하는 코드를 작성합니다. 

```python
from collections import Counter
from itertools import combinations as C
cand_keys = [] # 후보키 리스트
for cnt in range(1,M+1): # 피쳐 1~M개 조합
    for tmp_key in C(features.keys(), cnt): # 각 조합된 임시키에 대해
        tmp_features = list(zip(*[features[key] for key in tmp_key]))
        
out: # tmp_features
('100',) ('200',) ('300',) ('400',) ('500',) ('600',)

('ryan',) ('apeach',) ('tube',) ('con',) ('muzi',) ('apeach',)

...

('music', '2') ('math', '2') ('computer', '3') ('computer', '4') ('music', '3') ('music', '2')

('100', 'ryan', 'music') ('200', 'apeach', 'math') ('300', 'tube', 'computer') ('400', 'con', 'computer') ('500', 'muzi', 'music') ('600', 'apeach', 'music')

...

('100', 'ryan', 'music', '2') ('200', 'apeach', 'math', '2') ('300', 'tube', 'computer', '3') ('400', 'con', 'computer', '4') ('500', 'muzi', 'music', '3') ('600', 'apeach', 'music', '2')
```

tmp_features는 피쳐의 조합으로 생성된 임시 키(tmp_key)의 각 속성값들의 조합입니다. 

이 속성값은 유일성 검사를 하기 위해 생성됩니다. 

<br>

**3. 유일성 검사와 최소성 검사**

이제 유일성과 최소성 검사를 합니다. 

유일성은 앞에서 생성된 tmp_features의 값들 중 겹치는 값이 있는지 검사함으로써 할 수 있습니다. 

최소성은 후보키들(cand_keys) 중 **임시키(tmp_key)에 모두 포함되는 후보키가 있다면** 최소성을 만족하지 못하는 것입니다. 즉 최소성을 만족하는 후보키가 이미 있다면, 그 후보키에 필요없는 다른 키가 조합된 임시키는 최소성을 만족하지 않는 것이죠. 

```python
# 겹치는 값이 없으면(유일성)
if Counter(tmp_features).most_common(1)[0][1] == 1:
    # 최소성을 만족하는 지 검사
    for cand_key in cand_keys:
        for feature in cand_key:
            if feature not in tmp_key: break # 해당 키조합이 없다면 break(계속 검사)
        else: break # 해당 키 조합이 있다면 break
    else: cand_keys.append(tmp_key) # 해당 키 조합이 없다면 append 
```

그리고 두 조건 모두 만족한다면 후보키 리스트에 임시키를 추가합니다. 

<br>

<br>

아래는 전체 코드입니다. 

```python
def solution(relations):
    M = len(relations[0]) # 피쳐 개수
    features = dict() # 피쳐 딕셔너리
    for key,col in enumerate(range(M)):
        features[str(key)] = [row[col] for row in relations]

    from collections import Counter
    from itertools import combinations as C
    cand_keys = [] # 후보키 리스트
    for cnt in range(1,M+1): # 피쳐 1~M개 조합
        for tmp_key in C(features.keys(), cnt): # 각 조합된 임시키에 대해
            tmp_features = list(zip(*[features[key] for key in tmp_key]))
            print(*tmp_features, end='\n\n')
            # 겹치는 값이 없으면(유일성)
            if Counter(tmp_features).most_common(1)[0][1] == 1:
                # 최소성을 만족하는 지 검사
                for cand_key in cand_keys:
                    for feature in cand_key:
                        if feature not in tmp_key: break # 해당 키조합이 없다면 break(계속 검사)
                    else: break # 해당 키 조합이 있다면 break
                else: cand_keys.append(tmp_key) # 해당 키 조합이 없다면 append
                
    return len(cand_keys)
```

<br>

<br>

### [추가] 다른 방법으로 최소성 검사

---

위에서는 임시키가 생성될 때마다 각 후보키들이 임시키에 포함되는 지 검사했습니다. 

이번에는 다른 방식으로, 후보키가 생성될 때마다 그 후보키를 포함하는 임시키를 모두 생성해, 최소성을 만족하는 하나의 후보키를 제외하고 모두 impossible_keys라는 후보키가 될 수 없는 리스트에 추가할 것입니다. 

이렇게 하면 임시키가 생성될 때마다 impossible_keys에 있는지의 여부로 최소성 검사를 할 수 있겠죠. 

<br>

전체 코드는 아래와 같습니다. 

```python

def solution(relations):
    M = len(relations[0]) # 피쳐 개수
    features = dict() # 피처 딕셔너리
    for key,col in enumerate(range(M)):
        features[str(key)] = [row[col] for row in relations]

    from collections import Counter  
    from itertools import combinations as C
    cand_keys = [] # 후보키 리스트
    impossible_keys = dict() # 후보키가 될 수 없는 키 해쉬맵(최소성 검사)
    # impossible_keys = [] 
    for cnt in range(1,M+1): # 피처 1~M개 조합
        for tmp_key in C(features.keys(), cnt): # 각 조합된 임시키에 대해
            tmp_features = list(zip(*[features[key] for key in tmp_key]))
            # 겹치는 값이 없고(유일성) impossible_key에 포함되지 않는다면(최소성),
            if Counter(tmp_features).most_common(1)[0][1] == 1 and tmp_key not in impossible_keys:
                cand_keys.append(tmp_key)
                # tmp_key조합과 나머지 키들의 조합을 만들어 최소성을 만족하지 않는 조합의 키들을 생성
                for tmp_cnt in range(1,M-len(tmp_key)+1):
                    comb_feature_list = list(features.keys())
                    for feature in tmp_key: 
                        comb_feature_list.remove(feature)
                    for tmp_comb in C(comb_feature_list, tmp_cnt):
                        impossible_keys[tuple(sorted(list(tmp_comb)+list(tmp_key)))] = 1
                        # impossible_keys.append(sorted(list(tmp_comb)+list(tmp_key)))
    
    return len(cand_keys)
```

<br>

####  <span style="color:red">🔥 이렇게 미리 불가능한 조합들을 생성해서 해쉬맵을 생성해 탐색의 속도를 **O(1)**으로 가져가는 방법은 아래 문제에서 영감을 받았습니다. </span>

[[Programmers] 순위 검색](https://wowo0709.github.io/Programmers-%EC%88%9C%EC%9C%84-%EA%B2%80%EC%83%89/)

리스트에서 원소가 있는 지 탐색하는 것의 시간 복잡도는 **O(n)**이지만, 해쉬맵(딕셔너리)에서 키가 있는지 탐색하는 것의 시간 복잡도는 **O(1)**입니다. 

따라서 `in` 키워드를 사용해야 하는 상황이라면 값들을 리스트가 아니라 딕셔너리의 키로 저장해놓는 것은 어떨까요?

<br>

이만 포스팅 마치겠습니다. 
