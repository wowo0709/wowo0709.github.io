---
layout: single
title: "[Programmers] 순위 검색"
categories: ['Algorithm', 'Programmers']
---

# 순위 검색

###### 문제 설명

**[본 문제는 정확성과 효율성 테스트 각각 점수가 있는 문제입니다.]**

카카오는 하반기 경력 개발자 공개채용을 진행 중에 있으며 현재 지원서 접수와 코딩테스트가 종료되었습니다. 이번 채용에서 지원자는 지원서 작성 시 아래와 같이 4가지 항목을 반드시 선택하도록 하였습니다.

* 코딩테스트 참여 개발언어 항목에 cpp, java, python 중 하나를 선택해야 합니다.
* 지원 직군 항목에 backend와 frontend 중 하나를 선택해야 합니다.
* 지원 경력구분 항목에 junior와 senior 중 하나를 선택해야 합니다.
* 선호하는 소울푸드로 chicken과 pizza 중 하나를 선택해야 합니다.

인재영입팀에 근무하고 있는 `니니즈`는 코딩테스트 결과를 분석하여 채용에 참여한 개발팀들에 제공하기 위해 지원자들의 지원 조건을 선택하면 해당 조건에 맞는 지원자가 몇 명인 지 쉽게 알 수 있는 도구를 만들고 있습니다.
예를 들어, 개발팀에서 궁금해하는 문의사항은 다음과 같은 형태가 될 수 있습니다.
`코딩테스트에 java로 참여했으며, backend 직군을 선택했고, junior 경력이면서, 소울푸드로 pizza를 선택한 사람 중 코딩테스트 점수를 50점 이상 받은 지원자는 몇 명인가?`

물론 이 외에도 각 개발팀의 상황에 따라 아래와 같이 다양한 형태의 문의가 있을 수 있습니다.

* 코딩테스트에 python으로 참여했으며, frontend 직군을 선택했고, senior 경력이면서, 소울푸드로 chicken을 선택한 사람 중 코딩테스트 점수를 100점 이상 받은 사람은 모두 몇 명인가?
* 코딩테스트에 cpp로 참여했으며, senior 경력이면서, 소울푸드로 pizza를 선택한 사람 중 코딩테스트 점수를 100점 이상 받은 사람은 모두 몇 명인가?
* backend 직군을 선택했고, senior 경력이면서 코딩테스트 점수를 200점 이상 받은 사람은 모두 몇 명인가?
* 소울푸드로 chicken을 선택한 사람 중 코딩테스트 점수를 250점 이상 받은 사람은 모두 몇 명인가?
* 코딩테스트 점수를 150점 이상 받은 사람은 모두 몇 명인가?

즉, 개발팀에서 궁금해하는 내용은 다음과 같은 형태를 갖습니다.

```
* [조건]을 만족하는 사람 중 코딩테스트 점수를 X점 이상 받은 사람은 모두 몇 명인가?
```

------

#### **[문제]**

지원자가 지원서에 입력한 4가지의 정보와 획득한 코딩테스트 점수를 하나의 문자열로 구성한 값의 배열 info, 개발팀이 궁금해하는 문의조건이 문자열 형태로 담긴 배열 query가 매개변수로 주어질 때,
각 문의조건에 해당하는 사람들의 숫자를 순서대로 배열에 담아 return 하도록 solution 함수를 완성해 주세요.

#### **[제한사항]**

* info 배열의 크기는 1 이상 50,000 이하입니다.
* info 배열 각 원소의 값은 지원자가 지원서에 입력한 4가지 값과 코딩테스트 점수를 합친 "개발언어 직군 경력 소울푸드 점수" 형식입니다.
    * 개발언어는 cpp, java, python 중 하나입니다.
    * 직군은 backend, frontend 중 하나입니다.
    * 경력은 junior, senior 중 하나입니다.
    * 소울푸드는 chicken, pizza 중 하나입니다.
    * 점수는 코딩테스트 점수를 의미하며, 1 이상 100,000 이하인 자연수입니다.
    * 각 단어는 공백문자(스페이스 바) 하나로 구분되어 있습니다.
* query 배열의 크기는 1 이상 100,000 이하입니다.
* query의 각 문자열은 "[조건] X" 형식입니다.
    * [조건]은 "개발언어 and 직군 and 경력 and 소울푸드" 형식의 문자열입니다.
    * 언어는 cpp, java, python, - 중 하나입니다.
    * 직군은 backend, frontend, - 중 하나입니다.
    * 경력은 junior, senior, - 중 하나입니다.
    * 소울푸드는 chicken, pizza, - 중 하나입니다.
    * '-' 표시는 해당 조건을 고려하지 않겠다는 의미입니다.
    * X는 코딩테스트 점수를 의미하며 조건을 만족하는 사람 중 X점 이상 받은 사람은 모두 몇 명인 지를 의미합니다.
    * 각 단어는 공백문자(스페이스 바) 하나로 구분되어 있습니다.
    * 예를 들면, "cpp and - and senior and pizza 500"은 "cpp로 코딩테스트를 봤으며, 경력은 senior 이면서 소울푸드로 pizza를 선택한 지원자 중 코딩테스트 점수를 500점 이상 받은 사람은 모두 몇 명인가?"를 의미합니다.

------

##### **[입출력 예]**

| info                                                         | query                                                        | result        |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------- |
| `["java backend junior pizza 150","python frontend senior chicken 210","python frontend senior chicken 150","cpp backend senior pizza 260","java backend junior chicken 80","python backend senior chicken 50"]` | `["java and backend and junior and pizza 100","python and frontend and senior and chicken 200","cpp and - and senior and pizza 250","- and backend and senior and - 150","- and - and - and chicken 100","- and - and - and - 150"]` | [1,1,1,1,2,4] |

##### **입출력 예에 대한 설명**

지원자 정보를 표로 나타내면 다음과 같습니다.

| 언어   | 직군     | 경력   | 소울 푸드 | 점수 |
| ------ | -------- | ------ | --------- | ---- |
| java   | backend  | junior | pizza     | 150  |
| python | frontend | senior | chicken   | 210  |
| python | frontend | senior | chicken   | 150  |
| cpp    | backend  | senior | pizza     | 260  |
| java   | backend  | junior | chicken   | 80   |
| python | backend  | senior | chicken   | 50   |

* `"java and backend and junior and pizza 100"` : java로 코딩테스트를 봤으며, backend 직군을 선택했고 junior 경력이면서 소울푸드로 pizza를 선택한 지원자 중 코딩테스트 점수를 100점 이상 받은 지원자는 1명 입니다.
* `"python and frontend and senior and chicken 200"` : python으로 코딩테스트를 봤으며, frontend 직군을 선택했고, senior 경력이면서 소울 푸드로 chicken을 선택한 지원자 중 코딩테스트 점수를 200점 이상 받은 지원자는 1명 입니다.
* `"cpp and - and senior and pizza 250"` : cpp로 코딩테스트를 봤으며, senior 경력이면서 소울푸드로 pizza를 선택한 지원자 중 코딩테스트 점수를 250점 이상 받은 지원자는 1명 입니다.
* `"- and backend and senior and - 150"` : backend 직군을 선택했고, senior 경력인 지원자 중 코딩테스트 점수를 150점 이상 받은 지원자는 1명 입니다.
* `"- and - and - and chicken 100"` : 소울푸드로 chicken을 선택한 지원자 중 코딩테스트 점수를 100점 이상을 받은 지원자는 2명 입니다.
* `"- and - and - and - 150"` : 코딩테스트 점수를 150점 이상 받은 지원자는 4명 입니다.

<br>



### 문제 풀이

---

아마 프로그래머스 2단계 문제들 중 가장 오래 걸린 문제가 아닐까 싶습니다...

대부분의 분들이 그러셨겠듯이, 정확도 테스트를 통과하는 것을 어렵지 않습니다. 

다만, **이 문제의 핵심은 효율성 테스트**입니다. query 안의 문자열들에 대해 info에 만족하는 문자열이 있는지 하나하나 비교(완전 탐색)하면 안됩니다. info를 먼저 점수 기준으로 정렬하고 점수 제한을 만족하는 원소들에 대해서만 비교해도 안됩니다. 

**이 문제가 원하는 것은 순차 탐색이 아니라는 것입니다. **

<br>

우선, 제 첫번째 코드를 간단히 소개하겠습니다. 

```python
# 정확성 성공, 효율성 실패 (초기 코드)
def solution(infos, querys):
    # 각 원소를 리스트로 변환
    infos = sorted(list(map(lambda i: i.split(), infos)), key = lambda i:int(i[4]))
    score_list = [int(i[4]) for i in infos]
    # 조건 비교
    ans = [0 for _ in range(len(querys))]
    for e,q in enumerate(querys):
        q = list(ch for ch in q.split() if ch != 'and')
        # 최저 점수 커트라인 
        from bisect import bisect_left
        limit_idx = bisect_left(score_list,int(q[4]))
        # 커트라인 이상인 대상들에 한해 조건 비교
        for i in infos[limit_idx:]:
            for idx in range(4):
                if q[idx] == '-': continue
                if q[idx] != i[idx]: break
            else: 
                ans[e] += 1

    return ans
```

보시면, 먼저 infos의 각 원소를 split하여 리스트로 만들고 점수기준으로 정렬하죠. 그리고 점수 리스트를 따로 생성합니다. 

querys 안의 문자열들에 대해, **bisect_left** 메서드로 점수 제한을 만족하는 원소의 최소 위치를 찾아냅니다. 점수를 만족하는 커트라인을 찾는 것이죠. 

이후에는 추려낸 원소들에 대해서만 쿼리의 문자열과 비교하며 조건을 만족한다면 ans를 1 증가시킵니다. 

<br>

이 코드에 대한 정확성 및 효율성 테스트 결과는 아래와 같습니다. 

```python
정확성  테스트
테스트 1 〉	통과 (0.48ms, 10.3MB)
테스트 2 〉	통과 (0.54ms, 10.4MB)
테스트 3 〉	통과 (1.57ms, 10.4MB)
테스트 4 〉	통과 (6.01ms, 10.4MB)
테스트 5 〉	통과 (42.09ms, 10.5MB)
테스트 6 〉	통과 (66.00ms, 10.6MB)
테스트 7 〉	통과 (24.66ms, 10.6MB)
테스트 8 〉	통과 (133.67ms, 13.2MB)
테스트 9 〉	통과 (129.67ms, 13.4MB)
테스트 10 〉	통과 (120.34ms, 13.5MB)
테스트 11 〉	통과 (19.80ms, 10.5MB)
테스트 12 〉	통과 (82.27ms, 10.7MB)
테스트 13 〉	통과 (21.95ms, 10.5MB)
테스트 14 〉	통과 (95.18ms, 11.9MB)
테스트 15 〉	통과 (119.80ms, 11.8MB)
테스트 16 〉	통과 (21.95ms, 10.5MB)
테스트 17 〉	통과 (44.35ms, 10.6MB)
테스트 18 〉	통과 (95.29ms, 11.9MB)
효율성  테스트
테스트 1 〉	실패 (시간 초과)
테스트 2 〉	실패 (시간 초과)
테스트 3 〉	실패 (시간 초과)
테스트 4 〉	실패 (시간 초과)
채점 결과
정확성: 40.0
효율성: 0.0
합계: 40.0 / 100.0
```

효율성 테스트가 모두 **실패**인 것을 알 수 있죠...

<br>

<br>

## 그럼 어떻게 해야 할까요?

순차 탐색 알고리즘으로는 이 문제를 통과하지 못합니다. **그렇다면 다른 알고리즘/자료구조 를 활용해야겠죠?**

<br>

저는 우선 아래와 같이 이 문제의 키워드를 정리하고 싶습니다. 

### \# 해쉬맵 \# 조합 \# 이분 탐색

<br>

해쉬맵은 사실 그렇게 친숙한 자료구조는 아닙니다. 코딩 테스트를 준비하다 보면 **해쉬맵** 자료구조는 별로 없어요...😂

게다가 이런 고난도 해쉬 맵 문제는 거의 없죠. 그래서 더 어렵게 느껴지지 않았나 싶습니다. 

<br>

천천히 순서대로 설명해보죠. 

**1. 해쉬맵 생성하기**

우리는 파라미터로 주어진 infos 리스트에서 info_dict라는 딕셔너리(해쉬맵)를 만들겁니다. 

그 이유는? querys를 보면, 각 4가지의 조건(점수 제외)과 함께 ' - ' 가 섞여있죠.  우리가 infos에서 4가지 조건과 함께 ' - ' 가 섞인 조건을 해쉬맵의 키로 미리 만들어놓는다면, querys를 만족하는 원소가 있는지는 **query에 해당하는 조건이 info_dict의 키에 존재하는 지**만 보면 되는 것이죠. 이것으로 **탐색에 대한 시간 복잡도를 O(1)으로 줄이는 것입니다.**

키를 만들면, 해당 키의 값은 점수가 됩니다. 이것으로 마지막에 탐색을 할 때 **해당 조건을 만족하는 사람이 있는지 바로 찾아간 후에, 있다면 정렬된 점수 리스트에서 이분 탐색 알고리즘을 이용하여 점수까지 만족하는 사람은 몇 명인지 찾아낼 것**입니다. 

먼저 코드를 보시죠. 

```python
    from itertools import combinations
    info_dict = dict()
    for info in infos:
        conds = info.split()
        for n in range(5):                            # "-" 를 0~4개 삽입 가능
            idx_list = list(combinations(range(4),n)) # "-"를 삽입하는 위치에 대한 콤비네이션
            for idxs in idx_list:      # 각 위치(인덱스 리스트)에 대해
                tmp = conds[:4].copy() # 조건(점수 제외) 복사
                for idx in idxs:   # 각 인덱스에 대해
                    tmp[idx] = "-" # 인덱스에 "-" 삽입
                info_dict_key = "_".join(tmp)
                if info_dict_key in info_dict: info_dict[info_dict_key].append(int(conds[4]))
                else: info_dict[info_dict_key] = [int(conds[4])]
```

info의 4가지 조건과 " - "를 조합할 때는 **combinations** 클래스를 이용합니다. 

combinations 로 " - "가 삽입될 인덱스 리스트를 만들고 각 자리에 해당 조건 대신 " - "를 삽입하는 것이죠. " - "은 0 ~ 4 개가 삽입될 수 있으니 range(5)에 대하여 for문을 도는 것입니다. 

예를 들어, **combinations([1,2,3], 2)**는 다음 리스트를 반환합니다. 

```python
in:
  combinations([1,2,3], 2)
out:
  [(1,2), (1,3), (2,3)]
```

위 예시의 경우 차례로 " - "를 인덱스 1과 2, 인덱스 1과 3, 인덱스 2와 3 위치에 각각 삽입하는 조합을 만들어내는 것이죠. 

<br>

**2. 딕셔너리의 각 값을 정렬**

위에서 말했듯, 4가지 조건을 만족하는 점수들(사람들)이 있다면 그 점수 리스트에서 이분 탐색을 할 것이라고 했습니다. 

이분 탐색을 하려면 리스트가 sorted list이어야 하므로 각 values를 먼저 정렬해줍니다. 

100,000 개 이하의 쿼리가 존재하기 때문에(매우 많습니다) 매 쿼리마다 그때서야 정렬을 하는 것은 좋지 않습니다. 

```python
    # 점수 정렬 -> 이진 탐색 알고리즘
    for values in info_dict.values(): 
        values.sort()
```

<br>

**3. 조건 비교하기**

이제 마지막 단계입니다. 

querys 리스트 안의 각 query에 대하여 4가지 조건(점수 제외)을 만족하는 사람들이 있는 지(조건과 일치하는 키가 있는지) info_dict를 참조합니다. 

만약 있다면 정렬되어 있는 점수 리스트에 대해 점수 커트라인을 찾아내서 만족하는 점수의 개수를 구하여 ans 리스트에 append 합니다. 

```python
    # 조건 비교
    from bisect import bisect_left
    ans = []
    for query in querys:
        query = [q for q in query.split() if q != "and"]
        info_dict_key = "_".join(query[:4])
        if info_dict_key in info_dict: # 조건을 만족하는 후보에 한해 커트라인 점수를 적용 
            candidates = info_dict[info_dict_key]
            cnt = len(candidates) - bisect_left(candidates,int(query[4]))
        else:                          # 조건을 만족하는 후보가 없음
            cnt = 0
        ans.append(cnt)

    return ans
```

<br>

<br>

마지막으로 아래는 전체 코드입니다. 

```python
def solution(infos, querys):
    # 조건 해쉬맵 생성
    from itertools import combinations
    info_dict = dict()
    for info in infos:
        conds = info.split()
        for n in range(5):                            # "-" 를 0~4개 삽입 가능
            idx_list = list(combinations(range(4),n)) # "-"를 삽입하는 위치에 대한 콤비네이션
            for idxs in idx_list:      # 각 위치(인덱스 리스트)에 대해
                tmp = conds[:4].copy() # 조건(점수 제외) 복사
                for idx in idxs:   # 각 인덱스에 대해
                    tmp[idx] = "-" # 인덱스에 "-" 삽입
                info_dict_key = "_".join(tmp)
                '''dict.get() 쓰면 시간 2배 이상 걸림'''
                # info_dict[info_dict_key] = info_dict.get(info_dict_key,[]) + [int(conds[4])]
                if info_dict_key in info_dict: info_dict[info_dict_key].append(int(conds[4]))
                else: info_dict[info_dict_key] = [int(conds[4])]
    '''먼저 정렬하지 않고 탐색 시마다 정렬하면 시간 초과'''
    # 점수 정렬 -> 이진 탐색 알고리즘
    for values in info_dict.values(): 
        values.sort()
    
    # 조건 비교
    from bisect import bisect_left
    ans = []
    for query in querys:
        query = [q for q in query.split() if q != "and"]
        info_dict_key = "_".join(query[:4])
        if info_dict_key in info_dict: # 조건을 만족하는 후보에 한해 커트라인 점수를 적용 
            candidates = info_dict[info_dict_key]
            cnt = len(candidates) - bisect_left(candidates,int(query[4]))
        else:                          # 조건을 만족하는 후보가 없음
            cnt = 0
        ans.append(cnt)

    return ans
```

<br>

✋ <span style="color:rgb(244,24,24)">**14번째 줄을 보면 조건문 대신 딕셔너리의 get() 메서드를 쓰면 시간이 2배 이상 걸린다고 되어있죠? 네 저는 get 메서드가 코드를 간결하게 해주어서 자주 애용했는데, 이번에 그 메서드 하나 때문에 시간초과가 나는 것을 보고 매우 놀랐어요 ㅎㅎㅎ 진짜 마지막으로 참고 삼아 조건문을 썼을 때와 get 메서드를 썼을 때의 테스트 결과를 올리면서 마치겠습니다. (이제 get 안써야지...)**</span>

```
''' dict.get() 안 썼을 때
정확성  테스트
테스트 1 〉	통과 (0.92ms, 10.3MB)
테스트 2 〉	통과 (2.26ms, 10.4MB)
테스트 3 〉	통과 (1.24ms, 10.4MB)
테스트 4 〉	통과 (1.89ms, 10.5MB)
테스트 5 〉	통과 (3.16ms, 10.4MB)
테스트 6 〉	통과 (91.08ms, 10.5MB)
테스트 7 〉	통과 (4.74ms, 10.7MB)
테스트 8 〉	통과 (83.77ms, 11.5MB)
테스트 9 〉	통과 (116.94ms, 13.3MB)
테스트 10 〉	통과 (75.92ms, 13.9MB)
테스트 11 〉	통과 (3.78ms, 10.5MB)
테스트 12 〉	통과 (10.95ms, 10.7MB)
테스트 13 〉	통과 (3.74ms, 10.7MB)
테스트 14 〉	통과 (56.83ms, 12.1MB)
테스트 15 〉	통과 (53.62ms, 12MB)
테스트 16 〉	통과 (5.76ms, 10.4MB)
테스트 17 〉	통과 (8.37ms, 10.7MB)
테스트 18 〉	통과 (43.91ms, 12.1MB)
효율성  테스트
테스트 1 〉	통과 (982.25ms, 64.4MB)
테스트 2 〉	통과 (992.74ms, 63.9MB)
테스트 3 〉	통과 (984.64ms, 63.8MB)
테스트 4 〉	통과 (875.99ms, 64.4MB)
채점 결과
정확성: 40.0
효율성: 60.0
합계: 100.0 / 100.0
'''

''' dict.get() 썼을 대
정확성  테스트
테스트 1 〉	통과 (1.07ms, 10.4MB)
테스트 2 〉	통과 (1.10ms, 10.4MB)
테스트 3 〉	통과 (1.66ms, 10.3MB)
테스트 4 〉	통과 (3.68ms, 10.5MB)
테스트 5 〉	통과 (3.37ms, 10.4MB)
테스트 6 〉	통과 (18.59ms, 10.4MB)
테스트 7 〉	통과 (3.81ms, 10.7MB)
테스트 8 〉	통과 (307.80ms, 11.6MB)
테스트 9 〉	통과 (290.81ms, 13.4MB)
테스트 10 〉	통과 (429.54ms, 14.1MB)
테스트 11 〉	통과 (5.84ms, 10.5MB)
테스트 12 〉	통과 (19.18ms, 10.8MB)
테스트 13 〉	통과 (6.81ms, 10.7MB)
테스트 14 〉	통과 (117.25ms, 12.2MB)
테스트 15 〉	통과 (121.46ms, 12.1MB)
테스트 16 〉	통과 (4.99ms, 10.5MB)
테스트 17 〉	통과 (41.35ms, 10.7MB)
테스트 18 〉	통과 (107.22ms, 12.1MB)
효율성  테스트
테스트 1 〉	실패 (시간 초과)
테스트 2 〉	실패 (시간 초과)
테스트 3 〉	실패 (시간 초과)
테스트 4 〉	실패 (시간 초과)
채점 결과
정확성: 40.0
효율성: 0.0
합계: 40.0 / 100.0
'''
```

