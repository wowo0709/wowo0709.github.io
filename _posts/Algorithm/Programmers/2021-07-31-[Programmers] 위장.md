---
layout: single
title: "[Programmers] 위장"
categories: ['Algorithm', 'Implementation']
tag: ['Combinations']
---

# 위장

###### 문제 설명

스파이들은 매일 다른 옷을 조합하여 입어 자신을 위장합니다.

예를 들어 스파이가 가진 옷이 아래와 같고 오늘 스파이가 동그란 안경, 긴 코트, 파란색 티셔츠를 입었다면 다음날은 청바지를 추가로 입거나 동그란 안경 대신 검정 선글라스를 착용하거나 해야 합니다.

| 종류 | 이름                       |
| ---- | -------------------------- |
| 얼굴 | 동그란 안경, 검정 선글라스 |
| 상의 | 파란색 티셔츠              |
| 하의 | 청바지                     |
| 겉옷 | 긴 코트                    |

스파이가 가진 의상들이 담긴 2차원 배열 clothes가 주어질 때 서로 다른 옷의 조합의 수를 return 하도록 solution 함수를 작성해주세요.

##### 제한사항

* clothes의 각 행은 [의상의 이름, 의상의 종류]로 이루어져 있습니다.
* 스파이가 가진 의상의 수는 1개 이상 30개 이하입니다.
* 같은 이름을 가진 의상은 존재하지 않습니다.
* clothes의 모든 원소는 문자열로 이루어져 있습니다.
* 모든 문자열의 길이는 1 이상 20 이하인 자연수이고 알파벳 소문자 또는 '_' 로만 이루어져 있습니다.
* 스파이는 하루에 최소 한 개의 의상은 입습니다.

##### 입출력 예

| clothes                                                      | return |
| ------------------------------------------------------------ | ------ |
| [["yellowhat", "headgear"], ["bluesunglasses", "eyewear"], ["green_turban", "headgear"]] | 5      |
| [["crowmask", "face"], ["bluesunglasses", "face"], ["smoky_makeup", "face"]] | 3      |

##### 입출력 예 설명

예제 #1
headgear에 해당하는 의상이 yellow_hat, green_turban이고 eyewear에 해당하는 의상이 blue_sunglasses이므로 아래와 같이 5개의 조합이 가능합니다.

```
1. yellow_hat
2. blue_sunglasses
3. green_turban
4. yellow_hat + blue_sunglasses
5. green_turban + blue_sunglasses
```

예제 #2
face에 해당하는 의상이 crow_mask, blue_sunglasses, smoky_makeup이므로 아래와 같이 3개의 조합이 가능합니다.

```
1. crow_mask
2. blue_sunglasses
3. smoky_makeup
```

[출처](http://2013.bapc.eu/)

<br>



### 문제 풀이

---

**1번 풀이. 시간 초과**

대다수의 사람들이 최초에는 이 방법으로 접근할 것이라 생각합니다. 

주석으로 달아놓은 것처럼 먼저 몇 개를 입을 것인지 정하고, 그 다음 combination 함수로 입을 종류를 정하고, 마지막으로 입을 옷을 정하는 것이죠. 

그런데... 이렇게 하면 시간초과가 납니다 😂😂

문제 의도가 이게 아니라는 거고, 훨씬 나은 방법이 있다는 것이겠죠!

```python
def solution(clothes):
    from itertools import combinations

    cloth_dict = dict()
    for name, type in clothes:
        cloth_dict[type] = cloth_dict.get(type,0) + 1
    # 몇 개를 입을 것인지 정하고, 입을 종류를 정하고, 입을 옷을 정한다. 
    answer = 0
    for i in range(len(cloth_dict)+1):
        for j in list(combinations(cloth_dict.keys(),i)):
            tmp = 1
            for k in j: tmp *= cloth_dict[k]
            answer += tmp
        
    return answer - 1 # 0개 입는 경우 제외
```

<br>

**2번 풀이. 통과**

그래서! 수정한 풀이는 아래 보시는 것 처럼 경우의 수를 구하는 부분이 모두 사라진 코드입니다. 

사실 이 문제는 수학적인 사고를 조금만 해보면 해답이 나오는 문제인데, 각 의상 개수를 (실제 옷의 개수 + 안 입는 경우(1)) 로 초기화하는 것입니다. 그래서 최종적으로 각 의상 종류의 개수들을 모두 곱하고, 모두 안 입는 것은 안되므로 1을 빼주면 됩니다 😁

```python
def solution(clothes):
    cloth_dict = dict()
    for name, type in clothes:
        cloth_dict[type] = cloth_dict.get(type,1) + 1 # 안 입는 경우를 하나의 경우의 수로 친다. 

    answer = 1
    for i in cloth_dict.values(): answer *= i  
    return answer - 1 # 0개 입는 경우 제외
```

<br>

아래 코드는 라이브러리를 이용하여 좀 더 간단하게 짠 코드입니다. 

Counter 클래스로 각 의상 종류마다 개수가 몇 개인지 딕셔너리를 만들고, reduce 함수로 (실제 옷의 개수 + 1)을 모두 곱하고 마지막에 1을 빼서 답을 구합니다. 

```python
def solution(clothes):
    from collections import Counter
    from functools import reduce
    cloth_dict = Counter([type for name, type in clothes])
    # 안 입는 경우를 하나의 경우의 수로 치고 0개 입는 경우는 제외
    return reduce(lambda x, y: x*(y+1), cloth_dict.values(), 1) - 1
```

<br>

<br>

**[사용된 파이썬 스킬들]**

* [itertools, 원소의 경우의 수 (순열, 조합) 추출하기](https://yganalyst.github.io/etc/memo_18_itertools/)
* [functools 모듈의 reduce 함수](https://codepractice.tistory.com/86)
* [collections 모듈의 Counter 클래스 사용법](https://www.daleseo.com/python-collections-counter/)

