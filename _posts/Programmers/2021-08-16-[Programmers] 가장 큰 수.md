---
layout: single
title: "[Programmers] 가장 큰 수"
categories: ['Algorithm', 'Programmers']
---

# 가장 큰 수

### 문제 설명

---

##### 문제 설명

0 또는 양의 정수가 주어졌을 때, 정수를 이어 붙여 만들 수 있는 가장 큰 수를 알아내 주세요.

예를 들어, 주어진 정수가 [6, 10, 2]라면 [6102, 6210, 1062, 1026, 2610, 2106]를 만들 수 있고, 이중 가장 큰 수는 6210입니다.

0 또는 양의 정수가 담긴 배열 numbers가 매개변수로 주어질 때, 순서를 재배치하여 만들 수 있는 가장 큰 수를 문자열로 바꾸어 return 하도록 solution 함수를 작성해주세요.

##### 제한 사항

* numbers의 길이는 1 이상 100,000 이하입니다.
* numbers의 원소는 0 이상 1,000 이하입니다.
* 정답이 너무 클 수 있으니 문자열로 바꾸어 return 합니다.

##### 입출력 예

| numbers           | return    |
| ----------------- | --------- |
| [6, 10, 2]        | "6210"    |
| [3, 30, 34, 5, 9] | "9534330" |

<br>



### 문제 풀이

---

규칙을 찾아서 최적의 정렬 방법을 수행해야 하는 문제였습니다. 

이 문제의 규칙은 다음과 같습니다. 예를 들어 15와 151이 있다면, 두 숫자를 길이가 4가 될 때까지 복사해서 서로 비교하면 됩니다. 

이게 무슨 말이냐면, 

* 15 ➡ 1515
* 151 ➡ 1511(51)

이 되어서 15가 앞에 와야 합니다. 

길이가 4가 될때까지인 이유는 이 문제에서 숫자의 최댓값이 1,000이기 때문입니다. 

<br>

이러한 규칙을 찾지 않고 **서로 인접한 두 수를 비교하면서 정렬을 한다면(버블 정렬),** numbers의 길이가 100,000이나 되기 때문에 **O(n<sup>2</sup>)** 의 시간 복잡도로 시간 초과가 납니다. 

예를 들면 아래 코드와 같이 말이죠😂

```python
# 시간 초과
def solution(numbers):
    if sum(numbers) == 0: 
        return '0'
    numbers = sorted(list(map(str, numbers)), reverse=True)

    for i in range(len(numbers)-1):
        for j in range(len(numbers)-1-i):
            if int(numbers[j]+numbers[j+1]) < int(numbers[j+1]+numbers[j]):
                numbers[j], numbers[j+1] = numbers[j+1], numbers[j]
    
    return ''.join(numbers)
```

<br>

찾은 규칙을 반영한 코드는 아래와 같습니다. 

```python
def solution(nums):
    if sum(nums) == 0: return '0'
    nums = list(map(str,nums))
    for i in range(len(nums)): nums[i] = [(nums[i]*4)[:4],len(nums[i])]
    nums.sort(key=lambda x:int(x[0]),reverse=True)
    return ''.join(list(map(lambda x:x[0][:x[1]],nums)))
```

각 nums의 원소를 길이 4가 될때까지 복사하여 늘린 수로 대체하고, 동시에 그 길이의 원래 길이를 저장합니다. 

이는 정답을 반환할 때 원래 숫자의 길이만큼 잘라서 반환하기 위함입니다. 

**그런데...!!! 이렇게 하지 않아도 됩니다.** 아래 코드를 보시죠. 

<br>

```python
def solution(nums):
    if sum(nums) == 0: return '0'
    nums = list(map(str,nums))
    nums.sort(key=lambda num:num*3, reverse=True)
    return ''.join(nums) 
```

사실 그냥 정렬만 찾은 기준으로 하면 되죠. 너무 어렵게 생각했던 것 같습니다...

저의 처음 코드는 쓸데없는 코드들을 양산했습니다. 

* **리스트의 각 원소를 대체할 필요없이 정렬만 수행하면 된다.**
* **대체하지 않는다면 길이도 저장할 필요가 없고, 정렬하여 바로 반환하면 된다.**

문자열 정렬의 경우 각 문자열 인덱스의 아스키 코드값을 기준으로 정렬하기 때문에 숫자로 변환해야 할 필요도 없습니다. 

<br>

✋ **만약 '999'와 '999999' 를 비교하면 '999'까지는 두 문자열이 모두 같기 때문에 길이가 더 긴 '999999'의 정렬 우선 순위가 더 높습니다. 어쨌든 이 문제에서는 두 숫자 중 뭐가 앞에 와도 상관없습니다(정렬 우선순위가 같습니다). **

<br>

**num*4**를 기준으로 정렬해야 하지 않냐고 하실 분들도 계실텐데, 문제에서 최댓값인 1,000이 4자리 수이긴 하지만 한자리 수인 1~9 보다 정렬 기준에서 모두 작으므로 **num*3**으로만 비교해도 됩니다. 

<br>

#### 우리 모두 쓸 데 없는 코드를 양산하지 않도록 노력합시다..!!! 😁



