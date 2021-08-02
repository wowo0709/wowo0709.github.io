# 위장

##### 문제 설명

초 단위로 기록된 주식가격이 담긴 배열 prices가 매개변수로 주어질 때, 가격이 떨어지지 않은 기간은 몇 초인지를 return 하도록 solution 함수를 완성하세요.

##### 제한사항

* prices의 각 가격은 1 이상 10,000 이하인 자연수입니다.
* prices의 길이는 2 이상 100,000 이하입니다.

##### 입출력 예

| prices          | return          |
| --------------- | --------------- |
| [1, 2, 3, 2, 3] | [4, 3, 1, 1, 0] |

##### 입출력 예 설명

* 1초 시점의 ₩1은 끝까지 가격이 떨어지지 않았습니다.
* 2초 시점의 ₩2은 끝까지 가격이 떨어지지 않았습니다.
* 3초 시점의 ₩3은 1초뒤에 가격이 떨어집니다. 따라서 1초간 가격이 떨어지지 않은 것으로 봅니다.
* 4초 시점의 ₩2은 1초간 가격이 떨어지지 않았습니다.
* 5초 시점의 ₩3은 0초간 가격이 떨어지지 않았습니다.

※ 공지 - 2019년 2월 28일 지문이 리뉴얼되었습니다.

<br>



### 문제 풀이

---

이중 for문을 돌며 가격이 떨어진다면 그 시간 차를 answer에 append하는 쉬운 방식. 

```python
def solution(prices):
    answer = []
    N = len(prices)
    for i in range(N):
        for j in range(i,N):
            if prices[i] > prices[j] or j == N-1:
                answer.append(j-i)
                break

    return answer
```

<br>

우선 answer 리스트를 '길이-1' 부터 0까지 내림차순으로 초기화한다. 끝까지 떨어지지 않았을 경우 그대로 값을 가져가면 되도록 이렇게 초기화한다. 

스택 s를 사용하여 각 time에 대해 s에 아직 떨어진 시각이 없을 경우 그대로 time을 스택에 push하고 떨어진 경우 떨어지지 않은 시각까지  스택에서 pop한 이후에 스택에 push한다. 

pop하면서 그 시각으로 해당 시각의 answer 값을 초기화한다.  

```python
def solution(prices):
    answer = [i for i in range(len(prices)-1,-1,-1)]
    s = []
    for time in range(len(prices)):
        while tmp and prices[time] < prices[tmp[-1]]:
                t = tmp.pop()
                answer[t] = time - t
        tmp.append(time)
    return answer
```

<br>

<br>

<br>

