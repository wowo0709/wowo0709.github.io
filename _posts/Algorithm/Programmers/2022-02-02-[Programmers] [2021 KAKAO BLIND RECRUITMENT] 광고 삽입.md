---
layout: single
title: "[Programmers] [2021 KAKAO BLIND RECRUITMENT] 광고 삽입"
categories: ['Algorithm', 'SequentialSearch', 'DynamicProgramming']
toc: true
toc_sticky: true
tag: ['Memoization']
---



<br>

## 문제 설명

### 문제 설명

`카카오TV`에서 유명한 크리에이터로 활동 중인 `죠르디`는 환경 단체로부터 자신의 가장 인기있는 동영상에 지구온난화의 심각성을 알리기 위한 공익광고를 넣어 달라는 요청을 받았습니다. 평소에 환경 문제에 관심을 가지고 있던 "죠르디"는 요청을 받아들였고 광고효과를 높이기 위해 시청자들이 가장 많이 보는 구간에 공익광고를 넣으려고 합니다. "죠르디"는 시청자들이 해당 동영상의 어떤 구간을 재생했는 지 알 수 있는 재생구간 기록을 구했고, 해당 기록을 바탕으로 공익광고가 삽입될 최적의 위치를 고를 수 있었습니다.
참고로 광고는 재생 중인 동영상의 오른쪽 아래에서 원래 영상과 `동시에 재생되는` PIP(Picture in Picture) 형태로 제공됩니다.

![2021_kakao_cf_01.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/597ec277-4451-4289-8817-2970be644a69/2021_kakao_cf_01.png)

다음은 "죠르디"가 공익광고가 삽입될 최적의 위치를 고르는 과정을 그림으로 설명한 것입니다.
![2021_kakao_cf_02.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/e733fafb-1e6b-4d30-bbab-a22f366229e7/2021_kakao_cf_02.png)

- 그림의 파란색 선은 광고를 검토 중인 "죠르디" 동영상의 전체 재생 구간을 나타냅니다.
  - 위 그림에서, "죠르디" 동영상의 총 재생시간은 `02시간 03분 55초` 입니다.
- 그림의 검은색 선들은 각 시청자들이 "죠르디"의 동영상을 재생한 구간의 위치를 표시하고 있습니다.
  - 검은색 선의 가운데 숫자는 각 재생 기록을 구분하는 ID를 나타냅니다.
  - 검은색 선에 표기된 왼쪽 끝 숫자와 오른쪽 끝 숫자는 시청자들이 재생한 동영상 구간의 시작 시각과 종료 시각을 나타냅니다.
  - 위 그림에서, 3번 재생 기록은 `00시 25분 50초` 부터 `00시 48분 29초` 까지 총 `00시간 22분 39초` 동안 죠르디의 동영상을 재생했습니다. [1](https://programmers.co.kr/learn/courses/30/lessons/72414#fn1)
  - 위 그림에서, 1번 재생 기록은 `01시 20분 15초` 부터 `01시 45분 14초` 까지 총 `00시간 24분 59초` 동안 죠르디의 동영상을 재생했습니다.
- 그림의 빨간색 선은 "죠르디"가 선택한 최적의 공익광고 위치를 나타냅니다.
  - 만약 공익광고의 재생시간이 `00시간 14분 15초`라면, 위의 그림처럼 `01시 30분 59초` 부터 `01시 45분 14초` 까지 공익광고를 삽입하는 것이 가장 좋습니다. 이 구간을 시청한 시청자들의 누적 재생시간이 가장 크기 때문입니다.
  - `01시 30분 59초` 부터 `01시 45분 14초`까지의 누적 재생시간은 다음과 같이 계산됩니다.
    - `01시 30분 59초` 부터 `01시 37분 44초` 까지 : 4번, 1번 재생 기록이 두차례 있으므로 재생시간의 합은 `00시간 06분 45초` X 2 = `00시간 13분 30초`
    - `01시 37분 44초` 부터 `01시 45분 14초` 까지 : 4번, 1번, 5번 재생 기록이 세차례 있으므로 재생시간의 합은 `00시간 07분 30초` X 3 = `00시간 22분 30초`
    - 따라서, 이 구간 시청자들의 누적 재생시간은 `00시간 13분 30초` + `00시간 22분 30초` = `00시간 36분 00초`입니다.

### **[문제]**

"죠르디"의 동영상 재생시간 길이 play_time, 공익광고의 재생시간 길이 adv_time, 시청자들이 해당 동영상을 재생했던 구간 정보 logs가 매개변수로 주어질 때, 시청자들의 누적 재생시간이 가장 많이 나오는 곳에 공익광고를 삽입하려고 합니다. 이때, 공익광고가 들어갈 `시작 시각`을 구해서 return 하도록 solution 함수를 완성해주세요. 만약, 시청자들의 누적 재생시간이 가장 많은 곳이 여러 곳이라면, 그 중에서 `가장 빠른 시작 시각`을 return 하도록 합니다.

### **[제한사항]**

- play_time, adv_time은 길이 8로 고정된 문자열입니다.
  - play_time, adv_time은 `HH:MM:SS` 형식이며, `00:00:01` 이상 `99:59:59` 이하입니다.
  - 즉, 동영상 재생시간과 공익광고 재생시간은 `00시간 00분 01초` 이상 `99시간 59분 59초` 이하입니다.
  - 공익광고 재생시간은 동영상 재생시간보다 짧거나 같게 주어집니다.
- logs는 크기가 1 이상 300,000 이하인 문자열 배열입니다.
  - logs 배열의 각 원소는 시청자의 재생 구간을 나타냅니다.
  - logs 배열의 각 원소는 길이가 17로 고정된 문자열입니다.
  - logs 배열의 각 원소는 `H1:M1:S1-H2:M2:S2` 형식입니다.
    - `H1:M1:S1`은 동영상이 시작된 시각, `H2:M2:S2`는 동영상이 종료된 시각을 나타냅니다.
    - `H1:M1:S1`는 `H2:M2:S2`보다 1초 이상 이전 시각으로 주어집니다.
    - `H1:M1:S1`와 `H2:M2:S2`는 play_time 이내의 시각입니다.
- 시간을 나타내는 `HH, H1, H2`의 범위는 00~99, 분을 나타내는 `MM, M1, M2`의 범위는 00~59, 초를 나타내는 `SS, S1, S2`의 범위는 00~59까지 사용됩니다. 잘못된 시각은 입력으로 주어지지 않습니다. (예: `04:60:24`, `11:12:78`, `123:12:45` 등)
- return 값의 형식
  - 공익광고를 삽입할 시각을 `HH:MM:SS` 형식의 8자리 문자열로 반환합니다.

### **[입출력 예]**

| play_time    | adv_time     | logs                                                         | result       |
| ------------ | ------------ | ------------------------------------------------------------ | ------------ |
| `"02:03:55"` | `"00:14:15"` | `["01:20:15-01:45:14", "00:40:31-01:00:00", "00:25:50-00:48:29", "01:30:59-01:53:29", "01:37:44-02:02:30"]` | `"01:30:59"` |
| `"99:59:59"` | `"25:00:00"` | `["69:59:59-89:59:59", "01:00:00-21:00:00", "79:59:59-99:59:59", "11:00:00-31:00:00"]` | `"01:00:00"` |
| `"50:00:00"` | `"50:00:00"` | `["15:36:51-38:21:49", "10:14:18-15:36:51", "38:21:49-42:51:45"]` | `"00:00:00"` |

#### **입출력 예에 대한 설명**

------

**입출력 예 #1**
문제 예시와 같습니다.

**입출력 예 #2**
![2021_kakao_cf_03.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/0e58c7f5-2b81-43f2-95e1-c504f17aab9b/2021_kakao_cf_03.png)

`01:00:00`에 공익광고를 삽입하면 `26:00:00`까지 재생되며, 이곳이 가장 좋은 위치입니다. 이 구간의 시청자 누적 재생시간은 다음과 같습니다.

- `01:00:00-11:00:00` : 해당 구간이 1회(2번 기록) 재생되었으므로 누적 재생시간은 `10시간 00분 00초` 입니다.
- `11:00:00-21:00:00` : 해당 구간이 2회(2번, 4번 기록) 재생되었으므로 누적 재생시간은 `20시간 00분 00초` 입니다.
- `21:00:00-26:00:00` : 해당 구간이 1회(4번 기록) 재생되었으므로 누적 재생시간은 `05시간 00분 00초` 입니다.
- 따라서, 이 구간의 시청자 누적 재생시간은 `10시간 00분 00초` + `20시간 00분 00초` + `05시간 00분 00초` = `35시간 00분 00초` 입니다.
- 초록색으로 표시된 구간(`69:59:59-94:59:59`)에 광고를 삽입해도 동일한 결과를 얻을 수 있으나, `01:00:00`이 `69:59:59` 보다 빠른 시각이므로, `"01:00:00"`을 return 합니다.

**입출력 예 #3**
![2021_kakao_cf_04.png](https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/8e564c82-00ce-4e1a-80fc-5cd96e465a69/2021_kakao_cf_04.png)

동영상 재생시간과 공익광고 재생시간이 같으므로, 삽입할 수 있는 위치는 맨 처음(`00:00:00`)이 유일합니다.

------

1. `동영상 재생시간 = 재생이 종료된 시각 - 재생이 시작된 시각`(예를 들어, `00시 00분 01초`부터 `00시 00분 10초`까지 동영상이 재생되었다면, 동영상 재생시간은 `9초` 입니다.) [↩](https://programmers.co.kr/learn/courses/30/lessons/72414#fnref1)

<br>

## 문제 풀이

### \# SequentialSearch \# DynamicProgramming

<br>

### 풀이 과정

카카오 기출에서 자주 나오는 시간을 활용한 문제입니다. 



<br>

### 전체 코드

😂 **1번 풀이: 실패**

문제를 처음 봤을 때 2가지 풀이법이 떠올랐습니다. 

1. 모든 시간(1초 간격)에 대해 리스트를 만들어서 누적 플레이 시간을 구한다. -> 메모리 초과 위험..?
2. 플레이어 유입 시각을 기준으로 딕셔너리를 만든다. 나중에 각 플레이 시간과 키 값의 시간을 비교하여 누적 플레이 시간을 구하다. -> 시간 초과 위험..?

이 문제의 입력 크기가 300,000이기 때문에 2번째 방법의 경우 최악의 경우 (300,000)^2 만큼의 시간 복잡도가 발생합니다. 일단 시도는 해봤지만... 당연히 시간초과

그리고 **광고 시작 시간은 플레이어 유입 시각 중 하나일 것이다**라는 가정도 틀린 것 같네요 😂

```python
def solution(play_hms, adv_hms, logs):
    '''
    1. 최적 구간의 시작 시각은 어떤 사용자의 재생시작 시각과 같다라는 가설을 세운다. 
    2. 각 플레이어의 시작 시각에 대해 '시작 시각~끝 시각'(길이는 광고 재생시간) 해쉬 맵 생성
    3. 각 로그를 하나씩 보면서 시작/끝 시각을 비교하면서 해당 최적 구간 후보의 값을 더해간다. 
    4. 최종적으로 가장 값이 큰 최적 구간 후보의 시작 시각이 정답.
    -> 최대 시간복잡도는 300000(로그의 수)*300000(최적구간 후보의 수)
    '''
    def convert_hms_to_time(hms):
        h, m, s = map(int,hms.split(':'))    
        return 3600*h + 60*m + s
    def convert_time_to_hms(time):
        h, m = divmod(time, 3600)
        m, s = divmod(m, 60)
        return ':'.join([t.zfill(2) for t in map(str,[h,m,s])])

    adv_time = convert_hms_to_time(adv_hms)
    play_time = convert_hms_to_time(play_hms)
    cumulative_times = {(0, adv_time): 0}
    for e, log in enumerate(logs):
        start_hms, end_hms = log.split('-')
        start_time, end_time = convert_hms_to_time(start_hms), convert_hms_to_time(end_hms)
        logs[e] = (start_time, end_time)
        if start_time+adv_time <= play_time:
            cumulative_times[(start_time, start_time+adv_time)] = 0
    for play_start_time, play_end_time in logs:
        for (adv_start_time, adv_end_time), v in cumulative_times.items():
            cumulative_times[(adv_start_time, adv_end_time)] += max(0,(min(play_end_time,adv_end_time)-max(play_start_time,adv_start_time)))

    return convert_time_to_hms(sorted(cumulative_times.keys(), key=lambda x:(cumulative_times[x],-x[0]))[-1][0])

'''
정확성  테스트
테스트 1 〉	통과 (150.19ms, 10.5MB)
테스트 2 〉	통과 (1329.96ms, 11MB)
테스트 3 〉	실패 (시간 초과)
테스트 4 〉	실패 (시간 초과)
테스트 5 〉	실패 (시간 초과)
테스트 6 〉	통과 (529.68ms, 10.7MB)
테스트 7 〉	실패 (시간 초과)
테스트 8 〉	실패 (시간 초과)
테스트 9 〉	통과 (922.70ms, 49.6MB)
테스트 10 〉	실패 (시간 초과)
테스트 11 〉	실패 (시간 초과)
테스트 12 〉	실패 (시간 초과)
테스트 13 〉	실패 (시간 초과)
테스트 14 〉	실패 (시간 초과)
테스트 15 〉	실패 (52.79ms, 10.5MB)
테스트 16 〉	실패 (시간 초과)
테스트 17 〉	실패 (시간 초과)
테스트 18 〉	실패 (시간 초과)
테스트 19 〉	통과 (49.15ms, 10.5MB)
테스트 20 〉	통과 (5.83ms, 10.5MB)
테스트 21 〉	통과 (2980.61ms, 20.3MB)
테스트 22 〉	통과 (217.89ms, 20.3MB)
테스트 23 〉	실패 (시간 초과)
테스트 24 〉	실패 (시간 초과)
테스트 25 〉	통과 (1.53ms, 10.5MB)
테스트 26 〉	실패 (0.05ms, 10.5MB)
테스트 27 〉	실패 (0.04ms, 10.5MB)
테스트 28 〉	통과 (0.04ms, 10.5MB)
테스트 29 〉	실패 (0.05ms, 10.4MB)
테스트 30 〉	실패 (0.04ms, 10.5MB)
테스트 31 〉	통과 (0.05ms, 10.5MB)
채점 결과
정확성: 35.5
합계: 35.5 / 100.0
'''
```

😂 **2번 풀이: 실패**

그래서! 앞서 생각한 1번째 방법으로 시도해봤습니다. 

모든 시각(1초 간격)에 대해 리스트 `all_times`를 만들고(최대 크기 100*3600 = 3,600,000), 각 플레이어의 재생 시간 내의 각 리스트 원소에 +1을 해서 해당 시간에 존재하는 플레이어 수를 구한다. 

그리고 이 값을 이용해 마지막에 누적 플레이 수를 구한다. 

그런데 이 방법 또한 **시간 초과**가 발생합니다. 그렇다면 어떻게 해서 시간을 줄일 수 있을까요??

```python
def solution(play_hms, adv_hms, logs):
    def convert_hms_to_time(hms):
        h, m, s = map(int,hms.split(':'))    
        return 3600*h + 60*m + s
    def convert_time_to_hms(time):
        h, m = divmod(time, 3600)
        m, s = divmod(m, 60)
        return ':'.join([t.zfill(2) for t in map(str,[h,m,s])])

    adv_time = convert_hms_to_time(adv_hms)
    play_time = convert_hms_to_time(play_hms)
    cumulative_time = [0 for _ in range(play_time+1)]
    for log in logs:
        start_time, end_time = map(convert_hms_to_time, log.split('-'))
        for time in range(start_time, end_time):
            cumulative_time[time] += 1
            
    most_playing_num = sum(cumulative_time[:adv_time])
    playing_num = sum(cumulative_time[:adv_time])
    most_playing_time = 0
    for time in range(play_time-adv_time):
        playing_num = playing_num - cumulative_time[time-1] + cumulative_time[time+adv_time]
        if most_playing_num < playing_num:
            most_playing_num = playing_num
            most_playing_time = time
    return convert_time_to_hms(most_playing_time)

'''
정확성  테스트
테스트 1 〉	실패 (3.72ms, 10.5MB)
테스트 2 〉	실패 (63.93ms, 10.6MB)
테스트 3 〉	통과 (215.74ms, 11.4MB)
테스트 4 〉	실패 (시간 초과)
테스트 5 〉	통과 (234.79ms, 23.2MB)
테스트 6 〉	실패 (54.13ms, 13.2MB)
테스트 7 〉	실패 (시간 초과)
테스트 8 〉	실패 (시간 초과)
테스트 9 〉	실패 (시간 초과)
테스트 10 〉	실패 (시간 초과)
테스트 11 〉	실패 (시간 초과)
테스트 12 〉	실패 (1629.38ms, 40.9MB)
테스트 13 〉	실패 (시간 초과)
테스트 14 〉	실패 (시간 초과)
테스트 15 〉	실패 (112.80ms, 11.3MB)
테스트 16 〉	실패 (시간 초과)
테스트 17 〉	실패 (시간 초과)
테스트 18 〉	실패 (788.57ms, 40.9MB)
테스트 19 〉	통과 (1.13ms, 10.5MB)
테스트 20 〉	통과 (8.60ms, 10.5MB)
테스트 21 〉	실패 (274.84ms, 20.2MB)
테스트 22 〉	통과 (257.67ms, 20.3MB)
테스트 23 〉	실패 (시간 초과)
테스트 24 〉	통과 (918.09ms, 40.9MB)
테스트 25 〉	실패 (40.56ms, 12.4MB)
테스트 26 〉	실패 (31.80ms, 12.1MB)
테스트 27 〉	실패 (29.60ms, 12.9MB)
테스트 28 〉	실패 (25.19ms, 13.1MB)
테스트 29 〉	실패 (24.22ms, 13.3MB)
테스트 30 〉	실패 (21.20ms, 12.4MB)
테스트 31 〉	통과 (23.28ms, 12.6MB)
채점 결과
정확성: 22.6
합계: 22.6 / 100.0
'''
```





😁 **3번 풀이: 성공**

성공한 3번째 풀이..!!

2번째 풀이와의 차이점은 `all_time` 리스트에 플레이어 수를 기록하는 방식에 있습니다. 

2번째 풀이에서는 매 log마다 해당 구간 내의 원소들에 모두 +1을 해주었기 때문에 그 과정에 **(log의 수) x (각 log의 길이)** 만큼의 시간이 소요되며, 이는 최대 **(300,000) x (3,600,000)**의 시간입니다. 

하지만 3번째 풀이에서는 처음에 매 log마다 시작과 종료 시각에 해당하는 원소에만 각각 +1, -1을 해주고, 다음으로 순차 탐색을 하며 `all_time[i] - all_time[i-1]`이라는 식으로 그 시간에 존재하는 플레이어 수를 구합니다. 이렇게 하면 이 과정에 필요한 시간이 **(log의 수) + (3,600,000)**로 줄어들게 됩니다. 

비슷한 풀이에 대한 상세한 해설을 적어놓은 포스팅을 [여기](https://dev-note-97.tistory.com/156)에 남깁니다. 

```python
def solution(play_hms, adv_hms, logs):
    def convert_hms_to_time(hms):
        h, m, s = map(int,hms.split(':'))    
        return 3600*h + 60*m + s
    def convert_time_to_hms(time):
        h, m = divmod(time, 3600)
        m, s = divmod(m, 60)
        return ':'.join([t.zfill(2) for t in map(str,[h,m,s])])

    adv_time = convert_hms_to_time(adv_hms)
    play_time = convert_hms_to_time(play_hms)
    all_time = [0 for _ in range(play_time+1)]
    for log in logs: # 플레이어가 유입/유출되는 시점만 기록
        start_time, end_time = map(convert_hms_to_time, log.split('-'))
        all_time[start_time] += 1
        all_time[end_time] -= 1
    for i in range(1, len(all_time)): # 1초(i-1 ~ i) 동안 존재하는 플레이어 수
        all_time[i] = all_time[i] + all_time[i-1]
    for i in range(1, len(all_time)): # i초(0 ~ i) 동안 누적 플레이 시간
        all_time[i] = all_time[i] + all_time[i-1]

    most_playing_cnt = all_time[adv_time]
    most_playing_time = 0
    for i in range(play_time-adv_time+1): # i ~ i+adv_time 동안 누적 플레이 수가 가장 많은 구간을 탐색
        if most_playing_cnt < all_time[i+adv_time] - all_time[i]:
            most_playing_cnt = all_time[i+adv_time] - all_time[i]
            most_playing_time = i+1
    return convert_time_to_hms(most_playing_time)

'''
정확성  테스트
테스트 1 〉	통과 (1.60ms, 10.5MB)
테스트 2 〉	통과 (8.10ms, 10.6MB)
테스트 3 〉	통과 (16.49ms, 11.2MB)
테스트 4 〉	통과 (196.10ms, 28MB)
테스트 5 〉	통과 (282.78ms, 34.3MB)
테스트 6 〉	통과 (113.73ms, 21.6MB)
테스트 7 〉	통과 (429.99ms, 41.1MB)
테스트 8 〉	통과 (458.26ms, 45.9MB)
테스트 9 〉	통과 (589.98ms, 54.3MB)
테스트 10 〉	통과 (633.74ms, 54.7MB)
테스트 11 〉	통과 (663.94ms, 52.2MB)
테스트 12 〉	통과 (695.72ms, 49.7MB)
테스트 13 〉	통과 (685.04ms, 54.6MB)
테스트 14 〉	통과 (496.30ms, 40.9MB)
테스트 15 〉	통과 (42.91ms, 15.2MB)
테스트 16 〉	통과 (539.50ms, 41MB)
테스트 17 〉	통과 (678.09ms, 54.8MB)
테스트 18 〉	통과 (606.79ms, 42.3MB)
테스트 19 〉	통과 (1.54ms, 10.6MB)
테스트 20 〉	통과 (1.66ms, 10.5MB)
테스트 21 〉	통과 (167.97ms, 20.3MB)
테스트 22 〉	통과 (171.27ms, 20.3MB)
테스트 23 〉	통과 (603.32ms, 47.1MB)
테스트 24 〉	통과 (676.28ms, 40.9MB)
테스트 25 〉	통과 (86.71ms, 19.6MB)
테스트 26 〉	통과 (84.54ms, 14.9MB)
테스트 27 〉	통과 (109.61ms, 17.5MB)
테스트 28 〉	통과 (93.36ms, 17MB)
테스트 29 〉	통과 (61.06ms, 16.8MB)
테스트 30 〉	통과 (42.37ms, 14.1MB)
테스트 31 〉	통과 (48.22ms, 14.9MB)
채점 결과
정확성: 100.0
합계: 100.0 / 100.0
'''
```

<br>

## 정리

* **메모리**와 **시간** 제한은 (물론 문제마다 다르지만) 일반적으로 약 **N x 10,000,000 (1 <= N <= 9)**까지는 허용되는 것 같다. 이를 고려하여, 문제의 입력 크기를 보고 메모리 사용량과 문제 풀이 방식의 방향을 잡을 수 있을 것이다. 
* 각 데이터에 대해 그 구간 내의 모든 원소 값을 바로 더하는 것이 아니라, **처음에는 시작과 종료 값만 표시(유입과 유출만 표시)해놓고 순차탐색(dp[i] = dp[i] + dp[i-1])**으로 해당 시각에 존재하는 사람의 수를 구한다. 크... 정말 좋은 방법인 듯 합니다. 이렇게 **해당 시각에 존재하는 사용자 수 또는 특정 구간의 누적 사용자 수**를 구할 때 꼭 기억해두면 좋을 듯!!!
