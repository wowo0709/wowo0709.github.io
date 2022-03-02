---
layout: single
title: "[Programmers] 카펫"
categories: ['Algorithm', 'Implementation', 'Bruteforce', 'Math']
---

# 카펫

##### 문제 설명

Leo는 카펫을 사러 갔다가 아래 그림과 같이 중앙에는 노란색으로 칠해져 있고 테두리 1줄은 갈색으로 칠해져 있는 격자 모양 카펫을 봤습니다.

<img src="https://grepp-programmers.s3.ap-northeast-2.amazonaws.com/files/production/b1ebb809-f333-4df2-bc81-02682900dc2d/carpet.png" alt="carpet.png" style="zoom:50%;" />

Leo는 집으로 돌아와서 아까 본 카펫의 노란색과 갈색으로 색칠된 격자의 개수는 기억했지만, 전체 카펫의 크기는 기억하지 못했습니다.

Leo가 본 카펫에서 갈색 격자의 수 brown, 노란색 격자의 수 yellow가 매개변수로 주어질 때 카펫의 가로, 세로 크기를 순서대로 배열에 담아 return 하도록 solution 함수를 작성해주세요.

##### 제한사항

* 갈색 격자의 수 brown은 8 이상 5,000 이하인 자연수입니다.
* 노란색 격자의 수 yellow는 1 이상 2,000,000 이하인 자연수입니다.
* 카펫의 가로 길이는 세로 길이와 같거나, 세로 길이보다 깁니다.

##### 입출력 예

| brown | yellow | return |
| ----- | ------ | ------ |
| 10    | 2      | [4, 3] |
| 8     | 1      | [3, 3] |
| 24    | 24     | [8, 6] |

[출처](http://hsin.hr/coci/archive/2010_2011/contest4_tasks.pdf)

※ 공지 - 2020년 2월 3일 테스트케이스가 추가되었습니다.
※ 공지 - 2020년 5월 11일 웹접근성을 고려하여 빨간색을 노란색으로 수정하였습니다.

<br>



### 문제 풀이

---

음...2단계치고는 굉장히 쉬운 문제! 왜 2단계인지 모르겠는 문제!

이 문제는 푸는 방법이 상당히 다양한 듯 합니다. 넓이로 풀이, 둘레로 풀이, 방정식을 이용한 근의 공식으로 풀이 ...등등

저는 그 중 가장 보편적(?)이고 떠올리기 쉬운 넓이로 접근해보았습니다. 

<br>

너비가 높이보다 크거나 같다고 했으니 w는 size부터 1씩 작아지면서 for문을 돕니다. (어차피 w는 h(size/w)보다 큰 값에서 찾아지기 때문에 range에 하한을 둘 필요는 없습니다.)

for 문 안에서는 w가 size의 약수인지와 넓이에 부합하는지 검사하고, 맞다면 너비와 높이를 리턴합니다. 

```python
def solution(brown, yellow):
    size = brown + yellow
    for w in range(size,0,-1):
        if size % w == 0 and (w-2)*(size//w-2)==yellow:
            return [w,size//w] # h = size / w
```

