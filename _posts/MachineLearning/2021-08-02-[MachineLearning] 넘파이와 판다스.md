---
layout: single
title: "[MachineLearning] 넘파이와 판다스"
---



# 파이썬 머신러닝 완벽가이드

<br>

### 1장. 넘파이와 판다스

---

* 머신러닝 개요
    * 지도학습: 회귀, 분류
    * 비지도 학습: 군집화
    * 강화 학습

* 넘파이
    * 선형 대수, 통계 등을 사용할 때 사용하는 라이브러리로, 처리할 데이터의 수가 많아질 수록 파이썬, C++ 보다도 좋은 빠른 성능을 보임
    * ndarray: 기본적인 데이터 타입
    * arange: ndarray 인스턴스 생성
    * zeros, ones: 형상과 데이터 타입을 전달
    * reshape: 형상을 재조정
    * slicing: ndarray 일부를 인덱스로 가져옴
    * np.sort(ndarray), ndarray.sort(): 정렬을 수행하는데, 정렬된 리스트를 리턴하는지 내부에서 정렬하여 None을 리턴하는지의 차이
    * dot, transpose: 내적과 전치
* 판다스
    * dataframe: 기본적인 데이터 타입
    * read_csv: csv 파일을 읽어와 dataframe 인스턴스로 생성
    * info, describe, head, tail: 데이터 프레임의 기본적인 정보들, 수치들 등을 보여줌
    * pd.DataFrame(ndarray, columns): ndarray를 dataframe으로 변환
    * dataframe[열이름 | [열이름1, 열이름2], ...]: 원하는 컬럼을 가져옴 (데이터 셀렉션)
    * sort_values(by=[열이름]): 데이터 프레임 정렬, 기준 전달
    * 데이터프레임[컬럼명].apply(lambda 식): 데이터프레임에 람다식 적용
