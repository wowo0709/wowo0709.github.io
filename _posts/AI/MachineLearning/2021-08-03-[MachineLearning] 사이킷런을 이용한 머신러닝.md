---
layout: single
title: "[MachineLearning] 사이킷런을 이용한 머신러닝"
categories: ['AI', 'MachineLearning']
---



# 파이썬 머신러닝 완벽가이드

<br>

### 2장. 사이킷런을 이용한 머신러닝

---

* ​	사이킷런
    * 넘파이와 사이파이 기반 위에서 구축된 라이브러리로 머신러닝을 위한 ㅐ우 다양한 알고리즘과 개발을 위한 편리한 프레임워크와 API 제공
    * 데이터 세트 분리 -> 모델 학습 -> 예측 수행 -> 평가
    
    * train_test_split(입력 데이터, 라벨, test_size, random_state)
    * DecisionTreeClassifier: 분류기, fit: 학습 진행, predict: 예측
    * Estimator는 분류(Classifier)와 회귀 전용(Regressor)으로 나뉨. 학습은 fit, 평가는 predict
    * import sklearn.metrics import accuracy_score(예측 라벨, 실제 라벨) -> 정확도 출력
    * 오버 피팅을 피하기 위해 검증 데이터 사용. 학습 데이터를 학습과 검증용으로 분리. 
    * K 폴드 교차검증이 가장 보편적. import sklearn.model_selection import KFold(n_splits)
    * Stratified KFold: 데이터 크기에 비해 찾아내야 하는 데이터가 적은 경우에 사용. 학습과 검증 데이터의 데이터 분포를 비슷하게 만들어줌. 
    * cross_val_score(분류기, 검증라벨, 실제 라벨)
    * GridSearchCV: 파라미터 탐색
    
* 데이터 전처리

    * Null 값을 허용하지 않음
        * null이 적은 경우 평균값, 최빈값 등으로 대체
        * null이 많은 경우 해당 feature를 드랍
    * 문자열 값을 입력값으로 허용하지 않음 - 숫자로 변환 필요
        * 레이블 인코딩: from sklearn.preprocessing import LabelEncoder
        * 원-핫 인코딩: from sklearn.preprocessing import OneHotEncoder
    * 피쳐 스케일링
        * 표준화: 정규분포를 가지도록 변환: StandardScaler, MinMaxScaler
        * 정규화: 분산과 평균으로 변환
