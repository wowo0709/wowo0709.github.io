---
layout: single
title: "[AITech][Semantic Segmentation] 20220425 - Introduction"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['COCO dataset', 'EDA', 'metric']
---



<br>

_**본 포스팅은 KAIST의 '김현우' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Introduction

이번 포스팅에서는 `Semantic Segmentation` 강의를 시작하면서 기본적으로 Segmentation에서 사용하는 데이터 포맷, 수행할 수 있는 EDA, 평가 metric 들에 대해 알아보도록 하겠습니다. 

![image-20220425093410139](https://user-images.githubusercontent.com/70505378/165006435-533ae3c0-6090-438e-9ec0-9c11c4d88d79.png)

## COCO Dataset

Semantic segmentation에서는 object detection과 동일하게 COCO dataset 포맷을 사용합니다. 

대회에서 주어지는 데이터는 아래와 같이 총 4개의 json 파일과 이미지 파일들입니다. 

![image-20220425093656074](https://user-images.githubusercontent.com/70505378/165006437-64f76322-d436-4080-ba2e-b57006d81e94.png)

`train.json` 파일에는 아래와 같이 총 5개의 key 값이 존재합니다. 

![image-20220425093758703](https://user-images.githubusercontent.com/70505378/165006441-3c0ba49f-52e5-4e5e-8a52-908a0e4fb7e3.png)

* info: dataset에 대한 high-level의 정보
* licenses: image의 license 목록
* images: dataset의 image 목록 및 각각의 width, height, file_name, id(image_id) 등을 포함
* categories: class에 해당하는 id, name 및 supercategory가 포함되어 있음
* annotations: segmentation 어노테이션 정보 (각 클래스 별)

아래는 json 파일의 annotations 예시입니다. 

![image-20220425094059435](https://user-images.githubusercontent.com/70505378/165006442-226036c7-c882-488d-9c73-0e6558fac14d.png)

<br>

## EDA

대회에서 주어진 데이터는 총 11개 클래스(10개 클래스 + 1개 배경)를 가지는 재활용 쓰레기 이미지 데이터입니다. 

**Class 분석**

하나의 Image 안에 등장하는 object의 개수 - Train (전체 3272개)

![image-20220425095131834](https://user-images.githubusercontent.com/70505378/165006445-3cc12544-76d6-4f7e-ac87-26a73fe97a8d.png)

![image-20220425095144631](https://user-images.githubusercontent.com/70505378/165006446-d0299c8e-efa7-4ad4-84d8-cf63034cd098.png)

**가설 검정**

* Plastic과 plastic bag은 따로 라벨링 되었을까?
  * NO! 'Plastic' 한 카테고리로만 라벨링

![image-20220425095346405](https://user-images.githubusercontent.com/70505378/165006449-304b11ce-29cd-46bb-9e35-3ab0f8faccce.png)

* 'Plastic bag' 내부에 있는 재활용들도 라벨링 되었을까?

  * 'Plastic bag' 내부에 다른 재활용이 보여도 모두 담겨져있으면 'Plastic bag'로만 표기

  ![image-20220425095519014](https://user-images.githubusercontent.com/70505378/165006451-bfe83f44-510f-41e5-a815-a972b25bcdfc.png)

  * 하지만, 'Plastic bag'에 덮혀져 있지 않은 부분들은 보통은 다른 클래스로 분류

  ![image-20220425095632566](https://user-images.githubusercontent.com/70505378/165006454-abedec50-bdd9-484f-aeee-fc0d4c672352.png)

* 색으로 클래스를 구분하는 것이 가능할까? 'Plastic bag' 안에 흰색 'Paper bag'이 있는 경우에 육안으로도 식별이 어려운데 모델이 구분할 수 있을까?

![image-20220425095821462](https://user-images.githubusercontent.com/70505378/165006455-052c320d-4e92-4e3b-8b25-3586f56e15c6.png)

* 배경이 'Paper bag'인 건 어떻게 알 수 있을까?

![image-20220425095904906](https://user-images.githubusercontent.com/70505378/165006458-7e76fe7e-706d-4e0c-8ad2-0dbab7d05f4a.png)

* 매우 작은 크기의 object 라벨링 존재

![image-20220425095947958](https://user-images.githubusercontent.com/70505378/165006459-435d015d-e96b-49ce-93c1-e0e109992d0d.png)

* 돌이 하나의 object에 포함되어 라벨링 된 경우 존재

![image-20220425100014743](https://user-images.githubusercontent.com/70505378/165006462-eb124ec4-49d8-45ed-8369-4632bc5d2063.png)

* Test EDA
  * Inference 후 결과를 직접 시각화하여 train dataset과 비교해서 어떤 부분에 대한 성능이 낮은지 확인!

![image-20220425102742443](https://user-images.githubusercontent.com/70505378/165006464-2316fd51-0fbc-485d-8b0f-fd86f52322b3.png)

<br>

## 평가 Metric

본 대회에서는 semantic segmentation의 metric으로 `mIoU(mean IoU)`를 사용합니다. 

![image-20220425101932027](https://user-images.githubusercontent.com/70505378/165006463-54ea5144-73e4-42ce-90b5-42b214cdce94.png)













<br>

<br>

# 참고 자료

* 
