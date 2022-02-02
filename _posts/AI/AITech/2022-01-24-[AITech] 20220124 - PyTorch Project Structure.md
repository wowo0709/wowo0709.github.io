---
layout: single
title: "[AITech] 20220124 - PyTorch Project Structure"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Pytorch Template']
---



<br>

## 학습 내용 정리

### Pytorch 프로젝트 구조 이해하기

* 개발 초기 단계에서는 대화식 개발 과정이 유리
  * 학습과정과 디버깅 등 지속적인 확인
* 배포 및 공유 단계에서는 notebook 공유의 어려움
  * 쉬운 재현의 어려움, 실행순서 꼬임
* DL 코드도 하나의 프로그램
  * 개발 용이성 확보와 유지보수 향상 필요
* 다양한 프로젝트 템플릿이 존재하며, 사용자 필요에 따라 선택 및 수정
* 실행, 데이터, 모델, 설정, 로깅, 지표, 유틸리티 등 다양한 모듈들을 분리하여 프로젝트 템플릿 화

> 프로젝트 템플릿: [GitHub - victoresque/pytorch-template: PyTorch deep learning projects made easy.](https://github.com/victoresque/pytorch-template)
>
> 위 소스코드들에 주석을 달아가면서 공부하는 것이 많은 도움이 될 것이다. 

![image-20220125153216945](https://user-images.githubusercontent.com/70505378/150939796-91052ae4-77ff-46e5-9d04-2b96f16db7f7.png)







<br>
