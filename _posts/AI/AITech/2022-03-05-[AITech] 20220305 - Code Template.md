---
layout: single
title: "[AITech][Image Classification][P stage] 20220305 - Code Template"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Image Classification']
---



<br>

# Code Template

## 들어가기 전에

이번 포스팅에서는 PyTorch Project를 구성하는 Code Template에 대하여 다룹니다. EDA 시 유용하게 활용하는 jupyter notebook file 외에, 실제로 프로젝트를 작성할 때는 python file(.py)을 사용하게 됩니다. 그리고 모든 코드를 한 파일 안에 작성하는 것이 아닌, 역할들 마다 따로 파일을 생성하고 그 안에 코드를 작성합니다. 

다만, 프로젝트 템플릿을 구성하는 방법은 다양합니다. 누군가가 분리해놓은 파일을 누군가는 분리하지 않을 수도 있고, 누군가가 생성한 파일을 누군가는 필요없어서 생성하지 않을 수도 있고, 그냥 파일 별로 구분한 것을 디렉토리 별로 따로 담아둘 수도 있습니다. 다양하고 잘 작성해 놓은 PyTorch Code Template들이 있으니 참고하시길 바라고, 여기서 작성한 Code Template은 AI Tech 측에서 대회 참여를 위해 제공해준 템플릿을 사용했다는 것을 밝힙니다. 이 코드도 상당히 깔끔하고 가독성 좋게 작성된 템플릿이라고 생각합니다. 

각 코드에 대한 세세한 리뷰는 Code Review 포스팅에서 진행합니다. 



<br>

## 대회 개요

Lavel 1 P-stage에서는 **이미지 분류 대회**를 진행했습니다. 분류하고자 하는 이미지는 Mask Dataset이고, 단순히 Mask on/Mask off를 분류하는 binary classification 문제가 아닌 mask(착용/미착용/올바르지 않은 착용), gender(남/여), age(~29, 30~59, 60~) 를 고려하여 총 18개의 class로 분류해야 하는 multi label classification 문제였습니다. 

Train data로는 약 20,000장 정도가 주어졌고, 데이터의 특징으로는 mask별, age별 불균형이 심한 데이터라는 것이었습니다. 

**Class description**

![image-20220305223739923](https://user-images.githubusercontent.com/70505378/156886594-400fb96f-12f1-4d4b-af7b-eb4ca1ce6f01.png)

**Class distribution**

![image-20220305222638146](https://user-images.githubusercontent.com/70505378/156886596-a49b6b22-0969-4628-b6e6-5d63827b91ce.png)



<br>

## 코드 템플릿

### EDA.ipynb

첫번째로 EDA 용 jupyter notebook 파일입니다. 이 파일은 Project에 직접적으로 포함되는 파일이라기 보다는, 데이터를 이해하고 특징을 파악하기 위한 과정에서 생성되는 파일입니다. 

### dataset.py

두번째로 `dataset.py` 파일입니다. 이 파일에서는 Dataset, Augmentation, Custom Transform 등의 클래스가 포함됩니다. 

Dataset은 Train용 dataset과 Test용 dataset 클래스를 생성합니다. Validation용 dataset은 Train용 dataset 클래스 내에 split 메서드를 구현함으로써 달성할 수 있습니다. 

Augmentation과 Custom transform 클래스의 경우 데이터셋 자체라기 보다는 변형을 주기 위한 클래스이므로, 따로 python file을 만들어 관리해도 좋습니다. 

### loss.py

다음은 `loss.py` 파일입니다. 여기에는 loss 클래스들이 구현되어 있습니다. 

Loss 클래스를 불러와 객체를 생성하는 경우 인자를 전달해야 하는 경우가 많으므로, 이를 위한 편의 함수들도 정의해놓습니다. 자세한 설명은 Code Review 포스팅에서 합니다.  

### model.py

`model.py` 파일에서는 각종 model 클래스들이 구현되어 있습니다. 직접 구현한 모델, 불러온 pretrained 모델들 모두 여기에 클래스로 정의해놓습니다. 

### train.py

실질적인 학습/검증 코드가 구현되어 있는 `train.py` 파일입니다. Logging, Visualization 코드나 Dataset/DataLoader 불러오기 ~ 학습한 모델 save까지 일련의 코드들이 모두 이 파일에 들어있습니다. 

또한 터미널에서 실질적으로 호출하여 실행하는 코드이기도 합니다. 따라서 argparse 모듈을 이용해 다양한 argument들을 선언하고, 터미널에서 지정하여 전달할 수 있도록 합니다. 

앞에서도 밝혔듯이, logging이나 visualization 등과 같은 세부적인 코드들도 따로 python file로 만들어 관리할 수도 있습니다. 

### inference.py

추론을 수행하는 `inference.py` 파일입니다. 

저장된 모델을 불러오고, test dataset을 생성해서 ground truth를 이용하여 모델 성능을 확인하거나, 추론 결과를 저장할 수 있습니다. 

### utils.py

이외의 편의를 위한 클래스들이 선언되어 있는 `utils.py` 파일입니다. 대표적으로 EarlyStopping 등과 같은 편의 코드들이 여기에 정의되어 있습니다. 

### requirements.txt

마지막으로 `requirements.txt`입니다. 이 텍스트 파일도 PyTorch Project에 직접적으로 관여하는 파일은 아니고, 환경 구축을 위해 필요한 파일입니다. 필요한 라이브러리와 버전들을 명시하고, 다른 사용자가 `pip install -r requirements.txt` 명령어로 환경 구축을 할 수 있도록 도와줍니다. 

<br>

<br>

이상으로 PyTorch Project에는 어떤 python file들이 존재하는지, 각각의 역할은 무엇인지에 대해 살펴보았습니다. 다음 포스팅에서는 각 파일들의 code review를 통해 안에 정의되는 클래스나 함수, 코드 흐름들을 어떻게 작성할 수 있는지 알아보겠습니다. 























<br>
