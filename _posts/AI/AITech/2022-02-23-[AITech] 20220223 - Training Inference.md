---

layout: single
title: "[AITech][Image Classification] 20220223 - Training&Inference"
categories: ['AI', 'AITech', 'ImageClassification']
toc: true
toc_sticky: true
tag: []
---



<br>

**_본 포스팅은 번개장터의 '김태진' 강사 님의 강의를 바탕으로 제작되었습니다._** 

# 학습 내용

이번 포스팅에서는 Training과 Inference 과정에서 필요한 지식에 대해 알아봅니다. 

## Loss

`Loss`도 nn.Module 을 상속받은 클래스입니다. 이는 곧 Loss 함수의 backward 메서드를 사용하면 모델 내의 각 layer들의 backward 메서드를 한 번에 호출할 수 있다는 얘기입니다. (앞선 Model 포스팅에서 이야기했습니다)

`loss.backward()` 를 호출하면 모델의 파라미터의 grad 값이 업데이트됩니다. (아직 파라미터가 갱신되지는 않습니다)

![image-20220223213534878](https://user-images.githubusercontent.com/70505378/155326457-c5c0887a-c03d-40d1-ba94-5003aed9a99d.png)

### Example: 조금 특별한 loss

* **Focal Loss**: Class Imbalance 문제가 있는 경우, 맞춘 확률이 높은 Class는 조금의 loss를, 맞춘 확률이 낮은 Class는 Loss를 훨씬 높게 부여
* **Label Smoothing Loss**: Class target label을 Onehot 표현으로 사용하기 보다는 ex) [0,1,0,0,0,0…], 조금 Soft하게 표현해서 일반화 성능을 높이기 위함 ex) [0.025, 0.9, 0.025, 0.025, …]  



<br>

## Optimizer

`Optimizer`는 모델의 가중치 업데이트를 결정하는 알고리즘입니다. 유명한 최적화 알고리즘들에 대해서는 전에도 많이 봤으니, 여기서는 Optimizer의 Learning rate를 동적으로 조절해주는 Schedular들에는 무엇이 있는지 보겠습니다. 

**StepLR**

특정 Step마다 LR를 감소시킵니다. 

![image-20220223214138403](https://user-images.githubusercontent.com/70505378/155326461-df8620c6-abc3-451c-871d-6b870428ea33.png)

**CosineAnnealingLR**

조금 이상하게 보일 수도 있겠지만, 규칙적으로 변화하는 LR를 사용해서 local minima에 빠지지 않게 할 수 있다고 합니다. 

![image-20220223214158581](https://user-images.githubusercontent.com/70505378/155326464-68f8c69a-b2c4-496b-9086-e6b09d01bca2.png)

**ReduceLROnPlateau**

더 이상 성능 향상이 없을 때 LR를 감소시킵니다. 

![image-20220223214224151](https://user-images.githubusercontent.com/70505378/155326466-2a32a1f3-49c8-4997-95ea-8ea03004aa58.png)







<br>

## Metric

모델 평가 지표로는 다음의 것들이 있습니다. 

* **Classification**: Accuracy, F1-score, precision, recall, ROC&AUC
* **Regression**: MAE, MSE
* **Ranking**: MRR, NDCG, MAP

학습에 직접적으로 사용되는 것은 아니지만, 학습된 모델을 객관적으로 평가할 수 있는 지표가 필요합니다. 

아래 표를 비교해보면 오른쪽 모델의 성능이 훨씬 좋은 듯 합니다. 하지만 실제로 그럴까요?

![image-20220223214721244](https://user-images.githubusercontent.com/70505378/155326469-2ca1c06b-a3c6-4600-9c8d-74914ffc18cf.png)

전체 데이터에 대한 accuracy는 오른쪽 모델이 높지만, 이것이 accuracy metric의 맹점입니다. 오른쪽 모델은 클래스 1에 대해서는 50%의 정확도 밖에 보이지 못합니다. 

따라서, 데이터 상태에 따라 적절한 Metric을 선택하는 것이 중요합니다. Image Classification에서는 대표적으로 다음의 두 가지 metric을 사용합니다. 

* **Accuracy**: Class 별로 밸런스가 적절히 분포
* **F1-score**: Class 별 밸런스가 좋지 않아서 각 클래스 별로 성능을 잘 낼 수 있는지 확인 필요





<br>

## Training Process

앞에서 모델 training에 필요한 3가지 요소인 Loss, Optimizer, Metric에 대해 살펴보았습니다. 이번 Training Process 세션에서는 실제 PyTorch 코드 상에서 model training process를 이해하는 것을 목표로 합니다. 

![image-20220223215205276](https://user-images.githubusercontent.com/70505378/155326470-49d78857-94fc-4084-b1df-f8a375eb3348.png)

* **model.train()**: 모델이 훈련 하기 전에 train 모드로 바꿔줘야 합니다. 
* **optimizer.zero_grad()**: loss를 미분하면서 생기는 grad가 누적되지 않도록 clear해주는 역할을 합니다. 
* **loss = criterion(output, labels)**: 손실 함수에 output과 label을 전달해서 손실 함수 값을 계산합니다. 
* **loss.backward()**: 앞에서 구한 손실 함수 값을 미분하면서 backpropagation을 수행합니다. 각 layer parameter들의 grad 값을 구합니다. 
* **optimizer.step()**: 앞에서 구한 grad 값을 이용해 parameter들을 갱신합니다. 

### More: Gradient Accumulation

Training process를 이해한다면, 이를 응용해 아래와 같은 코드를 작성하는 것도 가능합니다. 

아래 코드는 특정 step마다 parameter를 갱신하고 grad를 초기화해주는데, 이는 batch_size를 크게 설정할 때의 효과를 기대할 수 있다고 합니다. 

![image-20220223220417266](https://user-images.githubusercontent.com/70505378/155326475-8bae02d7-3739-4e84-a6ff-830a65be431c.png)



<br>

## Inference Process

![image-20220223215212729](https://user-images.githubusercontent.com/70505378/155326472-9054cfa7-38b9-48e0-b8de-0838ebaece90.png)

* **model.eval()**: 모델이 예측(inference)하기 전에 eval 모드로 바꿔줘야 합니다. 
* **with torch.no_grad()**: grad가 계산되거나 parameter가 갱신되는 것을 막아서 리소스를 절약합니다. 

### Validation

추론 과정에 Validation set이 들어가면 그게 검증 과정입니다. 

![image-20220223220731027](https://user-images.githubusercontent.com/70505378/155326477-a5c78207-d64a-4a65-95bf-4c141d7162de.png)

### Checkpoint

모델이 최고 성능을 보일 때의 모델 정보를 저장합니다. 이는 간단히 아래와 같이 작성하면 됩니다. 

![image-20220223220815612](https://user-images.githubusercontent.com/70505378/155326479-7f7d3a9f-47d4-4d02-af6f-ee15b0fc7474.png)









<br>

## PyTorch Lightning

PyTorch Lightning은 PyTorch를 간단하게 사용할 수 있게 해주는 High-level API입니다. 마치 Tensorflow의 Keras 같은 존재라고 보면 됩니다. 

![image-20220223221216177](https://user-images.githubusercontent.com/70505378/155326480-846d12dc-68ef-4951-9f7f-b08daa462091.png)

다만, 강의를 진행해주신 강사 분은 PyTorch에 대한 충분한 이해와 실습이 진행되고 난 후에 PyTorch Lightning을 사용할 것을 당부하셨습니다. PyTorch Lightning은 코드 생산성 측면에서는 아주 좋을지 몰라도, 프로세스를 이해하기에는 부족합니다. PyTorch를 충분히 사용해보고 code level에서 machine learning process를 충분히 이해한 후에 사용하는 것이 좋을 것 같습니다!







<br>

<br>

# 참고 자료

* 





<br>
