---
layout: single
title: "[AITech][Data Annotation] 20220414 - Text Detection metrics"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 Upstage의 '이활석' 마스터 님의 강의를 바탕으로 작성되었습니다.**_

# Text Detection metrics

Text detection 성능 평가는 아래 두 단계에 걸쳐 진행됩니다. 

1. 테스트 이미지에 대해 결과값을 추출
2. 예측 결과와 정답 간 matching/scoring 과정

즉, 크게 **matching**과 **scoring** 과정이 필요합니다. 이때 영역 간 matching을 나타내는 행렬을 매칭 행렬(matching matrix)이라고 합니다. 

![image-20220422104801953](https://user-images.githubusercontent.com/70505378/164601409-fe8cc21b-88ec-44f7-82f3-08050b7068d3.png)

그러면 이를 위한 방법에 무엇이 있는지 알아보겠습니다. 

## Glossary

각 metric들에 대해 보기 전에, detection metric에서 사용하는 기본적인 용어들에는 무엇이 있는지 보도록 하겠습니다. 

**IoU(Intersection over Union)**

`IoU`는 두 영역이 겹치는 정도를 나타내는 평가 지표입니다. 

![image-20220422105454555](https://user-images.githubusercontent.com/70505378/164601412-f74756bb-c0a1-4b57-9efe-6fb1e5df26bc.png)

**Area Recall/Area Precision**

Recall과 Precision의 개념을 area로 가져와서, area recall과 area precision을 계산할 수 있습니다. 

![image-20220422105545089](https://user-images.githubusercontent.com/70505378/164601414-998ac2ff-cd6e-429f-b3a0-e6768653670a.png)

**One-to-One/One-to-Many/Many-to-One Match**

Text detection에서는 하나의 영역을 여러개 영역으로 예측하거나, 여러개 영역을 하나의 영역으로 예측하는 일이 빈번하게 일어납니다. 

이를 제대로 평가하기 위해 세 가지 경우를 정의합니다. 

![image-20220422105726762](https://user-images.githubusercontent.com/70505378/164601417-29db55c2-30d5-48d9-9be9-e31092cfbbae.png)

그럼 이제 본격적으로 실제로 사용되는 metric에 대해 살펴보도록 하겠습니다. 







<br>

<br>

## DetEval

`DetEval`은 text detection task에서 가장 많이 사용되는 metric 중 하나입니다. 

먼저 matching 행렬을 채우기 위해 각 cell 마다 area recall과 area precision 값을 구합니다. 

그리고 다음으로 area recall이 0.8 이상이고, area precision이 0.4 이상인 cell의 값은 1로, 아닌 cell의 값은 0으로 이진화합니다. 

위 과정까지 수행하면 one-to-one, many-to-one, one-to-many를 알 수 있습니다. one-to-many (split) matching을 지양하기 위해서 해당 값은 1 대신 0.8로 바꿉니다. 

![image-20220422111049918](https://user-images.githubusercontent.com/70505378/164601422-5265ef03-23a5-4f4d-b7fe-3d4a04b7ff70.png)

아래 그림은 예시입니다. 

![image-20220422111207327](https://user-images.githubusercontent.com/70505378/164601427-156c91f2-56e1-4c3c-ba0e-5727004424ca.png)

우리가 하고 있는 task의 목적에 따라 merge를 허용할지, split을 허용할지 달리질 수 있습니다. 

최종적으로 우리는 하나의 이미지에 대응하는 DetEval 값을 계산할 수 있습니다. 이를 위해 먼저 하나의 이미지에 대응하는 recall과 precision 값을 구해야 합니다. 

Recall은 정답 기준으로 score의 평균값, precision은 예측 기준으로 score의 평균값을 사용합니다. 아래 그림은 예시입니다. 

![image-20220422111612026](https://user-images.githubusercontent.com/70505378/164601430-b4ceb717-4302-40aa-b419-1daff34f785e.png)

이렇게 구한 recall과 precision 값으로 F1-score를 계산하여 하나의 이미지에 대응하는 score 값을 구할 수 있습니다. 





<br>

<br>

## IoU

`IoU`는 더 엄격한 metric입니다. IoU metric에서는 매칭 행렬을 채울 때 IoU > 0.5 이면 1, 아니면 0으로 채웁니다. 

또한 IoU에서는 many-to-one, one-to-many 의 경우를 모두 제한(score=0)하고 one-to-one matching만을 허용(score=1)합니다. 

![image-20220422112126234](https://user-images.githubusercontent.com/70505378/164601433-7924147d-3865-48dc-87aa-ed6c05e0d3c6.png)

아래는 예시입니다. 

![image-20220422112136299](https://user-images.githubusercontent.com/70505378/164601436-2b14ef84-5fe9-4506-8ad7-4f0f3e4ef45c.png)

만약 위 그림의 마지막 경우(split case)에서 두 prediction 중 하나의 prediction 만 IoU > 0.5일 경우, recall=1, precision=0.5, score=0.67이 됩니다. 하지만 두 prediction 모두 IoU > 0.5라면 one-to-many match에 해당하므로 score=0이 됩니다. 

IoU 평가방식은 merge와 split matching을 허용하지 않는다는 것, IoU가 0.5 이상이면 어느 정도 겹쳤는지가 점수에 반영되지 않는다는 문제 등이 있습니다. 









<br>

<br>

## TIoU

`TIoU`(Tighness-aware IoU)는 DetEval과 IoU metric이 대응하지 못 하는 경우를 해결하기 위해 제안된 metric입니다. 

TIoU는 전반적인 과정은 IoU와 동일하되, 부족하거나 초과된 영역 크기에 비례하여 IoU 점수에 penalty를 부여합니다. 즉, TIoU에서는 예측 박스의 tightness를 점수에 반영합니다. 

하지만 이 때문에 TIoU에서는 정답 자체가 mislabeling되었을 경우 그 영향이 크게 나타납니다. 

![image-20220422113335096](https://user-images.githubusercontent.com/70505378/164601440-50feab6e-b7fd-4b2f-bbf2-254215c7e380.png)

예측 박스에 대한 TIoU recall과 TIoU precision을 구한 뒤 최종 TIoU 값은 F1 score로 구합니다. 

또한 IoU에서는 many-to-one, one-to-many matching을 모두 허용하지 않았지만 TIoU에서는 별도의 line annotation을 사용해 허용할 수 있습니다. 

아래는 TIoU 방식을 사용할 때의 예시이자 문제점입니다. 아래 두 경우와 같이 한 경우에는 모든 텍스트를 예측하고, 다른 경우에는 그렇지 못했음에도 영역 기반의 점수 때문에 두 경우에 같은 점수를 부여하게 됩니다. 

![image-20220422113917810](https://user-images.githubusercontent.com/70505378/164601443-a26e7040-06d9-4870-a92a-a26a3732b376.png)





<br>

<br>

## CLEval

TIoU의 문제점을 극복하기 위해 등장한 metric이 바로 `CLEval`입니다. 

CLEval의 핵심 철학은 검출기의 성능 평가는 인식기의 관점에서 바라보아야 한다는 것입니다. 

* 얼마나 많은 글자(Character)를 맞추고 틀렸느냐를 가지고 평가. 
* Detection 뿐 아니라 end-to-end, recognition에 대해서도 평가 가능

그런데 CLEval 평가를 위해서는 **각 글자마다 정답 영역이 필요**합니다. 이를 위해서 별도 모듈을 이용해 단어 단위의 정답 영역으로부터 글자 단위의 정답 영역을 추정하도록 했습니다. 

이는 매우 간단한 과정에 의해 수행됩니다. 정답 영역 라벨에는 글자 수에 대한 정보가 있으므로, 해당 영역을 글자 수만큼 등분하여 각 등분된 영역의 중심에 점을 찍습니다. 이를 `PCC`(Pseudo Character Center)라고 하며, 하나의 점은 하나의 글자 영역을 나타냅니다. 

![image-20220422114804279](https://user-images.githubusercontent.com/70505378/164601445-17fc206c-66ac-4265-85bf-26c58304a806.png)

이제 CLEval에서 recall, precision, score를 구하는 방법에 대해 알아보겠습니다. 

**Recall**은 정답 기준의 점수로, `(CorrectNum - GranualPenalty) / TotalNum` 값으로 계산합니다. 

* CorrectNum: 정답 영역 내 PCC 중 어느 에측 영역이라도 속하게 된 PCC 개수
* GranualNum: 정답 영역 내 PCC를 포함하는 예측 영역의 개수 - 1
* TotalNum: 정답 영역 내 PCC 개수

![image-20220422115533892](https://user-images.githubusercontent.com/70505378/164601453-cf0bc188-11b2-43b0-b463-ef3213aa98d4.png)

**Precision**은 예측 기준의 점수로, 계산 식은 recall과 같습니다. 다만 각 항의 의미가 다릅니다. 

* CorrectNum: 이 예측 영역이 포함하고 있는 각 PCC 별로, 해당 PCC를 포함하는 예측 영역의 개수로 나눈 값을 모두 합함
* GranualPenalty: 예측 영역과 연관된 정답 영역의 개수 - 1
  * '연관'은 정답 영역의 PCC를 예측 영역이 포함하고 있으면 연관있다고 간주
* TotalNum: 이 예측 영역이 포함하고 있는 PCC 개수

![image-20220422115840328](https://user-images.githubusercontent.com/70505378/164601457-89541669-bb0b-4990-85ca-1300792551b7.png)

**Score**의 경우, 한 cell에 대한 score는 앞서 구한 cell의 recall과 precision의 조화평균(F1-score)으로 구하면 됩니다. 

**이미지에 대한 최종 score**의 경우, 먼저 모든 정답 박스와 예측 박스에 대해 CorrectNum, GranualPenalty, TotalNum 값을 구해서 이미지에 대한 recall과 precision을 구한 뒤 조화평균으로 구합니다. 

![image-20220422131243487](https://user-images.githubusercontent.com/70505378/164601462-8c5ae350-8926-4909-a191-c1ad810c0253.png)

이미지에 대한 score를 구할 때는 각 cell의 recall, precision의 평균을 취하는 것이 아니라 CorrectNum, GranualPenalty, TotalNum의 총합을 사용한다는 것이 특징적입니다. 







<br>

<br>

## Summary

아래 그림은 앞서 소개한 4개 text detection metric으로 계산한 점수를 비교한 모습입니다. (\*: 별도의 line annotation 필요)

여러분 눈에는 어떤 평가 방식이 가장 합리적으로 보이나요?

![image-20220422131521686](https://user-images.githubusercontent.com/70505378/164601399-7bbbe45e-b8a2-4b6e-ab5f-a299319919b3.png)

좋은 Metric은 정량적 평가와 정성적 평가를 모두 고루 포함할 수 있어야 합니다. 여러분들도 이에 대한 고민을 하시면서 어떤 metric이 좋은지에 대해 생각해보시기 바랍니다. 



























<br>

<br>

# 참고 자료

* 
