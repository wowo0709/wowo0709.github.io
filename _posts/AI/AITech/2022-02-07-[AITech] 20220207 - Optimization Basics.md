---

layout: single
title: "[AITech] 20220207 - Optimization Basics"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Concepts', 'Optimizers', 'Regularization']
---



<br>

## 학습 내용

### Important Concepts in Optimization

**Generalization**

Generalization(일반화)이란 **모델이 학습 데이터에서 내는 성능만큼 테스트 데이터에서도 내는 것**을 말합니다. 즉, 일반화 성능이 높다면 학습-테스트 데이터로 예측을 했을 때 그 차이가 적을 것이고, 일반화 성능이 낮다면 그 차이가 크겠죠. 

헷갈리지 말아야 할 것은, **일반화 성능이 좋다고 그 모델의 성능이 좋은 것은 아닙니다.** 일반화 성능이 아무리 좋아도, 그 모델이 학습 데이터에 대한 성능조차 낮다면 그 모델의 성능은 낮은 것입니다. 

![image-20220207145629667](https://user-images.githubusercontent.com/70505378/152759733-435e08bf-88ab-4693-86db-e020b2eb7505.png)

**Underfitting vs. Overfitting**

그리고 일반화 성능 얘기에서 빠지지 않고 나오는 것이 underfitting과 overfitting입니다. 

아래 그림처럼 underfitting(과소적합)이란 아직 모델 자체의 성능이 높지 않은 것이고, overfitting(과대적합)이란 학습 데이터에 대한 성능은 좋으나 테스트 데이터에 대한 성능은 오히려 떨어지는 것을 의미합니다. 

![image-20220207145910025](https://user-images.githubusercontent.com/70505378/152759738-944b14b0-8838-4da0-affe-980fb444424b.png)

**Bias-Variance Tradeoff**

이렇게 과소적합된 모델을 **편향이 크다**라고 하고 과대적합된 모델은 **분산이 크다**라고 합니다. 

우리가 모델의 손실값을 줄인다는 것은 입력값과 모델의 출력값을 이용해 손실 함수 값을 줄인다는 것입니다. 그리고 이 손실 함수 값에는 크게 **편향(bias), 분산(variance), 노이즈(noise)** 항이 있으며, 손실 값을 줄인다는 것은 세 항의 총 합을 줄인다는 것입니다. 

일반적으로 **편향과 분산**은 하나의 값이 작아지면 다른 하나의 값은 커지는 tradeoff 관계를 가집니다. 따라서 모델 훈련 시에는 둘 간의 균형을 고려하여 적절하게 학습시키는 것이 매우 중요합니다. 

**Cross-validation**

Cross validation(교차 검증)이란 모델을 검증할 때 사용하는 테크닉인데요, train dataset을 train VS validation data로 나눌 때 N개의 partition으로 먼저 쪼개고 각각의 partition을 모두 한 번씩 validation data로 쓰는 것, 즉, 총 N번의 훈련-검증을 수행할 수 있는 형태의 검증 방법입니다. 이 때 validation data 로 사용되지 않는 데이터들은 train data로 사용됩니다. 

중요한 것은, 어찌되었든 **test dataset**은 모델 훈련 과정에서 절대로 사용되면 안된다는 것입니다. 

![image-20220207150857037](https://user-images.githubusercontent.com/70505378/152759679-ae86e41f-5ad1-42f0-a556-ff51d3e069c9.png)

**Bootstrapping**

Bootstrapping은 random sampling과 replacement를 사용하는 일련의 모든 test, metric 들을 통칭하는 말입니다. 우리가 여기서 주목할 것은 **Bagging**과 **Boosting**입니다. 

* Bagging
  * 모델의 성능을 높이는 병렬적 방법입니다. 
  * 전체 dataset에서 random sampling된 일부 dataset으로 학습한 모델을 여러 개 만들고, 예측 시 그 모델들의 예측치를 평균한 값을 최종 예측치로 사용하는 것입니다. (Ensemble)
  * 최종 모델을 만들 때는 voting 방법과 average 방법이 있습니다. 
* Boosting
  * 모델의 성능을 높이는 직렬적 방법입니다. 
  * 전체 dataset에서 random sampling된 일부 dataset으로 학습한 모델을 먼저 하나 만들고, 그 모델이 낮은 성능을 보이는 부분 위주로 resampling하여 모델을 재학습시킵니다. 이 과정을 반복합니다. 

![image-20220207151458300](https://user-images.githubusercontent.com/70505378/152759686-79991f5e-ad90-436b-b228-84128bf44c0d.png)







<br>

### Practical Gradient Descent Methods

#### Batch-size Matters

학습 시킬 때 다들 배치 크기, 배치 크기하는데.. 이 배치 크기는 왜 중요하고 어떻게 설정해야 하는 것일까요?

배치 크기에 대한 이야기는 매우 많은데, 이것이 중요한 이유는 **모델의 일반화 성능**과 직결되기 때문이라고 합니다. 논문에 따르면, 

> 큰 배치 사이즈는 **sharp minimizer**에 빠르게 수렴할 확률이 높고, 작은 배치 사이즈는 **flat minimizer**에 서서히 수렴할 확률이 높다

고 합니다. 

그리고 이는 그래디언트 계산 과정에서 발생하는 내재적인 노이즈 때문이라고 하는데요, sharp minimizer와 flat minimizer는 다음과 같습니다. 

![image-20220207151956516](https://user-images.githubusercontent.com/70505378/152759694-b79d2ec8-f11f-48f7-b788-1a532394aa28.png)

위 그림을 보고 나면, 어떤 배치 사이즈가 좋은 것인지 알 수 있을 것입니다. **큰 배치 사이즈보다는 작은 배치 사이즈가 좋다는 것을요!!**

다만, 우리가 1, 2 크기의 배치 사이즈를 사용하지 못하는 것은 그렇게 되면 모델이 제대로된 minima에 수렴하기까지 시간이 너무 오래 걸리기 때문이죠. 따라서 적절한 크기인 32, 64 정도의 배치 사이즈를 선택하는 것입니다. 

#### Gradient Descent Methods

여기서는 여러 gradient descent method들, 즉 여러 옵티마이저 들을 수식적으로 소개합니다. 

* Gradient Descent

  * 이전 단계의 그래디언트 값이 가장 많이 감소하는 방향으로 이동합니다. 

  ![image-20220207152511390](https://user-images.githubusercontent.com/70505378/152759697-f714dffe-f5d6-4d4a-a983-8f8308ab819e.png)

* Momentum

  * `a`라는 항을 삽인하는데, 이 a는 그래디언트와 더불어 이전 단계의 a항을 반영합니다. 즉, 이번 step의 그래디언트와 이전 step의 그래디언트 정보를 함께 고려한다고 할 수 있겠습니다. 
  * 이전 그래디언트에 대한 참조 정도는 하이퍼파라미터 beta로 조절합니다. 

  ![image-20220207152519636](https://user-images.githubusercontent.com/70505378/152759701-8db2b37c-7c69-4a76-9173-bb8300533233.png)

* Nesterov Accelerated Gradient(NAG)

  * Momentum과 비슷한데, momentum이 현재 위치에서 '현재 위치에서의 이번 그래디언트+저번 그래디언트'를 고려했다면 NAG는 '저번 그래디언트대로 움직인 위치에서의 이번 그래디언트+저번 그래디언트'를 고려한다고 할 수 있습니다. (뭐... 사실 저도 아직 이해가 말끔히 되지는 않은 것 같습니다)

  ![image-20220207152529767](https://user-images.githubusercontent.com/70505378/152759703-5fd91b0a-e292-4d43-8289-48c297d1272a.png)

  ![image-20220207153210852](https://user-images.githubusercontent.com/70505378/152759717-c75f147d-8ad0-400f-a3f5-2e9adb48f738.png)

* Adagrad

  * 지금까지의 매 그래디언트 값들을 저장해서, 변화가 덜 일어난 가중치들을 더 많이 업데이트 하도록 합니다. 
  * 아주 오랜 시간 학습 시, 분모에 있는 `Gt` 항이 계속 커지기 때문에 가중치 업데이트가 거의 일어나지 않는 문제가 발생합니다. 

  ![image-20220207152539107](https://user-images.githubusercontent.com/70505378/152759704-6eebf19a-4db4-4301-b56a-e62ef2964108.png)

* Adadelta

  * Adagrad의 문제를 해결하기 위한 형태입니다. 특이한 것은, Adadelta에는 learning rate를 지정할 수 없습니다. (그래서 잘 사용하지 않습니다)
  * EMA라는 용어가 나왔는데, 이는 Exponential Moving Average로 오래된 값보다 최근의 값의 가중치를 더 주어서 평균을 구하는 방식입니다. 

  ![image-20220207152551182](https://user-images.githubusercontent.com/70505378/152759706-e1a28d89-844c-4135-b1fc-cca06f65acb8.png)

* RMSProp

  * Adadelta에서 learning rate를 지정할 수 있도록 개선한 형태입니다. 

  ![image-20220207152601145](https://user-images.githubusercontent.com/70505378/152759707-94145d7f-2115-40d2-a292-317fbf15ab66.png)

* Adam

  * 웬만한 경우에 다른 옵티마이저들보다 더 나은 결과를 보여준다는 Adam!! Momentum과 RMSProp을 융합한 형태입니다. 

  ![image-20220207152619562](https://user-images.githubusercontent.com/70505378/152759711-4af725ab-ba7c-46f7-8607-eaaa653344bd.png)

이렇게 여러 옵티마이저들에 대해 살펴보았는데요, 많은 경우에 Adam을 사용하면 짧은 시간 내에 어느 정도의 성능을 보장해주기 때문에 Adam을 많이 사용합니다. 

다만, 충분한 시간과 리소스가 보장되어 있다면 다른 옵티마이저들이 더 좋은 성능을 보일 수도 있습니다. 







<br>

### Regularization

**Early Stopping**

앞서서 모델은 일반화 성능이 중요하다고 했습니다. Early Stopping(조기 종료)은 바로 이 일반화 성능을 고려하여 학습을 조기에 종료하는 것입니다. 

조기 종료 기법을 사용하려면 당연히 validation data를 마련해야 합니다. 

![image-20220207155500744](https://user-images.githubusercontent.com/70505378/152759721-bc423522-3b98-4936-a4b0-9d3f44e942c8.png)

**Parameter Norm Penalty (Weight Regularizaton)**

Parameter Norm Penalty는 손실 함수에 가중치의 합을 나타내는 가중치 항을 추가하여 가중치가 너무 커지지 않도록 조절하는 것을 말합니다. 

가중치가 너무 커지지 않도록 조절한다는 것은 모델의 복잡도가 너무 커지지 않도록 규제한다는 것이고, 즉 일반화 성능을 고려한다는 것입니다. 

가중치 규제에는 1-Norm을 사용하는 Lasso, 2-Norm을 사용하는 Ridge, 두가지를 모두 사용하는 elastic net 규제 방법이 있으며 아래 예시는 그 중 Ridge 규제를 나타낸 것입니다. 

![image-20220207155836010](https://user-images.githubusercontent.com/70505378/152759722-5fec6146-4599-45c6-8636-a5ddb271f56f.png)

**Data Augmentation**

Data Augmentation(데이터 증강)은 말그대로 가지고 있는 데이터를 어떤 방법을 사용하여 더 많이 만들어내는 것을 말합니다. 예를 들면 회전, 확대/축소, 자르기, 뒤집기(반전) 등이 있을 수 있죠. 

머신러닝/딥러닝에서 **데이터**는 너무나 중요한 요소이고, **많을수록 좋습니다.** 따라서 데이터 증강은 더 나은 모델을 만들 수 있는 가장 좋은 방법일지도 몰라요!

![image-20220207172333268](https://user-images.githubusercontent.com/70505378/152759726-49742229-ed98-4045-9e0c-bd9517db1960.png)

**Noise Robustness**

Noise Robustness는 데이터에 일부러 노이즈를 추가하여, 더욱 강건하고 일반화된 모델을 만드는 기법입니다. 

![image-20220207172419783](https://user-images.githubusercontent.com/70505378/152759727-d07f818e-e28a-4855-99f5-ef5982e98f49.png)

**Label Smooting (Mix up)**

Label smoothing이란 어떤 데이터를 개/고양이 처럼 이분법적으로 구분하지 않고, 개 0.5 고양이 0.5 처럼 그 비율에 대해 라벨링하는 것을 말합니다. **이 기법을 사용하면 모델 성능이 많이 개선된다고 합니다!**

이러한 Label smoothing 기법으로 Mix up, Cut Mix 등의 여러 방법이 있고 더 자세히 알고 싶으신 분들은 아래 참고자료에서 논문을 참고하시는 것을 추천드립니다. 

![image-20220207172701066](https://user-images.githubusercontent.com/70505378/152759728-e38c9a5a-e63c-43b9-9121-8f26eceb498b.png)

**Dropout**

Dropout은 학습 시에 몇 개의 뉴런을 비활성화시키는 것을 말합니다. 이 때의 비활성화란, 계산 과정에 아예 참여하지 않는 것을 말해요. 

학습할 때 순전파 시에는 이렇게 드롭아웃을 적용하고, 역전파나 추론을 할 때는 드롭아웃을 사용하지 않습니다. 

![image-20220207172850523](https://user-images.githubusercontent.com/70505378/152759731-0a698fb8-eccb-41a6-a17a-07f964df14d1.png) 

**Batch Normalization**

Batch Normalization은 처음 들으면 무척이나 생소한 기법인데요, 배치 정규화의 목적은 각 미니 배치의 평균과 분산을 비슷하게 해서 **학습 속도를 빠르게 하고, 더 안정적으로 학습하도록** 하는 것에 있습니다.

간단히 말하면 BN 층은 다음의 과정을 수행합니다. 

1. 미니 배치에 대해 Wx+b로 계산된 값들의 평균 `mu`와 표준편차 `sigma`를 구합니다. 
2. 계산된 초기값 `xi`를 평균 `mu`로 빼고 표준편차 `sigma` 로 나눠서 normalize합니다. 이 값을 `xi'` 이라고 하겠습니다. 
3. 정규화된 데이터 `xi'`에 scaling parameter `gamma`를 곱하고 shifting parameter `beta`를 더합니다. 여기까지 연산이 진행된 최종적인 데이터를 `yi`라고 하고, 이 데이터를 다음 층에서 사용합니다. 

BN 층에서 각 뉴런은 개별적인 `beta`와 `gamma`를 하이퍼파라미터로 가집니다. (CNN의 경우 각 필터마다 개별적인 값을 가집니다)

위 과정은 미니 배치를 사용하는 **train 시**에 진행되는 과정이고, **test 시**에는 '각 미니배치에 대해 구했던 `gamma`와 `beta`의 각각의 평균값을 test 시의 `gamma`와 `beta`로 사용'합니다. (참고로, test 시에는 분산값에 **N/(N-1)**을 곱해줍니다. 실제 모분산과 분산의 평균을 계산하는 과정에서 생기는 차이입니다.)

그런데 사실 위에서 구한 test 시의 `gamma`와 `beta`의 **평균**은 그냥 이동 평균이 아닌 **지수 이동 평균(Exponential Moving Average, EMA)**입니다. EMA는 옛날 값에는 가중치를 적게 주고 최근 값에는 가중치를 크게 주어서 구하는 평균입니다. 

이러한 평균을 사용하는 이유는 무엇일까요? 옛날 값 즉 초기값은 충분히 학습이 진행되기 전의 값이기 때문에 상대적으로 부정확하고 그 의미가 덜합니다. 최근의 값일수록 충분히 학습이 진행된 후의 값이기 때문에 더 유의미하겠죠. 따라서 최근 값에 가중치를 더 부여하는 평균인 지수 이동 평균을 사용하는 것입니다. 

<br>

그렇다면 이 배치 정규화가 어떻게 효과를 보이느냐? 

생각해보면 이상하지 않나요? BN은 어떤 값에 **'빼고 나눈 다음에 곱하고 더합니다'.** 그러면 똑같은 거 아닌가..? 기존의 가중치랑 편향 파라미터로도 충분히 할 수 있을거 같은데 왜 굳이 저 과정을 반복하는거지..?

우리 시그모이드 함수를 생각해봅시다. 

![image-20220207182050923](https://user-images.githubusercontent.com/70505378/152760190-cbb21ff1-75ff-45e4-8f49-a6a9241912c2.png)

위와 같은 시그모이드 함수가 활성화 함수로서 그 사용 빈도가 낮아진 것은 바로 **양 끝에서는 그래디언트 값이 거의 없기 때문**입니다. 이 때문에 학습이 제대로 되지 못하죠. 

BN층은 바로 **2번 과정**에서 이 문제를 해결하기 위해 **여러 값에 걸쳐 나타나는 데이터들을 N(0, 1)로 모아버립니다.** 이렇게 모으면 그래디언트 값이 큰 가운데 부분으로 데이터들이 모이기 때문에 기울기 소실 문제가 없겠죠?

그러면 2번 과정까지만 수행하면 되지 왜 굳이 기껏 모은 데이터들을 다시 뿌리냐? 이에 대한 연구를 진행한 논문도 있었는데요, 결과는 다음과 같습니다. 

> 데이터들이 가운데에만 모이게 되면, 사실상 시그모이드(탄젠트 하이퍼볼릭 등) 함수의 가운데 부분은 선형 함수에 가깝기 때문에 비선형성을 잃게 된다. 따라서 **그래디언트 값이 살아있도록 데이터들을 가운데로 모으되, 비선형성을 유지하기 위해 이를 적절하게 다시 뿌린다.**

따라서 **3번 과정**이 필요하게 됩니다. 이제 BN 층의 역할에 대해 감이 조금 오시나요?

논문에서는, 마지막으로 이러한 메커니즘에 의해 변환된 데이터들은 전체적으로 비슷한 양상을 띄게 되기 때문에 **학습율을 크게 잡아도 된다**라고 말하고 있습니다. 

✋ 해당 배치정규화에 대한 정리 내용은 아래 유튜브 영상을 참고하여 작성하였으니, 이해가 잘 가지 않는 분들은 아래 영상을 보시는 것을 추천드립니다. 

* [혁펜하임의 배치 정규화 17분 설명](https://www.youtube.com/watch?v=m61OSJfxL0U)

<br>

<br>

## 참고 자료

* **Important Concepts in Optimization**
  * [과대적합과 과소적합, 편향-분산 트레이드 오프](https://ukb1og.tistory.com/28)
  * [Cross Validation in Machine Learning]([Cross Validation in Machine Learning Trading Models (quantinsti.com)](https://blog.quantinsti.com/cross-validation-machine-learning-trading-models/))
  * [Adaboost Tutorial](https://www.datacamp.com/community/tutorials/adaboost-classifier-python)
* **Practical Gradient Descent Methods**
  * 논문: On Large-batch Training for Deep Learning: Generalization Gap and Sharp Minima, 2017  
* **Regularization**
  * 논문: mixup: Beyond Empirical Risk Minimization, 2018  
  * 논문: CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features, 2019  
  * 논문: Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift, 2015  
  * [혁펜하임의 배치 정규화 17분 설명](https://www.youtube.com/watch?v=m61OSJfxL0U)

















<br>
