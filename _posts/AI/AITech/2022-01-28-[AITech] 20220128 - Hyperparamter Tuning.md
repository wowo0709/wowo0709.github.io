---
layout: single
title: "[AITech] 20220128 - Hyperparameter Tuning"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Ray']
---



<br>

## 학습 내용 정리

### Hyperparameter Tuning

모델의 성능을 높이는 데에는 크게 다음의 3가지 방법이 있습니다. 

1. 모델의 구조 개선: 현실적으로 큰 변화를 만들기 어렵다. 
2. 데이터 증강/보강: 가장 중요하면서 큰 효과를 볼 수 있다. 
3. 하이퍼파라미터 튜닝: 아주 큰 차이를 일으키지는 않지만 시도해 볼 만 하다. 

이 중 **하이퍼파라미터 튜닝**은 사실 이전에는 그 값에 의해 성능이 크게 좌우될 때도 있었지만, 요즘은 그렇지는 않다고 합니다. 

하지만 learning rate, 모델의 크기, batch size, optimizer 등 여러 하이퍼파라미터들을 튜닝하는 방법은 **마지막으로 모델의 성능을 조금 더 끌어올리고 싶을 때** 사용해 볼 만한 방법입니다. 

하이퍼 파라미터 튜닝 방법에는 전통적으로 사용되어 온 **grid search**와 **random search**가 있으며, 보통 random search로 튜닝을 하다가 성능이 좋은 부분이 발견되면 그 부분 부근에서 grid search를 수행하는 식으로 수행되었다고 합니다. 

![image-20220128115041782](https://user-images.githubusercontent.com/70505378/151489753-b580ccda-42df-428a-b598-dbe863baa8ec.png)

최근에는 두 방법 외에 **베이지안 기법**들이 주도하고 있습니다. 이에 대해 `BOHB(Baesian Optimization Hyper Band) 2018`이라는 논문을 읽어보면 도움이 될 것입니다. 

이번 포스팅에서는 이 하이퍼파라미터 튜닝 과정을 간소화 시켜주는 **Ray**라는 모듈에 대해 소개하고 사용하는 방법을 보려합니다. 

#### Ray

* Multi-node multi processing를 지원하며 ML/DL의 병렬 처리를 위해 개발된 모듈
* 기본적으로 현재의 분산 병렬 ML/DL 모듈의 표준
* Hyperparameter search를 위한 다양한 모듈 제공

```python
data_dir = os.path.abspath("./data")
load_data(data_dir)
# 1. config에 search space 지정
config = {
    "l1": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    "l2": tune.sample_from(lambda _: 2 ** np.random.randint(2, 9)),
    "lr": tune.loguniform(1e-4, 1e-1),
    "batch_size": tune.choice([2, 4, 8, 16])
}
# 2. 학습 스케줄링 알고리즘 지정
scheduler = ASHAScheduler(
                metric="loss", mode="min", max_t=max_num_epochs, grace_period=1,
                reduction_factor=2)
# 3. 결과 출력 양식 지정
reporter = CLIReporter(metric_columns=["loss", "accuracy", "training_iteration"])
# 4. 병렬 처리 양식으로 학습 수행
result = tune.run(partial(train_cifar, data_dir=data_dir),
                resources_per_trial={"cpu": 2, "gpu": gpus_per_trial},
                config=config, num_samples=num_samples,
                scheduler=scheduler,
                progress_reporter=reporter)
```

Ray 모듈은 hyperparameter search를 수행할 때 처음에는 모든 경우로 학습을 시행하다가 성능이 좋지 않으면 해당 경우는 더 이상 학습을 시행하지 않고, 성능이 좋은 경우들로만 학습을 계속 수행하는 방식으로 리소스를 절약합니다. 

![image-20220128115742600](https://user-images.githubusercontent.com/70505378/151489755-8a135c5e-865f-4a99-98c1-4c29fdab9608.png)



또한 지난 포스팅에서 소개한 wandb 와 함께 사용하면 그 결과를 확인하기 더욱 좋기 때문에 두 모듈을 함께 활용하면 어렵지 않게 hyperparameter tuning을 수행할 수 있을 것으로 기대합니다. 

<br>

## 참고 자료

* **Hyperparameter Tuning**
  * [Hyperparameter Tuning with Ray Tune](https://pytorch.org/tutorials/beginner/hyperparameter_tuning_tutorial.html)
