---
layout: single
title: "[AITech][Object Detection] 20220329 - MMDetection - PyTorch Randomness 제어하기"
categories: ['AI', 'AITech', 'PyTorch']
toc: true
toc_sticky: true
tag: ['mmdetection']
---



<br>

# MMDetection - PyTorch Randomness 제어하기

현재 진행 중인 object detection competition에서 reproducibility를 위해 팀원들끼리 seed number를 정하여 고정시켰는데, 실험 과정에서 재현이 되지 않는 것을 발견하였습니다. 

이번 포스팅에서는 pytorch에서 randomness를 결정하는 요소들에는 무엇이 있는지 보고, 각각이 무엇을 의미하는 지, 그리고 어떻게 제어할 수 있는지까지 정리해보도록 하겠습니다. 

## 개요

현재 object detection competition에서는 `mmdetection` 라이브러리를 이용해 대회를 진행하고 있습니다. 

mmdetection의 `tools/train.py` 파일에 randomness를 제어하는 부분이 있는데, 해당 코드는 아래와 같습니다. 

```python
    # set random seeds
    seed = init_random_seed(1333)
    args.deterministic = True
    logger.info(f'Set random seed to {seed}, '
                f'deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed
    meta['seed'] = seed
    meta['exp_name'] = osp.basename(args.config)

    # ...
    
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta)
```

크게 세 부분을 확인할 수 있습니다. 

* `seed = init_random_seed(seed)`
* `args.deterministic = True`
* `cfg.seed = seed`

그러면 각 부분에 의해 제어되는 randomness들을 살펴보도록 하겠습니다. 

<br>

## seed = init_random_seed(seed)

`init_random_seed` 함수의 코드는 아래와 같습니다. 

```python
def init_random_seed(seed=None, device='cuda'):
    """Initialize random seed.

    If the seed is not set, the seed will be automatically randomized,
    and then broadcast to all processes to prevent some potential bugs.

    Args:
        seed (int, Optional): The seed. Default to None.
        device (str): The device where the seed will be put on.
            Default to 'cuda'.

    Returns:
        int: Seed to be used.
    """
    if seed is not None:
        return seed

    # Make sure all ranks share the same random seed to prevent
    # some potential bugs. Please refer to
    # https://github.com/open-mmlab/mmdetection/issues/6339
    rank, world_size = get_dist_info()
    seed = np.random.randint(2**31)
    if world_size == 1:
        return seed

    if rank == 0:
        random_num = torch.tensor(seed, dtype=torch.int32, device=device)
    else:
        random_num = torch.tensor(0, dtype=torch.int32, device=device)
    dist.broadcast(random_num, src=0)
    return random_num.item()
```

말 그대로 random number를 반환해주는 함수입니다. 우리가 seed 인자에 숫자를 지정해줬을 경우 해당 숫자를 그대로 return 합니다. 

여기서 얻은 random number는 seed에 저장되고, 이 seed는 `set_random_seed` 함수 호출 시 전달됩니다. 

```python
seed = init_random_seed(1333)
# ...
set_random_seed(seed, deterministic=args.deterministic)
```

그럼 이제 `set_random_seed` 함수 내부를 보도록 하겠습니다. 

```python
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

함수 내부는 위와 같습니다. deterministic 인자에 대한 이야기는 뒤에서 하고, 먼저 seed로 설정되는 4가지 부분을 각각 보도록 하겠습니다. 

### Python: random.seed

`random.seed(seed)` 함수는 파이썬 random 라이브러리의 seed를 고정시킵니다. 

파이썬 random 라이브러리를 사용할 경우 고정해주어야 하며, 특히 **torchvision의 transforms** 사용 시 RandomCrop, RandomFlip 등의 data augmentation을 적용할 때 python의 random 라이브러리를 사용하기 때문에 필수적으로 고정시켜 주어야 합니다. 

```python
random.seed(seed)
```

### NumPy: np.random.seed

`np.random.seed(seed)` 함수는 numpy 라이브러리의 seed를 고정시킵니다. 

딥러닝의 주요 라이브러리나 프레임워크들에서 모두 numpy를 사용하기 때문에 필수적으로 고정시켜 주어야 합니다. 

```python
np.random.seed(seed)
```

### Torch: torch.manual_seed

`torch.manual_seed(seed)` 함수는 PyTorch의 seed를 고정시킵니다. 

torch에서 사용하는 랜덤성 함수들인 rand, randint 외에도 torch.Tensor.index_add(), torch.nn.functional.interpolate() 등의 함수들을 사용할 때도 모두 torch의 seed에 의해 제어됩니다. 

```python
torch.manual_seed(seed)
```

### Cuda: torch.cuda.manual_seed_all

`torch.cuda.manual_seed_all` 함수는 cuda 라이브러리의 seed를 고정시킵니다. 

cuda의 randomness는 모델 학습 시 backpropagation 과정에서 드러납니다. 이를 고정해주지 않으면 같은 데이터로 학습하더라도 다른 모델 성능으로 귀결될 수 있습니다. 

```python
torch.cuda.manual_seed_all(seed)
```



<br>

## args.deterministic = True

이번에는 `deterministic` 인자가 제어하는 randomness에 대해 살펴보도록 하겠습니다. 

deterministic=True로 지정해 주어야 randomness의 제어가 가능해집니다. 

```python
args.deterministic = True
# ...
set_random_seed(seed, deterministic=args.deterministic)
```

해당 인자도 앞에서와 동일하게 `set_random_seed` 함수에 전달됩니다. 

```python
def set_random_seed(seed, deterministic=False):
    """Set random seed.

    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

이번에는 `if deterministic` 부분을 보도록 하겠습니다. 

### CuDNN: torch.backends.cudnn.deterministic/benchmark

`torch.backends.cudnn.deterministic/benchmark` 프로퍼티는 cudnn의 randomness를 제어합니다. 

기본적으로는 `torch.backends.cudnn.benchmark = True`로 지정되어 있습니다. 이는 모델 학습 시 선택되는 알고리즘 혹은 연산 방법이 **학습 당시 하드웨어 환경 등에 최적화된(가장 빠른) 방법으로 선택**되도록 합니다. 따라서 우리 모델은 학습 시 마다 같은 결과를 불러오는 연산이더라도, 당시 환경에 따라 조금씩 달라지는 알고리즘에 의해 수행되었던 것입니다. 

이를 제어하려면,  `torch.backends.cudnn.benchmark = False`로 지정하고 `torch.backends.cudnn.deterministic = True` 로 지정해주면 됩니다. deterministic 프로퍼티는 정해진 알고리즘으로 모델을 학습시킵니다. 

다만 여기서 문제는, **연산에 따라 deterministic 연산이 정의되어 있지 않을 수도 있다**는 것입니다. 그렇기 때문에 우리가 deterministic 연산이 정의되어 있지 않은 연산을 사용하고 있다면, 안타깝게도 **완벽한 randomness 제어는 불가능**합니다. 

하지만, 여기까지의 과정만으로도 거의 모든 결과가 유사하게 재형 가능하다고 합니다. 

```python
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

추가적으로 `torch.use_deterministic_algorithms(True)` 함수를 사용하는 방법도 있는데, 해당 함수는 연산에 deterministic 알고리즘이 정의되어 있지 않으면 **RuntimeError**를 일으킵니다. 

```python
torch.use_deterministic_algorithms(True) # throws an error when there is no deterministic algorithm in operation
```





<br>

## cfg.seed = seed

마지막으로 앞서 만든 seed로 `cfg.seed = seed` 코드로 cfg의 seed를 지정하고, cfg를 train_detector 함수 호출 시 전달합니다. 

```python
cfg.seed = seed
meta['seed'] = seed
meta['exp_name'] = osp.basename(args.config)

# ...
train_detector(
    model,
    datasets,
    cfg,
    distributed=distributed,
    validate=(not args.no_validate),
    timestamp=timestamp,
    meta=meta)
```

train_detector 함수는 내부에서 `build_dataloader`를 호출 할 때 cfg.seed를 전달합니다. 

```python
data_loaders = [
        build_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))
        for ds in dataset
    ]
```

그리고 build_detector 내부에서는 Sampler나 multi cpu/gpu worker 생성 시에 해당 seed로 randomness를 제어합니다. 

<br>

<br>

이상 mmdetection 사용 시에 randomness를 제어하는 방법에 대해 알아보았습니다. 

해당 내용들은 pytorch에서 reproducibility가 필요할 때 필수적으로 사용되는 내용들이기 때문에 이해해두면 좋을 것 같습니다. 

일반적으로는 **python, numpy, pytorch, cuda, cudnn** 이렇게 5가지의 randomness 제어를 잊지 말고 하면 될 것 같네요!!

```python
# seed 고정
random_seed = 21
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```









<br>

<br>

# 참고 자료

* https://hoya012.github.io/blog/reproducible_pytorch/
* https://tempdev.tistory.com/28
* https://pytorch.org/docs/stable/notes/randomness.html
* https://antilibrary.org/2481
* https://docs.python.org/ko/3/library/random.html

