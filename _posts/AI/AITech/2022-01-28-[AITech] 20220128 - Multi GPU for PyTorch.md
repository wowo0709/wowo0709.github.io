---
layout: single
title: "[AITech] 20220128 - Multi GPU for PyTorch"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Model Parallel', 'Data Parallel']
---



<br>

## 학습 내용 정리

### Multi-GPU 학습

#### 개념 정리

* `Single VS Multi`: 1개 VS 2개 이상
* `GPU VS Node`: GPU VS 컴퓨터
* `Single Node Single GPU`: 컴퓨터 1대에 GPU 1개
* `Single Node Multi GPU`: 컴퓨터 1대에 GPU 여러 개
* `Multi Node Multi GPU`: 컴퓨터 여러 대에 GPU 여러 대

#### Model Parallel

다중 GPU에 학습을 분산하는 방법에는 **모델을 나누는 방법**과 **데이터를 나누는 방법**이 있다. 

모델을 나누는 것은 비교적 예전부터 사용해온 기법(AlexNet)이지만, 모델의 병목이나 파이프라인의 어려움으로 인해 모델 병렬화는 곡난이도 과제이다. 

![image-20220128111218264](https://user-images.githubusercontent.com/70505378/151489744-cbbf842e-76e7-4dec-9281-932bcd8e3764.png)

* 예시 코드

```python
class ModelParallelResNet50(ResNet):
    def __init__(self, *args, **kwargs):
        super(ModelParallelResNet50, self).__init__(
        	Bottleneck, [3, 4, 6, 3], num_classes=num_classes, *args, **kwargs)
        
        self.seq1 = nn.Sequential(
        	self.conv1, self.bn1, self.relu, self.maxpool, self.layer1, self.layer2
        ).to('cuda:0')
        
        self.seq2 = nn.Sequential(
        	self.layer3, self.layer4, self.avgpool,
        ).to('cuda:1')
        
        self.fc.to('cuda:1')
        
    def forward(self, x):
        x = self.seq2(self.seq1(x).to('cuda:1'))
        return self.fc(x.view(x.size(0), -1))
```





#### Data Parallel

Data Parallel 기법은 데이터를 나눠 GPU에 할당한 후 결과의 평균을 취하는 방법입니다. 

![image-20220128112840914](https://user-images.githubusercontent.com/70505378/151489746-44d3d6e0-0f29-449f-b502-ed1e06867d4b.png)

위 그림을 보면 'Forward 시 분배가 일어나고 Backward가 완료된 후 취합'하는 것이 아니라, **중간에 Forward의 결과를 하나의 GPU가 취합한 후 gradient를 계산하고, 다시 분배하는 과정**이 일어나게 됩니다. 

이는 **Global Interpreter Lock**이라고 하는 파이썬의 멀티 프로세싱 상의 제약 사항 때문이라고 합니다. 

위와 같은 Data Parallel 기법은 파이토치에서 제공하는 DataParallel 클래스를 사용하여 간단히 구현할 수 있습니다. 

```python
parallel_model = torch.nn.DataParallel(model) # 이게 전부!!

# Forward ~ Loss Computation
predictions = parallel_model(inputs) # Forward pass on multi-GPUs
loss = loss_function(predictions, labels) # Compute loss function

# Gradient Backward propagation
loss.mean().backward() # Average GPU-losses + backward pass
optimizer.step() # Optimizer step

predictions = parallel_model(inputs) # Forward pass with new parameters
```

그런데 `DataParallel` 클래스는 위에서 말했듯이, 단순히 데이터를 분배한 후 평균을 취하고 다시 분배를 해주는 동작을 수행합니다. 

이는 **GPU 사용 불균형 문제**나 **Batch 사이즈 감소(취합하는 하나의 GPU의 병목)** 등의 문제를 야기합니다. 

<br>

이를 해결하는 방법으로 `DistributedDataParallel` 클래스를 사용할 수 있고, 해당 클래스는 **각 CPU마다 개별 process를 생성하여 GPU에 할당**함으로써 **중간에 취합하는 과정을 없앨 수 있습니다.**

사용하는 방법은 조금 더 복잡하지만 뛰어난 병렬화 효과를 볼 수 있습니다. 

```python
train_sampler = torch.utils.data.distributed.DistributedSampler(train_data)
shuffle = False
pin_memory = True

trainloader = torch.utils.data.DataLoader(train_data, batch_size=20, shuffle=True
										pin_memory=pin_memory, num_workers=3,
										shuffle=shuffle, sampler=train_sampler)

def main():
    n_gpus = torch.cuda.device_count()
    torch.multiprocessing.spawn(main_worker, nprocs=n_gpus, args=(n_gpus, ))
    
def main_worker(gpu, n_gpus):
    image_size = 224
    batch_size = 512
    num_worker = 8
    epochs = ...
    
    batch_size = int(batch_size / n_gpus)
    num_worker = int(num_worker / n_gpus)
    # 멀티 프로세싱 통신 규약 정의
    torch.distributed.init_process_group(
    		backend='nccl’ , init_method='tcp://127.0.0.1:2568’ , world_size=n_gpus, rank=gpu)
    
    model = MODEL
    # Distributed data parallel 정의
    torch.cuda.set_device(gpu)
    model = model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
```

 ✋ 파이썬의 멀티프로세싱 코드

```python
from multiprocessing import Pool

def f(x):
	return x*x

if __name__ == '__main__':
    with Pool(5) as p:
        print(p.map(f, [1, 2, 3]))
```

<br>

## 참고 자료

* **Multi-GPU**
  * [PyTorch Lightning Multi GPU 학습](https://pytorch-lightning.readthedocs.io/en/stable/advanced/multi_gpu.html)
  * [DDP Tutorial](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
