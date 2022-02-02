---
layout: single
title: "[AITech] 20220128 - PyTorch Troubleshooting"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Out of Memory']
---



<br>

## 학습 내용 정리

### PyTorch Troubleshooting

이 섹션에서는 모델 학습 과정에서 가장 많이 만나게 되는 에러이자, 해결하기 어려운 `OOM(Out of Memory)` 에러에 대해 얘기해보고자 합니다. 

#### **OOM이 해결하기 어려운 이유**

* 왜, 어디서 발생했는지 알기 어렵다. 
* Error backtracking이 이상한 데로 간다. 
* 메모리의 이전 상황의 파악이 어렵다. 

OOM을 해결하기 위해 가장 쉽게 시도해 볼 수 있는 방법으로는 **Batch size를 줄이는 시도**가 있습니다. 아, 그리고 batch size를 조정한 후에는 GPU clean(kernel restart) 과정을 해야 한다는 것을 잊지 마세요!

#### OOM 해결을 위한 방법들

**torch.cuda.empty_cache()**

empty_cache() 함수는 사용되지 않고 있는 GPU 상 cache를 정리합니다. (가비지 컬렉터를 호출하는 것으로 볼 수 있습니다)

이렇게 함으로써 가용 메모리를 확보할 수 있습니다. (메모리 주소의 참조를 끊는 del 과는 구분됩니다)

empty_cache() 함수는 학습 시작 전에 한 번 호출하는 것이 좋다고 합니다 ^_^

```python
import torch
from GPUtil import showUtilization as gpu_usage

print("Initial GPU Usage")
gpu_usage()
'''
Initial GPU Usage
| ID | GPU | MEM |
------------------
| 0 | 0% | 0% |
| 1 | 0% | 0% |
| 2 | 0% | 0% |
| 3 | 0% | 0% |
GPU Usage after allcoating a bunch of Tensors
'''
tensorList = []
for x in range(10):
	tensorList.append(torch.randn(10000000,10).cuda())
print("GPU Usage after allcoating a bunch of Tensors")
gpu_usage()
'''
GPU Usage after allcoating a bunch of Tensors
| ID | GPU | MEM |
------------------
| 0 | 0% | 40% |
| 1 | 0% | 0% |
| 2 | 0% | 0% |
| 3 | 0% | 0% |
'''
del tensorList
print("GPU Usage after deleting the Tensors")
gpu_usage()
'''
GPU Usage after deleting the Tensors
| ID | GPU | MEM |
------------------
| 0 | 0% | 40% |
| 1 | 0% | 0% |
| 2 | 0% | 0% |
| 3 | 0% | 0% |
'''
torch.cuda.empty_cache()
print("GPU Usage after emptying the cache")
gpu_usage()
'''
GPU Usage after emptying the cache
| ID | GPU | MEM |
------------------
| 0 | 0% | 5% |
| 1 | 0% | 0% |
| 2 | 0% | 0% |
| 3 | 0% | 0% |
'''
```

**training loop에 tensor로 축적되는 변수 확인**

tensor로 처리되는 변수들은 GPU 상에서 메모리를 사용하고, 해당 변수 loop 안에 연산이 있을 때 GPU에 computational graph를 생성하면서 메모리를 잠식해갑니다. 

따라서 이런 경우에는 1-d tensor의 경우 파이썬의 기본 객체(int, float, list 등)로 변환하여 처리할 것이 권장됩니다. 

```python
total_loss = 0

for x in range(10):
    # assume loss is computed
    iter_loss = torch.randn(3,4).mean()
    iter_loss.requires_grad = True
    # total_loss += iter_loss 대신, 
    iter_loss += iter_loss.item # 또는 float(iter_loss)
```

**del 명령어의 적절한 사용**

필요가 없어진 변수를 적절히 삭제하는 것도 방법이 될 수 있습니다. 

```python
for i in range(5):
    intermediate = f(input[i])
    result += g(intermediate)
    
del intermediate # del
output = h(result)
del result # del
return output
```

**배치 사이즈를 줄여보기(1로 해보기)**

**torch.no_grad()**

모델 추론 시점에는 역전파 과정이 필요 없으므로, `torch.no_grad()` context를 사용하여 backward 과정으로 인해 사용되는 메모리를 확보할 수 있습니다. 

```python
with torch.no_grad(): # torch.no_grad()
    for data, target in test_loader:
        output = network(data)
        test_loss += F.nll_loss(output, target, size_average=False).item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).sum()
```

**tensor의 precision 줄이기**

tensor의 float precision을 8, 16bit 수준으로 줄이는 것도 하나의 방법입니다. 

그러나 이 방법은 모델의 성능에 직접적인 영향을 줄 수 있고, 매우 큰 모델을 돌리는 것이 아니라면 권장되지 않기 때문에 '최후의 수단' 정도로 생각해 두는 것이 좋을 듯 합니다. 

<br>

이외에도 **CUDNN_STATUS_NOT_INIT**이나 **device-side-assert** 등의 에러도 cuda와 관련하여 OOM의 일종이라고 할 수 있고, 역시 적절한 코드의 처리가 필요합니다. 이에 대해 참고할 만한 내용은 아래 참고자료 _GPU 에러 정리_ 에서 확인하실 수 있습니다. 

<br>

## 참고 자료

* **PyTorch Troubleshooting**
  * [Pytorch에서 자주 발생하는 에러 질문들](https://pytorch.org/docs/stable/notes/faq.html)
  * [OOM시에 GPU 메모리 flush하기](https://discuss.pytorch.org/t/how-to-clean-gpu-memory-after-a-runtimeerror/28781)
  * [GPU 에러 정리](https://brstar96.github.io/shoveling/device_error_summary/)
