---
layout: single
title: "[AITech] 20220124 - PyTorch Basics"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Introduction', 'Tensor', 'AutoGrad']
---



<br>

## 학습 내용 정리

### Introduction to PyTorch

딥러닝 코드를 짤 대는 **남이 만들어 놓은 걸 가져다 씁니다.** 자료도 많고, 관리도 잘 되고, 표준이기 때문입니다. 

현재는 많은 종류의 딥러닝 프레임워크들이 있으며, 선두 주자는 **Facebook의 PyTorch**와 **Google의 TensorFlow**입니다. 

![image-20220125134904933](https://user-images.githubusercontent.com/70505378/150939752-4834358a-4439-4ea4-b07e-a87079ddd7a6.png)

파이토치는 **Dynamic computation graphs**, 즉 실행을 하면서 그래프를 생성한다는 장점이 있어서 사용량이 계속해서 크게 증가하고 있습니다. 텐서플로의 경우 그래프를 먼저 정의하고, 실행 시점에 데이터를 feed시켜 주는 방식을 사용하고 있습니다. 

따라서 파이토치는 중간중간 결과를 확인할 수 있기 때문에 **debugging** 측면에서 큰 이점을 가집니다. 따라서 사용하기가 편리합니다. 

반면 텐서플로는 production과 scalability 측면에서 장점을 가진다고 할 수 있습니다. 

* **PyTorch = Numpy + AutoGrad + Function**
  * Numpy 구조를 가지는 Tensor 객체로 array를 표현. 넘파이에서 지원되는 연산을 거의 지원 가능. 
  * 자동미분을 지원하여 DL 연산을 지원
  * 다양한 형태의 DL을 지원하는 함수와 모델을 지원

<br>

### PyTorch Basics

#### Tensor Operations

* Tensor

```python
'''Array 생성하기'''
# 넘파이
import numpy as np
n_array = np.arange(10).reshape(2,5)
print(n_array, n_array.dtype)
# 파이토치
import torch
t_array = torch.FloatTensor(n_array)
print(t_array, t_array.dtype)
'''
[[0 1 2 3 4]
 [5 6 7 8 9]] int64
tensor([[0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.]]) torch.float32
'''

'''list/ndarray에서 Tensor 변환'''
# torch.tensor, torch.from_numpy, torch.IntTensor, torch.FloatTensor, ...
# data to tensor
data = [[3,5],[10,5]]
x_data = torch.tensor(data)
# ndarray to tensor
nd_array_ex = np.array(data)
tensor_array = torch.from_numpy(nd_array_ex)
```

* Tensor basic operations

```python
'''Basic operations'''
data = [[3,5,20], [10,5,50], [1,5,10]]
x_data = torch.tensor(data)

x_data[1:]
x_data[:2, 1:]
x_data.flatten()
torch.ones_like(x_data)
x_data.numpy()
x_data.shape
x_data.dtype
x_data + x_data
x_data + 10

# Pytorch의 tensor는 GPU에 올려서 사용 가능
x_data.device
# device(type='cpu')

if torch.cuda.is_available():
    x_data_cuda = x_data.to('cuda')
x_data_cuda.device
# device(type='cuda', index=0)
```

* Tensor handling
  * view: reshape과 기능은 동일하지만 view의 경우 기존 객체와 같은 메모리 주소를 가리키고 표현 방식만 다름. reshape은 값만 동일하고 다른 메모리 주소에 copy함. 
  * squeeze: 차원의 개수가 1인 차원을 삭제(압축)
  * unsqueeze: 차원의 개수가 1인 차원을 추가(확장)

```python
'''view VS reshape'''
# view
a = torch.zeros(3,2)
b = a.view(6)
a.fill_(1)
print(a,b)
# reshape
a = torch.zeros(3,2)
b = a.t().reshape(6)
a.fill_(1)
print(a, b)
'''
tensor([[1., 1.],
        [1., 1.],
        [1., 1.]]) tensor([1., 1., 1., 1., 1., 1.])
tensor([[1., 1.],
        [1., 1.],
        [1., 1.]]) tensor([0., 0., 0., 0., 0., 0.])
'''

'''squeeze VS unsqueeze'''
# squeeze
tensor_ex = torch.rand(size=(2,1,2))
print(tensor_ex, tensor_ex.shape)

t1 = tensor_ex.squeeze()
print(t1, t1.shape)
'''
tensor([[[0.5553, 0.4658]],
        [[0.8090, 0.0953]]]) torch.Size([2, 1, 2])
tensor([[0.5553, 0.4658],
        [0.8090, 0.0953]]) torch.Size([2, 2])
'''
# unsqueeze
# unsqueeze
tensor_ex = torch.rand(size=(2,2))
print(tensor_ex.shape)
print(tensor_ex.unsqueeze(0).shape)
print(tensor_ex.unsqueeze(1).shape)
print(tensor_ex.unsqueeze(2).shape)
'''
torch.Size([2, 2])
torch.Size([1, 2, 2])
torch.Size([2, 1, 2])
torch.Size([2, 2, 1])
'''
```

* 행렬 곱셈 연산은 dot이 아닌 `mm` 사용

```python
n1 = np.arange(10).reshape(2,5)
t1 = torch.FloatTensor(n1)
n2 = np.arange(10).reshape(5,2)
t2 = torch.FloatTensor(n2)

print(t1.mm(t2))
# print(t1.dot(t2)) # RuntimeError
print(t1.matmul(t2))
'''
tensor([[ 60.,  70.],
        [160., 195.]])
tensor([[ 60.,  70.],
        [160., 195.]])
'''

'''mm과 matmul은 broadcasting 지원 차이'''
a = torch.rand(5,2,3)
b = torch.rand(3)
# a.mm(b) # RuntimeError: a, b must be a matrix
print(a[0].mm(torch.unsqueeze(b,1))) # (2,3) x (3,1) = (2,1)
print(a[1].mm(torch.unsqueeze(b,1)))
print(a[2].mm(torch.unsqueeze(b,1)))
print(a[3].mm(torch.unsqueeze(b,1)))
print(a[4].mm(torch.unsqueeze(b,1)))

print(a.matmul(b))
'''
tensor([[0.0171],
        [0.1744]])
tensor([[0.0630],
        [0.3709]])
tensor([[0.1253],
        [0.1102]])
tensor([[0.6091],
        [0.4064]])
tensor([[0.3232],
        [0.3015]])
        
tensor([[0.0171, 0.1744],
        [0.0630, 0.3709],
        [0.1253, 0.1102],
        [0.6091, 0.4064],
        [0.3232, 0.3015]])
'''
```

* Tensor operations for ML/DL formula

```python
import torch
import torch.nn.functional as F

tensor = torch.FloatTensor([0.5, 0.7, 0.1])
h_tensor = F.softmax(tensor, dim=0)
print(h_tensor)
# tensor([0.3458, 0.4224, 0.2318])

y = torch.randint(5, (10,5))
y_label = y.argmax(dim=1)
print(F.one_hot(y_label))
'''
tensor([[0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0],
        [1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]])
'''
```







#### AutoGrad

PyTorch의 핵심은 **자동 미분의 지원**이다. backward 함수를 사용함으로써 자동 미분을 수행할 수 있다. 

```python
w = torch.tensor(2.0, requires_grad = True)
y = w**2
z = 10*y + 2
z.backward()
print(w.grad) # dz/dw = (dz/dy)(dy/dw) = 10 * 2w = 20w
# tensor(40.)

a = torch.tensor([2., 3.], requires_grad=True)
b = torch.tensor([6., 4.], requires_grad=True)
Q = 3*a**3 - b**2
external_grad = torch.tensor([1., 1.])
Q.backward(gradient=external_grad)

print(a.grad) # dQ/da = 9*a**2
# tensor([36., 81.])
print(b.grad) # dQ/db = -2*b
# tensor([-12.,  -8.])
```

* backward 메서드의 gradient 파라미터
  * 순전파 시 입력 텐서의 값이 존재해야 하는 것처럼 역전파 시 단말 노드(최종 출력 노드)의 gradient 값을 정의해주어야 합니다. 
  * default 값은 1로, 입력값이 scalar 일 때(첫번째 예시의 경우처럼)는 gradient 값을 따로 지정해주지 않아도 됩니다. 
  * 입력이 2개 이상의 값을 가질 경우(두번째 예시의 경우처럼) 값의 개수와 동일한 크기의 최초 gradient를 지정해주어야 역전파가 수행됩니다.  

<br>



## 참고 자료

* [torch.Tensor.backward document](https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html)

<br>
