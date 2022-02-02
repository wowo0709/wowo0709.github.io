---
layout: single
title: "[AITech] 20220126 - Custom Model 개발하기"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['파이토치', '실습','BasicOperations','nn.Module']
---



<br>

## 학습 내용 정리

### Custom Model 개발하기

#### Basic Operations

* torch.tensor와 torch.Tensor의 차이

* 텐서 인덱싱

  * [index_select(tensor, axis, indices: 1D tensor)](https://pytorch.org/docs/stable/generated/torch.index_select.html#torch.index_select)

    * axis(dim)은 이동하는 방향이 아니라 선택 후보의 방향이다!

    ```python
    A = torch.Tensor([[1, 2],
                      [3, 4]])
    
    indices = torch.tensor([0])
    output = torch.squeeze(torch.index_select(A, 1, indices))
    print(output)
    # tensor([1., 3.])
    ```

    

  * numpy style indexing

* 해당 인덱스의 값 모으기

  * [gather(tensor, axis, indices_tensor: index-value tensor)](https://pytorch.org/docs/stable/generated/torch.gather.html#torch.gather)

    * 예

    > A = torch.Tensor([[1,2], [3,4]])
    >
    > torch.gather(A, dim=0, index=torch.tensor([[0,1], [1,1]])) 이라면, 
    >
    > 각 인덱스에 대해 차례로
    >
    > \[0\]\[0\] => A\[index\[0\]\[0\]\]\[0\] = A\[0\]\[0\] = 1
    >
    > \[0\]\[1\] => A\[index\[0\]\[1\]\]\[1\] = A\[1\]\[1\] = 4
    >
    > \[1\]\[0\] => A\[index\[1\]\[0\]\]\[0\] = A\[1\]\[0\] = 3
    >
    > \[1\]\[1\] => A\[index\[1\]\[1\]\]\[1\] = A\[1\]\[1\] = 4
    >
    > <br>
    >
    > 즉, dim에 해당하는 차원의 인덱스 값은 index로 전달된 텐서의 값으로 대체되고, 나머지 인덱스 값은 index의 위치값을 그대로 유지합니다. 

    

* 자료구조 확인

  * is_tensor, is_storage, is_complex, ...
  * numel(tensor) :원소의 개수 반환

* Tensor 함수

  * Creating
    * from_numpy(ndarray)
    * zeros(*shape)
    * zeros_like(tensor)
    * full(shape, value)
    * [Tensor.expand(*size)](https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html?highlight=expand)
      * 셀프 broadcasting을 시켜주는 함수로 볼 수 있음. 
  * Indexing, Slicing, Joining, Mutating
    * [chunk(input, chunks, dim=0)](https://pytorch.org/docs/stable/generated/torch.chunk.html#torch.chunk)
    * [tensor_split(input, indices_or_sections, dim=0)](https://pytorch.org/docs/stable/generated/torch.tensor_split.html#torch.tensor_split)
    * [swapdims(input, dim0, dim1)](https://pytorch.org/docs/stable/generated/torch.swapdims.html#torch.swapdims)
    * [tensor.scatter_(dim, index, src, reduce=None)](https://pytorch.org/docs/stable/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter_)
  * Random sampling
    * [normal(mean: tensor, std: tensor)/ normal(mean: float, std: float, size)](https://pytorch.org/docs/stable/generated/torch.normal.html#torch.normal)
      * mean, std를 tensor 타입으로 전달할 경우 output tensor의 각 원소의 mean, std를 각각 지정
      * mean, std를 float 타입으로 전달할 경우 output tensor의 전체 원소의 mean, std를 지정하고 size를 전달
    * [rand(*size)](https://pytorch.org/docs/stable/generated/torch.rand.html#torch.rand)
      * 0~1 사이 랜덤한 값을 uniform distribution으로 생성
    * [randint(low, high, size)](https://pytorch.org/docs/stable/generated/torch.randint.html#torch.randint)
      * low~high 사이 랜덤한 정수 값을 생성
    * [randn(*size)](https://pytorch.org/docs/stable/generated/torch.randn.html#torch.randn)
      * mean=0, std=1 인 정규분포를 따르는 랜덤한 값을 생성
  * Math operations - Pointwise ops
    * abs(tensor)
    * add(input: tensor, other: tensor or number, alpha: number)
    * [addcdiv(input, tensor1, tensor2, value)](https://pytorch.org/docs/stable/generated/torch.addcdiv.html#torch.addcdiv)
    * [addcmul(input, tensor1, tensor2, value)](https://pytorch.org/docs/stable/generated/torch.addcmul.html#torch.addcmul)
  * Reduction ops
    * [argmax(input, dim)](https://pytorch.org/docs/stable/generated/torch.argmax.html#torch.argmax), [argmin(input, dim)](https://pytorch.org/docs/stable/generated/torch.argmin.html#torch.argmin)
    * [amax(input, dim)](https://pytorch.org/docs/stable/generated/torch.amax.html#torch.amax), [amin(input, dim)](https://pytorch.org/docs/stable/generated/torch.amin.html#torch.amin)
    * [all(input, dim)](https://pytorch.org/docs/stable/generated/torch.all.html#torch.all), [any(input, dim)](https://pytorch.org/docs/stable/generated/torch.any.html#torch.any)
  * Math operations - Comparison ops
    * [allclose(input, other, rtol, atol)](https://pytorch.org/docs/stable/generated/torch.allclose.html#torch.allclose)
    * [argsort(input, dim, descending=False)](https://pytorch.org/docs/stable/generated/torch.argsort.html#torch.argsort)
    * [eq(input, other)](https://pytorch.org/docs/stable/generated/torch.eq.html#torch.eq), [equal(input, other)](https://pytorch.org/docs/stable/generated/torch.equal.html#torch.equal)
      * elementwise equality: tensor VS tensor equality: bool
  * Math operations - Other ops
    * [einsum(equation, *operands)](https://pytorch.org/docs/stable/generated/torch.einsum.html#torch.einsum)
      * einstein summation의 약자
    * atleast_1d(input: tensor or a list of tensors), atleast_2d(input: tensor or a list of tensors), atleast_3d(input: tensor or a list of tensors)
    * [broadcast_tensors(*tensors)](https://pytorch.org/docs/stable/generated/torch.broadcast_tensors.html#torch.broadcast_tensors), [broadcast_to(input: tensor, shape)](https://pytorch.org/docs/stable/generated/torch.broadcast_to.html#torch.broadcast_to)
  * Math operations - BLAS and LAPACK ops
    * 선형대수학과 연관된 함수
    * BLAS: Basic Linear Algebra Subprograms
    * LAPACK: Linear Algebra PACKage
    * 현재는 torch.linalg 모듈의 함수들을 사용할 것이 권장됨

* [torch.linalg](https://pytorch.org/docs/stable/linalg.html#)

  * 선형대수학 관련 함수들을 모아놓은 모듈

* [torch.nn](https://pytorch.org/docs/stable/nn.html)

  * 딥러닝 모델을 만들기 위한 Basic Building Block들을 미리 만들어 놓은 모듈
  * Linear와 LazyLinear의 차이는?
    * LazyLinear는 아직 in_features가 결정되지 않은 Linear layer입니다. 최초 실행 시 in_features가 결정되면서 Linear와 동일하게 동작합니다. 

<br>

#### nn.Module

파이토치의 [nn.Module](https://pytorch.org/docs/stable/generated/torch.nn.Module.html) 클래스는 여러 기능들을 한 곳에 모아놓는 상자의 역할을 합니다. 

nn.Module 이라는 상자는 또 다른 nn.Module 상자를 포함할 수도 있으며, 어떻게 사용하느냐에 따라 다른 의미를 가집니다. 

- `nn.Module`이라는 상자에 `기능`들을 가득 모아놓은 경우 `basic building block`
- `nn.Module`이라는 상자에 `basic building block`인 `nn.Module`들을 가득 모아놓은 경우 `딥러닝 모델`
- `nn.Module`이라는 상자에 `딥러닝 모델`인 `nn.Module`들을 가득 모아놓은 경우 `더욱 큰 딥러닝 모델`

<br>

* **Containers**

  * nn.Module 블록들을 묶어서 관리하는 클래스들

  * [nn.Sequential](https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html#torch.nn.Sequential)

    * 순차적으로 forward를 실행할 때 사용

    ```python
    class Add(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
    
        def forward(self, x):
            return x + self.value
    
    #        y = x + 3 + 2 + 5
    from collections import OrderedDict
    calculator = nn.Sequential(OrderedDict([
                                     ('plus1', Add(3)),
                                     ('plus2', Add(2)),
                                     ('plus3', Add(5))
    ]))
    
    
    x = torch.tensor([1])
    
    output = calculator(x)
    ```

  * [nn.ModuleList](https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html#torch.nn.ModuleList)

    * 모듈들을 모아놓고 사용하고 싶은 모듈만 선택해서 사용
    * 인덱스로 관리

    ```python
    class Add(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
    
        def forward(self, x):
            return x + self.value
    
    
    class Calculator(nn.Module):
        def __init__(self):
            super().__init__()
            self.add_list = nn.ModuleList([Add(2), Add(3), Add(5)])
    
        def forward(self, x):
            #        y = ((x + 3) + 2) + 5 
            x = self.add_list[1](x)
            x = self.add_list[0](x)
            x = self.add_list[2](x)
            return x
    
    
    x = torch.tensor([1])
    
    calculator = Calculator()
    output = calculator(x)
    ```

    

  * [nn.ModuleDict](https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html#torch.nn.ModuleDict)

    * key 값을 이용해 모듈들을 관리할 때 사용

    ```python
    class Add(nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
    
        def forward(self, x):
            return x + self.value
    
    
    class Calculator(nn.Module):
        def __init__(self):
            super().__init__()
            self.add_dict = nn.ModuleDict({'add2': Add(2),
                                           'add3': Add(3),
                                           'add5': Add(5)})
    
        def forward(self, x):
            #        y = ((x + 3) + 2) + 5 
            x = self.add_dict['add3'](x)
            x = self.add_dict['add2'](x)
            x = self.add_dict['add5'](x)
            return x
    
    
    x = torch.tensor([1])
    
    calculator = Calculator()
    output = calculator(x)
    ```

* **Parameters**: [nn.parameter.Parameter](https://pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html?highlight=parameter)

  * nn.parameter.Parameter(data, requires_grad=True)
  * [self.register_parameter(name, tensor)](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=register_parameter#torch.nn.Module.register_parameter): 모듈에 새로운 파라미터 등록
  
  ```python
  class Linear(nn.Module):
      def __init__(self, in_features, out_features):
          super().__init__()
  
          self.W = Parameter(torch.ones(out_features, in_features))
          self.b = Parameter(torch.ones(out_features))
  
      def forward(self, x):
          output = torch.addmm(self.b, x, self.W.T)
  
          return output
  ```
  
  * **Tensor** 대신 **Parameter**를 사용하는 이유!!
  
    * Tensor는 모델 저장 시 저장되지 않는다!
  
    * Gradient가 계산되지 않는다!
  
    * 혹시 갱신되지는 않지만 저장하고 싶은 Tensor 값이 있다면, **buffer**에 tensor를 등록한다!
      * [self.register_buffer(name, tensor, persistent=True)](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=register_buffer#torch.nn.Module.register_buffer)
      * 위에서 self는 nn.Module을 상속받은 커스텀 모델 클래스

**nn.Module 분석하기**

* **SubModule**

  * SubModule 표시하기: named_modules() VS named_children()
    * [named_modules()](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=named#torch.nn.Module.named_modules): 자신에 속하는 전체 하위 모듈을 표시
    * [named_children()](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=child#torch.nn.Module.named_children): 하나 아래 단계의 하위 모듈까지만 표시
    * 이름 없이 그냥 모듈만 가져올 경우 modules(), children() 사용
  * SubModule 가져오기: [get_submodule(target_name)](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=get_submodule#torch.nn.Module.get_submodule)

* **Parameter**

  * Parameter 표시하기: parameters(), named_parameters()
  * Parameter 가져오기: get_parameter(target_name)

* **Buffer**

  * Buffer 표시하기: buffers(), named_buffers()
  * Buffer 가져오기: get_buffer(target_name)

* 같은 레벨(함수 레벨, basic block 레벨 등)에서는 서로의 참조를 허용하지 않는다!!!

  ```python
  class Function_C(nn.Module):
      def __init__(self):
          super().__init__()
          self.register_buffer('duck', torch.Tensor([7]), persistent=True)
  
      def forward(self, x):
          x = x * self.duck
          
          return x
  
  class Function_D(nn.Module):
      def __init__(self):
          super().__init__()
          self.W1 = Parameter(torch.Tensor([3]))
          self.W2 = Parameter(torch.Tensor([5]))
          # self.c = Function_C()
  
      def forward(self, x):
          x = x + self.W1
          x = Function_C().forward(x) # self.c(x)
          x = x / self.W2
  
          return x
  ```

* 모듈의 이름(이를 **repr**이라 함)을 재설정 해주고 싶다면, 해당 모듈 내에서 `extra_repr(self): return ' '`을 오버라이딩 해준다. 

* [Docstring](https://www.datacamp.com/community/tutorials/docstrings-python)

  * Docstring은 함수 또는 클래스의 맨 위에 해당 함수/클래스에 대한 정보(파라미터, 반환값 등)를 적시하는 것으로, 코멘트(주석)와는 다르다. 
  * `__doc__` 프로퍼티로 module에 대한 docstring을 볼 수 있다. 
  * `help(module)`로 module에 대한 더 자세한 설명(메서드, 프로퍼티 등)을 볼 수 있다. 
  * Documentation이 없는 모델이라면 Docstring을 Documentation처럼 여기고 꼼꼼히 보아야 한다. 

  ```python
  def string_reverse(str1):
      '''
      Returns the reversed String.
  
      Parameters:
          str1 (str):The string which is to be reversed.
  
      Returns:
          reverse(str1):The string which gets reversed.   
      '''
  
      reverse_str1 = ''
      i = len(str1)
      while i > 0:
          reverse_str1 += str1[i - 1]
          i = i- 1
      return reverse_str1
  ```

**nn.Module 더 알아보기**

* **hook**

  * hook은 패키지화된 다른 코드에서 다른 프로그래머가 custom 코드를 중간에 실행시킬 수 있도록 만들어놓은 인터페이스입니다. 
    * 프로그램의 실행 로직을 분석하거나, 
    * 프로그램에 추가적인 기능을 제공하고 싶을 때
  * 사용합니다. 
  * 기본적으로 hook은 아래와 같이 동작합니다. 

  ```python
  class Package(object):
      """프로그램 A와 B를 묶어놓은 패키지 코드"""
      def __init__(self):
          self.programs = [program_A, program_B]
          self.hooks = []
  
      def __call__(self, x):
          for program in self.programs:
              x = program(x)
  
              # Package를 사용하는 사람이 자신만의 custom program을
              # 등록할 수 있도록 미리 만들어놓은 인터페이스 hook
              if self.hooks:
                  for hook in self.hooks:
                      output = hook(x)
  
                      # return 값이 있는 hook의 경우에만 x를 업데이트 한다
                      if output:
                          x = output
  
          return x
  ```

  * hook을 어디에 심어놓을 지는 package를 설계하는 설계자에게 달려있습니다. 
  * Tensor의 hook
    * Tensor의 hook에는 `tensor._backward_hooks` 만이 존재하고, 등록은 [tensor.register_hook(hook)](https://pytorch.org/docs/stable/generated/torch.Tensor.register_hook.html#torch.Tensor.register_hook)을 사용하여 할 수 있습니다. 
  * Module의 hook
    * [register_forward_pre_hook(hook)](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=hook#torch.nn.Module.register_forward_pre_hook)
    * [register_forward_hook(hook)](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=hook#torch.nn.Module.register_forward_hook)
    * [register_full_backward_hook(hook)](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=register_full#torch.nn.Module.register_full_backward_hook)
    * module의 `__dict__` attribute에서 parameter, hook 등을 모두 볼 수 있습니다. 
      * backward_hooks의 경우 full_backward_hooks의 전신으로 현재 deprecated 상태이고, state_dict_hooks의 경우 모듈이 내부적으로 사용하는 hook입니다. 
  * [PyTorch hooks 사용 사례 보기](https://medium.com/the-dl/how-to-use-pytorch-hooks-5041d777f904)
    * gradient의 값의 변화를 시각화
    * gradient값이 특정 임곗값을 넘으면 gradient exploding 경고 알림
    * 특정 tensor의 gradient 값이 너무 커지거나 작아지는 현상이 관측되면 해당 tensor 한정으로 gradient clipping

* **[apply](https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=apply#torch.nn.Module.apply) -> applied module**

  * 모델에 custom 함수를 적용시켜 그 하위 모듈들에도 모두 적용되도록 하고 싶을 때 사용합니다. 

  * apply를 통해서 적용하는 함수는 module을 입력으로 받으며, 모델의 모든 module들을 순차적으로 입력받아 처리합니다. 

  * 주로 가중치 초기화에 많이 사용됩니다. 

  * apply는 Postorder Traversal 방식(후위탐색)으로 함수를 module에 적용합니다. 

    ![image-20220126215452221](https://user-images.githubusercontent.com/70505378/151172647-561ff86e-4871-4bdc-8c10-4c1090269758.png)

  * [How to initialize weights in PyTorch?](https://stackoverflow.com/questions/49433936/how-to-initialize-weights-in-pytorch)

    ```python
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    
    net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
    net.apply(init_weights)
    ```

    

<br>



## 참고 자료

* [functools.partial](https://hamait.tistory.com/823)



<br>

<br>
