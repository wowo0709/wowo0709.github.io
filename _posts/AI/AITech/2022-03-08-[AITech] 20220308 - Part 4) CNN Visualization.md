---
layout: single
title: "[AITech][CV] 20220308 - Part 4) CNN Visualization"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Filter', 'Saliency Map', 'Activation Map', 'Backpropagate features', 'Grad-CAM']
---



<br>

_**본 포스팅은 POSTECH '오태현' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# CNN Visualization

이번 포스팅에서는 `CNN Visualization`에 대해 알아보겠습니다. 

## Visualizing CNN

**CNN Visualization**이란 무엇이고 왜 필요할까요?

![image-20220312223437777](https://user-images.githubusercontent.com/70505378/158145746-a8cef94c-fcb6-405d-88f0-8b03205edf24.png)

* CNN의 black box 안에는 어떤 정보가 들어있을까요?
* CNN 네트워크는 어떻게 그토록 좋은 성능을 낼 수 있는 것일까요?
* CNN의 성능을 어떻게 향상시킬 수 있을까요?

위 질문들에 대한 해법을 찾는 방법들 중 하나가 CNN Visualization입니다. Black box에 들어있는 정보들을 시각화함으로써 **설명가능하한 모델(XAI, eXplainable AI)** 만드려는 시도인 것입니다. 

그렇다면 어떤 정보를 시각화해야 할까요?

정해진 것은 없습니다. 아래와 같이 다양한 정보와 기법들을 사용하여 CNN Visualization을 하려는 시도들이 많아지고 있습니다. 

![image-20220312223953234](https://user-images.githubusercontent.com/70505378/158145749-5081457e-629a-4144-8f29-76211d28b02b.png)



<br>

## Analysis of model behaviors

### Embedding feature analysis

CNN 구조에서 low level의 convolution layer들은 선이나 면 같은 low level feature들을, high level의 convolution layer들은 패턴 등의 추상적인 high level feature들을 학습하게 됩니다. 

![image-20220312225500235](https://user-images.githubusercontent.com/70505378/158145750-e0b5a30c-d2b7-412e-9840-112b63f62f96.png)

그런데 이런 high level feature들은 매우 고차원이기 때문에, 직관적으로 이해하기 어렵습니다. 이를 우리가 이해 가능한 형태로 만들기 위해 사용할 수 있는 기법이 **차원 축소(dimension reduction)** 기법입니다. 

![image-20220312225923104](https://user-images.githubusercontent.com/70505378/158145753-8eb686e5-0847-4aa0-95e3-991afe74fe82.png)

그 중에서도 t-SNE(t-distributed stochastic neighbor embedding) 기법이 많이 사용됩니다. 아래는 고차원의 특징 정보를 2차원으로 축소시켜 시각화한 것입니다. 보시면 2차원에서도 어느정도 각 cluster를 구분할 수 있는 것을 볼 수 있습니다. 이렇게 고차원의 정보를 저차원으로 내렸을 때 그 정보가 남아있다는 것을 '매니폴드 정리'라 합니다. 

![image-20220312230106856](https://user-images.githubusercontent.com/70505378/158145645-bbb29037-186c-43c8-b660-c06912549aa0.png)

위 그림을 보면 3, 5, 8에 해당하는 cluster들의 거리가 가까운 것을 볼 수 있는데, 이로부터 실제로 3, 5, 8이 공유하는 특징이 유사하다는 것을 알 수도 있습니다. 



### Activation investigation

여기서는 모델이 알고 있는 정보를 어떻게 시각화 할 지에 대한 방법에 대해 알아봅니다. 

Activation map은 CNN의 중간 단계에서 CNN, Pooling, Activation function 등을 거친 중간 결과 feature map을 말합니다. 이 activation map을 이용해 어떻게 CNN visualization을 수행하는 지 알아봅시다. 

**Layer activation**

중간 activation map의 특정 채널의 map을 하나 가져옵니다. map의 해상도를 input image 해상도로 resizing하고, 값이 큰 곳은 밝게 작은 곳은 어둡게 해줍니다. 그리고 input image와 후처리된 map을 overlap합니다. 

그러면 아래와 같은 결과를 얻을 수 있습니다. 

![image-20220312231250949](https://user-images.githubusercontent.com/70505378/158145655-f9908af7-ed9b-4272-a32f-8795f8c3c544.png)

**Maximally activating patches**

다른 방법으로 activation map이 큰 값을 갖는 부분의 input image patch를 뜯어서 확인하는 방법도 있습니다. 

![image-20220312231348524](https://user-images.githubusercontent.com/70505378/158145662-fa9d2a45-8d45-44c9-aaff-56c1b2932d8c.png)

과정은 아래와 같습니다. 

1. 학습된 CNN 모델의 중간 activation map에서 관찰할 특정 채널을 선택합니다. 
2. Input image를 주면서 선택한 activation map의 채널을 기록(저장)해둡니다. 
3. Input image에서 앞서 기록된 map이 큰 값을 가지는 부분의 patch를 가져와 확인합니다. 이 때 map과 input image의 해상도가 다른데, receptive field에 해당하는 patch를 가져오면 됩니다. 

**Class visualization - Gradient ascent**

Class visualization은 모델이 알고 있는 class에 대한 정보로, **class에 해당하는 output이 나올 수 있도록 input을 업데이트**하는 과정입니다. (Generate a synthetic image that triggers maximal class activation)

여기서 output으로 부터 input을 갱신할 때 사용하는 기법이 Gradient ascent이고, 손실 함수로는 다음과 같은 형태의 함수를 사용합니다. 

![image-20220312232908297](https://user-images.githubusercontent.com/70505378/158145666-dcada241-a9bc-46f3-9ef5-68c773304c1c.png)

과정은 아래와 같습니다. 

1. Initial random image에 대해 모델 추론을 수행합니다. 
2. Target class score를 maximizing 하도록 input image까지 backpropagation을 수행합니다. 
3. Input image를 업데이트합니다. 
4. 업데이트된 current input image에 관해 1~3 과정을 반복합니다. 

위 과정을 거치면 아래와 같은 결과를 얻을 수 있습니다. 

![image-20220312233150745](https://user-images.githubusercontent.com/70505378/158145670-1060d0b8-ee18-4505-a94c-34be3aea4ea1.png)

이로부터 모델이 어떤 데이터로 학습이 되었고, 클래스를 판별할 때 무엇을 보고 판단하는지 등을 확인할 수 있습니다. 





<br>

## Model decision explanation

여기서는 모델이 input image에서 어떤 부분을 주목하여 바라보고 있는 지를 시각화하는 방법에 대해 알아봅니다. 

### Saliency test

**Occlusion map**

Input image에서 특정 부분을 가려서 모델의 입력으로 넣게 되면 모델이 에측한 class score가 달라질 것입니다. 이런 방식으로 input image의 각 부분을 가려 모델의 input으로 넣어보고, 각각의 prediction score를 map 형태로 나타낸 것을 **occlusion map**이라 하고, occlusion map에서 낮은 값을 보이는 부분을 **Salient part**라고 합니다. 즉, salient part는 모델이 해당 이미지를 예측할 때 주목하는 영역을 의미합니다. 

![image-20220312234421457](https://user-images.githubusercontent.com/70505378/158145676-1240f6cd-dc25-4668-bd22-9792d9120303.png)

**via Backpropagation**

두번째 saliency test 방법으로 backpropagation을 통해 모델이 input image의 어느 부분을 주목하고 있는지 확인하는 방법이 있습니다. 

![image-20220312234552702](https://user-images.githubusercontent.com/70505378/158145679-62377775-b28a-44d1-a762-283e55bdd4fe.png)

과정은 아래와 같습니다. 

1. Input image의 모델 예측 결과를 얻습니다. 
2. Input domain까지 back propagation을 수행합니다. 
3. 얻은 gradient magnitude map을 시각화합니다. 절대값이 큰 부분은 밝게, 작은 부분은 어둡게해서 시각화하면 위와 같은 결과를 얻을 수 있습니다. 

이때 부호를 무시한 절댓값을 사용하는 이유는 부호와 상관 없이 값의 크기가 큰 부분이 해당 이미지에서 예측값의 변화를 크게 일으키는 주요 부분이라는 의미이기 때문입니다. 



### Backpropagate features

**Guided backpropagation**

**Standard backpropagation**은 아래와 같이 동작합니다. Forward pass 과정에서 ReLU에 의해 masking된 부분을 저장해서, Backward pass 시에는 gradient 값에 따라 masking을 하는 것이 아닌 Forward pass 시 저장했던 mask를 이용해 masking을 하는 것입니다. 

![image-20220312235709968](https://user-images.githubusercontent.com/70505378/158145683-4bec357a-071b-4811-8348-a82efdd5dacd.png)

**Deconvolution**은 이와 달리 backward pass 시 gradient 값에 따라 masking을 수행하는 기법입니다. 

![image-20220312235805315](https://user-images.githubusercontent.com/70505378/158145685-c953f350-79e7-4a6f-8f1f-69bf61926bb8.png)

그리고 **Guided backpropagation**은 두 기법을 모두 적용하는 것입니다. 즉, forward pass 시에 저장했던 mask에 의한 masking과 backward pass 시 gradient 값에 따른 masking을 모두 수행하는 것입니다. 

![image-20220313000038644](https://user-images.githubusercontent.com/70505378/158145690-d6317cf9-8303-4f5f-b324-cd0dd713a5f4.png)

이때 '왜?'라고 묻는다면 명확한 설명을 하기에는 어렵습니다. 다만, 실험적으로 이 guided backpropagation 방법이 가장 직관적인 시각화 정보를 제공한다는 것을 확인할 수 있습니다. 

![image-20220313000151938](https://user-images.githubusercontent.com/70505378/158145691-aee982e6-929b-44d9-bb17-78ce4ee5dad6.png)





### Class activation mapping (CAM)

**CAM** 기법은 유명하고 가장 많이 사용되는 CNN visualization 기법입니다. 

**Class activation mapping (CAM)**

CAM 기법은 아래와 같이 input image에서 모델이 주목한 부분을 히트맵 형태로 표시해주는 결과를 만들어냅니다. 

![image-20220313001006710](https://user-images.githubusercontent.com/70505378/158145693-2caa046f-7a19-4343-8ec3-276426f8ca2a.png)

이 CAM 기법을 적용하기 위해서는 2가지 과정이 필요합니다. 

1. CNN 아키텍쳐의 마지막 부분을 GAP(Global Average Pooling)과 FC layer로 교체합니다. 
2. 교체된 CNN 아키텍쳐를 재학습시킵니다. 

![image-20220313001149675](https://user-images.githubusercontent.com/70505378/158145701-65d98d6e-094a-441b-92f5-873fa05d818a.png)

위 그림과 같이 GAP의 output인 feature vector를 weighted sum해서 CAM을 확인하고 싶은 클래스에 연결해줍니다. 이를 수식적으로 보면 아래와 같은데, 

![image-20220313001323765](https://user-images.githubusercontent.com/70505378/158145705-15941a92-e0a5-4855-907b-eb6cccf96667.png)

S<sub>c</sub>는 FC layer 결과, w는 weight, F는 feature vector의 값을 나타냅니다. 이 수식을 변형하면 마지막 수식과 같이 표현할 수 있는데, GAP이 적용되기 전에는 spatial information이 보존되어 있기 때문에 이를 CAM으로 visualization하는 것입니다. 

![image-20220313001610565](https://user-images.githubusercontent.com/70505378/158145707-e8b46cec-cab1-47ca-bf22-077350c4824b.png)

**Grad-CAM**

하지만 모든 네트워크의 마지막 부분이 GAP과 FC layer로 변경될 수 있는 것은 아닙니다. Grad-CAM은 **모델 아키텍쳐를 변경하지 않고도  CNN backbone만 이용한다면 CAM을 얻을 수 있는 기법**으로 제안되었습니다. 

앞서 CAM을 얻기 위해서는 마지막 feature map과 결합되는 importance weight가 필요했습니다. 바로 이 **importance weight를 어떻게 얻는지**가 Grad-CAM의 핵심입니다. 

![image-20220313002112009](https://user-images.githubusercontent.com/70505378/158145709-7b15d650-c1c8-42f7-967f-f427e4af19c2.png)

Grad-CAM에서의 weight는 앞서 본 weight와 조금 다르기 때문에 alpha로 표현합니다. 이 alpha는 backpropagation을 통해 구하게 되는데, 앞서 살펴본 saliency map을 구할 때와 다르게 input domain이 아니라 **필요한 activation map까지만 backprapagation하고 GAP을 적용**함으로써 구할 수 있습니다. 수식에서 y<sup>c</sup>는 살펴보고자 하는 class score의 loss, A는 관심이 있는 activation map입니다. 

![image-20220313003314584](https://user-images.githubusercontent.com/70505378/158145713-bc8a7487-e79b-4982-80af-d268b3caf0c8.png)

그리고 alpha를 activation map(A)과 linear combination을 통해 곱하고, ReLU를 적용함으로써 Grad-CAM 결과를 얻을 수 있습니다. 

![image-20220313003542218](https://user-images.githubusercontent.com/70505378/158145716-dd20bbe9-acb4-4e3b-9b3e-b3b30032c6f9.png)

Grad-CAM은 CNN Backbone만 있다면 어느 아키텍쳐에서도 이용할 수 있다는 것이 큰 장점입니다. 이와 함께 rough하지만 클래스에 대해 민감한 결과를 얻어내는 Grad-CAM과 클래스에 대한 구분성은 떨어지지만 sharp한 결과를 얻어내는 Guided backpropagation의 결과를 곱해서 더 일반화된 결과를 얻어내는 CNN visualization 방법도 고안되었습니다. 

![image-20220313004315835](https://user-images.githubusercontent.com/70505378/158145721-9271671c-efa6-4cc3-8ad4-762752b9b2f9.png)

**SCOUTER**

최근에는 모델이 왜 이미지를 해당 클래스로 분류했는지를 확인하는 것 뿐 아니라, **왜 다른 클래스로 분류하지 않았는지**에 대해 시각화하는 기법들로도 확장이 되었습니다. 

![image-20220313004506164](https://user-images.githubusercontent.com/70505378/158145727-2fd6678e-4420-491d-a26d-e39d35e06f3f.png)



**GAN dissection**

지금까지의 내용들을 통해 CNN 모델이 우리도 모르게 우리가 충분히 납득할 만한 정보들을 학습하고 가지고 있다는 것을 알 수 있었습니다. 

이를 이용하면 CNN 모델에 대한 정보를 확인하는 것 뿐 아니라, 이를 활용하여 우리가 커스터마이징 할 수 있는 모델을 만들어낼 수도 있을 것입니다. 

GAN과 같은 생성 모델과 함께 사용한다면, 아래와 같이 우리가 원하는 부분을 변경하거나 생성해내는 것과 같이 컨트롤이 가능한 모델을 만들어 낼 수도 있습니다. 

![image-20220313005058715](https://user-images.githubusercontent.com/70505378/158145733-2b8aad1e-7c82-403f-97f6-4e383e06600a.png)

<br>

## 실습) CNN Visualization

CNN visualization 실습을 위한 이미지 분류 모델로는 지난 Semantic segmentation 포스팅의 실습에서 구축했던 VGG-11 모델을 사용합니다.  데이터셋도 그대로 마스크 데이터셋을 사용합니다. 

```python
model_root = './model.pth'

model = VGG11Classification()
model.load_state_dict(torch.load(model_root))
```

>  **_아래 나오는 코드들은 전체 코드가 아닌 핵심 코드들만 발췌했기 때문에, 포스팅의 내용과 비교하며 CNN Visualization이 코드 상에서는 어떤 식으로 어떤 흐름에 따라 구현되는 지를 중심적으로 보는 것을 추천드립니다._**

### Filter visualization

처음으로는 Filter visualization입니다. 다음의 3가지 task로 나눌 수 있습니다. 

* **TO DO (1)**: 주어진 모듈의 parameter 개수를 return하는 **get_params_num** 코드를 완성해주세요.
* **TO DO (2)**: 모델에서 **conv1_filters_data**를 얻는 코드를 완성해주세요.
  * 모델의 맨 첫번째 convolution layer는 입력 이미지의 채널 수와 같은 RGB 3채널을 갖게 됩니다. 따라서 시각화가 용이하고, 이외의 filter들은 더 높은 채널 차원수를 갖기 때문에 시각화가 어렵습니다. 
* **TO DO (3)**: Activation을 target layer에 시각화하기 위해 hook function을 register해주세요.

**TODO (1)**

```python
def get_module_params_num(module):
  """
  Return the parameter number of modules
  With parameter in module in shape of (H,W,D), the size of such parameter would be HxWxD

  Keyword arguments:
  module: the module is composed of several named parameters
  """
  param_num = 0

  for _, param in module.named_parameters():
    
    '''==========================================================='''
    '''======================== TO DO (1) ========================'''

    param_size = 1
    for size in list(param.size()):
      param_size *= size
    param_num += param_size

    '''==========================================================='''
    '''======================== TO DO (1) ========================'''

  return param_num

def get_model_params_num(model):
  module_num = 0
  for name, module in model._modules.items():
    module_num += get_module_params_num(module)
  return module_num


num_params = get_model_params_num(model)
print(f"Number of parameters in customed-VGG11: {num_params}")
# Number of parameters in customed-VGG11: 9229575
```

**TODO (2)**

첫번째 convolution layer의 filter를 시각화합니다. 

```python
def plot_filters(data, title=None):
    """
    Take a Tensor of shape (n, K, height, width) or (n, K, height, width)
    and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
    """
    
    if data.size(1) > 3:
      data = data.view(-1, 1, data.size(2), data.size(3))
        
    data = image_tensor_to_numpy(data)
        
    # normalize data for display
    data = (data - data.min()) / (data.max() - data.min())
    
    # force the number of filters to be square
    n = int(np.ceil(np.sqrt(data.shape[0])))
    padding = (((0, n ** 2 - data.shape[0]),
               (0, 2), (0, 2))                 # add some space between filters
               + ((0, 0),) * (data.ndim - 3))  # don't pad the last dimension (if there is one)
    data = np.pad(data, padding, mode='constant', constant_values=1)  # pad with ones (white)
    
    # tile the filters into an image
    data = data.reshape((n, n) + data.shape[1:]).transpose((0, 2, 1, 3) + tuple(range(4, data.ndim + 1)))
    data = data.reshape((n * data.shape[1], n * data.shape[3]) + data.shape[4:])
    data = data.squeeze()
    
    # plot it
    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.title(title)
    plt.imshow(data)
    

'''==========================================================='''
'''======================== TO DO (2) ========================'''

conv1_filters_data = model.backbone.conv1.weight.data

'''==========================================================='''
'''======================== TO DO (2) ========================'''

plot_filters(conv1_filters_data, title="Filters in conv1 layer")
```

![image-20220314181919737](https://user-images.githubusercontent.com/70505378/158145734-56764bc6-98ba-42ab-ba56-2ba18b761046.png)

**TODO (3)**

이번엔 filter가 아닌 모델의 중간 결과 activation map을 시각화합니다. 

```python
plot_activations = plot_filters
activation_list = []

def show_activations_hook(name, module, input, output):
  # conv/relu layer outputs (BxCxHxW)
  if output.dim() == 4:
    activation_list.append(output)
    plot_activations(output, f"Activations on: {name}")

# Register the hook on the select set of modules
module_list  = [model.backbone.conv1, model.backbone.bn4_1]
module_names = ["conv1", "bn4_1"]

# You may use functools.partial to make function already filled with target module name
for idx, (name, module) in enumerate(zip(module_names, module_list)):
  '''==========================================================='''
  '''======================== TO DO (3) ========================'''

  hook = functools.partial(show_activations_hook, name)
  module.register_forward_hook(hook)

  '''==========================================================='''
  '''======================== TO DO (3) ========================'''

_ = model(img)
np.shape(activation_list[0])
# torch.Size([1, 64, 224, 224])
```

![image-20220314182254785](https://user-images.githubusercontent.com/70505378/158145738-9fe462ba-099b-46a5-9300-2184b3c7dd2e.png)

<br>

### Saliency map

Saliency map은 CNN이 최종 결과를 내리기까지 각 pixel이 기여하고 있는 정도를 시각화하여 나타낸 map입니다. 

Saliency map은 아래와 같은 수식을 통해 구할 수 있으며, 이때 s<sub>y</sub>는 class y에 대한 logit입니다. (fully connected layer를 통과한 다음, 즉 softmax layer를 통과하기 이전의 값)

Gradient를 계산한 다음에, 해당 값들을 시각화함으로써 input image에 대한 saliency를 확인할 수 있습니다. 자세한 내용은 아래 논문의 본문을 참고해주세요!

[[1\] Simonyan et al., Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps, ICLR 2014](https://arxiv.org/pdf/1312.6034.pdf)

```python
def compute_gradient_score(scores, image, class_idx):
    """
    Returns the gradient of s_y (the score at index class_idx) w.r.t the input image (data), ds_y / dI. 

    class_idx에 해당하는 class에 대한 gradient인 s_y를 계산해야 합니다.
    전체 class의 개수의 길이를 갖는 scores에서 원하는 index의 score를 s_y로 얻은 다음, 해당 s_y를 back-propagate하여 gradient를 계산하는 코드를 완성해주세요.
    """
    grad = torch.zeros_like(image)

    '''==========================================================='''
    '''======================== TO DO (4) ========================'''

    s_y = scores[idx]
    s_y.backward()

    '''==========================================================='''
    '''======================== TO DO (4) ========================'''

    grad = image.grad
    assert tuple(grad.shape) == (1, 3, 224, 224)

    return grad[0]

def visualize_saliency(image, model):
    input = Variable(image.unsqueeze(0), requires_grad=True)
    output = model(input)[0]
    max_score, max_idx = torch.max(output, 0)

    grad = compute_gradient_score(output, input, max_idx)

    vis = grad ** 2
    vis, _ = torch.max(vis, 0)
    
    return vis


model = VGG11Classification()
model.load_state_dict(torch.load(model_root))
model.double()

input_images = []
saliency_maps = []
  
for _, sample in enumerate(mask_dataset):
  saliency_map = visualize_saliency(sample, model)
  assert list(saliency_map.shape) == [224, 224]
    
  saliency_maps.append(saliency_map.unsqueeze(0))
  input_images.append(sample)

row_list = list(zip(input_images, saliency_maps))
show_images(row_list)
```

![image-20220314182803340](https://user-images.githubusercontent.com/70505378/158145741-29090818-2d88-4598-8e94-fe0d69f03bb4.png)



<br>

### Grad-CAM

이번에는 Grad-CAM을 시각화 해보겠습니다. 다음 두 가지 과정이 필요합니다. 

* **TO DO (5)**: 함수 **vis_gradcam**을 완성해주세요. (1) Layer의 activation을 저장할 function을 hook하고 (2) forward하고 (3) gradients를 저장하기 위해 hook을 register한 다음 (4) 출력에 대한 최댓값에 해당하는 score를 backward해야 합니다. (1) - (3) - (2) - (4)의 순서로 Grad-CAM을 시각화할 수 있습니다.

- **TO DO (6)**: 함수 **vis_gradcam**을 완성해주세요. 아래의 텍스트에 설명되어 있는 **Grad-CAM**의 값을 계산한 다음 (2) 원본 이미지의 크기에 맞게 upsampling 해야합니다.

```python
save_feat=[]
def hook_feat(module, input, output):
  save_feat.append(output)
  return output


save_grad=[]
def hook_grad(grad):
  """
  get a gradient from intermediate layers (dy / dA).
  See the .register-hook function for usage.
  :return grad: (Variable) gradient dy / dA
  """ 
  save_grad.append(grad)
  return grad


def vis_gradcam(vgg, img):
  """
  Imshow the grad_CAM.
  :param vgg: VGG11Customed model
  :param img: a dog image
  output : plt.imshow(grad_CAM)
  """
  vgg.eval()

  '''==========================================================='''
  '''======================== TO DO (5) ========================'''
  # (1) Reister hook for storing layer activation of the target layer (bn5_2 in backbone)
  vgg.backbone.bn5_2.register_forward_hook(hook_feat)
  
  # (2) Forward pass to hook features
  img = img.unsqueeze(0)
  s = vgg(img)[0]

  # (3) Register hook for storing gradients
  # print(save_feat)
  save_feat[0].register_hook(hook_grad)
  
  # (4) Backward score
  y = torch.argmax(s).item()
  s_y = s[y]
  s_y.backward()

  '''==========================================================='''
  '''======================== TO DO (5) ========================'''



  # Compute activation at global-average-pooling layer
  gap_layer  = torch.nn.AdaptiveAvgPool2d(1)
  alpha = gap_layer(save_grad[0][0].squeeze())
  A = save_feat[0].squeeze()



  '''==========================================================='''
  '''======================== TO DO (6) ========================'''
  # (1) Compute grad_CAM 
  # (You may need to use .squeeze() to feed weighted_sum into into relu_layer)
  relu_layer = torch.nn.ReLU()

  weighted_sum = torch.sum(alpha*A, dim=0)
  grad_CAM = relu_layer(weighted_sum)

  # print(grad_CAM)
  grad_CAM = grad_CAM.unsqueeze(0)
  grad_CAM = grad_CAM.unsqueeze(0)

  # (2) Upscale grad_CAM
  # (You may use defined upscale_layer)
  upscale_layer = torch.nn.Upsample(scale_factor=img.shape[-1]/grad_CAM.shape[-1], mode='bilinear')

  grad_CAM = upscale_layer(grad_CAM)
  grad_CAM = grad_CAM / torch.max(grad_CAM)


  '''==========================================================='''
  '''======================== TO DO (6) ========================'''



  # Plotting
  img_np = image_tensor_to_numpy(img)
  if len(img_np.shape) > 3:
    img_np = img_np[0]
  img_np = normalize(img_np)
  
  grad_CAM = grad_CAM.squeeze().detach().numpy()

  plt.figure(figsize=(8, 8))
  plt.imshow(img_np)
  plt.imshow(grad_CAM, cmap='jet', alpha = 0.5)
  plt.show

  return grad_CAM


model = VGG11Classification()
model.load_state_dict(torch.load(model_root))
model.double()

img = mask_dataset[1]
res = vis_gradcam(model, img)
```

![image-20220314183132701](https://user-images.githubusercontent.com/70505378/158145745-da4d6e29-324a-4c9e-ab93-328e6f06d3c7.png)























<br>

<br>

# 참고 자료

* 
