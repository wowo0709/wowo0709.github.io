---
layout: single
title: "[AITech][CV] 20220308 - Part 8) 3D Understanding"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 POSTECH '오태현' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# 3D Understanding

이번 포스팅의 내용은 `3D Understanding`입니다. 

우리가 살고 있는 세계는 3D 좌표계에 있습니다. 따라서, 자율주행 자동차나 3D 프린터 등 우리가 실생활의 편의를 위해 개발하는 인공지능 역시 3D 환경에서 잘 동작할 수 있어야 하며, 그러기 위해서는 3D 환경을 잘 이해하고 다룰 수 있는 컴퓨터 비전 모델이 필요합니다. 

이번 강의에서는 컴퓨터에서 3D data의 표현 방법, 다양한 3D dataset들 및 대표적인 3D task (3D object detection, 3D segmentation…)들과 해당하는 모델들을 소개합니다.

먼저 포스팅하기 앞서, 본 강의에서 추천해주신 **Multiple View Geometry**라는 책을 소개하겠습니다. 난이도가 쉽지는 않지만, computer vision 분야의 3D understanding에서의 바이블 같은 책이자 쉽게 pdf를 구할 수 있다고 하니 관심 있으신 분들은 아래 책을 보시는 것을 추천드립니다. 

![image-20220317114217897](https://user-images.githubusercontent.com/70505378/158726015-4272d977-e8fa-42aa-ad1e-8f889778d43b.png)

## Seeing the world in 3D perspective

**3D data representation**

우리가 살고 있는 세상은 3D 세상이기 때문에 공간을 인식하는 것은 매우 중요한 일입니다. 이런 3D application에는 AR/VR, 3D printing, Medical applications, image projection 등 다양한 분야들이 있습니다. 

그렇다면 이러한 3D data를 표현하는 방법에는 무엇이 있을까요? 대표적으로 다음의 6가지 방법을 들 수 있습니다. 

* Multi-view images: 물체를 다각도에서 바라본 여러 장의 2D 이미지로 표현
* Volumetric(voxel): 3차원 공간을 격자로 나누어 물체가 있는 격자는 1, 없는 격자는 0으로 표현
* Part assembly: 물체의 여러 개의 도형의 조합으로 표현
* Point cloud: 물체 표면을 여러 개의 좌표 (x, y, z) 로 표현
* Mesh(Graph CNN): 물체 표면을 여러 개의 좌표 (x, y, z)로 표현한 점을 node, 그 node들은 이은 선분을 edge로 graph 구조로 표현. 많은 경우에 3개의 node를 연결해 삼각형으로 면을 표현. 
* Implicit shape: 물체 표면을 어떤 함수와 0 축과의 교점(평면)으로 표현

![image-20220316233953119](https://user-images.githubusercontent.com/70505378/158725301-d525f3aa-df5b-4ccd-ac49-a174784aa624.png)

**3D datasets**

3D 형태의 데이터셋에는 무엇이 있는지 알아보겠습니다. 

우선 **물체를 3D 형태**로 나타낸 데이터셋으로는 **ShapeNet(51,300 3D models with 55 categories), PartNet(573,585 part instances in 26,671 3D models)**이 있습니다. 그 중 PartNet은 3D 물체를 instance segmentation처럼 각 부분으로 나누어 표현해서, segmentation task에 유용하게 사용됩니다. 

![image-20220316234312077](https://user-images.githubusercontent.com/70505378/158725303-652d4a31-4e5c-4f24-9ca1-27f2869c9268.png)

**실내 공간을 3D 형태**로 나타낸 데이터셋에는 **SceneNet(5M RGB-Depth synthetic indoor images), ScanNet(RGB-Depth dataset with 2.5M view obtained from more than 1500 scans)**이 있습니다. 

![image-20220316234636243](https://user-images.githubusercontent.com/70505378/158725304-611ff495-9908-4cfa-b974-f8bedc7acb2a.png)

마지막으로 **실외 공간을 3D 형태**로 나타낸 데이터셋에는 **KITTI, Semantic KITTI, Waymo Open Dataset** 등이 있습니다. 

![image-20220316234722827](https://user-images.githubusercontent.com/70505378/158725305-2e75a36f-5bfb-43b3-8339-022828b4448f.png)





<br>

## 3D tasks

2D image를 가지고 했던 task들을 3D data를 가지고도 할 수 있습니다. 

**3D recognition**

3D data에 Volumetric CNN 모델을 사용하여 Classification task를 수행할 수 있습니다. 

![image-20220316235459502](https://user-images.githubusercontent.com/70505378/158725306-d3711ee7-6686-4247-ad56-5501fe3fd746.png)



**3D object detection**

3D object detection도 가능합니다. 주로 자율 주행 분야에 유용하게 사용됩니다. 

![image-20220316235551680](https://user-images.githubusercontent.com/70505378/158725308-0a046148-c013-46f9-a802-dbed6a33b8d5.png)

**3D sementic segmentation**

마찬가지로 3D semantic segmentation도 가능합니다. 

![image-20220316235623696](https://user-images.githubusercontent.com/70505378/158725309-248ae597-49da-47ce-a1b4-a9fa419c6635.png)

**Conditional 3D generation**

3D 생성 모델도 만들 수 있습니다. 

먼저 소개할 모델은 **Mesh R-CNN**으로, 2D image를 input으로 넣어주면 3D Mesh를 output으로 출력하는 모델입니다. 

![image-20220316235820072](https://user-images.githubusercontent.com/70505378/158725310-fd2cf7a2-f976-4af9-8c5c-63d0fb2c7c62.png)

R-CNN 계열의 모델인 만큼, Mask R-CNN에 **3D branch**를 추가하는 것 만으로 간단히 구현할 수 있습니다. 

![image-20220316235959226](https://user-images.githubusercontent.com/70505378/158725311-0c3350b8-c3ba-467b-8457-31f44417d1f2.png)

또 다른 접근 방식을 사용하는 모델들도 있습니다. 

대표적으로 2D Image를 Multi-task head로 물리적으로 의미있는(physically meaningful) 여러 개의 특징(Surface normal, depth, silhouette, ...)으로 나누어서 구한 뒤에, 그 결과들을 취합하여 더욱 정교화된 결과를 구할 수도 있습니다. 

또는 Spherical Map이라는 것을 중간 단계에 구해서 이를 이용하는 모델도 있습니다. 공통적인 것은 중간 단계에서 physically meaningful한 feature들을 추출(sub-task)하여 이로부터 3D reconstruction을 수행(goal)하는 것입니다. 

![image-20220317000628196](https://user-images.githubusercontent.com/70505378/158725312-59481a37-3a6e-4eab-bb80-a0c8cceb760a.png)

**Photo refocusing**

마지막으로 살펴 볼 것은 photo refocusing입니다. Photo refocusing은 아래와 같이 우리가 설정한 거리 범위 내에 있는 물체에만 focusing하고 나머지 물체는 defocusing하는 것입니다. 

![image-20220317001853217](https://user-images.githubusercontent.com/70505378/158725314-ae681a8a-cba4-4dbe-9cae-f90274d7bfdc.png)

Photo refocusing은 depth map을 사용하여 수행 할 수 있습니다. 그 과정을 순서에 따라 보도록 하겠습니다. 

_1. Set a depth threshold range [D<sub>min</sub>, D<sub>max</sub>] you want to focus_

첫번째 단계에서는 focusing하고 싶은 거리 범위를 설정합니다. 우리가 사용할 depth map의 depth 값은 보통 0~255 사이의 정수값으로 nomalize하여 사용합니다. 

헷갈릴 수 있는데, **depth map의 값은 0에 가까울수록 어둡고(멀고) 255에 가까울수록 밝습니다(가깝습니다).**

![image-20220317002254787](https://user-images.githubusercontent.com/70505378/158725316-1bd578a5-fca4-4ce3-b0a3-24d978514b7a.png)

_2. Compute a mask of "focusing area" and "defocusing area" by depth map thresholding_

두번째 단계에서는 앞서 설정한 거리 범위를 이용해 depth map으로부터 "focusing area mask"와 "defocusing area mask"를 구합니다. 

![image-20220317002407983](https://user-images.githubusercontent.com/70505378/158725318-5198eba9-c6eb-49d5-84fe-c2cfe36c9327.png)

```python
# 여기서는 D_max 값만 사용
focus_mask = depth_map[..., :] > threshold_value
defocus_mask = depth_map[..., :] <= threshold_value
```



_3. Generate a blurred version of the input image_

세번째 단계에서는 4단계에서 사용할 image의 blurred version을 생성합니다. kernel size는 하이퍼파라미터입니다. 

![image-20220317002647654](https://user-images.githubusercontent.com/70505378/158725320-e4c482a2-3351-468d-a0ea-a7a49167160b.png)

```python
blurred_image = cv2.blur(original_image, (20,20))
```



_4. Compute "Masked focused image" and "Masked defocused image"_

Masked focus image는 focusing area mask에 original image를 곱해서 구합니다. 

Masked defocused image는 defocusing area mask에 blurred image를 곱해서 구합니다. 

![image-20220317002910313](https://user-images.githubusercontent.com/70505378/158725325-8e3d1047-a018-410d-bf76-bedffbfe2703.png)

```python
focused_with_mask = focus_mask * original_image
defocused_with_mask = defocus_mask * blurred_image
```



_5. Blend masked images to generate a refocused image_

마지막 5단계에서는 4단계에서 구한 두 이미지를 더해서 최종 refocused image를 생성합니다. 

![image-20220317003133249](https://user-images.githubusercontent.com/70505378/158725286-48c7ff6a-e38c-4f82-9b56-dffdd96a8359.png)

```python
defocused_image = focused_with_mask + defocused_with_mask
```









<br>

## 실습) Re-focusing Using Depth Map

앞에서 살펴 본 **photo refocusing**에 대한 실습을 진행합니다. 

**이미지 가져오기**

Photo refucusing을 위해서는 original image와 그에 대응하는 depth map이 있어야 합니다. 

![image-20220317153138793](https://user-images.githubusercontent.com/70505378/158752154-8b276193-5429-46f4-ad31-cfc75ce43592.png)

**Depth map histogram**

물체들의 거리 분포를 알고 싶을 때는 depth map을 histogram으로 표현하여 시각화하면 많은 도움이 됩니다. 

```python
plt.hist(depth_map.reshape(-1), bins=100)
plt.ylabel('Frequency')
plt.xlabel('Pixel intensity')
plt.show()
```

![image-20220317153246627](https://user-images.githubusercontent.com/70505378/158752160-2ce6a7f9-9373-4781-be0e-640605364508.png)

크게 [0, 50], [50, 150], [150, 255]의 세 구간으로 나눌 수 있습니다. 다시 한 번 말하지만, 값이 클수록 가까이에 있는 것입니다. 

**Focusing the object in the middle of the image**

이번 실습에서는 가운데 있는 물체에 focusing 해보겠습니다. 아래와 같은 코드로 [50, 150] 구간에 있는 물체를 나타내는 `object_mask`와 그 밖에 있는 물체를 나타내는 `background_mask`를 구합니다. 

```python
object_mask = np.where((50 < depth_map) & (depth_map < 150), 1, 0).astype("uint8")
background_mask = (np.invert(object_mask) + 2).astype("uint8")
# object_mask = (50 < depth_map) & (depth_map < 150)
# background_mask = np.invert(object_mask)
```

그리고 아래와 같은 코드로 `refocused_image`를 얻을 수 있습니다. 

```python
refocused_image = (original_image*object_mask) + (blurred_image*background_mask)
```

![image-20220317153831383](https://user-images.githubusercontent.com/70505378/158752162-0d851614-ed46-4528-ad1f-23b51ff8de01.png)





**Alpha-blending**

앞서 만든 focusing 이미지는 단순히 각 영역에 대해 원본 이미지의 값을 사용할 지 혹은 blur 처리된 값을 사용할 지 양자택일한 결과이기 때문에 focusing 정도를 조절할 수 없습니다. 

따라서 0~1 사이의 alpha 값을 이용하여 focusing 정도를 조절하고자 합니다 (1에 가까울수록 강하게 focusing).

가장 앞에 있는 동상에 focusing 정도를 조절하도록 간단하게 아래와 같은 코드로 구현할 수 있습니다. 

```python
for alpha in [0.2, 0.4, 0.6, 0.8, 1]:
  object_mask_w_alpha = alpha * (depth_map > 150).astype(np.float32)
  background_mask_w_alpha = (1-object_mask_w_alpha)

  defocused_image_w_alpha = (object_mask_w_alpha * original_image) + (background_mask_w_alpha * blurred_image)
  defocused_image_w_alpha = defocused_image_w_alpha.astype(np.uint8)

  show_image(defocused_image_w_alpha, 'Refocused Image with Alpha Blending (alpha=%s)' % alpha)
```

![image-20220317154330808](https://user-images.githubusercontent.com/70505378/158752165-f4ce366c-bb63-4439-b6e2-76aab001ea55.png)

![image-20220317154341444](https://user-images.githubusercontent.com/70505378/158752168-ee52f751-3aa4-46d5-8596-767c79032c7c.png)

![image-20220317154353244](https://user-images.githubusercontent.com/70505378/158752170-642f2366-76ca-4456-87be-252ff4d60c65.png)

















<br>

<br>

# 참고 자료


* 3D understanding

  * Chang et al., ShapeNet: An Information-Rich 3D Model Repository, ArXiv 2015 
  * Mo et al., PartNet: A Large-Scale Benchmark for Fine-Grained and Hierarchical Part-Level 3D Object Understanding, CVPR 2019
  * McCormac et al., SceneNet RGB-D: Can 5M Synthetic Images Beat Generic ImageNet Pre-Training on Indoor Segmentation?, ICCV 2017 
  * Dai et al., ScanNet: Richly-Annotated 3D Reconstructions of Indoor Scenes, CVPR 2017  
* 3D tasks

  * Gkioxari et al., Mesh R-CNN, ICCV 2019 
  * Wu et al., MarrNet: 3D Shape Reconstruction via 2.5D Sketches, NeurIPS 2017 
  * Zhang et al., Learning to Reconstruct Shapes from Unseen Classes, NeurIPS 2018  




<br>

