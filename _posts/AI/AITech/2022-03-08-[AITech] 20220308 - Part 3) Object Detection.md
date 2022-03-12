---
layout: single
title: "[AITech][CV] 20220308 - Part 3) Object Detection"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['Single-stage detector', 'Single-stage detector', 'Focal loss', 'RetinaNet', 'DETR']
---



<br>

_**본 포스팅은 POSTECH '오태현' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# Object Detection

이번 포스팅에서는 한층 더 복잡하고 발전된 CV task인 object detection에 대해 알아보겠습니다. 

## Object detection

Object detection은 Classification에 더하여 **Localization**이라는 또 하나의 업무를 수행하는 task입니다. 

![image-20220312151018095](https://user-images.githubusercontent.com/70505378/158012834-03136670-1ed8-4761-82d8-afac87b71867.png)

Object detection 또한 자율 주행, OCR(Optical Character Recognition) 기술 등에 활발히 적용되고 있는 기술입니다. 





<br>

## Two-stage detector

### R-CNN family

Two-stage detector는 Localization과 Classification을 두 단계로 나누어 차례로 수행하는 모델을 말합니다. 

대표적으로 R-CNN, Fast R-CNN, Faster R-CNN의 R-CNN family 계열의 모델들이 있고, 특징은 속도가 느리지만 더 정확하다는 것입니다. 

![image-20220208175442234](https://user-images.githubusercontent.com/70505378/152952758-0c06a06e-257c-4583-8b4b-9762c304bdbd.png)

R-CNN 모델들의 아이디어는 쉽지 않기 때문에, 제가 정리해놓은 논문 리뷰 포스팅을 보는 것을 추천드립니다. 

* [RCNN 시리즈 논문 분석](https://wowo0709.github.io/ai/computervision/Computer-Vision-RCNN-%EC%8B%9C%EB%A6%AC%EC%A6%88-%EB%85%BC%EB%AC%B8-%EB%B6%84%EC%84%9D/)
  * [AI - Computer Vision - RCNN 시리즈 논문 분석]에 있습니다. 







<br>

## Single-Stage detector

다음으로 Single-Stage detector입니다. Single-Stage detector는 말 그대로 localizaiton과 classification을 동시에 수행하는 모델이고, 따라서 정확도는 비교적 조금 낮지만 매우 빠른 속도를 자랑합니다. 

대표적인 Single-Stage detector 모델로는 YOLO와 SSD가 있습니다. 

### YOLO

YOLO(You Only Look once)는 대표적인 single-stage detector 모델로, 기본 45fps ~ 최대 155fps의 추론 속도를 낼 수 있습니다. 

![image-20220312152040012](https://user-images.githubusercontent.com/70505378/158012836-b6338859-c433-4201-8bea-3ecdfb4a9bee.png)

![image-20220208173303961](https://user-images.githubusercontent.com/70505378/152952768-d83e07c8-4297-4083-b81f-03ceb4271bc1.png)



아래 그림은 YOLO가 detection을 수행하는 과정을 간단히 흐름에 따라 정리한 것입니다. 

![image-20220208174633482](https://user-images.githubusercontent.com/70505378/152952771-d80f5979-3353-4aaf-a086-3d17553474c2.png)

아래는 YOLO와 R-CNN의 성능을 비교한 표입니다. 

![image-20220312152131793](https://user-images.githubusercontent.com/70505378/158012837-90ff9f71-b070-4248-8aa0-3c940f17134c.png)

mAP와 FPS 간의 trade-off 관계를 확인할 수 있습니다. 





### SSD

SSD(Single Shot Detector)는 YOLO 이후에 나온 single-stage detector로, 정확도와 속도 면에서 모두 우수한 성능을 보여줍니다. 

YOLO의 한계는 맨 마지막 단계에서만 prediction을 수행하기 때문에 작은 크기의 object들에 대한 정확도가 비교적 떨어진다는 것이었습니다. 

SSD의 아이디어는 **중간 과정에서 생성되는 feature map의 해상도에 따라 다른 box들로 prediction을 수행하여 그 결과들을 모두 이용**하는 것입니다. 이렇게 multiple feature map으로부터 multi-scale output을 이용함으로써 다양한 크기의 object들을 잘 탐지할 수 있게 되었습니다. 

![image-20220312153449356](https://user-images.githubusercontent.com/70505378/158012838-4f76b8e9-270b-43ea-acc1-c27e0701d2fc.png)

아래는 Faster R-CNN, SSD, YOLO의 정확도와 속도를 비교한 표인데, SSD가 정확도와 속도 면에서 다른 두 모델보다 더 나은 성능을 보이는 것을 알 수 있습니다. 

![image-20220312153603019](https://user-images.githubusercontent.com/70505378/158012839-371950d2-ccd6-42f6-86e8-b305713b0f32.png)



<br>

## Single-stage detector vs two-stage detector

### Focal loss

Single-Stage detector가 겪게 되는 필연적인 문제로 **Class imbalance problem**이 있습니다. 많은 경우에 이미지 내에서 우리가 찾고자 하는 물체는 아주 일부에 해당하고, 나머지는 배경에 해당합니다. 이러한 배경은 모델 학습에 도움을 주지 못 함에도 불구하고 계속해서 loss를 발생시켜 모델의 올바른 학습을 방해합니다. 

![image-20220312181809720](https://user-images.githubusercontent.com/70505378/158012840-99025937-ee1a-46a0-beac-3b4f8a95d89c.png)

이럴 때 사용할 수 있는 손실 함수로 **Focal loss**라는 것이 있습니다. Focal loss는 class imbalance problem을 완화하기 위한 식으로 구성되어 있고, 맞힐 확률을 높은 클래스에 대해서는 낮은 loss 값을, 맞힐 확률이 낮은 클래스에 대해서는 높은 loss 값을 발생시킵니다. 

![image-20220312182006161](https://user-images.githubusercontent.com/70505378/158012842-336938ed-13d0-4e56-86a4-a3c8b5207456.png)

### RetinaNet

`RetinaNet`은 SSD보다도 더 발전된 형태의 single-stage detector입니다. 

여러 해상도의 중간 activation map의 정보를 활용한다는 점에서 공통적인 모습을 보이며, RetinaNet은 이를 pyramid 구조로 구성하였고 각각의 activation map을 차례로 더해가며 class subnet과 box subnet을 거쳐 중간 결과들을 뽑아냅니다. 

![image-20220312182048676](https://user-images.githubusercontent.com/70505378/158012844-509ee4c6-1752-4eb3-9141-7799b32c1c4b.png)

아래 표를 보면 정확도와 속도 면에서 모두 한층 더 발전된 모습을 보여준다는 것을 확인할 수 있습니다. 

![image-20220312182320391](https://user-images.githubusercontent.com/70505378/158012846-7fabccf1-a901-407f-a6aa-37cef0d70f9d.png)



<br>

## Detection with Transformer

### DETR

자연어 처리 분야에서 혁명적인 반향을 불러일으킨 모델로 Attention 구조를 사용한 Transformer 모델이 있죠. 이후 Transformer 모델이 어떻게 하면 CV 분야의 여러 task에도 적용이 될 수 있을 지에 대한 연구가 많이 진행되고 있습니다. 

* [Transformer 포스팅 보러가기](https://wowo0709.github.io/ai/aitech/AITech-20220209-Attention&Transformer/)

Transformer를 Image classification에 적용한 모델로 ViT(Vision Transformer), DEiT(Data-Efficient image Transformer) 등의 모델이 있습니다. 

* [ViT 포스팅 보러가기](https://wowo0709.github.io/ai/aitech/AITech-20220210-Vit-Visual-Transformer-%EC%8B%A4%EC%8A%B5/)

그 중 Detection task를 위한 Transformer 모델로 `DETR(DEtection TRansformer)` 모델이 있습니다. 

구조는 아래와 같습니다. 

![image-20220312183205580](https://user-images.githubusercontent.com/70505378/158012847-39568667-31f7-4e32-8927-fc655236e8f7.png)



<br>

## Further reading

이외에도 bounding box를 찾을 때 (x, y, h, w)가 아닌 중심점과 크기를 이용해 찾는다든가, 박스를 조정해야 할 방향을 이용해 찾는 등의 다양한 연구들이 진행되고 있습니다. 

이에 대한 자세한 내용은 이후 포스팅에서 다뤄질 예정입니다. 

![image-20220312183753262](https://user-images.githubusercontent.com/70505378/158012849-211d5fb5-767f-4271-a8d9-0fee1bf22abb.png)











<br>

<br>

# 참고 자료


* 



<br>

