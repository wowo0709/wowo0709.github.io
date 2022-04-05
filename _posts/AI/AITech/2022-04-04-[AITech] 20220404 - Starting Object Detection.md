---
layout: single
title: "[AITech][Object Detection][P stage] 20220404 - Starting Object Detection"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['COCO', 'EDA', 'Library', 'Metric']
---



<br>

# Starting Object Detection

## COCO dataset

다들 `COCO 데이터셋`이라는 말은 많이 들어보셨으리라 생각합니다. 저도 COCO dataset은 object detection에서 모델의 성능을 평가할 때 사용하는 벤치마킹 데이터셋이고, 우리가 모델을 학습시킬 때도 대부분의 모델들이 COCO dataset format을 요구한다고는 알고 있었지만, 실제로 어떤 구조로 되어 있는지, 그리고 어떻게 만들 수 있는지에 대한 지식은 없었습니다. 

여기서는 실제 데이터셋 구조와 어떻게 만들 수 있는지, 그리고 COCO dataset을 쉽게 처리할 수 있게 해주는 pycocotools에 대해 알아보려고 합니다. 

### Structure of COCO

먼저 COCO의 구조를 살펴보겠습니다. COCO는 json 포맷으로 되어 있습니다. 

* `info`: 버전, 만들어진 시간, 저자 등 데이터셋 자체에 대한 정보들이 있습니다. 
* `licenses`: 데이터셋의 저작권에 대한 정보들이 있습니다. 
* `category`: 클래스 정보가 들어있습니다. 
  * 10개의 클래스를 가질 경우 각 클래스는 0~9의 id에 대응합니다. 
  * 각 상위 카테고리 안에 하위 카테고리 클래스를 넣을 수도 있습니다. 
* `images`: 이미지 정보가 들어있습니다. 
  * Width, height, file name, date captured 같은 정보들이 들어있고, 각 이미지는 category와 마찬가지로 0부터 id를 부여받습니다. 
* `annotations`: 각 이미지가 가지고 있는 객체(bbox/segmentation/key points)에 대한 정보들이 들어있습니다. 
  * 위에서 부여한 category id와 image id를 사용합니다. 
  * area, bbox([xmin, ymin, xmax, ymax]) 정보를 가집니다. (Segmentation의 경우 segmentation(object outline) 정보를, pose estimation의 경우 key points 정보를 가집니다)
  * iscrowd는 0 또는 1 값을 가지는 binary parameter로, 1이면 여러 객체들을 따로 구분할 수 없어서 묶어서 bbox/segmentation을 특정했음을 말합니다. 
  * 각 annotation 또한 id를 가집니다. 

```json
{
    "info": {
        "year": "2021",
        "version": "1.0",
        "description": "Exported from FiftyOne",
        "contributor": "Voxel51",
        "url": "https://fiftyone.ai",
        "date_created": "2021-01-19T09:48:27"
    },
    "licenses": [
        {
          "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/",
          "id": 1,
          "name": "Attribution-NonCommercial-ShareAlike License"
        },
        ...   
    ],
    "categories": [
        ...
        {
            "id": 2,
            "name": "cat",
            "supercategory": "animal"
        },
        ...
    ],
    "images": [
        {
            "id": 0,
            "license": 1,
            "file_name": "<filename0>.<ext>",
            "height": 480,
            "width": 640,
            "date_captured": null
        },
        ...
    ],
    "annotations": [
        {
            "id": 0,
            "image_id": 0,
            "category_id": 2,
            "bbox": [260, 177, 231, 199],
            "segmentation": [...],
            "area": 45969,
            "iscrowd": 0
        },
        ...
    ]
}
```

<br>

### Pycocotools

위 json 파일에서 우리가 주로 사용하게 되는 데이터는 **Images**와 **Annotations**입니다. 

이 두 정보를 잘 가공해야 하는데, `pycocotools` 라이브러리가 이를 도와줍니다. Pycocotools는 여러 클래스를 제공하는데, 그 중 COCO 클래스를 사용하겠습니다. 

**COCO 객체 생성**

```python
from pycocotools.coco import COCO
coco = COCO('파일경로')
```

**getAnnIds**

image id 또는 category id를 인자로 전달하면 그에 해당하는 annotation id를 반환해줍니다. 

* 0번 image에 있는 annotation을 출력합니다. 

```python
coco.getAnnIds(imgIds=0)
# [0, 1]
```

* 9번 카테고리에 해당하는 annotation을 출력합니다. 

```python
coco.getAnnIds(catIds=9)
# [245, 364, ..., 2998]
```

**getCatIds**

Category 이름을 인자로 전달하면 그에 해당하는 category id를 반환합니다. 

```python
coco.getCatIds(catNms='Battery') # supNms, catIds
# 9
```

인자로 아무것도 넘기지 않으면 전체 category id를 반환합니다. 

**getImgIds**

Image id 또는 Category id를 인자로 전달하면 그에 해당하는 Image id를 반환합니다. 

```python
coco.getImgIds(catIds=9)
# [711, ..., 826]
```

**loadAnns**

Annotation id를 인자로 전달하면 그에 해당하는 annotation dict 전체를 반환합니다. 

```python
coco.loadAnns(3)
'''
[{'image_id': 1,
  'category_id': 4,
  'area': 69096.17,
  'bbox': [722.3, 313.4, 274.3, 251.9],
  'iscrowd': 0,
  'id': 3}]
'''
```

**loadCats**

Category id를 인자로 전달하면 그에 해당하는 category dict 전체를 반환합니다. 

```python
coco.loadCats(0)
# [{'id': 0, 'name': 'General trash', 'supercategory': 'General trash'}]
```

**loadImgs**

Image id를 인자로 전달하면 그에 해당하는 image dict 전체를 반환합니다. 

```python
coco.loadImgs(2)
'''
[{'width': 1024,
  'height': 1024,
  'file_name': 'train/0002.jpg',
  'license': 0,
  'flickr_url': None,
  'coco_url': None,
  'date_captured': '2020-12-27 17:55:52',
  'id': 2}]
'''
```

**showAnns**

Annotation dict가 segmentation, keypoints, caption 키를 가지고 있다면 사용할 수 있습니다. 인자로 전달한 annotation들을 시각화해줍니다. 

<br>

여기까지 COCO structure와 pycocotools에 대해 살펴보았습니다! 이 두가지를 잘 알고 있어야 object detection을 쉽게 시작할 수 있는 것 같습니다. 

그럼 다음 섹션에서는 이 pycocotools를 이용해 EDA를 어떤 식으로 할 수 있는지 보도록 하겠습니다. 

<br>

<br>

## EDA

모든 task에 앞서 충분한 EDA는 필수적입니다. 

Object detection에서도 여러 EDA 방법들이 있겠지만, 그러한 방법들은 모두 적기에는 너무 많으니 kaggle의 discussion과 같은 곳에서 참고하는 것으로 하고 여기서는 대회 진행 중에 수행한 EDA들에 대해 보겠습니다. 

### Data Preparation









<br>

### Object per Image









<br>

### Object per Category









<br>

### Bbox size/ratio









<br>

### Visualization













<br>

<br>

## Object detection library















<br>

<br>

## Metric





















<br>

<br>

# 참고 자료

* 
