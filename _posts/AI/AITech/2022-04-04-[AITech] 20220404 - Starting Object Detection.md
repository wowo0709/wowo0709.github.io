---
layout: single
title: "[AITech][Object Detection][P stage] 20220404 - Starting Object Detection"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['COCO', 'EDA', 'Library', 'Loss']
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
  * area, bbox([xmin, ymin, xmax, ymax]) 정보를 가집니다. (Segmentation의 경우 segmentation(object outline) 정보를, pose estimation의 경우 keypoints 정보를 가집니다)
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

`COCO` 클래스를 사용하여 데이터들을 불러와서 데이터프레임 형태로 만들었습니다. 

```python
GT_JSON = '/opt/ml/detection/dataset/train.json'
image_path = '/opt/ml/detection/dataset'
coco = COCO(GT_JSON)

data = pd.DataFrame(columns = ['image_id','id','category_id','area','width','height'])
for i in range(23144):
    image_id = coco.loadAnns(i)[0]['image_id']
    id = coco.loadAnns(i)[0]['id']
    category_id = coco.loadAnns(i)[0]['category_id']
    area = coco.loadAnns(i)[0]['area']
    width = coco.loadAnns(i)[0]['bbox'][2]
    height = coco.loadAnns(i)[0]['bbox'][3]
    
    data.loc[i] = [image_id, id, category_id, area, width, height]
    
data.head()
```

![image-20220407131531884](https://user-images.githubusercontent.com/70505378/162124791-ce38682f-ffeb-44dc-a3f0-09c336290e64.png)







<br>

### Object per Image

각 이미지가 몇 개의 객체를 포함하는 지 시각화합니다. 

```python
from collections import Counter

fig, ax = plt.subplots(1, 1, figsize=(30, 6))
# ax.hist(data['image_id'].value_counts(),bins=72,edgecolor='black',linewidth=1.5,color='royalblue')
xs = sorted(data['image_id'].value_counts().unique())
ys = sorted(Counter(data['image_id'].value_counts().values).most_common())
ys = [y[1] for y in ys]

ax.bar(xs, ys, 
        width=0.7, 
        edgecolor='black', 
        linewidth=2, 
        color='royalblue',
        zorder=10
        )

for s in ['top', 'right']:
    ax.spines[s].set_visible(False)

total = sum(ys)
for idx, value in zip(xs, ys):
    ax.text(idx, value+10, s=value,
            ha='center', 
            fontweight='bold',
            fontsize=10
            )

plt.title('bbox per image')
plt.show()
print('total:', total) # 4883
```

![image-20220407131634301](https://user-images.githubusercontent.com/70505378/162124792-f4e1eaf6-8a02-4e90-8040-1b721171266b.png)

대부분이 10개 이하인데, 70개가 넘는 객체를 포함하는 이미지도 있네요...ㄷㄷ





<br>

### Object per Category

이번 대회에서는 총 10개의 클래스가 주어졌습니다. 각 클래스에 해당하는 객체가 몇 개나 있는지 확인합니다. 

```python
fig, ax = plt.subplots(1, 1, figsize=(10, 6))
xs = data['category_id'].value_counts().index
ys = data['category_id'].value_counts().values

ax.bar(x=xs, height=ys,edgecolor='black',linewidth=1.5,color='royalblue')
ax.set_xticks([x for x in range(10)])
ax.set_xticklabels([category[x]['name'] for x in range(10)],rotation=-45)
for s in ['top', 'right', 'left']:
    ax.spines[s].set_visible(False)
ax.yaxis.set_visible(False)

total = sum(ys)
for idx, value in zip(xs, ys):
    ax.text(idx, value+10, s=value,
            ha='center', 
            fontweight='bold',
            fontsize=10
            )

plt.show()
print('total:', total) # 23144
```

![image-20220407131815628](https://user-images.githubusercontent.com/70505378/162124794-b97526ab-5bc3-4129-9db1-ee0c1df8c77e.png)

클래스 불균형이 꽤 있네요! Augmentation이나 Focal loss 등을 통해 조절해주어야 할 것 같습니다. 

<br>

팀원분께서 수행하신 EDA인데, 대체로 하나의 이미지 내에서 여러 개 클래스들이 복합적으로 존재하기 때문에 같이 등장하는 횟수를 시각화하는 시도도 해 볼 수 있습니다. 

```python
li = [[0 for _ in range(10)] for _ in range(10)]
for i in range(4883):
    l = sorted(data[data['image_id']==i]['category_id'].values)
    for x in range(len(l)-1):
        for y in range(x+1,len(l)):
            li[int(l[y])][int(l[x])] += 1
            
# 절대적 개수 시각화.
fig, ax = plt.subplots(1,1 ,figsize=(10, 9))
mask = np.zeros_like(li)
mask[np.triu_indices_from(mask,k=1)] = True
sns.heatmap(li,annot=True,fmt='.0f',mask=mask,xticklabels=category_name,yticklabels=category_name,linewidth=0.1)
plt.show() 
```

![image-20220407132139719](https://user-images.githubusercontent.com/70505378/162124795-95f9af3b-f067-4e50-97b1-eccc8e1cdf63.png)



<br>

### Bbox size/ratio

Object detection에서 bbox의 size와 ratio는 매우 중요합니다. 

먼저 bbox의 size의 분포를 시각화해보도록 하겠습니다. 

```python
fig, ax = plt.subplots(1,1, figsize=(12, 5))
sns.kdeplot(x='area', data=data, ax=ax,color='royalblue',fill=True)
ax.set_title('bbox area')
plt.show()
```

![image-20220407132304200](https://user-images.githubusercontent.com/70505378/162124776-9229ff9d-0675-45c1-a65c-f96ec5988db3.png)

bbox ratio도 시각화해봅시다. 

```python
fig, axes = plt.subplots(2,5, figsize=(20, 8),sharey=True,sharex=True)
for i in range(2):
    for j in range(5):
        sns.scatterplot(x='width',y='height',data=data[data['category_id']==5*i+j],ax=axes[i][j],color='royalblue')
        x1,x2 = [0, 1024], [0, 1024*0.5]
        y1,y2 = [0, 1024*0.5], [0, 1024]
        
        axes[i][j].plot(x1, y1,color = 'tomato')
        axes[i][j].plot(x2, y2,color = 'tomato')
        axes[i][j].set_title(category[5*i+j]['name'])
plt.show()
```

![image-20220407132354224](https://user-images.githubusercontent.com/70505378/162124778-072ab400-9655-401b-b5ae-8ce267a83349.png)





<br>

### Visualization

실제로 이미지 내에서의 Bbox들을 시각화하는 것도 좋을 것 같습니다. 아래와 같은 util 함수를 정의합니다. 

```python
def showbox(id,size=6,ann=True):
    image = np.array(Image.open(os.path.join(image_path, coco.loadImgs(id)[0]['file_name'])))
    fig, ax = plt.subplots(figsize=(size, size))
    ax.imshow(image)
    plt.axis('off')
    box = [x['bbox'] for x in coco.loadAnns(coco.getAnnIds(id))]
    cat = [coco.loadCats(x['category_id'])[0]['name'] for x in coco.loadAnns(coco.getAnnIds(id))]
    for (x, y, w, h), c in zip(box,cat):
        ax.add_patch(patches.Rectangle((x, y), w, h,edgecolor='red',linewidth=1.5,fill=False))
        if ann:
            ax.text(x,y-5,c,fontsize=12,color='red')
```

 아래와 같이 시각화 할 수 있습니다. 

```python
showbox(1160,size=4)
```

![image-20220407132725897](https://user-images.githubusercontent.com/70505378/162124779-508caa44-e998-4eaa-a33f-6fe445640a04.png)

```python
showbox(160)
```

![image-20220407132749799](https://user-images.githubusercontent.com/70505378/162124783-bbe5c17f-6579-431f-b93b-324206acd0bc.png)







<br>

<br>

## Object detection library

![image-20220407133739269](https://user-images.githubusercontent.com/70505378/162124789-44088a62-f73f-49c6-9e71-bbf807819d8a.png)

Object detection library는 Image classification library와 다르게, 통합된 라이브러리가 없습니다. 그래서 분산되어 존재하는 라이브러리들의 모델을 사용해야 합니다. 

실무/캐글에서는 주로 `MMDetection` 라이브러리와 `Detectron2` 라이브러리를 주로 활용합니다. 이 중 이번 대회에서는 MMDetection 라이브러리를 사용했고, MMDetection에 대한 자세한 내용들은 다음 포스팅에서 다루도록 하겠습니다. 

여기서는 간단하게 두 라이브러리에 대한 소개 만을 하도록 하겠습니다. 

![image-20220407133308866](https://user-images.githubusercontent.com/70505378/162124787-bafe9545-cbc1-4224-98cd-d0f856c9f1ac.png)













<br>

<br>

## Loss

Object detection task에는 classification loss와 bbox regression loss가 존재합니다. 이 중 bbox regression loss에서 사용하는 개념으로 `IoU`가 있습니다. 

사용할 수 있는 loss 척도로 아래와 같은 것들이 있습니다. 이에 대해 잘 정리해 놓은 블로그가 있어 주소를 남깁니다. 

* MSE (Mean squared error)
  * 박스의 좌표 간 거리 차이
  * 박스의 overlap 정도를 반영하지 못 함
* IOU (Intersection over Union)
  * 박스가 overlap되는 정도 차이
  * 동일한 IOU를 가지는 수 많은 박스 존재
  * 박스가 아예 overlap되지 않을 경우 gradient vanishing 문제 발생
* GIOU (Generalized-IOU)
  * IOU + (두 박스를 포함하는 최소 크기의 박스 C 활용)
  * 박스가 overlap되지 않을 경우의 gradient vanishing 해결
  * 수렴 시간이 오래 걸리고, horizontal-vertical 정보를 포함하지 못 함
* DIOU (Distance-IOU)
  * GIOU + (두 박스의 중심좌표 간 거리 차이)
  * 수렴 속도가 빠르고, 박스의 형태에 비슷하게 예측
* CIOU (Complete-IOU)
  * DIOU + (두 박스의 비율(aspect ratio) 차이)
  * Bbox loss에 overlap area, central point distance, aspect ratio를 모두 고려

참조: [IOU 개념 정리](https://silhyeonha-git.tistory.com/3)

















<br>

<br>

# 참고 자료

* Coco dataset, https://medium.com/mlearning-ai/coco-dataset-what-is-it-and-how-can-we-use-it-e34a5b0c6ecd 
* Coco dataset, https://ukayzm.github.io/cocodataset/ 
* Coco data format&Pycocotools, https://comlini8-8.tistory.com/67 
* github - cocoapi, https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py 
* IOU loss, https://silhyeonha-git.tistory.com/3 
