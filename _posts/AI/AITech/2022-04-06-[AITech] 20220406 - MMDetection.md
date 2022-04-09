---
layout: single
title: "[AITech][Object Detection][P stage] 20220406 - MMDetection"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

# MMDetection

`MMDetection`은 object detection task에서 사용할 수 있는 detection model library입니다. 

처음 `MMDetection`에 대한 설명을 들으면 **config 파일만 수정하면 된다**라는 말을 듣습니다. MMDetection의 구조를 파헤쳐보면서 그것이 무엇을 의미하는지 보도록 하겠습니다. 

![image-20220409183828261](https://user-images.githubusercontent.com/70505378/162576635-3e02be22-00c2-4284-8966-f06c42a98172.png)

## MMDetection Structure

[mmdetection 보러 가기](https://github.com/open-mmlab/mmdetection)

mmdetection의 구조는 아래와 같습니다. 

![image-20220409183717522](https://user-images.githubusercontent.com/70505378/162576634-33071071-e6df-4d9d-a1be-df755f45a55a.png)

위 폴더들 중 우리가 많이 사용하게 되는 폴더는 `configs`, `mmdet`, `tools` 폴더입니다. 그 중에서도 단연 많이 사용하는 것은 **configs** 폴더입니다. 

mmdetection 라이브러리를 설치해서 import 하여 사용하거나, repository의 폴더들을 로컬로 가져와서 사용할 수 있습니다. 여러 커스터마이징을 위해서는 clone하여 사용하거나, zip file로 다운로드 받아서 전체 폴더를 가져오는 것이 좋다고 생각합니다. 

저희 조에서는 `level2-object-detection-level2-cv-10`이라는 repository를 파서 그 안에 mmdetection 폴더를 포함시키는 식으로 작업했습니다. 폴더 구조를 보려면 아래 주소를 참조해주세요. 

[level2-object-detection-level2-cv-10](https://github.com/boostcampaitech3/level2-object-detection-level2-cv-10/tree/experiment)

추가적으로, mmdetection repository를 clone 했다면 `.git` 폴더로 로컬과 리모트가 연결되어 있는 상태이기 때문에 이 파일들을 지워서 연결을 끊어줍니다. 







<br>

<br>

## config folder

먼저 우리가 가장 많이 사용하게 되는 `config` 폴더를 보겠습니다. 

config 폴더 안에는 여러 모델의 구조가 미리 폴더별, 파일별로 정의되어 있습니다. 

![image-20220409184745371](https://user-images.githubusercontent.com/70505378/162576636-333a2361-8625-4160-ba6c-3431f1b4eb21.png)

저희 조는 config 폴더 안에 각자의 실험을 위한 personal folder를 하나씩 만들었습니다. `_base_`를 제외하고 `_youngwoo_`와 같이 앞뒤로 '_'가 붙어있는 폴더들이 각자의 실험용 폴더입니다. 이에 대한 이야기는 뒤에서 추가적으로 하도록 하겠습니다. 

기본적으로 2-stage model의 구조는 아래 그림을 따릅니다. 2-stage model의 config 파일 안에는 크게 model, backbone, neck, rpn_head, roi_head 가 정의되어 있습니다. 

![image-20220409185709952](https://user-images.githubusercontent.com/70505378/162576640-057f00c8-3438-46a8-8601-d6cec45f972e.png)

그럼 모델 폴더 안에는 어떤 파일들이 있는지 보도록 하겠습니다. 아래는 casecade_rcnn 폴더 내부의 모습입니다. 

![image-20220409185115276](https://user-images.githubusercontent.com/70505378/162576638-e94027f4-9a22-4d92-b2d0-1a64275bfb28.png)

위 파일들 중 가장 기본이 되는 파일은 `cascade_rcnn_r50_fpn_1x_coco.py` 파일입니다. 파일명은 일반적으로 아래와 같이 구성됩니다. 

```
{model}_[model setting]_{backbone}_{neck}_[norm setting]_[misc]_[gpu x batch_per_gpu]_{schedule}_{dataset}
```

* `cascade_rcnn`: 전체 모델 명
* `r50`: backbone 모델 명. r50은 resnet 50을 가리킴. 
* `fpn`: Neck 모델 명. 
* `1x`: 학습 epoch 수
* `coco`: dataset 포맷. 

여기에 구조의 변경이 들어가거나, 1-stage 모델과 같이 구조가 다른 모델일 경우 파일명은 그에 따라 달라집니다. 

`cascade_rcnn_r50_fpn_1x_coco.py` 파일은 아래왁 같이 작성되어 있습니다. 

```python
_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
```

별다른 코드 없이, `_base_` 라는 변수에 사용할 파일들만 정의되어 있습니다. 보통 `_base_` 폴더에는 크게 `models`, `datasets`, `schedules`, `runtime` 파일들이 정의되어 있습니다. 이 `_base_` 폴더 안에 기본 모델이 정의되어 있다면 `_base_` 폴더의 파일을 가져와서 사용하고, `_base_` 폴더 안에 기본 모델이 정의되어 있지 않은 경우 모델 파일에서 `_base_` 폴더의 파일을 가져오지 않고 코드가 직접 작성되어 있을 수 있습니다. 

예를 들어 **cornernet** 모델의 경우 `_base_` 폴더에서는 runtime과 dataset 파일만 가져오고, model과 scheduler(optimizer)는 직접 정의되어 있습니다. 

```python
_base_ = [
    '../_base_/default_runtime.py', '../_base_/datasets/coco_detection.py'
]

# model settings
model = dict(
    type='CornerNet',
    backbone=dict(
        type='HourglassNet',
        downsample_times=5,
        num_stacks=2,
        stage_channels=[256, 256, 384, 384, 384, 512],
        stage_blocks=[2, 2, 2, 2, 2, 4],
        norm_cfg=dict(type='BN', requires_grad=True)),
    neck=None,
    bbox_head=dict(
        type='CornerHead',
        num_classes=80,
        in_channels=256,
        num_feat_levels=2,
        corner_emb_channels=1,
        loss_heatmap=dict(
            type='GaussianFocalLoss', alpha=2.0, gamma=4.0, loss_weight=1),
        loss_embedding=dict(
            type='AssociativeEmbeddingLoss',
            pull_weight=0.10,
            push_weight=0.10),
        loss_offset=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1)),
    # training and testing settings
    train_cfg=None,
    test_cfg=dict(
        corner_topk=100,
        local_maximum_kernel=3,
        distance_threshold=0.5,
        score_thr=0.05,
        max_per_img=100,
        nms=dict(type='soft_nms', iou_threshold=0.5, method='gaussian')))
# ...
optimizer = dict(type='Adam', lr=0.0005)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    step=[180])
runner = dict(type='EpochBasedRunner', max_epochs=210)
```





이처럼, 기본적인 뼈대 파일을 `_base_` 변수에 담아두고(없다면 생략하고), 여러가지 구조를 변경해서 새로운 모델을 만드는 것이 위에서 얘기한 **config 파일만 수정하면 된다**의 의미입니다. 

이 `_base_` 폴더의 구조가 어떻게 생겼는지는 아래에서 보도록 하겠습니다. 

<br>

예를 들어 기본이 되는 파일인 `cascade_rcnn_r50_fpn_1x_coco.py` 파일의 backbone을 `resnet 101`로 수정한 `cascade_rcnn_r101_fpn_1x_coco.py` 파일은 아래와 같이 작성되어 있습니다. 

```python
_base_ = './cascade_rcnn_r50_fpn_1x_coco.py'
model = dict(
    backbone=dict(
        depth=101,
        init_cfg=dict(type='Pretrained',
                      checkpoint='torchvision://resnet101')))
```

다른 모든 설정들은 cascade_rcnn_r50_fpn_1x_coco.py 에서 가지고 오되, model의 backbone 부분을 새롭게 정의하여 오바라이딩 할 수 있습니다. 





### 개인 config 폴더 구성하기

저희 조는 각자 실험을 진행한 config 파일을 기록/저장하기 위해 개인 config 폴더(`_youngwoo_`)를 만들었습니다. 개인 config 폴더 내부는 아래와 같이 구성되어 있습니다. (작성자의 파일이 중간에 날아가버려서... 아래 이미지는 다른 팀원 분의 폴더 구성입니다)

![image-20220409211631144](https://user-images.githubusercontent.com/70505378/162576641-786d02c8-58df-420b-ab9f-73aabc0c34e9.png)

상위 config 폴더와 마찬가지로, 기본적인 뼈대 코드가 들어있는 `_base_` 폴더와 커스텀한 각 모델들이 들어있는 모델 폴더(여기서는 `dyhead`, `retinanet`, `universenet`)가 있습니다. 각 폴더 내부 구조는 상위 config 폴더에 있는 폴더들의 내부 구조와 동일합니다. 

아까부터 계속 `_base_`  폴더 이야기를 했는데, 이제 이 폴더 내부 구조는 어떻게 되어 있는지 살펴보겠습니다. 

<br>

<br>

## \_base\_ folder

위에서 말했듯이,  `_base_` 폴더에는 기본이 되는 `models`, `datasets`, `schedules`, `runtime` 파일들이 정의되어 있습니다.

![image-20220409211940930](https://user-images.githubusercontent.com/70505378/162576643-ab3e5101-3801-418c-b228-d42a522b2e57.png)

### datasets

![image-20220409212021564](https://user-images.githubusercontent.com/70505378/162576626-3ba8534e-4c2d-40d0-970c-8295f9e70d7a.png)

task, dataset format에 따라 다른 dataset 파일들이 존재합니다. 이번 대회에서 사용했던 `coco_detection.py` 파일 내부를 보겠습니다. 

```python
# dataset settings
dataset_type = 'CocoDataset'
data_root = 'data/coco/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))
evaluation = dict(interval=1, metric='bbox')
```

coco dataset format의 detection dataset을 사용할 때 사용하는 파일입니다. 코드는 상당히 직관적으로 짜여져 있습니다. 

우리의 데이터셋을 사용하기 위해서는, 위 코드에 몇 가지 수정이 필요합니다. 이렇게 수정된 `coco_detection.py` 파일은 위에서 본 개인 config 폴더 안의 `datasets` 폴더 안에 넣습니다. 수정된 코드는 아래와 같습니다. 

```python
# dataset settings
dataset_type = 'CocoDataset'
data_root = '/opt/ml/detection/dataset/'

# 우리 데이터셋에 맞도록 classes 추가
classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass",
            "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1024, 1024), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=4, # batch size 2 -> 4
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'train0.json', # fold0
        img_prefix=data_root,
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'val0.json', # fold0
        img_prefix=data_root,
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'test.json',
        img_prefix=data_root,
        pipeline=test_pipeline))

evaluation = dict(interval=1, metric='bbox', classwise=True, save_best='bbox_mAP_50')
```

필수적으로 수정이 필요한 부분은 아래와 같습니다. 

* `data_root`: 우리가 사용할 데이터셋이 위치한 경로(폴더)
* `classes`: 우리가 사용할 데이터셋의 클래스(라벨) 목록 정의
* `data`: dataset config 지정
  * train/val/test: classes, ann_file, img_prefix

위 세 가지는 필수적으로 우리 데이터셋에 맞게 수정해주어야 합니다. 이외의 요소들은 실험을 통해 적절히 변경하면 됩니다. 

추가적인 팁으로, 맨 마지막 줄에 있는 `evaluation` 에 classwise와 save_best를 추가할 수 있습니다. 각 key의 의미는 아래와 같습니다. 

* classwise: 매 validation마다 각 클래스의 AP를 출력해줍니다. 
* save_best: 기준이 될 metric을 전달하면 그 metric이 최적일 때 model checkpoint를 저장해줍니다. 

<br>

### models

`models` 폴더 안에는 일부 많이 사용되는 모델들에 대한 baseline 코드 파일들이 작성되어 있습니다. 

![image-20220409215209568](https://user-images.githubusercontent.com/70505378/162576627-d9e4fafa-fd7b-4a20-9e88-918c329b8bff.png)

예를 들어 `cascade_rcnn_r50_fpn.py` 파일을 보도록 하겠습니다. 

```python
model = dict(
    type='CascadeRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
    roi_head=dict(
        type='CascadeRoIHead',
        num_stages=3,
        stage_loss_weights=[1, 0.5, 0.25],
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=80,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    # model training and testing settings
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
```

기본적인 모델의 뼈대가 작성되어 있습니다. Cascade RCNN 모델을 사용하는 파일의 경우, 이 파일을 `_base_` 리스트 변수에 넣고 수정하고 싶은 부분을 직접 정의하여 오버라이딩하면 됩니다. 





<br>

### schedules

`schedules` 폴더에는 기본적인 optimizer와 scheduler가 작성되어 있습니다. 

![image-20220409215544608](https://user-images.githubusercontent.com/70505378/162576628-2f90e258-2d28-46fc-be93-b89539a649ac.png)

가장 기본이 되는 `schedule_1x.py` 파일 내부는 아래와 같이 작성되어 있습니다. 

```python
# optimizer
optimizer = dict(type='SGD', lr=0.02, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[8, 11])
runner = dict(type='EpochBasedRunner', max_epochs=12)
```

마찬가지로, 이 baseline code를 사용하는 경우 해당 파일의 `_base_` 리스트 변수에 이 파일명을 넣고 각 부분을 새롭게 정의하여 오버라이딩하면 됩니다. (아니면 아예 다른 파일에 복붙하고 수정해서 새로운 파일을 만들어도 됩니다)





<br>

### default_runtime.py

마지막으로 `default_runtime.py` 파일입니다. 해당 파일은 폴더 안에 들어있지는 않고 독립적인 파일로 존재합니다. 

파일 내부는 아래와 같이 작성되어 있습니다. 

```python
checkpoint_config = dict(interval=1)
# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        # dict(type='TensorboardLoggerHook')
    ])
# yapf:enable
custom_hooks = [dict(type='NumClassCheckHook')]

dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]

# disable opencv multithreading to avoid system being overloaded
opencv_num_threads = 0
# set multi-process start method as `fork` to speed up the training
mp_start_method = 'fork'
```

runtime 파일의 경우 위 파일을 그대로 사용하면 됩니다. 

추가적으로, 다른 팀원분께서 공유해주신 꿀팁을 적어볼까 합니다. 모델을 학습시키면 매 epoch마다 model checkpoint 파일이 저장되는데요, 이 파일이 쌓이다 보면 out of memory가 발생할 수 있습니다. **checkpoint_config** 변수 안의 **max_keep_ckpts**를 지정해서 마지막 n개의 checkpoint 파일만 저장되도록 할 수 있습니다. 

```python
checkpoint_config = dict(max_keep_ckpts=3, interval=1) # max number of saving checkpoints
```







<br>

<br>

## train.py

모델 학습을 시킬 때는 `tools` 폴더 안에 있는 `train.py` 파일을 실행합니다. (tools 폴더는 mmdetection 폴더 안에 있습니다)

코드는 너무 길어서 올리지는 않고, 아래 주소에서 확인해주세요. 

[mmdetection - train.py](https://github.com/open-mmlab/mmdetection/blob/master/tools/train.py)

현재 터미널 위치가 mmdetection 폴더에 있다고 할 때, 아래 커맨드를 통해 모델 학습을 수행합니다. 

```
python tools/train.py <실행할 config 폴더 경로> --work-dir <model checkpoint를 저장할 폴더 경로> --resume-from <학습을 재개할 model checkpoint 파일 경로>
```

저는 보통 위와 같은 커맨드를 사용했습니다. 기본적으로는 **실행할 config 폴더 경로**만 지정하면 되고, 나머지는 필요한 경우 추가적으로 지정합니다. 











<br>

<br>

## Random seed

팀원들과 협업을 진행할 때 재현 가능한 학습을 하고 싶다면, random seed를 고정해주어야 합니다. 이에 대한 내용은 별도의 포스팅으로 작성했으니 아래 포스팅을 참고해주세요. 

[MMDetection - PyTorch Randomness 제어하기](https://wowo0709.github.io/ai/aitech/pytorch/AITech-20220329-MMDetection-PyTorch-Randomness-%EC%A0%9C%EC%96%B4%ED%95%98%EA%B8%B0/)

















<br>

<br>

## mmdet folder

마지막으로 볼 것은 `mmdet` 폴더입니다. mmdet 폴더 내부 구성은 아래와 같습니다. 

![image-20220409221735266](https://user-images.githubusercontent.com/70505378/162576630-3062ca2e-721f-41d7-9928-6b51e1f8c5f3.png)

mmdet 폴더에서는 mmdetection에 어떤 파일들이 존재하는지를 볼 수 있고, 각 파일의 구현을 볼 수 있습니다. 예를 들어 `models/necks` 에 가보면 각 neck들이 구현된 파일을 확인할 수 있습니다. 

![image-20220409222200739](https://user-images.githubusercontent.com/70505378/162576631-68f94485-ccce-40fd-a7aa-479fad55a0bf.png)

### Adding new module

mmdetection에서 지원하지 않는 모듈을 사용하고 싶을 때 이 mmdet 폴더의 파일들을 수정합니다. 

예를 들어 저는 이번 대회에서 neck에 BiFPN을 추가하여 사용했는데요, neck 추가를 위해서는 `models/necks` 위치에 bifpn이 구현된 python 파일을 추가하고, `__init__.py` 파일에 bifpn을 추가해주면 됩니다. 

**bifpn.py 파일 추가**

![image-20220409222354829](https://user-images.githubusercontent.com/70505378/162576632-a078527c-37ac-4e3f-99c6-224adc7a06f7.png)

**\_\_init\_\_.py 파일 수정**

```python
# Copyright (c) OpenMMLab. All rights reserved.
from .bfp import BFP
from .channel_mapper import ChannelMapper
from .ct_resnet_neck import CTResNetNeck
from .dilated_encoder import DilatedEncoder
from .dyhead import DyHead
from .fpg import FPG
from .fpn import FPN
from .fpn_carafe import FPN_CARAFE
from .hrfpn import HRFPN
from .nas_fpn import NASFPN
from .nasfcos_fpn import NASFCOS_FPN
from .pafpn import PAFPN
from .rfp import RFP
from .ssd_neck import SSDNeck
from .yolo_neck import YOLOV3Neck
from .yolox_pafpn import YOLOXPAFPN
from .bifpn import BIFPN

__all__ = [
    'FPN', 'BFP', 'ChannelMapper', 'HRFPN', 'NASFPN', 'FPN_CARAFE', 'PAFPN',
    'NASFCOS_FPN', 'RFP', 'YOLOV3Neck', 'FPG', 'DilatedEncoder',
    'CTResNetNeck', 'SSDNeck', 'YOLOXPAFPN', 'DyHead',
    'BIFPN'
]
```





<br>

이상으로 mmdetection을 처음 시작할 때 필수적으로 알아야 할 부분들에 대해 보았습니다. 이번 포스팅에서 다룬 내용들은 입문자를 위한 간단한 내용이고, mmdetection을 제대로 이용하고 커스텀하기 위해서는 **모델 자체에 대한 이해**와 **mmcv 라이브러리에 대한 이해**가 동반되어야 합니다. 

앞으로 mmdetection을 사용하는 데 있어서 이 포스팅이 조금이나마 도움이 되었으면 좋겠습니다! 😊







<br>

<br>

# 참고 자료

* 
