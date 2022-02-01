---
layout: single
title: "[Computer Vision] 4(5). 전이학습 예제"
categories: ['AI', 'ComputerVision']
---



<br>

# 전이학습 예제

이번 포스팅에서는 케라스를 이용해 전이학습을 처음부터 끝까지 수행해본다. 

ImageNet 데이터셋으로 학습을 완료한 케라스 애플리케이션 모델을 사용해 전이학습을 수행할 것이며, 이것이 앞에서 본 랜덤 가중치 초기화 방법과 어떻게 다른지 확인해 볼 것이다.


```python
import tensorflow as tf
import os
import matplotlib.pyplot as plt
import math

# 하이퍼 파라미터
batch_size = 32
num_epochs = 300
random_seed = 42
```

<br>

<br>

### 데이터 준비

---

여기서는 테스트 데이터로 벤치마크 데이터셋인 CIFAR-100을 사용할 것입니다. 


```python
# !pip install tensorflow-datasets
```

이 책의 깃허브 저장소에서 제공해주는 cifar_utils.py 소스코드 파일을 다운로드합니다. 

[cifar_utils.py 깃허브 저장소](https://github.com/wikibook/dl-vision/blob/master/Chapter04/cifar_utils.py)

위 파일을 다운로드 받고 같은 디렉터리 내에 포함시킵니다. 이제 위의 소스파일을 **Input Pipeline**으로 사용합니다. 


```python
import cifar_utils

cifar_info = cifar_utils.get_info()
print(cifar_info)

# 클래스 수
num_classes = cifar_info.features['label'].num_classes

# 이미지 수
num_train_imgs = cifar_info.splits['train'].num_examples
num_val_imgs = cifar_info.splits['test'].num_examples
# 배치 단위
train_steps_per_epoch = math.ceil(num_train_imgs / batch_size)
val_steps_per_epoch = math.ceil(num_val_imgs / batch_size)

# 입력 데이터 형상
input_shape = [224,224,3]
```

    tfds.core.DatasetInfo(
        name='cifar100',
        full_name='cifar100/3.0.2',
        description="""
        This dataset is just like the CIFAR-10, except it has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class. The 100 classes in the CIFAR-100 are grouped into 20 superclasses. Each image comes with a "fine" label (the class to which it belongs) and a "coarse" label (the superclass to which it belongs).
        """,
        homepage='https://www.cs.toronto.edu/~kriz/cifar.html',
        data_path='C:\\Users\\wjsdu\\tensorflow_datasets\\cifar100\\3.0.2',
        download_size=160.71 MiB,
        dataset_size=132.03 MiB,
        features=FeaturesDict({
            'coarse_label': ClassLabel(shape=(), dtype=tf.int64, num_classes=20),
            'id': Text(shape=(), dtype=tf.string),
            'image': Image(shape=(32, 32, 3), dtype=tf.uint8),
            'label': ClassLabel(shape=(), dtype=tf.int64, num_classes=100),
        }),
        supervised_keys=('image', 'label'),
        disable_shuffling=False,
        splits={
            'test': <SplitInfo num_examples=10000, num_shards=1>,
            'train': <SplitInfo num_examples=50000, num_shards=1>,
        },
        citation="""@TECHREPORT{Krizhevsky09learningmultiple,
            author = {Alex Krizhevsky},
            title = {Learning multiple layers of features from tiny images},
            institution = {},
            year = {2009}
        }""",
    )



```python
# 훈련 데이터 가져오기
train_cifar_dataset = cifar_utils.get_dataset(phase='train', batch_size=batch_size, 
                                              num_epochs=num_epochs, shuffle=True, 
                                              input_shape=input_shape, seed=random_seed)
# 검증 데이터 가져오기
val_cifar_dataset = cifar_utils.get_dataset(phase='test', batch_size=batch_size, 
                                              num_epochs=1, shuffle=False, 
                                              input_shape=input_shape, seed=random_seed)
```

<br>

<br>

### 사전학습된 케라스 애플리케이션 모델으로 새로운 분류기 만들기

---

여기서는 ImageNet 데이터셋으로 사전학습된 케라스 애플리케이션의 ResNet-50 모델을 특징 추출기로서 사용합니다. 그리고 계층에 마지막에 우리가 분류할 데이터인 CIFAR에 맞게 계층을 추가해줄 것입니다. 

이 때 특징 추출기의 계층들은 학습을 하지 않도록 계층을 고정시키고, 분류를 위해 새로 추가한 밀집 계층들만 학습을 하도록 할 것입니다. 


```python
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

resnet50_feature_extractor = tf.keras.applications.resnet50.ResNet50(
    include_top=False, weights='imagenet', input_shape=input_shape)
# resnet50_feature_extractor.summary()
```

    Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
    94773248/94765736 [==============================] - 8s 0us/step

케라스 애플리케이션에서 ResNet50을 특징 추출기로 가져왔으니, 이제 계층을 고정시킵니다. 계층을 고정시키는 이유는 크게 2가지라고 할 수 있습니다. 

- 사전학습된 신경망의 가중치를 온전히 사용하고 싶다. 
- 사전학습에 사용된 데이터셋이 새로운 학습에 사용할 데이터셋보다 훨씬 많다. 

하지만 여기서 주의해야 할 것은, **정규화 계층과 같은 계층들은 고정시키지 않아야 한다는 것**입니다. 배치 정규화 계층 같은 계층들은, 새로운 데이터셋의 특징들을 추출하도록 학습되어야 합니다. 


```python
frozen_layers, trainable_layers = [], []
for layer in resnet50_feature_extractor.layers:
    if isinstance(layer, tf.keras.layers.Conv2D):
        # 합성곱 계층 고정시키기
        layer.trainable = False
        frozen_layers.append(layer.name)
    else:
        if len(layer.trainable_weights) > 0:
            # 학습할 파라미터가 존재하는 계층들만 추가
            trainable_layers.append(layer.name)
```

고정시킬 계층과 훈련시킬 계층을 나눴습니다. 

고정시킬 계층은 합성곱 계층입니다. 훈련시킬 계층은 합성곱 계층을 제외한 계층들 중, **학습할 파라미터가 존재하는** 계층들입니다. 

즉, 풀링 계층 같은 학습할 파라미터가 존재하지 않는 계층들은 포함되지 않습니다. 


```python
log_begin_red, log_begin_blue, log_begin_green = '\033[91m', '\n\033[94m', '\033[92m'
log_begin_bold, log_begin_underline = '\033[1m', '\033[4m'
log_end_format = '\033[0m'

# 고정/학습 계층들 출력
print("{2}Layers we froze:{4} {0} ({3}total = {1}{4}).".format(
    frozen_layers, len(frozen_layers), log_begin_red, log_begin_bold, log_end_format))
print("\n{2}Layers which will be fine-tuned:{4} {0} ({3}total = {1}{4}).".format(
    trainable_layers, len(trainable_layers), log_begin_blue, log_begin_bold, log_end_format))
```

    Layers we froze: ['conv1_conv', 'conv2_block1_1_conv', 'conv2_block1_2_conv', 'conv2_block1_0_conv', 'conv2_block1_3_conv', 'conv2_block2_1_conv', 'conv2_block2_2_conv', 'conv2_block2_3_conv', 'conv2_block3_1_conv', 'conv2_block3_2_conv', 'conv2_block3_3_conv', 'conv3_block1_1_conv', 'conv3_block1_2_conv', 'conv3_block1_0_conv', 'conv3_block1_3_conv', 'conv3_block2_1_conv', 'conv3_block2_2_conv', 'conv3_block2_3_conv', 'conv3_block3_1_conv', 'conv3_block3_2_conv', 'conv3_block3_3_conv', 'conv3_block4_1_conv', 'conv3_block4_2_conv', 'conv3_block4_3_conv', 'conv4_block1_1_conv', 'conv4_block1_2_conv', 'conv4_block1_0_conv', 'conv4_block1_3_conv', 'conv4_block2_1_conv', 'conv4_block2_2_conv', 'conv4_block2_3_conv', 'conv4_block3_1_conv', 'conv4_block3_2_conv', 'conv4_block3_3_conv', 'conv4_block4_1_conv', 'conv4_block4_2_conv', 'conv4_block4_3_conv', 'conv4_block5_1_conv', 'conv4_block5_2_conv', 'conv4_block5_3_conv', 'conv4_block6_1_conv', 'conv4_block6_2_conv', 'conv4_block6_3_conv', 'conv5_block1_1_conv', 'conv5_block1_2_conv', 'conv5_block1_0_conv', 'conv5_block1_3_conv', 'conv5_block2_1_conv', 'conv5_block2_2_conv', 'conv5_block2_3_conv', 'conv5_block3_1_conv', 'conv5_block3_2_conv', 'conv5_block3_3_conv'] (total = 53).
    
    
    Layers which will be fine-tuned: ['conv1_bn', 'conv2_block1_1_bn', 'conv2_block1_2_bn', 'conv2_block1_0_bn', 'conv2_block1_3_bn', 'conv2_block2_1_bn', 'conv2_block2_2_bn', 'conv2_block2_3_bn', 'conv2_block3_1_bn', 'conv2_block3_2_bn', 'conv2_block3_3_bn', 'conv3_block1_1_bn', 'conv3_block1_2_bn', 'conv3_block1_0_bn', 'conv3_block1_3_bn', 'conv3_block2_1_bn', 'conv3_block2_2_bn', 'conv3_block2_3_bn', 'conv3_block3_1_bn', 'conv3_block3_2_bn', 'conv3_block3_3_bn', 'conv3_block4_1_bn', 'conv3_block4_2_bn', 'conv3_block4_3_bn', 'conv4_block1_1_bn', 'conv4_block1_2_bn', 'conv4_block1_0_bn', 'conv4_block1_3_bn', 'conv4_block2_1_bn', 'conv4_block2_2_bn', 'conv4_block2_3_bn', 'conv4_block3_1_bn', 'conv4_block3_2_bn', 'conv4_block3_3_bn', 'conv4_block4_1_bn', 'conv4_block4_2_bn', 'conv4_block4_3_bn', 'conv4_block5_1_bn', 'conv4_block5_2_bn', 'conv4_block5_3_bn', 'conv4_block6_1_bn', 'conv4_block6_2_bn', 'conv4_block6_3_bn', 'conv5_block1_1_bn', 'conv5_block1_2_bn', 'conv5_block1_0_bn', 'conv5_block1_3_bn', 'conv5_block2_1_bn', 'conv5_block2_2_bn', 'conv5_block2_3_bn', 'conv5_block3_1_bn', 'conv5_block3_2_bn', 'conv5_block3_3_bn'] (total = 53).


​    <br>


이제 분류를 수행할 상위 계층들을 추가합니다. 


```python
# 특징 추출기의 출력 특징맵
features = resnet50_feature_extractor.output
# 분류 상위 게층 추가
# GlobalAveragePooling으로 (7, 7, 28) 크기의 특징맵을 (1,1,28) 크기로 변환
avg_pool = GlobalAveragePooling2D(data_format='channels_last')(features)
# 최종 클래스 분류
predictions = Dense(num_classes, activation='softmax')(avg_pool)

# 모델 생성
resnet50_freeze = Model(resnet50_feature_extractor.input, predictions)
```

<br>

<br>

### 네트워크 훈련하기

---

모델 생성을 마쳤습니다. 

이제 최적화기, 손실함수, 성능지표, 콜백 등을 설정하고 훈련을 시작합니다. 

아래 코드 중 **keras_custom_callbacks** 모듈이 있는데, 이 모듈은 책의 저자들이 만든 커스텀 콜백 파일입니다. 이 소스파일 또한 아래에서 다운로드할 수 있습니다. 

[keras_custom_callbacks.py 깃허브 저장소](https://github.com/PacktPublishing/Hands-On-Computer-Vision-with-TensorFlow-2/blob/master/Chapter04/keras_custom_callbacks.py)


```python
import collections
import functools
from keras_custom_callbacks import SimpleLogCallback

# 케라스 콜백 설정
metrics_to_print = collections.OrderedDict([("loss", "loss"), 
                                            ("v-loss", "val_loss"), 
                                            ("acc", "acc"), 
                                            ("v-acc", "val_acc"), 
                                            ("top5-acc", "top5_acc"), 
                                            ("v-top5-acc", "val_top5_acc")])
# 모델 summary 저장 경로
model_dir = './models/resnet_keras_app_freeze_all'
# 콜백 설정
callbacks = [
    # 조기종료
    tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_acc', 
                                     restore_best_weights=True), 
    # 텐서보드 시각화
    tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0,
                                   write_graph=True),
    # 
    SimpleLogCallback(metrics_to_print, num_epochs=num_epochs, 
                      log_frequency=1),
    # 모델을 기록/저장할 체크포인트
    tf.keras.callbacks.ModelCheckpoint(
        os.path.join(model_dir, 'weights-epoch{epoch:02d}-loss{val-loss:.2f}.h5'), period=5)
]

# 모델 컴파일
optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)
resnet50_freeze.compile(optimizer=optimizer, 
                        loss='sparse_categorical_crossentropy', 
                        metrics=[
                            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
                            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5,name='top5_acc')
                        ])

# 모델 훈련
history_freeze = resnet50_freeze.fit(train_cifar_dataset, 
                                    epochs=num_epochs, 
                                    steps_per_epoch=train_steps_per_epoch, 
                                    validation_data=val_cifar_dataset, 
                                    validation_steps=val_steps_per_epoch,
                                    verbose=0, 
                                    callbacks=callbacks)
```

성능 지표들을 관찰해봅시다. 


```python
fig, ax = plt.subplots(3, 2, figsize=(15, 10), sharex='col')
ax[0, 0].set_title("loss")
ax[0, 1].set_title("val-loss")
ax[1, 0].set_title("acc")
ax[1, 1].set_title("val-acc")
ax[2, 0].set_title("top5-acc")
ax[2, 1].set_title("val-top5-acc")

ax[0, 0].plot(history_freeze.history['loss'])
ax[0, 1].plot(history_freeze.history['val_loss'])
ax[1, 0].plot(history_freeze.history['acc'])
ax[1, 1].plot(history_freeze.history['val_acc'])
ax[2, 0].plot(history_freeze.history['top5_acc'])
ax[2, 1].plot(history_freeze.history['val_top5_acc'])
```

전이 학습을 사용하지 않았을 때(top1: 65%, top5: 88%)와 비교하여 전이학습을 사용했을 때(top1: 78%, top5: 95%) 훨씬 좋은 성능을 보입니다. 

<br>

<br>

### 결과 시각화

---

마지막으로 앞서 훈련시킨 신경망으로 예측을 수행한다. 


```python
import glob
import numpy as np
from classification_utils import load_image, process_predictions, display_predictions

test_filenames = glob.glob(os.path.join('images', '*'))
# test_filenames = glob.glob(os.path.join('res', '*'))
test_images = np.asarray([load_image(file, size=input_shape[:2]) 
                          for file in test_filenames])

image_batch = test_images[:16]

# Our model was trained on CIFAR images, which originally are 32x32px. We scaled them up
# to 224x224px to train our model on, but this means the resulting images had important
# artifacts/low quality.
# To test on images of the same quality, we first resize them to 32x32px, then to the 
#expected input size (i.e., 224x224px):
cifar_original_image_size = cifar_info.features['image'].shape[:2]
class_readable_labels = cifar_info.features["label"].names

image_batch_low_quality = tf.image.resize(image_batch, cifar_original_image_size)
image_batch_low_quality = tf.image.resize(image_batch_low_quality, input_shape[:2])
    
predictions = resnet50_freeze.predict_on_batch(image_batch_low_quality)
top5_labels, top5_probabilities = process_predictions(predictions, class_readable_labels)

print("ResNet-50 trained on ImageNet and fine-tuned on CIFAR-100:")
display_predictions(image_batch, top5_labels, top5_probabilities)
```

<br>

<br>

### ResNet 특징 추출기 최적화

---

**Fine-tuning**이라는 것을 사용해보자. 

Fine-Tuning은 앞서 ResNet-50 특징 추출기의 모든 합성곱 계층을 고정시킨 것과 달리, 상위 일부 계층의 학습을 허락하여 조금 더 task-relavant features를 학습하도록 하는 것이다. 

**다만, 이 기법은 과대적합을 피할 수 있을 정도로 훈련 데이터가 충분히 클 때 사용해야 한다.**


```python
for layer in resnet50_feature_extractor.layers:
    if 'res5' in layer.name:
        # Keras developers named the layers in their ResNet implementation to explicitly 
        # identify which macro-block and block each layer belongs to.
        # If we reach a layer which has a name starting by 'resnet5', it means we reached 
        # the 4th macro-block / we are done with the 3rd one (see layer names listed previously):
        break
    if isinstance(layer, tf.keras.layers.Conv2D):
        layer.trainable = False
```


```python
num_macroblocks_to_freeze = [0, 1, 2, 3] # we already covered the "all 4 frozen" case above.

histories = dict()
histories['freeze all'] = history_freeze
for freeze_num in num_macroblocks_to_freeze:
        
    print("{1}{2}>> {3}ResNet-50 with {0} macro-block(s) frozen{4}:".format(
        freeze_num, log_begin_green, log_begin_bold, log_begin_underline, log_end_format))
    
    # ---------------------
    # 1. We instantiate a new classifier each time:
    resnet50_feature_extractor = tf.keras.applications.resnet50.ResNet50(
        include_top=False, weights='imagenet', 
        input_shape=input_shape, classes=num_classes)

    features = resnet50_feature_extractor.output
    avg_pool = GlobalAveragePooling2D(data_format='channels_last')(features)
    predictions = Dense(num_classes, activation='softmax')(avg_pool)

    resnet50_finetune = Model(resnet50_feature_extractor.input, predictions)
    
    # ---------------------
    # 2. We freeze the desired layers: 
    break_layer_name = 'res{}'.format(freeze_num + 2) if freeze_num > 0 else 'conv1'
    frozen_layers = []
    for layer in resnet50_finetune.layers:
        if break_layer_name in layer.name:
            break
        if isinstance(layer, tf.keras.layers.Conv2D):
            # If the layer is a convolution, and isn't after the 1st layer not to train:
            layer.trainable = False
            frozen_layers.append(layer.name)
    
    print("\t> {2}Layers we froze:{4} {0} ({3}total = {1}{4}).".format(
        frozen_layers, len(frozen_layers), log_begin_red, log_begin_bold, log_end_format))
    
    # ---------------------
    # 3. To start from the beginning the data iteration, 
    #    we re-instantiate the input pipelines (same parameters):
    train_cifar_dataset = cifar_utils.get_dataset(
    phase='train', batch_size=batch_size, num_epochs=num_epochs, shuffle=True,
    input_shape=input_shape, seed=random_seed)

    val_cifar_dataset = cifar_utils.get_dataset(
        phase='test', batch_size=batch_size, num_epochs=1, shuffle=False,
        input_shape=input_shape, seed=random_seed)

    # ---------------------
    # 4. We set up the training operations, and start the process:
    # We set a smaller learning rate for the fine-tuning:
    # optimizer = tf.keras.optimizers.SGD(lr=1e-4, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = tf.keras.optimizers.SGD(momentum=0.9, nesterov=True)

    model_dir = './models/resnet_keras_app_freeze_{}_mb'.format(freeze_num)
    callbacks = [
        # Callback to interrupt the training if the validation loss/metrics converged:
        # (we use a shorter patience here, just to shorten a bit the demonstration, already quite long...)
        tf.keras.callbacks.EarlyStopping(patience=8, monitor='val_acc', restore_best_weights=True),
        # Callback to log the graph, losses and metrics into TensorBoard:
        tf.keras.callbacks.TensorBoard(log_dir=model_dir, histogram_freq=0, write_graph=True),
        # Callback to save the model (e.g., every 5 epochs)::
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, 'weights-epoch{epoch:02d}-loss{val_loss:.2f}.h5'), period=5)
    ]
    
    # Compile:
    resnet50_finetune.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=[
            tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5_acc')
        ])

    # Train:
    print("\t> Training - {0}start{1} (logs = off)".format(log_begin_red, log_end_format))
    history = resnet50_finetune.fit(
        train_cifar_dataset,  epochs=num_epochs, steps_per_epoch=train_steps_per_epoch,
        validation_data=val_cifar_dataset, validation_steps=val_steps_per_epoch,
        verbose=0, callbacks=callbacks)
    print("\t> Training - {0}over{1}".format(log_begin_green, log_end_format))

    acc = history.history['acc'][-1] * 100
    top5 = history.history['top5_acc'][-1] * 100
    val_acc = history.history['val_acc'][-1] * 100
    val_top5 = history.history['val_top5_acc'][-1] * 100
    epochs = len(history.history['val_loss'])
    print("\t> Results after {5}{0}{6} epochs:\t{5}acc = {1:.2f}%; top5 = {2:.2f}%; val_acc = {3:.2f}%; val_top5 = {4:.2f}%{6}".format(
        epochs, acc, top5, val_acc, val_top5, log_begin_bold, log_end_format))

    histories['freeze {}'.format(freeze_num)] = history
```

<br>

<br>

### 텐서플로 허브 모델로 전이학습하기

---

텐서플로 허브 모델로 전이학습을 시키는 것은 모델을 케라스 애플리케이션이 아닌 tensorflow-hub 사이트에서 가져오는 것을 제외하면 동일하다. 

여기서는 텐서플로 허브의 inceptionV3 모델을 사용한다. 

**모델 가져오기**


```python
import tensorflow_hub as hub
# model_url = "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1"

# We need a TF2-compatible model:
module_url = "https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/2"
inception_expected_input_shape = [299, 299, 3]
inception_expected_output_shape = [2048]
```


```python
hub_feature_extractor = hub.KerasLayer(
    module_url, 
    trainable=False,                              # Flag to set the layers as trainable or not
    input_shape=inception_expected_input_shape,   # Expected input shape.
    output_shape=inception_expected_output_shape, # Output shape [batch_size, 2048].
    dtype=tf.float32)                             # Expected dtype

# Note: These parameters can be found on the webpage of tfhub Module, or can be fetched as follows:
# module_spec = hub.load_module_spec(model_url)
# expected_height, expected_width = hub.get_expected_image_size(module_spec)
# expected_input_shape = tf.convert_to_tensor([height, width, 3])

print(hub_feature_extractor)
```

    <tensorflow_hub.keras_layer.KerasLayer object at 0x0000025F0CEA59D0>

**상위 계층 추가하기**


```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense

inception_model = Sequential([
    hub_feature_extractor,
    Dense(num_classes, activation='softmax', name="logits_pred")
], name="inception_tf_hub")
```


```python
inception_model.summary()
```

    Model: "inception_tf_hub"
    _________________________________________________________________
    Layer (type)                 Output Shape              Param #   
    =================================================================
    keras_layer (KerasLayer)     (None, 2048)              21802784  
    _________________________________________________________________
    logits_pred (Dense)          (None, 100)               204900    
    =================================================================
    Total params: 22,007,684
    Trainable params: 204,900
    Non-trainable params: 21,802,784
    _________________________________________________________________

