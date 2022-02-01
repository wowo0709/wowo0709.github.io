---
layout: single
title: "[TFLite] TFLite 파일 regular tensorflow ops are not supported 에러"
categories: ['TFLite', 'Android']
toc: true
toc_sticky: true
tag: []
---



<br>

본 포스팅에서는 안드로이드 스튜디오에서 tflite 파일을 사용하는 과정에서 발생할 수 있는 오류를 해결하는 방법에 대해 알아보겠습니다. 

### 모델 변환

우선 저의 tflite 파일 변환 코드는 아래와 같습니다. 

```python
import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images

from tensorflow.python.eager import def_function
from tensorflow.python.framework import tensor_spec
from tensorflow.python.util import nest

# 변환할 가중치
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
# Tiny?
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
# 출력 경로
flags.DEFINE_string('output', './checkpoints/yolov3.tflite',
                    'path to saved_model')
# names 파일 경로
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
# 모의 테스트 이미지
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
# 클래스 개수
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
# 이미지 크기
flags.DEFINE_integer('size', 416, 'image size')


def main(_argv):
    # 모델 불러오기
    if FLAGS.tiny:
        yolo = YoloV3Tiny(size=FLAGS.size, classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(size=FLAGS.size, classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')
    # ====================================================================================================

    # 모델 변환
    converter = tf.lite.TFLiteConverter.from_keras_model(yolo)

    '''Fix from https://stackoverflow.com/questions/64490203/tf-lite-non-max-suppression'''
    # tflite OPs selection
    converter.experimental_new_converter = True
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
    # Post Training Quantization (Float16 Quantization)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    open(FLAGS.output, 'wb').write(tflite_model)
    logging.info("model saved to: {}".format(FLAGS.output))
    # ====================================================================================================

    # tflite 테스트
    interpreter = tf.lite.Interpreter(model_path=FLAGS.output)
    interpreter.allocate_tensors()
    logging.info('tflite model loaded')

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
    logging.info('classes loaded')

    img = tf.image.decode_image(open(FLAGS.image, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, 416)

    t1 = time.time()
    outputs = interpreter.set_tensor(input_details[0]['index'], img)

    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(output_data)

if __name__ == '__main__':
    app.run(main)
```

<br>

위 코드 중 **모델 변환**하는 부분의 코드는 아래의 과정을 수행합니다. 

1. 모델을 converter로 변환
2. converter의 연산 옵션 지정
3. Float16 양자화
4. tflite 파일 변환 후 저장

<br>

연산 옵션을 지정하는 부분에서 `tf.lite.OpsSet.TFLITE_BUILTINS`는 텐서플로 라이트에서 자체적으로 지원하는 최적화된 연산을 수행할 수 있도록 해주고, `tf.lite.OpsSet.SELECT_TF_OPS`는 텐서플로 라이트에서 아직 지원하지 않는 연산들을 텐서플로에서 직접 가져다 쓸 수 있도록 해줍니다. 

Float16 양자화는 모델의 **precision**을 32비트에서 16비트로 감소시키는 것으로, 학습 후 양자화 방법 중 하나입니다. 학습 후 양자화는 학습이 완료된 모델의 크기(용량)를 줄여주는 역할을 합니다. 

학습 후 양자화 방법에는 몇 가지가 있는데, 그 중 Float16 변환이 가장 코드도 간단하고 코드 대비 효율성도 좋습니다. 

<br>

<br>

### 모델 포함

아무튼, 모델 변환을 마치고 이 tflite 파일을 안드로이드 스튜디오 프로젝트의 `assets` 폴더에 포함시킵니다. 

![image-20211007102034524](https://user-images.githubusercontent.com/70505378/136307924-5dc12003-0058-4a73-82dc-3fdf06448e44.png)

`coco.names` 파일은 모델이 분류할 클래스의 목록입니다. 

<br>

### 에러 발생

여기까지 완료한 후, 앱을 실행시키면 모델 생성 런타임 과정에서 에러가 발생하며 아래와 같은 로그를 띄워줍니다. 

> # "Failed to run on the given Interpreter: Regular TensorFlow ops are not supported by this interpreter. Make sure you apply/link the Flex delegate before inference. Node number 62011 (FlexSize) failed to prepare."

위 에러에서 **Node number 62011 (FlexSize)** 부분은 본인이 사용하는 모델에 따라 다릅니다. 

에러는 **"현재 사용하는 Interpreter는 Regular TensorFlow ops를 지원하지 않는다"**라고 말하고 있습니다. 

<br>

아까 위에서 얘기한 것처럼, 텐서플로 라이트 라이브러리는 텐서플로의 모든 연산을 지원하지는 못합니다. 이는 아직 텐서플로 라이트 라이브러리가 완성된지 오래되지 않았고, 따라서 계속해서 활발하게 수정하고 추가하는 과정에 있기 때문입니다. 

따라서 우리는 이 문제를 해결해야 합니다. 

<br>

<br>

### 해결책 후보

저는 이 에러에 대한 해결책 후보로 총 4가지를 생각해보았습니다. 

**1. Tiny YOLO 사용**

YOLO 모델 대신 컴퓨팅 성능에 제한이 있는 엣지 디바이스에 최적화된 Tiny YOLO를 사용한다. 이것이 해결책이 될 수 있는 이유는 첫째로 Tiny YOLO는 에러를 일으키는 ‘특정 연산’을 사용하지 않고 있을 수 있고, 둘째로 Tiny YOLO의 크기가 일반 YOLO보다 작기 때문에 tflite 파일 변환 과정에서 양자화를 사용하지 않을 수 있기 때문이다. 

**2. 양자화 이외의 최적화 연산 적용**

YOLO 모델의 tflite 파일 크기는 약 236MB이다. 그리고 안드로이드 스튜디오에서는 200MB 이하의 파일만을 사용할 수 있다. 이 때문에 YOLO 모델의 경우 필연적으로 양자화 과정이 필요한데, 이 양자화를 사용하면 모델의 크기를 줄일 수 있지만 사용할 수 있는 연산이 제한이 생길 수 있다. 따라서 양자화 이외의 모델 크기를 줄여주는 최적화 기법을 사용하면 문제를 해결할 수도 있다. 

**3. 특정 연산을 연산 가능 목록에 추가**

텐서플로에서는 연산을 추가하는 기능을 제공하는 것으로 알고 있다. 이것이 가능할 지는 해봐야 알겠지만, 현재 에러를 일으키는 특정 연산을 수동으로 연산 가능 목록에 추가하는 것이 해결책이 될 수 있다. 

**4. 연산을 다른 연산으로 대체**

현재 문제를 일으키는 특정 연산은 YOLO 모델의 순전파 과정에 존재할 것이다. 이 부분의 코드를 찾아서 같은 결과는 내는 다른 연산들로 대체하는 것이 해결책이 될 수 있다. 

<br>

그리고 저는 위 후보들 중 **3번 해결책**으로 이 문제를 해결하였습니다. 

<br>

<br>

### 문제 해결 과정

어찌보면, 우리는 당연한 부분을 앞에서부터 놓치고 있었을지도 모릅니다. 

우리는 위에서 분명 텐서플로 라이트가 지원하지 않는 연산에 대비하여 텐서플로의 연산을 직접 가져다 쓸 수 있도록 연산 옵션에 `tf.lite.OpsSet.SELECT_TF_OPS`를 추가했었습니다. 

그런데 왜 안드로이드 런타임에 해당 연산을 지원할 수 없다는 에러가 발생하는 것일까요?

<br>

조십스럽게, 그렇다면 이는 **텐서플로에서 해당 연산을 가져오고 있지 못하기 때문**이라고 할 수 있습니다. 

<br>

자, 잠시 tflite 파일 변환 시로 돌아가겠습니다. 

저는 아나콘다 프롬프트로 커맨드를 입력해 변환을 수행하였는데, 이 때 아래와 같은 에러 메시지가 발생합니다. 

> Some ops are not supported by the native TFLite runtime, you can enable TF kernels fallback using TF Select. See instructions: https://www.tensorflow.org/lite/guide/ops_select
> TF Select ops: AddV2, Mul

그런데 이 에러 메시지는 실행을 종료시키지 않고 모델을 무사히 tflite 파일로 변환해줍니다. 

<br>

이 에러를 간과했던 것이 우리가 헤멘 이유였습니다...

위 에러는 우리에게 정답을 알려주고 있습니다. **"현재 몇 개의 연산이 native TFLite 런타임에 지원되지 않으니, TF Select를 이용해 TF kernels fallback을 사용하여 가능하게 할 수 있다"**라고 하면서 친절히 우리가 이 문제를 해결할 수 있는 사이트 `https://www.tensorflow.org/lite/guide/ops_select`까지 알려줍니다. 

<br>

그렇다면 위 사이트로 가보죠. 

![image-20211007104105969](https://user-images.githubusercontent.com/70505378/136307927-81e49f14-2198-4691-82ee-ecd2e6fb27b2.png)

위 사이트는 우리가 겪고 있는 문제인, **텐서플로 라이트에서 지원하지 않는 연산을 사용하지 못하는 문제**에 대한 해답을 알려줍니다. 

간단히 요약하면, 아래와 같습니다. 

**1. TFLite 파일 변환 시 converter에 `tf.lite.OpsSet.SELECT_TF_OPS`를 추가합니다.**

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

**2. 안드로이드 프로젝트 dependencies에 `TF op support` 라이브러리를 추가합니다.**

```groovy
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
    // This dependency adds the necessary TF op support.
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.0.0-nightly-SNAPSHOT'
}
```

**3. 안드로이드 프로젝트 allprojects의 repositories에 `Sonatype snapshot repository`를 추가합니다.**

```groovy
allprojects {
    repositories {      // should be already there
        mavenCentral()  // should be already there
        maven {         // add this repo to use snapshots
          name 'ossrh-snapshot'
          url 'http://oss.sonatype.org/content/repositories/snapshots'
        }
    }
}
```

<br>

제가 앞에서 언급한 것처럼, 저희는 텐서플로 연산을 사용하기 위한 과정 중 1번 과정만을 수행했을 뿐이어서 해당 연산을 가져오고 있지 못했던 것이었습니다.

따라서 위의 1~3번 과정을 수행하면 에러 없어 모델이 생성되는 것을 확인할 수 있습니다. 

<br>

**✋ 참고**

텐서플로 연산을 제공하는 라이브러리 전체를 추가하지 않고 본인이 필요한 연산만을 제공할 수 있도록 커스텀 라이브러리를 생성할 수 있다고 합니다. (이를 AAR이라고 하는 것 같은데... 잘은 모르겠습니다)

추후에 앱의 용량이 너무 크다면 이를 커스터마이징하는 것도 좋은 방법일 것 같습니다. 

이에 대한 정보는 마찬가지로 [여기1](https://www.tensorflow.org/lite/guide/ops_select) 와 [여기2](https://www.tensorflow.org/lite/guide/build_android#use_nightly_snapshots)에서 보실 수 있습니다. 

<br>













