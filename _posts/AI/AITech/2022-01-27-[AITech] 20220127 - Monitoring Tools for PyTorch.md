---
layout: single
title: "[AITech] 20220127 - Monitoring Tools for PyTorch"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: ['TensorBoard', 'Weight&Biases']
---



<br>

## 학습 내용 정리

### Monitoring tools for PyTorch

모델을 학습시키는 동안 학습 현황을 확인할 수 있는 많은 모니터링 도구들 중 대표적인 `TensorBoard`와 `weightandbiases`에 대해 살펴본다. 

#### Tensorboard

* **Tensorboard**

  * 특징

    * TensorFlow의 프로젝트로 만들어진 시각화 도구
    * 학습 그래프, metric, 학습 결과의 시각화 지원
    * PyTorch도 연결 가능 -> DL 시각화 핵심 도구

  * 데이터

    * scalar: metric 등 상수값의 연속을 표시
    * graph: 모델의 computational graph 표시
    * histogram: weight 등 값의 분포를 표현
    * Image/Text: 예측 값과 실제 값을 비교 표시
    * mesh: 3d 형태의 데이터를 표현하는 도구

  * 예시 코드

    ```python
    # Tensorboard 기록을 위한 directory 생성
    import os
    logs_base_dir = "logs"
    os.makedirs(logs_base_dir, exist_ok=True)
    
    # 기록 생성 객체 SummaryWriter 생성
    from torch.utils.tensorboard import SummaryWriter
    import numpy as np
    # add_scalar: scalar 값을 기록
    # Loss/train: loss category에 train 값
    # n_iter: x 축의 값
    writer = SummaryWriter(exp)
    for n_iter in range(100):
        writer.add_scalar('Loss/train', np.random.random(), n_iter)
        writer.add_scalar('Loss/test', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/train', np.random.random(), n_iter)
        writer.add_scalar('Accuracy/test', np.random.random(), n_iter)
    # 값 기록(disk에 쓰기)
    writer.flush()
    # jupyter/colab command(같은 명령어를 콘솔에서도 사용 가능)
    %load_ext tensorboard
    %tensorboard --logdir "logs"
    ```
    
  * Terminal command
  
    * `--logdir`: log가 저장된 경로
    * `--host`: 원격 서버에서 사용 시 0.0.0.0 (default: localhost)
    * `-port`: 포트 번호
  
    ```bash
    tensorboard --logdir PATH --host ADDR --port PORT
    ```
  
    

![image-20220226181054271](https://user-images.githubusercontent.com/70505378/155837514-30f45b04-a373-438d-8fa4-d8e8e7d638f7.png)

#### Weight & Biases

* **Weight & Biases**

  * 특징

    * 머신러닝 실험을 원활히 지원하기 위한 상용도구(유료, 기본 기능은 무료로 사용 가능)
    * 협업, code versioning, 실험 결과 기록 등 제공
    * MLOps의 대표적인 툴로 저변 확대 중
    * 하나의 프로젝트로 관리하기 때문에 코드나 실험 결과 공유 시 매우 유용

  * 사용 과정

    1. [weight & biases 사이트 접속](https://wandb.ai/site)

    2. 가입 후 API 키 확인(Settings -> API keys)

    3. 새로운 프로젝트 생성하기(Profile -> Create new project)

       * 프로젝트 이름은 모델과 연결 과정에서 사용됨

    4. wandb 연결

       * wanb.init()을 호출하면 API key를 입력

       ```python
       !pip install wandb -q
       
       import wandb
       wandb.init(project="my-test-project", entity='wowo0709') # 프로젝트명, 닉네임
       ```

    5. config 설정

       * 프로젝트 템플릿으로 따로 파일로 관리하면 매우 편리!

       ```python
       EPOCHS = 100
       BATCH_SIZE = 32
       LEARNING_RATE = 0.001
       
       config={"epochs": EPOCHS, "batch_size": BATCH_SIZE, "learning_rate" : LEARNING_RATE}
       wandb.config = config
       # wandb.init(project="my-test-project", config=config)
       ```

    6. 학습 기록

       ```python
       for e in range(1, EPOCHS+1):
           epoch_loss = 0
           epoch_acc = 0
           for X_batch, y_batch in train_dataset:
               X_batch, y_batch = X_batch.to(device), y_batch.to(device).type(torch.cuda.FloatTensor)
               optimizer.zero_grad()        
               y_pred = model(X_batch)
                      
               loss = criterion(y_pred, y_batch.unsqueeze(1))
               acc = binary_acc(y_pred, y_batch.unsqueeze(1))
               
               loss.backward()
               optimizer.step()
               
               epoch_loss += loss.item()
               epoch_acc += acc.item()
               
               
           train_loss = epoch_loss/len(train_dataset)
           train_acc = epoch_acc/len(train_dataset)
           print(f'Epoch {e+0:03}: | Loss: {train_loss:.5f} | Acc: {train_acc:.3f}')
           # 학습 기록!
           wandb.log({'accuracy': train_acc, 'loss': train_loss})
       ```

       

       ![image-20220127135843298](https://user-images.githubusercontent.com/70505378/151295520-afa06789-313a-4a2f-9f0d-2bfde985c7e8.png)





<br>
