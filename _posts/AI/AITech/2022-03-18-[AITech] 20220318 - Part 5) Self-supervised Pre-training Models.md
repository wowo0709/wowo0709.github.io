---
layout: single
title: "[AITech][NLP] 20220318 - Part 5) Self-supervised Pre-training Models"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 KAIST '주재걸' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# Self-supervised Pre-training Models

이번 강의는 `Self-supervised Pre-training Models`에 대한 내용입니다. 

Transformer 모델과 Self-Attention 모듈이 등장한 뒤, 모델 구조의 특별한 변화는 없이 Self-Attention 모듈을 여러 개(12개, 24개, 또는 그 이상) 쌓음으로써 모델의 성능을 높이는 연구들이 활발하게 진행되었습니다. 

이러한 모델들은 Self-supervised 메커니즘으로 학습되고, 다른 모델에 transfer-learning으로 쓰이는 등 큰 활용성을 보이고 있습니다. 

다만, 여전히 sequence generation 시 매 time step마다 앞 단어에서부터 순차적으로 단어를 생성해야 한다는 부분에서는 벗어나지 못 하고 있기도 합니다. 

## GPT-1

먼저 소개할 것은 `GPT-1`입니다. GPT-1은 Self-Attention 블록을 12개 연속해서 쌓아놓은 형태로, 특징은 아래와 같습니다. 

* 다양한 task에 활용하기 위해 \<S\>, \<E\>, $ 와 같은 special token들을 사용합니다. 
* 12-layer 모두 decoder 구조만을 사용합니다. 
* 12개의 multi head를 사용합니다. 
* GELU activation unit을 사용합니다. 

GPT-1은 다양한 NLP task에 적용할 수 있도록 학습되었습니다. 그래서 다른 task를 수행할 모델을 만들고 싶을 때, GPT-1의 Pretrained Attention modules를 가져오고 맨 뒤의 branch만 바꿔 fine-tuning 함으로써 적은 데이터로도 높은 성능을 낼 수 있게 되었습니다. 

![image-20220320170544870](https://user-images.githubusercontent.com/70505378/159172080-ea673e8c-ef05-4922-857d-b2e47a991600.png)

GPT-1은 학습 시에는 multi-task learning(text prediction&task classifier)을 수행합니다. 

위에서 GPT-1에는 decoder 구조만을 사용한다고 했는데, 이는 입력으로 input vector를 넣어주고 출력으로 그에 해당하는 output vector를 출력하도록 학습되면서, 현재 time step 이후의 단어들에 대한 참조가 불가능하도록 학습되기 때문입니다. 

위 그림을 보면 모델 구조 옆에 GPT-1의 pretrained attention modules(Transformer)로 수행할 수 있는 여러 task들이 안내되어 있습니다. 

그림이 의미하는 것은 input vector의 모양만 안내되어 있는대로 맞춰주면, Transformer(pretrained, 작은 learning rate 설정)와 Linear layer(random initiailzed, 큰 learning rate 설정)만 추가하여 fine-tuning함으로써 다양한 task에 pretrained GPT-1 모델이 사용될 수 있음을 보여줍니다. 



<br>

## BERT

두번째 pre-training model로 `BERT`에 대해 살펴보겠습니다. 

### Masked Language Model(MLM)

앞서 살펴본 GPT-1의 경우, pretraining 시 현재 time step 이전의 단어들만 이용해 예측을 하도록 학습되었습니다. 

하지만 실제 인간을 생각해보면, 단어 에측 시 앞에 있는 단어들 뿐만 아니라, 뒤에 나오는 단어들까지 모두 고려하여 현재 단어를 예측하는 것이 자연스러운 일이며, 실제로 정확도도 높을 것입니다. 이를 위하여 BERT는 **Masked Language Modeling**이라고 하는 pre-training task를 수행합니다. 

![image-20220320173312236](https://user-images.githubusercontent.com/70505378/159172081-e29d52c2-e555-49e5-ba4a-c502b51e505b.png)

MLM에서는 문장의 15%의 단어들을 실제 단어 대신 다른 토큰으로 대체하고, 이 토큰을 masked token이라고 합니다. 15%라는 수치는 실험적으로 결정된 것으로, 너무 크거나 작으면 문제가 생길 수 있습니다. 

* 너무 크다면, 문맥을 충분히 학습하기에 어렵다. 
* 너무 작다면, 학습해야 할 문맥이 많아지므로 학습 속도가 느려지고 많은 리소스가 필요하다. 

![image-20220320175647952](https://user-images.githubusercontent.com/70505378/159172087-1f65820f-3c2a-4661-8d2a-3d190e4f275c.png)

처음에는 masked token에 [MASK]라는 토큰을 대신 넣어, BERT 모델이 실제 단어를 잘 맞히도록 학습되었습니다. 그러나 이 경우 fine-tuning.시에는 [MASK]라는 토큰이 등장하지 않기 때문에, 성능에 좋지 않은 영향을 미치게 됩니다. 

따라서 masked token에는 아래와 같은 토큰을 넣는 것으로 개선하였습니다. 

* 80%는 [MASK] 토큰을 사용
  * went to the store à went to the [MASK]  
* 10%는 랜덤한 단어 토큰을 사용
  * went to the store à went to the running  
* 10%는 실제 단어 토큰을 사용(맞는 단어는 그대로 예측하도록)
  * went to the store à went to the store  

### Next Sentence Prediction(NSP)

단어 단위로 실제 있어야 하는 단어가 무엇인지 맞히는 masked languate modeling 이외에, 문장 레벨에서의 task에 대응하기 위한 **Next Sentence Prediction**이라는 기법도 pretraining 시 사용합니다. 

NSP에서는 문장 2개를 하나의 단위로, 뒷 문장이 앞 문장에 이어서 나오는 문장이 맞는지 binary classification을 수행합니다. 

![image-20220320175606415](https://user-images.githubusercontent.com/70505378/159172085-9d91f648-8524-40dd-80b0-b48301e60552.png)

[CLS] 토큰은 classification 토큰으로 하나의 분류 단위를 나타내고, [SEP] 토큰은 separation 토큰으로 하나의 문장 단위를 나타냅니다. 

### BERT Fine-tuning

BERT 모델은 다양한 downstream language task에 사용됩니다. 그 중, **Question answering task**에 대해 알아보도록 하겠습니다. 

![image-20220320184425040](https://user-images.githubusercontent.com/70505378/159172089-02713a92-ac55-488c-bd80-ae398cceed02.png)

Question answering에서 사용되는 데이터셋으로 **SQuAD**(Stanford Question Answering Dataset) 데이터셋이 있습니다. SQuAD 1.0은 글 속에 답이 있는 문제만 있는 데이터셋이고, SQuAD 2.0은 글 속에 답이 없는 문제도 추가된 데이터셋입니다. 

아래 그림은 SQuAD 2.0 데이터셋입니다. 

![image-20220320185117546](https://user-images.githubusercontent.com/70505378/159172092-d49ccf43-d77c-4d57-b70d-7adafb5e835f.png)

이 데이터셋을 이용해 모델을 학습시키는 과정은 아래와 같습니다. 

* 글 sequence와 질문 seqence를 준비합니다. 
* 모델의 입력은 `[cls] 글 [sep] 질문 [sep]`와 같이 주어집니다. 
* 모델은 글의 각 토큰들 마다 embedding vector를 출력하고 두 가지 classification task를 수행합니다. 
  * 정답에 해당하는 sequence의 시작 위치
  * 정답에 해당하는 sequence의 종료 위치
* 모델의 예측과 ground truth로 주어지는 정답의 시작-끝 위치를 비교하여 softmax loss로 학습을 수행합니다. 

또는, 아래와 같이 문장 뒤에 올 문장을 여러 개 후보 중에서 선택하는 task에도 적용할 수 있습니다. 

아래 문제의 경우, 학습은 각 4개 정답 후보에 대해 Next Sentence Prediction을 수행하는 것으로 이루어집니다. 

![image-20220320190222413](https://user-images.githubusercontent.com/70505378/159172094-aeda9a95-1225-4de5-8529-1c422afd5c91.png)

### BERT Summary

* **Model Architecture**

  * BERT BASE: L = 12, H = 768, A = 12
  * BERT LARGE: L = 24, H = 1024, A = 16
  * L은 self-attention block의 개수, H는 hidden state vector의 차원수, A는 multi head의 개수

* **Input Representation**

  * WordPiece embeddings (30,000 WordPiece)
    * 위에서는 '단어'라는 표현을 썼지만, BERT에서는 실제로 subword라는 토큰 단위를 사용합니다. 
  * Learned positional embedding
    * Transformer에서 사용하는 positional embedding은 미리 정해진 값을 더해주는 형태였다면, BERT에서는 positional vector 마저도 학습이 가능하게 했습니다. 
  * [CLS] – Classification embedding
  * Packed sentence embedding [SEP]
  * Segment Embedding
    * positional embedding이 단어 단위의 위치를 알려주는 기법이라면, segment embedding은 문장 단위의 위치를 알려줍니다. 

  ![image-20220320181303896](https://user-images.githubusercontent.com/70505378/159172088-7e9a796d-e088-4baf-a709-c31f18de4940.png)

* **Pre-training Tasks**

  * Masked Language Modeling
  * Next Sentence Prediction  

**BERT VS GPT-1**

* **Training data size**
  * GPT is trained on BookCorpus(800M words) ; BERT is trained on the BookCorpus and Wikipedia (2,500M words)  
* **Training special tokens during training**
  * BERT learns [SEP],[CLS], and sentence A/B embedding during pre-training  
* **Batch size**
  * BERT – 128,000 words ; GPT – 32,000 words  
* **Task-specific fine-tuning**
  * GPT uses the same learning rate of 5e-5 for all fine-tuning experiments; BERT chooses a task-specific fine-tuning learning rate.  

**Ablation Study**

BERT 연구진은 모델의 파라미터 수를 늘리고 모델의 크기를 키울수록 성능이 계속해서 좋아진다는 것을 발표하며, 리소스가 부족한 연구자들이 절망감을 느끼게 하기도 하였습니다 ㅎㅎ...

![image-20220320190422207](https://user-images.githubusercontent.com/70505378/159172095-f9547cce-f3e6-4e8c-a66e-9bf607bd2cee.png)

<br>

## GPT-2

`GPT-2`는 GPT 모델의 두번째 버전으로, 모델 구조 상에서의 특별한 변화는 없고 사용한 데이터셋에 대한 이야기가 많습니다. 

* 40GB의 텍스트 데이터로 학습
  * Reddit이라는 질의응답 사이트에서 잘 정제된 글들을 가져와 모델을 학습시켰습니다. 
* Classification task를 Question Answering task로 치환시키려는 시도를 했습니다. 
  * 가령 "I love this movie"라는 문장이 positive인지 negative인지 분류해야 하는 문제를 풀려고 한다면, 이 문제를 "Is this sentence positive or negative?"라는 질문에 대한 답을 생성하도록 하는 문제로 치환시켰습니다. 
  * 이를 이용하면 question answering, summarization, translation 등의 task를 별도의 dataset이나 fine-tuning 과정 없이도 바로 수행할 수 있습니다. 이를 zero-shot setting이라고 합니다. (그러나 여전히 Fine-tuning을 했을 때보다 성능이 떨어지기는 합니다)
* BPE(Byte Pair Encoding)를 사용했습니다. 
  * Subword 토큰화와 유사하게, 단어를 잘게 쪼개 많은 단어들을 효과적으로 표현할 수 있도록 하였습니다. 
* Modifications
  * Layer normalization 층이 각 sub-block의 입력 부분으로 이동했습니다. 
  * 추가적인 layer normalization 층이 최종 self-attention block 이후에 삽입되었습니다. 
  * 더 깊은 층으로 갈수록 residual layer의 initialization 값을 더 작게 했습니다. 
    * 정확히 말하면 layer의 층 수를 n이라고 할 때 1/root(n) 값으로 가중치를 초기화했습니다. 이는 더 깊은 층을 지날 때는 앞선 층들과 비교해 많은 것을 학습하거나 바꿀 필요가 없다는 데에 착안하였습니다. 

아래 그림은 입력으로 글의 첫 문단을 주었을 때 GPT-3가 만들어낸 글입니다. 

![image-20220320233548701](https://user-images.githubusercontent.com/70505378/159172066-40c68f64-c2d9-407e-aa85-decfabc562e2.png)

<br>

## GPT-3

`GPT-3` 모델은 GPT-2에 비해 훨씬 많아진 파라미터 수와 커진 모델 규모를 사용함으로써 그 성능을 한 층 더 끌어올렸습니다. GPT-3는 96개의 self-attention layer를 사용했고, 무려 3.2M이라는 batch size를 사용했습니다. 

![image-20220320233945965](https://user-images.githubusercontent.com/70505378/159172068-526489f1-94c8-499f-8495-4dea93c1f997.png)

### Few-Shot Learner

앞서 GPT-2에서 zero-shot setting에 대한 언급을 했었습니다. GPT-3가 보여준 놀라운 결과 중 하나로 few-shot setting을 들 수 있습니다. 

Zero-shot setting이 별도의 학습 데이터 전혀 없이 바로 다른 task를 language generating task로 치환시켜 푸는 것 처럼, GPT-3는 하나의 example을 제시하는 one-shot setting과 몇 개의 example을 제시하는 few-shot setting에 대한 실험을 진행했습니다. 

![image-20220320234924945](https://user-images.githubusercontent.com/70505378/159172069-459721ed-a169-4a28-8231-858595aa740a.png)

놀라운 것은, 이렇게 몇 개의 example을 제시해주는 것 만으로 모델의 성능이 점점 더 가파르게 올라간다는 것입니다. 

![image-20220320234954767](https://user-images.githubusercontent.com/70505378/159172070-08d6f022-87b5-430d-94c3-c72abe06c650.png)

<br>

## ALBERT

다음으로 소개할 것은 `ALBERT`(A Lite BERT)입니다. 

앞에서의 모델 발전사를 보면, 그 형태가 점점 더 많은 파라미터 수를 가지는 모델로 발전했다는 것을 알 수 있습니다. 당연하게도, 모델이 점점 더 커진다면 그만큼 많은 메모리가 필요하고, 학습 속도도 느려지게 됩니다. 

ALBERT는 BERT의 경량화된 버전으로서, 그 성능은 유지하면서도 모델의 크기를 줄인 모델입니다. 

### Factorized Embedding Parameterization

ALBERT에서 사용한 첫번째 기법은 `Factorized Embedding Parameterization`입니다. 

Self-attention layer를 사용하는 모델의 경우, 그 입력과 출력의 dimension을 동일하게 맞춰줘야 합니다. 이때 그 hidden state dimension이 크다면, 그만큼 필요한 파라미터 수와 연산량도 증가하게 됩니다. 

* `V`: Vacabulary size
* `H`: Hidden-state dimension
* `E`: word Embedding dimension

BERT가 V개의 단어를 embedded vector로 만들기 위해 H dimension을 사용한다면, 총 V \* H 크기의 파라미터 수가 필요합니다. 

ALBERT는 이 hidden state dimension을 먼저 E로 설정하고 embedded vector를 구한 다음, residual connection에서 input vector의 size와 맞춰주기 위해 H dimension으로의 변환을 한 번 더 적용함으로써 파라미터 수를 획기적으로 줄였습니다. 

![image-20220321001042605](https://user-images.githubusercontent.com/70505378/159172071-1fd1d322-83e1-4c32-8663-a3925ad62f11.png)

BERT에서 하나의 embedding layer가 VH 개의 파라미터 수를 가진다면, ALBERT의 embedding layer는 E(V+H) 개의 파라미터 수를 가지게 됩니다. 

### Cross-layer Parameter Sharing

모델 내의 각각의 self-attention layer들은 개별적인 parameter를 가지고 있습니다. `Cross-layer Parameter Sharing`은 이 각 층의 parameter 들을 공유하여 사용한다는 것입니다. 즉, 총 12개의 self-attention layer 각각이 8개의 multi-head를 사용한다면 기존에는 8x12  만큼의 parameter set가 필요했던 것을, ALBERT에서는 8개의 parameter set만을 이용해 각 층이 sharing하도록 했습니다. 

ALBERT는 이 sharing을 다음 세 가지 방법으로 시도해보았습니다. 

* Shared-FFN
* Shared-Attention
* All-Shared

그 결과는 아래와 같습니다. 어느 정도 성능 하락이 발생하긴 하지만, 그 파라미터 수의 감소폭과 비교한다면 하락폭이 크지 않은 것을 볼 수 있씁니다. 

![image-20220321002348569](https://user-images.githubusercontent.com/70505378/159172072-a59da33a-14eb-4805-9f66-d6747f7fbd1d.png)





### Setence Order Prediction

BERT 모델은 Masked Language Modeling과 Next Sentence Prediction이라는 pretraining task를 수행하여 학습하였습니다. 

하지만 그 이후 진행된 연구들은 Next sentence prediction을 수행하지 않은 모델과 수행한 모델 사이에 큰 성능 차이가 없다는 것을 근거로 들며, Next sentence prediction이 pretraining task에 있어 큰 실효성이 없다는 지적을 하였습니다. 

ALBERT에서는 이를 `Setence Order Prediction`이라는 새로운 pretraining task로 대신했습니다. Setence Order Prediction은 항상 연속된 두 문장을 입력으로 주되, 두 문장의 순서를 뒤바꿔가며 모델이 그 순서가 맞는 순서인지 아닌지를 맞히도록 하는 binary classification task입니다. 

![image-20220321003345662](https://user-images.githubusercontent.com/70505378/159172073-19feae97-7206-41cf-94ff-9c0f22558b3d.png)

<br>

아래 표는 다양한 데이터셋을 포함하는 benchmark dataset인 GLUE에 대한 다양한 모델들의 평가표입니다. 모델의 크기를 상당히 줄였음에도, ALBERT가 가장 좋은 성능을 보여주는 것을 확인할 수 있습니다. 

![image-20220321003751815](https://user-images.githubusercontent.com/70505378/159172076-38711d8b-61bb-43d6-b978-afce4c18a13a.png)







<br>

## ELECTRA

다음으로 소개할 pretraining model은 `ELECTRA`(Efficiently Learning an Encoder that Classifies Token Replacements Accurately) 모델입니다. 

ELECTRA는 기존의 BERT나 GPT 모델들처럼 적층된 self-attention layer들로 구성된 모델 구조를 보이지만, 조금 다른 형태로 pretrain된 모델입니다. 

ELECTRA의 pretraining 과정에서는 masked language model에 해당하는 **Generator** 모델과 앞에서 생성된 각 단어들을 original/replaced로 이진 분류하는 **Discriminator** 모델을 동시에 사용하여 적대적으로 학습시킵니다. 

![image-20220321004652561](https://user-images.githubusercontent.com/70505378/159172078-dd8ae7d5-4ffe-423c-8083-3eb7fefd643f.png)

두 모델 중 실제 pretrained model로 사용되는 것은 Discriminator에 해당하는 ELECTRA 모델입니다. 

ELECTRA는 실제로 같은 FLOPs 대비 다른 모델들 보다 더 나은 성능을 보여줍니다. 

![image-20220321005027504](https://user-images.githubusercontent.com/70505378/159172079-d262eb52-9982-4f1e-9cf2-331e112411ad.png)





<br>

## Afterward Researches

### Light-weight Models

많은 대규모의 모델들이 발전하면서, 동시에 이를 경량화하려는 시도들도 많이 진행되었습니다. 모델 경량화는 기존 모델의 성능은 비슷하게 유지하면서, 모델 크기를 줄이거나 학습 속도를 빠르게 하는 데에 그 목적이 있습니다. 

모델 경량화 기법 중 대표적인 Knowledge distillation을 적용하여 BERT 모델을 경량화한 `DistillBERT`, `TinyBERT` 등의 모델도 발표되었습니다. 

### Fusing Knowledge Graph into Language Model  

또 다른 최신 연구의 흐름은 기존의 MLM 모델에 knowledge graph를 결합하여 외부적인 정보를 모델에 결합하려는 시도입니다. 

Knowledge graph란 단어들 간의 관계를 그래프 형태로 모식화한 구조로, MLM 모델들이 데이터로부터 학습하지 못 한 부분들을 추가적으로 알려줄 수 있는 역할을 할 수 있습니다. 

예를 들어 다음의 두 문장이 있다고 하겠습니다. 

* 땅을 파서 꽃을 심었다. 
* 땅을 파서 집을 지었다. 

위의 두 문장에서 모델은 '무엇으로 땅을 팠는지'에 대한 정보를 학습하지 못 하지만, 이를 지식 그래프로부터 추가적인 정보를 전달받아 예측할 수 있습니다. 

이러한 식의 '모델이 학습하지 못 한 추가적인 외부 정보'를 지식 그래프가 전달해줌으로써, 성능을 한 층 더 높이려는 `ERNIE`나 `KagNET`과 같은 연구들이 활발히 진행되고 있습니다. 











<br>

<br>

# 참고 자료

* GPT-3
  * https://blog.openai.com/language-unsupervised/
* BERT : Pre-training of deep bidirectional transformers for language understanding,
  NAACL’19
  * https://arxiv.org/abs/1810.04805
* SQuAD: Stanford Question Answering Dataset
  * https://rajpurkar.github.io/SQuAD-explorer/
* SWAG: A Large-scale Adversarial Dataset for Grounded Commonsense Inference
  * https://leaderboard.allenai.org/swag/submissions/public
* How to Build OpenAI’s GPT-2: “ The AI That Was Too Dangerous to Release”
  * https://blog.floydhub.com/gpt2/
* decaNLP
  * https://decanlp.com/  
* GPT-2
  * https://openai.com/blog/better-language-models/
  * https://cdn.openai.com/better-language
    models/language_models_are_unsupervised_multitask_learners.pdf
* Language Models are Few-shot Learners, NeurIPS’20
  * https://arxiv.org/abs/2005.14165
* Illustrated Transformer
  * http://jalammar.github.io/illustrated-transformer/
* ALBERT: A Lite BERT for Self-supervised Learning of Language Representations, ICLR’20
  * https://arxiv.org/abs/1909.11942
* Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer, JMLR’20
  * https://arxiv.org/abs/1910.10683
* ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators, ICLR’20
  * https://arxiv.org/abs/2003.10555  
* DistillBERT, a distilled version of BERT: smaller, faster, cheaper and lighter
  * https://arxiv.org/abs/1910.01108
* TinyBERT: Distilling BERT for Natural Language Understanding, Findings of EMNLP’20
  * https://arxiv.org/abs/1909.10351
* ERNIE: Enhanced Language Representation with Informative Entities
  * https://arxiv.org/abs/1905.07129
* KagNet: Knowledge-Aware Graph Networks for Commonsense Reasoning
  * https://arxiv.org/abs/1909.02151  

