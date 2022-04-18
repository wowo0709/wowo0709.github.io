---
layout: single
title: "[Papers][CV][Face Clustering] Face Clustering 동향 분석"
categories: ['AI', 'AI-Papers']
tag: []
toc: true
toc_sticky: true
---



<br>

# Face Clustering 동향 분석

이번 포스팅은 `Face Clustering`에 대한 내용을 다뤄보려 합니다. 

하나의 논문을 깊게 다루기보다는 몇 개 논문의 핵심 아이디어를 살펴보고, 그 흐름이 어떤 식으로 발전하고 있는지에 대해 알아보려 합니다. 

본 포스팅에서 다루는 논문은 아래와 같습니다. 

* Linkage Based Face Clustering via Graph Convolution Network, [https://paperswithcode.com/paper/linkage-based-face-clustering-via-graph](https://paperswithcode.com/paper/linkage-based-face-clustering-via-graph) (Code Available)
* Video Face Clustering with Unknown Number of Clusters, [https://paperswithcode.com/paper/video-face-clustering-with-unknown-number-of-clusters](https://paperswithcode.com/paper/video-face-clustering-with-unknown-number-of) (Code Available)
* FaceMap: Towards Unsupervised Face Clustering via Map Equation, [https://paperswithcode.com/paper/facemap-towards-unsupervised-face-clustering](https://paperswithcode.com/paper/facemap-towards-unsupervised-face-clustering) (Code Available)
* Self-Supervised Video-centralised Transformer for Video Face Clustering, [https://arxiv.org/pdf/2203.13166.pdf](https://arxiv.org/pdf/2203.13166.pdf) (Code Unavailable)

## Face Clustering

먼저 Face clustering이 무엇인지 간단히 알아보겠습니다. 

* Face Detection: 이미지에서 face의 위치를 찾는 task
* Face recognition: 이미지에서 face를 찾고, 미리 결정된 class에 따라 분류하는 task
* Face clustering: 여러 face 이미지들을 미리 결정된 class 없이 identity에 따라 clustering하는 task

즉 Face clustering은 여러 얼굴 이미지들이 있을 때, 각 이미지들로부터 feature를 추출하고 이 feature 간의 유사성을 기반으로 각 이미지를 clustering합니다. 여기서 생성되는 하나의 cluster는 하나의 identity(사람)에 해당합니다. 

![image-20220411153448718](https://user-images.githubusercontent.com/70505378/162919007-970298b5-0623-4a5e-8011-7767a84fe5f4.png)

Face Clustering을 위해서 여러 방법론들이 연구되어 왔고, 이는 크게 supervised learning과 unsupervised learning으로 나눌 수 있습니다. 이번 포스팅에서는 크게 supervised learning에 해당하는 GCN과 Ball Clustering을 사용한 방법, 그리고 unsupervised learning에 해당하는 Map Equation을 사용한 방법에 대해 알아봅니다. 

<br>

## Linkage Based Face Clustering via Graph Convolution Network

첫번째로 소개할 논문은 GCN을 사용한 2019년 4월 발표된 논문입니다. 

본 논문에서는 face image의 feature map 상에서 두 feature가 하나의 identity에 해당한다면, 두 node(face) 사이 edge가 존재한다는 것에 기반을 둡니다. 

기존의 face clustering 방법들(K-Means, Spectral Clustering, DBSCAN 등)은 **다양하고 불균형한 face image**를 clustering하는 데 충분한 성능을 보이지 못 했습니다. 이에 대한 가장 큰 이유는 기존의 방법들은 **데이터 분포에 제약 사항이 존재**하기 때문입니다. 제약 사항으로는 cluster가 convex shape이어야 한다거나, cluster 내의 instance 수가 비슷해야 한다거나, density가 비슷해야 한다는 등의 사항들이 있습니다. 

본 논문에서 제안하는 GCN based face clustering은 데이터 분포에 제약 사항을 두지 않음으로써 훨씬 뛰어난 성능을 보였다고 주장합니다. 

또한 기존의 방법들이 heuristic 적으로 데이터의 분포를 제한했다면, GCN은 데이터의 분포를 '학습'합니다. 

<br>

본 논문에서 제안하는 face clustering 과정은 다음과 같습니다. 

1. Clustering problem을 link prediction problem으로 치환
2. 각 instance마다 다른 모든 instance들이 아닌, 주변 instance 들에 대해서만 link prediction을 수행
   * 이 과정에서 생성된 그래프를 IPS(Instance Pivot Subgraph)라고 합니다. 
3. GCN을 이용해 IPS들을 연결

논문에서는 IPS를 생성하기 위해 ARO(Approximate Rank-Order Clustering) 알고리즘을 사용하며, ARO는 instance 주변 k개의 이웃 노드(kNN)와의 연결성 만을 고려하여 시간 복잡도를 크게 줄인 알고리즘입니다. 여기서 kNN은 확률에 기반한 ANN(Approximate NN)으로 대체되어 그 복잡도를 더욱 줄일 수 있다고 합니다. 

생성된 IPS들은 (spatial-based) GCN의 input으로 들어가 최종적인 link prediction이 수행됩니다. 

![image-20220411155826384](https://user-images.githubusercontent.com/70505378/162919016-6630dda7-ca7b-4502-9d85-e53eebb00ea0.png)

<br>

### Construction of Instance Pivot Subgraph

![image-20220412000537506](https://user-images.githubusercontent.com/70505378/162919084-c30a79b6-6bf7-4103-ad63-cd7098cde9d4.png)

IPS가 어떻게 만들어지는지 알아보겠습니다. 

**Step 1. Node discovery**

Pivot instance p에 대하여 h-hop 노드까지를 IPS로 구성합니다. i-th hop에 해당하는 이웃 노드들의 집합을 k<sub>i</sub>라 하면, h=3인 경우 집합 V<sub>p</sub>는 {k1, k2, k3}로 표현됩니다. 즉, V<sub>p</sub>에 자기 자신(p)은 포함되지 않습니다. 

**Step 2. Node feature normalization**

V<sub>p</sub>에 해당하는 노드 중 하나를 q라고 하고 q의 feature를 x<sub>q</sub>라고 하겠습니다. 

두번째 단계에서는 V<sub>p</sub> 의 모든 노드들의 feature에 p의 feature x<sub>p</sub>를 빼서 normalize합니다. 그리고 normalize를 완료한 각 노드의 feature 집합을 F<sub>p</sub>라고 합니다. 여기서 F<sub>p</sub>는 pivot node와 neighbor node 간의 차이를 인코딩하고 있다고 할 수 있습니다. 

![image-20220411174703399](https://user-images.githubusercontent.com/70505378/162919017-06efb37c-42fa-477e-a59d-3cce3e85603b.png)

**Step 3. Adding edges among nodes**

V<sub>p</sub> 내의 모든 원소에 대해, 가장 가까운 u개의 이웃 노드(uNN)를 정의합니다. 이때 이웃 노드를 정의하는 범위는 V<sub>p</sub> 안에서 하는 것이 아니라 전체 instance에 대해 수행합니다. 

V<sub>p</sub>에 속하는 임의의 원소 q에 대해 uNN에 속하는 노드 r이 V<sub>p</sub>에 속한다면, (q, r)을 연결하는 edge를 추가합니다. 

이 과정을 V<sub>p</sub> 내의 모든 원소에 대해 반복하여 생성된 그래프를 IPS G<sub>p</sub>(V<sub>p</sub>, E<sub>p</sub>)라고 합니다. IPS의 구조적 형태는 인접 행렬 형태인 A<sub>p</sub>(size = \|V<sub>p</sub>\|x\|V<sub>p</sub>\|)로 표현되고, feature는 위에서 구한 F<sub>p</sub>로 표현됩니다. 

즉, 하나의 IPS는 A matrix와 F matrix로 표현됩니다. 

<br>

### Graph Convolutions on IPS

이제 생성된 IPS를 GCN의 input으로 넣어줍니다. 이제부터 F는 X라고 표기합니다. 

Graph convolution layer의 연산은 아래와 같은 수식으로 표현할 수 있습니다. 

![image-20220411175809442](https://user-images.githubusercontent.com/70505378/162919019-762e3fb4-7ca3-4b76-bcc4-a4ec0f8e3ba1.png)

_Calculate G_

G는 A로 부터 구합니다. 이때 A로부터 G를 구하는 방법에는 크게 3가지 방법을 사용할 수 있습니다. 

* Mean Aggregation
* Weighted Aggregation
* Attention Aggregation

각각에 대한 설명은 생략하겠습니다. 다만 아래로 갈수록 모델 성능이 높아지긴 하는데, 추론 시간 증가에 비해 그 상승폭이 크지 않아서 논문에서는 Mean Aggregation을 사용했습니다. 

_Multiply G and X_

G와 X를 행렬곱합니다. 행렬곱한 결과는 IPS에 대한 취합된 정보들을 인코딩하고 있는 것으로 해석할 수 있습니다. 

_Concat X and GX_

다음으로 X와 GX를 feature dimension으로 concat합니다. 

_Multiply W(Linear layer)_

이때의 W는 learnable parameter입니다. 

_apply relu activation(σ)_

  계산된 행렬은 relu activation을 통과합니다. 

<br>

GCN 학습 시 back propagation에서는 1-hop neighbor들에 대한 오차값만 사용합니다. 이는 1-hop neighbors 만을 사용할 때 positive sample과 negative sample의 수의 균형이 가장 잘 맞기 때문이라고 합니다. 여기서 positive sample이란 실제로 pivot과 동일한 identity의 node들이고, negative sample은 다른 identity의 node들입니다. 

아래 그림을 보면 2개의 graph convolution layer를 사용할 때 iteration에 따라 positive node들과 negative node들이 잘 분류되는 것을 확인할 수 있습니다. 

![image-20220411181225491](https://user-images.githubusercontent.com/70505378/162919026-3493391b-ae9b-4b51-ad1f-71db0b258159.png)

<br>

### Link Merging

GCN을 거치고 나면, 각 IPS의 각 edge에 대한 가중치을 얻게 됩니다. 이 가중치가 높을수록 edge가 연결하는 두 sample이 동일한 identity를 가질 확률이 높은 것입니다. 

최종 clustering된 graph를 만들기 위해 가장 쉬운 방법은 threshold를 설정하여 threshold 이하의 edge들은 끊어버리는 것이지만, 이는 threshold 값에 큰 영향을 받습니다. 

본 논문에서는 이보다 더 진보된 방식인 **pseudo label propagation**을 사용했습니다. 해당 기법은 iteration을 반복하면서 매 iteration마다 특정 threshold 이하의 edge는 연결을 끊되, 특정 size 이상의 cluster의 경우 연결을 끊지 않는 것입니다. 

Iteration이 증가할수록 threshold 값 또한 증가하고, 특정 size 이상의 cluster가 더 이상 없으면 iteration을 종료합니다. 

<br>

### Evaluation

본 논문에서는 metric으로 NMI(normalized mutual information)와 BCubed F-measure를 사용했습니다. 

_NMI_

![image-20220411182308633](https://user-images.githubusercontent.com/70505378/162919032-3017c715-df0f-47ce-9968-24241091879d.png)

H는 impurity, I는 mutual information을 나타냅니다. 

_Bcubed F-measure_

L이 ground truth label, C가 cluster label이라고 할 때, 두 sample의 pairwise correctness score는 아래와 같습니다. 

![image-20220411182739694](https://user-images.githubusercontent.com/70505378/162919035-d0ae9d15-a37b-451e-9367-ee1c9fc94c8c.png)

그리고 Precision과 Recall은 아래와 같이 구할 수 있습니다. 

![image-20220411182805366](https://user-images.githubusercontent.com/70505378/162919039-775dbbcf-340e-4d1f-be37-7adbe7b1327f.png)

최종 F-measure는 P와 R의 조화평균입니다. 

![image-20220411182857993](https://user-images.githubusercontent.com/70505378/162919043-d145fe2a-2b08-483c-9f16-99d640185903.png)

결과적으로 Bcubed F-measure는 각 sample에 대한 precision과 recall 값의 조화평균입니다. 

<br>

IPS construction 시에 설정할 수 있는 hyperparameter로 **h** (number of hops), **k<sub>i</sub>** (number of picked nearest neighbors in each hop), **u** (number of linked nearest neighbors u for picking edges)가  있습니다. 

본 논문에서는 이에 대한 실험을 진행했고, 아래와 같은 관찰 결과를 얻었습니다. 

* h가 3 이상일 때부터는 성능 증가가 거의 없다. 
* k1이 증가하면 recall 값이 증가하고, k2가 증가하면 precision 값이 증가한다. 
  * k가 증가하면 성능은 증가하지만, 추론 시간 또한 증가한다. 
* u는 모델 성능에 큰 영향을 주지 않는다. 

![image-20220411183521973](https://user-images.githubusercontent.com/70505378/162919046-1af300b2-a046-4995-8fe3-b9e956c29684.png)

결과적으로 본 논문에서는 h=2, k1=80, k2=5, u=5를 선택했습니다. 

<br>

아래는 GCN-based face clustering과 다른 기법들을 비교한 성능 평가표입니다. 

![image-20220411183712710](https://user-images.githubusercontent.com/70505378/162919049-3e7d0ddf-5b39-4c64-a04f-d28146451ce7.png)

<br>

### Contributions

* GCN을 사용해 face clustering을 link prediction task로 치환
* 데이터 분포에 대한 제약을 두지 않음으로써 성능 향상(singleton cluster 감소)
* Multi-view data, Multi-modal data의 활용 가능성
* SOTA 성능 달성

<br>

### Codes

* 학습데이터: face feature file(binary data), label(plain text), knn file(selective)
* [https://github.com/yl-1993/learn-to-cluster](https://github.com/yl-1993/learn-to-cluster)
* [https://github.com/Zhongdao/gcn_clustering](https://github.com/Zhongdao/gcn_clustering)



<br>

<br>

## Video Face Clustering with Unknown Number of Clusters (BCL)

다음으로 소개할 논문은 같은 해인 2019년 8월 발표된 Ball Cluster Learning(BCL)을 사용한 face clustering 논문입니다. 

`BCL`이란, embedding space의 각 cluster를 동일한 크기의 ball로 정교화하는 과정입니다. 그 과정에서 모든 sample들은 하나의 ball 안에 위치하게 되고, 각 ball들은 overlap되지 않습니다. 

![image-20220411184841157](https://user-images.githubusercontent.com/70505378/162919054-a9a78f3f-f7a4-4744-8f10-81ea0a4d7c22.png)

본 논문의 최종 목표는 아래와 같습니다. 

> 정해진 cluster의 개수 없이, video 내의 모든 인물(잠깐 등장한 인물을 모두 포함해서)을 동일한 크기의 ball에 할당하는 것

각 ball의 크기(반지름)는 학습 과정에서 결정되고, 반지름의 크기에 따라 알고리즘의 종료 시점이 결정됩니다. 또한 BCL의 장점은 batch size와 number of clusters에 linear한 시간 복잡도를 가진다는 것입니다. 

<br>

### Constraints

Constraint는 similar case와 dissimilar case로 나뉩니다. 여기서 **Similar**하다는 것은 두 sample x<sub>i</sub>와 x<sub>j</sub>가 같은 label을 가진다는 것을 말합니다. k번째 set of similar samples(즉, cluster)를 C<sub>k</sub>라고 하겠습니다. 

_Similar case_

먼저 용어부터 정리하겠습니다. 

* **µ<sub>k</sub>**는 cluster C<sub>k</sub>의 중심 좌표입니다.
* **f<sub>i</sub>**는 x<sub>i</sub>의 l2 normalized embedding으로, CNN module을 통해 face image를 feature representation x<sub>i</sub> 로 변환한 후에 한 번 더 embedding한 값에 해당합니다. (이로부터 약 2.5%의 성능 향상을 이뤘다고 합니다)
* **d<sup>2</sup>**은 euclidean distance입니다. 

이때 C<sub>k</sub>에 속하는 모든 sample x<sub>i</sub>는 아래 식을 만족해야 합니다. 

![image-20220411225533690](https://user-images.githubusercontent.com/70505378/162919058-2ab424ff-e125-4c25-91c7-de7671729696.png)

원의 방정식(x<sup>2</sup> + y<sup>2</sup> = r<sup>2</sup>)을 떠올려보면 b = r<sup>2</sup>에 해당함을 어렵지 않게 알 수 있습니다. 이때 b는 learnable parameter입니다. 

따라서 C<sub>k</sub>에 속하는 두 sample x<sub>i</sub>, x<sub>j</sub>는 아래식을 만족해야 합니다. 

![image-20220411225725617](https://user-images.githubusercontent.com/70505378/162919061-8372595d-b98f-4bd4-8604-23b8f80375e3.png)

이로부터 **2r**을 두 sample의 similar constraint로 사용할 수 있음을 알 수 있습니다. 

_Dissimilar case_

위로부터, dissimilar sample 사이 거리는 2r을 넘어야 하는 것을 알 수 있습니다. 그리고 따라서, C<sub>k</sub>에 속하는 sample f<sub>i</sub>와 또 다른 cluster C<sub>v</sub> 의 중심좌표 µ<sub>v</sub> 사이의 거리는 **3r**을 넘어야 합니다. r<sup>2</sup> = b 이기 때문에 이는 아래와 같이 b에 관해 나타낼 수 있습니다. 

![image-20220411230614700](https://user-images.githubusercontent.com/70505378/162919070-fb3f1915-85d3-4257-a6ff-783bd16ad83c.png)

그리고 논문에서는 γ = 9b + ε  로 나타내어 최종적으로 두 sample의 dissimilar constraint를 아래와 같이 정리합니다. 

![image-20220411230540064](https://user-images.githubusercontent.com/70505378/162919068-fd442142-f4bc-4a3a-9241-86675cae2576.png)







<br>

### Problem Formulation

위에서 구한 constraint들을 만족하는 b를 구하기 위해 loss function을 아래와 같이 정의합니다. 

![image-20220411230910115](https://user-images.githubusercontent.com/70505378/162919072-1d9c4b1e-7917-4e19-b849-9659704d337a.png)

여기서 L<sub>sim</sub>은 similar samples에 대한 loss이고, L<sub>dis</sub>는 dissimilar samples에 대한 loss입니다. 

![image-20220411230949664](https://user-images.githubusercontent.com/70505378/162919075-71690fc4-3ec3-434e-b0f9-5aae3aec4b04.png)

![image-20220411230958652](https://user-images.githubusercontent.com/70505378/162919078-fb787745-63a2-4052-be51-61b657f47745.png)







<br>

### Clustering Algorithm

본 논문에서는 Clustering algorithm으로 HAC(Hierarchical Aggolmerative Clustering) method를 사용합니다. 

HAC에서 각 sample들은 자기 자신만을 포함하는 cluster에서 시작하여, iteration을 진행하면서 유사한 cluster들끼리 merge됩니다. 여기서 **유사하다**의 기준은 아래 수식을 만족하는 cluster입니다. 

![image-20220411231533629](https://user-images.githubusercontent.com/70505378/162919080-755a25e1-7831-4cd3-bd09-13688b065a88.png)

그리고 앞에서 설정한 similar constraint에 따라, τ = (2r)<sup>2</sup> = 4b 입니다. 

위 수식을 만족하는 cluster가 없으면 HAC 알고리즘은 종료됩니다. 





<br>

### Video Face Track Clustering with BCL

BCL을 이용해 Video Face Track Clustering을 학습시킬 때는 아래 과정을 따릅니다. 학습 데이터로는 video tracking data가 사용됩니다. 

학습 시에는, 각 track마다 일련의 face image들 중 하나의 image를 추출하여 사용합니다. 이때 face image를 feature representation으로 나타내기 위해서는 pretrained fixed CNN 이 사용됩니다. 

추론 시에는 track 내의 모든 face image들을 CNN을 통과시켜 feature representation으로 만들고 평균을 내서 사용합니다. 

학습 시에 주의해야 하는 것은 동일한 identity가 서로 다른 dataset(train/validation/test)에 포함되지 않도록 하는 것입니다. 



<br>

### Evaluation

Evaluation metric으로는 세 가지가 사용됩니다. 

1. `#CI`: number of predicted clusters
2. `NMI`: normalized mutual information (위에서 봤던 metric)
3. `WCP`: weighted clustering purity, 각 예측된 cluster에 속한 sample의 수가 실제 각 ground truth의 sample 수와 유사한 정도

clustering 성능은 CNN model structure에 큰 영향을 받지 않는 것을 확인했습니다. embedding dimension은 32 이상부터는 성능 향상이 없었으며, batch size는 performance와 computational cost 사이 trade-off를 고려하여 2000으로 설정했습니다. 

BCL은 다른 학습 방법들보다 더 나은 성능을 보여줬습니다. 

![image-20220411234336173](https://user-images.githubusercontent.com/70505378/162919083-547c8eda-8c72-4dbe-9877-6bc5d65679bc.png)



<br>

### Contributions

* Ball cluster learning으로 cluster의 크기를 모델이 알아서 학습하도록 함
* 다른 학습 방법들보다 더 나은 성능을 보임







<br>

### Codes

* 학습 데이터: video face tracking data
* [https://github.com/makarandtapaswi/BallClustering_ICCV2019](https://github.com/makarandtapaswi/BallClustering_ICCV2019)

















<br>

<br>

## FaceMap: Towards Unsupervised Face Clustering via Map Equation

마지막으로 살펴볼 논문은 저번 달에 발표된 따끈따끈한 논문입니다. 본 논문은 Map Equation을 사용하여 Unsupervised face clustering을 수행했고, SOTA 성능을 달성하였습니다. 

본 논문은 두 가지 관점에서 기존 연구들이 부족하다고 이야기합니다. 

**Algorithm 관점**

![image-20220412153430816](https://user-images.githubusercontent.com/70505378/162919087-b6ab0660-33b6-4d27-aaeb-143c5b42f790.png)

* 기존 방법들은 feature extractor의 feature representation을 그대로 사용하고, 따라서 결함이 존재한다. 
* Singleton cluster가 너무 많이 발생한다. 
* Supervised method는 데이터를 만들기 힘들고, Unsupervised method는 성능이 떨어진다. 

논문에서는 이에 대한 대응으로 아래와 같은 방법들을 제시합니다. 

* 완벽하지 않은 feature representation을 보완하기 위해 OD(Outlier Detection) module 사용
* Face clustering task를 non-overlapping community detection task로 치환
  * Minimizing the entropy of information flows on a network of images

**Metric 관점**

![image-20220412153440698](https://user-images.githubusercontent.com/70505378/162919091-a4f9a086-f215-471c-abf2-7e00eec86ec2.png)

* 기존 metric들은 **number of predicted clusters와 number of gt identity의 수를 제대로 비교 평가하지 못 한다.** 
* **Large-size cluster에 편향된 평가**
* 따라서 algorithm은 performance를 올리기 위해 어려운 sample은 cluster에 넣기보다 singleton cluster로 분류하는 경향이 강하게 됨

논문에서는 기존 metric들의 두 가지 문제점을 보완하기 위해 3가지의 새로운 metric을 제안합니다. 

* `R#I`: (N/N*) x 100%, ratio of identity number
  * closer to 100%, the better
* `R#S`: (Ns/N*) x 100%, ratio of singleton cluster number
  * smaller, the better
* `FI(θ)`: F-score with threshold
  * sensitive to the small-size clusters with false identities (FP)

<br>

이에 본 논문에서 제안하는 Pipeline은 아래와 같습니다. 크게 **Community Detection part**와 **Outlier Detection part**로 나눌 수 있고, 각 부분에 대해 살펴보겠습니다. 

![image-20220412153456382](https://user-images.githubusercontent.com/70505378/162919093-c3e07227-5cfe-41cf-b433-907d1536ff32.png)

### Face Clustering as Community Detection

![image-20220412172205567](https://user-images.githubusercontent.com/70505378/162919095-3f447d17-779f-4527-a437-477fb9c46e5b.png)

첫번째로 Community Detection part입니다. 

이 단계에서 input으로는 face image의 feature representation들이 들어오고, output으로는 set of communities M이 반환됩니다. 여기서 말하는 M은 cluster의 개수 C와 동일한 의미입니다. 

먼저 kNN 알고리즘을 이용해 directed affinity graph를 생성하고, 이 graph를 인접 행렬 형태로 표현합니다. 이 인접 행렬을 A라고 합니다. 행렬 A는 논문의 표현을 빌리자면 'information flows within image', 즉 두 이미지 사이 정보의 양을 나타내고, 이는 similarity, 즉 유사도로 해석할 수 있습니다. 

이후에 행렬 A의 각 행에 normalization을 적용하여 trasition probability matrix를 생성합니다. 이 행렬을 P라고 합니다. 

<br>

### Key Observations

본 논문에서는 생성된 행렬 P로부터 몇 가지 중요한 사실을 발견하였습니다. 아래 이미지에서 3개의 column은 서로 다른 3 개의 sample을 나타냅니다. 또한 그림에서 검은색 점은 positive sample, 주황색 점은 negative sample입니다. 

![image-20220412170904226](https://user-images.githubusercontent.com/70505378/162919094-54d4bec0-70e2-40bb-bf31-2c29b8607077.png)

본 연구에서는 먼저 행렬 P의 임의의 행 pi를 값을 기준으로 내림차순 정렬하였습니다. 그리고 이를 pr_i라고 했습니다. 그림에서 pr_i 에 해당하는 그래프들을 보면, 특정 구간 전후로 positive sample들과 negative sample들이 나뉘는 것을 알 수 있습니다. 이는 face image를 feature representation으로 바꾸는 feature extractor의 불완전성에 기인합니다. 

본 연구에서는 이렇게 연산의 복잡도만 증가시키는 negative sample들을 제거하려 했습니다. 하지만 pr_i의 그래프들을 보면 아시다시피, trasition이 발생하는 절대적 값의 기준은 보이지 않습니다. 예를 들어 그래프 (a), (b)의 경우 그 값이 상대적으로 크고, 그래프 (c)의 경우 그 값이 상대적으로 작으면서 transition point가 늦게 나타납니다. 이제 transition point가 나타나는 구간을 mixed region이라고 하겠습니다. 

하지만 여기서 규칙을 찾을 수 있는 것이 바로 **기울기**입니다. negative sample들이 나타나기 시작하는 구간을 보면, 값의 기울기가 수렴하는 것을 볼 수 있습니다. 그래프 (d), (e), (f)가 이에 해당합니다. 

뒤에서 보겠지만, Outlier detection module에서는 기울기에 기반한 z-score를 기반으로 negative sample들을 제거합니다. 

마지막 (j), (k), (l) 그래프들은 negtive sample들이 clustering에 미치는 실제 영향을 보여줍니다. Mixed region이후의 sample들을 포함할 시, recall은 거의 증가하지 않으면서 precision은 크게 떨어집니다. 이는 mixed region 이후에 많은 False Positive sample들이 clustering의 성능을 떨어트리고 있음을 보여줍니다. 





<br>

### FaceMap with Outlier Detection

![image-20220412172240236](https://user-images.githubusercontent.com/70505378/162919098-345ed85b-9759-42fa-8d87-5aa0ffeeff66.png)

그래서 Outlier detection module에서는 negative sample들의 연결을 끊어서 불필요한 연산을 줄이면서 clustering의 성능을 높입니다. 

그 과정은 아래와 같습니다. 

1. 모든 pi를 내림차순 정렬
2. pi의 1차 미분 계산
3. pi의 1차 미분과 window size 'w'에 기반하여 각 구간의 z-score 계산
4. 가공된 new transition probability map 출력

결과적으로 OD module은 출력으로 새로운 transition probability map을 출력합니다. 그리고 이로부터 새로운 graph를 생성하고, 이 graph는 아래의 objective function에 기반하여 최종 cluster를 예측하게 됩니다. 

![image-20220412172359995](https://user-images.githubusercontent.com/70505378/162919103-47dfbc56-27d5-4bce-b78f-38dba6be52a7.png)

Objective function에 대한 자세한 설명은 생략하겠습니다. 그 의미를 보면 앞쪽의 항은 '서로 다른 cluster 간 entropy'를, 뒤쪽의 항은 '하나의 cluster 내부의 entropy'를 나타냅니다. 

즉, 이 objective function을 최소화하는 것은 최종 생성된 map의 entropy를 최소화하여 안정적인 최적의 cluster를 찾는다는 의미입니다. 







<br>

### Evaluation

Evaluation은 크게 두 가지로 진행되었습니다. 

첫번째로는 FaceMap이 다른 기존의 알고리즘들보다 더 나은 성능을 낸다는 것을 보여주었고, 

![image-20220412172716339](https://user-images.githubusercontent.com/70505378/162919106-c3a9b891-4344-4d94-9851-c359f2d23bdb.png)

두번째로는 GCN 대신 Outlier Detection module을 사용하는 것이 성능 향상에 도움을 준다는 것을 보여줬습니다. 

![image-20220412172754695](https://user-images.githubusercontent.com/70505378/162919110-7006fb8b-c376-4887-a0e1-e4d0f188a653.png)





<br>

### Contributions

본 논문이 기여한 바는 아래와 같이 정리할 수 있습니다. 

* Face clustering task를 Map equation을 이용한 community detection task로 치환
* Labeled data가 필요 없는 unsupervised 방법
* 기존 supervised, unsupervised method들보다 뛰어난 성능
* Large-scale dataset에 대해서 좋은 성능
* Hyper parameter(k, w)에 robust한 성능
* Cluster의 수와 singleton cluster의 수를 포함하는 새로운 metric 제안







<br>

### Codes

FaceMap은 unsupervised method이기 때문에 따로 학습 데이터가 필요하지 않습니다. 

* [https://github.com/bd-if/facemap](https://github.com/bd-if/facemap)











<br>

<br>

## Further Works

Face clustering 분야의 연구는 unsupervised, self-supervised learning의 방향으로 발전하고 있습니다. 

또한 FaceMap 논문보다도 이후에 나온 다른 논문은 face clustering에 Transformer 모델을 도입하려는 시도를 했습니다. 

* Unsupervised, Self-supervised
* Using Transformer model
* [https://arxiv.org/pdf/2203.13166.pdf](https://arxiv.org/pdf/2203.13166.pdf)



















<br>

<br>

_**논문에 대한 내용은 여기까지입니다. 아래부터는 개인적으로 새롭게 알고 느끼게 된 부분들을 정리하는 부분입니다.**_

<br>

# 새롭게 알게 된 것들

## Vocabulary

| Vocabulary   | meanings     |
| ------------ | ------------ |
| transductive | 변환         |
| inductuve    | 귀납적인     |
| auxiliary    | 보조, 보조자 |
| morphed      | 변형된       |



<br>

## Domain-specific word



<br>

## Others















