---

layout: single
title: "[AITech] 20220209 - Generative Model II"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

# 학습 내용

이번 포스팅에서는 지난 [Generative Model II] 포스팅에서 얘기한 것처럼,  대표적으로 사용되는 Generative model인 VAE와 GAN에 대해 수식을 위주로 알아보겠습니다. 

## VAE

### VAE and AE

강의에서는 먼저 이 질문을 먼저 던집니다. 

> "AutoEncoder는 generative model일까요?"

어떻게 생각하시나요? VAE가 생성 모델인 것처럼, AE도 생성 모델이라고 생각하시나요?

간단하게 VAE와 AE가 무엇인지부터 얘기해보겠습니다. 

AutoEncoder는 입력 `x`와 동일한 출력 `y`를 만들어내기 위해 학습되는 네트워크입니다. AutoEncoder 모델 내부의 latent space에서는 입력 x와 같은 출력 y를 만들기 위해 그 **'값'**을 저장합니다. 

반면 Variational AutoEncoder는 모델 내부의 latent space에서 입력 x들에 대한 **'확률 분포'**를 생성합니다. 

바로 이것이 AE와 VAE가 다른 점이자, 생성 모델이 아닌 AE에서 생성 모델인 VAE로 나아갈 수 있도록 해 준 부분입니다. 지난 포스팅에서 이야기했던 것 기억나시나요? Generative model이란, 입력 x의 확률 분포를 모델링하는 모델입니다. 따라서 VAE는 generative model이고, AE는 아닌 것이죠. 

![image-20220210102804040](https://user-images.githubusercontent.com/70505378/153326503-a7b7e579-eaa7-453a-b8f6-4a3d9107c69a.png)

### Objective of VAE

자, 이제 VAE 얘기를 해봅시다. VAE에서 하는 추론을 **Variational inference(VI)**라고 하는데요, 이 VI의 목적은 입력 데이터 x의 확률 분포인 posterior distribution `p`에 가장 잘 match하는 모델링 확률 분포 variational distribution `q`를 찾는 것입니다. 다시 말하면, 이는 **variational distribution과 posterior distribution 사이의 KL divergence를 최소화하는 것**이라고 할 수 있습니다. 

그런데 잠깐. 뭔가 이상하지 않나요?

목표 확률분포인 p(z|x)를 모르는데, 어떻게 그 확률분포에 가까운 q(z|x)를 찾아가죠? (모르는데 어떻게 가요;; 😢)

바로 여기서 사용할 수 있는 방법이 있고, 아래와 같이 수식이 전개됩니다. 

![image-20220210103609537](https://user-images.githubusercontent.com/70505378/153326506-14a03c82-4d6c-4ffa-b782-62420cc56706.png)

데이터의 확률 분포는 'ELBO(Evidence Lower Bound)' 항과 'KL Divergence' 항으로 나눠질 수 있습니다. 그리고 바로 이 ELBO 항은 tractable하기 때문에, 우리는 **ELBO 항을 최대화하는 것의 반대급부로 KL Divergence 항이 최소화되는 효과**를 얻을 수 있습니다. 

그리고 ELBO 항은 다시 아래와 같이 나눠질 수 있습니다. 

![image-20220210103902153](https://user-images.githubusercontent.com/70505378/153326508-fe32d418-bafc-4bdf-ab13-0bbec29ba2ac.png)

ELBO는 'Reconstruction Term'과 'Prior Fitting Term'으로 나눠지고, 결국 이 두 항을 최소화하는 것이 VAE의 목표입니다. 즉, 바로 이 수식이 VAE의 **손실 함수**가 되는 것입니다. (이렇게 수학적으로 모델 손실 함수가 도출되는 걸 볼 때마다 너무 신기...)

### Limitation of VAE

VAE에는 몇 가지 한계점이 있는데요, 그에 대해 살펴봅시다. 

* VAE는 intractable model이다. (가능도를 계산하기 어렵다)

  * 따라서 implicit model이다. 

* Prior fitting term은 미분이 되는 모양이어야 하는데(손실 함수이기 때문에), 이것이 많은 distribution 형태를 사용하지 못 하게 막는다. 

* 따라서 대부분, VAE에서는 'isotropic Gaussian'(모든 output distribution이 independent한 gaussian distribution)을 사용한다. 

  ![image-20220210104605558](https://user-images.githubusercontent.com/70505378/153326509-f4ad401f-7567-4240-bd4f-e8f2fca70f20.png)

다른 확률분포를 모델링하고 싶음에도 불구하고, VAE에서는 그 제약사항 때문에 Isotropic Gaussian 만을 사용해야 하는 것입니다. 

### AAE

**AAE(Adverserial Auto Encoder)**는 이러한 VAE의 한계를 극복합니다. 그 방법에 대해 여기서 논하지는 않지만 아래 참고자료에 논문 링크를 달아놨으니 궁금하신 분들은 한 번 보시면 좋을 것 같습니다. 

간단하게 말하면, AAE는 GAN 구조를 이용하여 prior fitting term을 GAN의 objective로 전환합니다. 이는 곧 sampling만 가능하다면(미분이 가능하지 않아도), 그 어떤 distribution 형태도 사용할 수 있음을 의미합니다. 

![image-20220210105030896](https://user-images.githubusercontent.com/70505378/153326513-463abd2b-d563-4e74-8310-c09c245712ff.png)

따라서 AAE는 우리가 그 어떤 임의의 distribution도 sampling만 가능하다면 사용할 수 있도록 해줍니다. 



<br>

## GAN

GAN입니다. 많은 분들이 Generative model하면 GAN을 많이 떠올리실 것 같습니다. 가장 매력적이고 강력한 네트워크니까요. 

이제는 어느정도 진부하게 느껴질 정도지만, GAN 구조를 설명할 때는 '지폐 위조범과 이를 구별해내는 경찰의 싸움'이라는 이야기를 많이 하죠. 즉 실제 이미지와 유사한 가짜 이미지를 만들어내는 'Generator'와 실제 이미지와 가짜 이미지를 구별해내는 'Discriminator'의 싸움입니다. GAN의 가장 큰 장점은 바로 이 구별해내는 Discriminator도 가만히 있는 것이 아닌, 학습을 한다는 것이죠. 이로써 Generator와 Discriminator의 성능은 동시에 함께 올라갑니다. 

아래는 GAN의 구조를 VAE의 구조와 비교하여 나타낸 그림입니다. 

![image-20220210105706197](https://user-images.githubusercontent.com/70505378/153326516-d79c9b89-686a-4b0d-afde-3d88adf1fbd5.png)



### Objective of GAN

GAN이 학습하는 과정은 discriminator와 generator 사이의 minmax game이라고 할 수 있습니다. 

![image-20220210110320764](https://user-images.githubusercontent.com/70505378/153326525-48363649-f1de-4e00-ae66-7c37fa77fcdb.png)

#### Discriminator

Discriminator의 objective는 아래와 같습니다. 

![image-20220210105904354](https://user-images.githubusercontent.com/70505378/153326518-73129960-1f16-40e7-b7cf-06cba63a7685.png)

그리고 이 식에서 Optimal discriminator에 대한 식을 뽑아내면 다음과 같습니다. 

![image-20220210105933242](https://user-images.githubusercontent.com/70505378/153326521-6c066b8b-577d-4d20-ade0-9efc689719cb.png)



#### Generator

Generator의 objective는 아래와 같습니다. 

![image-20220210105953373](https://user-images.githubusercontent.com/70505378/153326522-6e3004a1-f6aa-4c69-a0e7-8c1f42f3f262.png)

그리고 이 식에 위에서 구한 Optimial discriminator에 대한 식을 D(x)에 집어넣으면 식은 아래와 같이 전개됩니다. 

![image-20220210110100041](https://user-images.githubusercontent.com/70505378/153326523-8ba38e6a-c0ad-4885-94e4-e8803a74f459.png)

결국에, GAN의 objective는 실제 데이터 distribution과 모델링 distribution 사이의 JSD(Jenson-Shannon Divergence)를 최소화하는 것이라고 할 수 있습니다. 





### Various types of GAN

**DCGAN**

지금은 많이 사용되지 않는 고전적인 GAN 구조이지만, GAN이 발전하는 데 있어 많은 인사이트를 제공해준 네트워크입니다. 

![image-20220210110457113](https://user-images.githubusercontent.com/70505378/153326527-ad5ff3d9-8121-4ce1-8009-4c57939c37bd.png)

**Info-GAN**

어떤 무작위의 새로운 이미지를 생성해내는 것이 아니라, 내가 원하는 특정 class에 집중하여 생성할 수 있도록 하는 네트워크입니다. 

![image-20220210110548076](https://user-images.githubusercontent.com/70505378/153326528-a8005d4f-f96c-4b11-bb8e-44e67f655395.png)

**Text2Image**

주어진 텍스트로부터 이미지를 생성해내는 GAN 네트워크의 종류입니다. 

![image-20220210110615831](https://user-images.githubusercontent.com/70505378/153326529-15043793-d69f-4f8e-82ce-0d277b9c3166.png)

**Puzzle-GAN**

이 강의를 하신 교수님께서 참여한 논문이라고 하는데, 이미지의 일부분만으로 다시 원본 이미지를 복원해내는 네트워크입니다. 

![image-20220210110719581](https://user-images.githubusercontent.com/70505378/153326531-c27f3c83-00b2-4411-8508-e62697de17ee.png)

**Cycle GAN**

버클리 대학의 우리나라 대학원생 분이 저자로 참여한 논문으로, 이미지 사이의 domain 변화를 할 수 있는 유명한 GAN 네트워크 중 하나입니다. 

![image-20220210110753764](https://user-images.githubusercontent.com/70505378/153326533-72f1911e-eea1-4996-9fdf-e012e4e3f0ac.png)

GAN의 구조를 2개 사용하여, **Cycle-consistency loss**를 사용한다고 합니다. 이 cycle-consistency loss는 중요하다고 강조하셨으니, 한 번씩 찾아보시면 좋을 것 같습니다. 

![image-20220210110851274](https://user-images.githubusercontent.com/70505378/153326538-3d106a3b-8b65-4974-8726-ce2261ca5c09.png)

**Star-GAN**

저자가 우리나라 분들로만 구성된 논문입니다. 고려대학생이신 분도 참여한 논문이라고 하는데, 정말 대단하신 것 같네요... 아래 사진처럼 사람의 특정 부분을 변화시킬 수 있는 네트워크입니다. 

![image-20220210111035720](https://user-images.githubusercontent.com/70505378/153326540-adc3fe06-fe88-4e08-9ee6-72a0f93efe65.png)

**Progressive-GAN**

이미지를 한 번에 생성해내는 것이 아닌 서서히 조금씩 생성하여, 결국에는 아주 고차원의 세밀한 이미지까지 생성해낼 수 있는 네트워크입니다. 

![image-20220210111150868](https://user-images.githubusercontent.com/70505378/153326545-6c8bb85b-878b-4126-8e9e-61671164d85d.png)



**and Way More**

GAN과 관련한 논문의 숫자는 해가 갈수록 늘어나고 있고, 2018년 말에는 거의 한 달에 500편의 논문이 게재되었다고 하니 이 모든 GAN을 아는 것은 불가능하거니와 불필요합니다. 

다만, 기본적인 GAN의 구조와 목적을 이해하면 다른 GAN 네트워크를 볼 때 한결 수월할 것입니다. 



<br>

<br>

# 참고 자료

* D. Kingma, "Variational Inference and Deep Learning: A New Synthesis," Ph.D. Thesis  
* [Adverserial Auto Encoder 논문](https://arxiv.org/abs/1511.05644)
* [Generative Adverserial Network 논문](https://arxiv.org/abs/1406.2661)
  * [DCGAN 논문](https://arxiv.org/abs/1511.06434)
  * [Info-GAN 논문](https://arxiv.org/abs/1606.03657)
  * [Text2Image 논문](https://arxiv.org/abs/1605.05396)
  * [Puzzle-GAN 논문](https://arxiv.org/abs/1703.10730)
  * [CycleGAN 논문](https://arxiv.org/abs/1703.10593)
  * [Star-GAN 논문](https://arxiv.org/abs/1711.09020)
  * [Progressive-GAN 논문](https://arxiv.org/abs/1710.10196)

















<br>
