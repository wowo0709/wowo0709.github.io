---
layout: single
title: "[AITech][CV] 20220315 - Part 6) Conditional Generative Model"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

_**본 포스팅은 POSTECH '오태현' 강사 님의 강의를 바탕으로 작성되었습니다. **_

# Conditional Generative Model



## Conditional generative model

Conditional generative model이란 **condition**이 주어졌을 때 그에 대응하는 output을 generate하는 모델을 말합니다. 

좀 더 자세하게 말해보겠습니다. 기존의 Generative model은 class X가 주어지면, random sampling을 통해 X에 해당하는 이미지를 생성해내는 식이었습니다. 이에 반해 Conditional generative model은 주어진 condition을 고려한 sampling을 통해 이미지를 생성해냅니다. 이는 사용자의 의도를 반영한 생성 모델로, 높은 활용 가능성을 가집니다. 

![image-20220315134209858](https://user-images.githubusercontent.com/70505378/158335650-1f6d1561-d123-4d11-9a9f-aea5563490c7.png)

Basic generative model과 Conditional generative model을 구조 상 비교해보면 아래와 같습니다. 달라지는 부분은 Conditional generative model의 입력에 condition **c**가 함께 주어진다는 것만 다릅니다. 

![image-20220315134755558](https://user-images.githubusercontent.com/70505378/158335657-03f76aae-f100-4fda-93fc-06ca4401cfb2.png)

<br>

**Super resolution**

Conditional GAN의 활용 분야는 매우 넓습니다. Super resolution, Image-to-Image translation 등의 CV task 뿐 아니라 Machine translation, Article generation with the title 등 NLP 분야에도 활용될 수 있습니다. 

그 중 **Super resolution**에 대해 살펴보도록 하겠습니다. 

Super resolution task는 말 그대로 low resolution의 이미지가 들어왔을 때 high resolution의 이미지를 생성해내는 task입니다. 처음에는 이 task를 위해 아래와 같은 regression model에 MAE(또는 MSE) loss를 사용하려는 시도가 있었습니다. 

![image-20220315140402133](https://user-images.githubusercontent.com/70505378/158335665-afb9877a-51cc-4573-84e5-dd034aa588bd.png)

하지만 MAE(MSE) loss를 사용하는 regression model은 real image와 유사한 이미지를 생성해내지 못했습니다. 모델은 loss를 최소화하는 방향으로 학습하기 때문에, regression model은 image data의 평균치를 계산하려는 경향이 강하게 학습되게 됩니다. 

반면 GAN 모델의 경우, real image와 fake image를 판별해내는 Discriminator가 있기 때문에, Generator는 image data의 평균치와 상관없이 real image와 유사한 이미지를 생성 해낼 수 있게 됩니다. 

![image-20220315140700333](https://user-images.githubusercontent.com/70505378/158335667-becdffbb-645b-4880-ad42-2109a3c7173e.png)

Super resolution을 위한 기본 모델인 SRGAN의 결과물을 보면, bicubic 보간법이나 SRResNet 모델보다 훨씬 realistic한 이미지를 생성 해내는 것을 볼 수 있습니다. 

![image-20220315141158624](https://user-images.githubusercontent.com/70505378/158335670-37f7f31c-416b-4310-8342-582bc93563cf.png)







<br>

## Image translation GANs

여기서는 Image translation을 수행하는 GAN 모델에 대해 살펴보겠습니다. Image translation이란, 아래와 같이 input image를 다른 도메인의 output image로 생성해내는 것을 말합니다. 

![image-20220315141802309](https://user-images.githubusercontent.com/70505378/158335674-fccfad67-d97c-45ff-84b5-235a93e5530b.png)

**Pix2Pix**

`Pix2Pix` 모델은 2017년 발표된 image translation을 위한 모델입니다. 

Pix2Pix의 특징은 **Loss function에 cGAN loss와 L1 loss를 함께 사용**한다는 것입니다. 앞에서 봤듯이, L1 loss 만을 사용하면 좋은 성능을 내지 못 하지만, Pix2Pix 모델에서 **L1 loss는 target data와 비슷한 이미지를 생성할 수 있도록 하는 가이드 역할**을 수행합니다. Real data와 비슷한 생성은 cGAN loss에 의해 학습되고, L1 loss는 target domain의 real data와 크게 벗어나지 않도록 하는 가이드 역할을 하는 것이죠. 

아래 결과 사진을 보면 L1+cGAN loss를 사용했을 때 real image 같으면서도 target domain에 가장 가까운 이미지가 생성된 것을 확인할 수 있습니다. 

![image-20220315142428562](https://user-images.githubusercontent.com/70505378/158335677-f8e0c496-fd40-4e78-8247-fc9e7534a03d.png)

**CycleGAN**

Pix2Pix 모델의 한계는 항상 Paired dataset이 준비되어야 한다는 것입니다. 하지만 항상 paired dataset을 준비하기는 어렵기 때문에 이를 해결하려는 시도가 이어졌고, 이를 해결한 모델이 바로 `CycleGAN`입니다. CycleGAN은 아래 구조와 같이 2개의 GAN 모델을 함께 사용합니다. 

![image-20220315143248882](https://user-images.githubusercontent.com/70505378/158335678-77a97da8-5499-43cc-b95b-b67abec51c86.png)

![image-20220315144324406](https://user-images.githubusercontent.com/70505378/158335691-22bba340-3063-42fa-9b09-00d6d5a03850.png)

Cycle GAN은 image와 image 사이 관계를 학습하는 것이 아닌, domain과 domain 사이 관계를 학습합니다. CycleGAN에서 사용하는 Loss에 대해 살펴보겠습니다. 

* CycleGAN loss = GAN loss (int both direction) + **Cycle-consistency loss**

  * GAN loss: domain 간 변화 시 얼마나 target domain의 real data같은지
  * **Cycle-consistency loss**: 기존 이미지(X)와 변환 후 이미지(X -> Y -> X)가 얼마나 유사한지 

  ![image-20220315143422625](https://user-images.githubusercontent.com/70505378/158335683-fbbdd2f2-cc8f-4a55-8255-23dfd483bf71.png)

GAN loss만을 사용하면 안 되는 이유는, X -> Y로 변환된 이미지가 Y의 임의의 실제 데이터와 유사할 수는 있지만 그것이 X와 유사하지 않을 수 있기 때문입니다. 즉, GAN Loss는 '실제 데이터와의 유사성' 만을 보기 때문에, 변환 전 이미지와 유사해야 한다는 것을 학습할 수가 없습니다. 

아래 그림을 보면 이해가 갈 듯 합니다. 두 경우 모두에 GAN loss는 적절하게 변환이 되었다고 판단합니다. 

![image-20220315143917893](https://user-images.githubusercontent.com/70505378/158335686-ab526d2a-a693-4e68-afcf-9e0a597df6f9.png)

따라서 이 문제를 해결하기 위해 **Cycle-consistency loss**가 필요합니다. GAN loss의 맹점을 보완하기 위해, cycle-consistency loss는 기존 이미지(X)와 변환 후 이미지(X->Y->X)가 서로 유사하도록 학습하게 합니다. 

![image-20220315144209469](https://user-images.githubusercontent.com/70505378/158335689-60f4f70f-7484-4a5a-bc2f-bc4f20d57f0f.png)

CycleGAN은 입력 X에 대해 타겟 y가 매칭되어 있지 않아도 되고, 단지 두 도메인의 데이터만 충분히 주어져 있으면 되기 때문에 Self-supervised model이라고 할 수 있습니다. 

**Perceptual loss**

마지막으로 알아볼 것은 `Perceptual loss`입니다. Perceptual loss는 GAN loss(Adversarial loss)의 어려움을 극복하려는 목적을 갖고 있습니다. 

* GAN loss
  * Generator와 Discriminator가 각각 적대적으로 학습되기 때문에, 학습 방법이나 코드 구현이 어렵다. 
  * Pre-trained network를 필요로 하지 않는다. 
* Perceptual loss
  * **GAN의 Generator만 사용하기 때문에 학습 방법이나 코드 구현이 매우 쉬워진다.**
  * **Discriminator의 역할을 대신 할 pre-trained network가 필요하다.**

그렇다면 perceptual loss를 사용하는 GAN의 학습은 어떻게 이루어지는 지 살펴보겠습니다. 

우선 아래는 Perceptual loss를 사용하는 GAN의 일반적 구조로, 앞 단에서는 생성용 GAN만 사용하고 뒷 단에서는 판별용 CNN으로 VGG-16을 사용하고 있습니다. 

![image-20220315151219837](https://user-images.githubusercontent.com/70505378/158335696-234d0e4c-db24-4021-ba79-21ce44fab9ae.png)

Perceptual loss를 사용하는 GAN에서는 **Content target**과 **Style target**의 존재가 핵심인데요, 이들에 대해 알아보도록 하겠습니다. 

<br>

**Content target과 Transformed image 사이의 loss**를 **Feature reconstruction loss**라고 합니다. Feature reconstruction loss는 Content target(보통 변환 전 이미지인 X를 그대로 입력으로 사용)에 있는 내용물이 변환 후 이미지인 Transformed image에도 그대로 들어가 있는지 검사합니다. 

두 이미지에서 CNN이 각각 feature map을 뽑아서, feature map들 간의 L2 loss를 loss 값으로 사용합니다. 

![image-20220315151629530](https://user-images.githubusercontent.com/70505378/158335698-99acd22c-94d9-407a-80b4-066cba174e2b.png)

그리고 **Style target과 Transformed image 사이의 loss**를 **Style reconstruction loss**라고 합니다. Style reconstruction loss는 말 그대로 Style target(target domain에 있는 임의의 이미지를 입력으로 사용)에 있는 스타일이 변환 후 이미지인 Transformed image에도 반영되어 있는지 검사합니다. 

마찬가지로 CNN을 이용해 두 이미지에 대한 feature map을 뽑아내는데, 이 feature map으로부터 'style'을 뽑아내기 위해 한 번의 연산을 더 수행합니다. 바로 **Gram matrix**를 계산하는 연산입니다. 

![image-20220315152117400](https://user-images.githubusercontent.com/70505378/158335702-ec0cfcdb-c42d-40ad-8b75-6b2b5acc8367.png)

Gram matrix는 결과적으로 이야기하면 각 이미지의 스타일 정보를 인코딩하고 있는 행렬입니다. 여기서 **'어떻게 인코딩하느냐'**가 매우 흥미롭습니다. 

Feature map의 크기를 (C, H, W)라고 하면, 이를 먼저 (C, H\*W) 모양으로 reshape해줍니다. 이 reshape된 matrix를 R이라고 하겠습니다. R(C, H\*W)과 R<sup>T</sup>(H\*W, C)를 행렬곱하면 (C, C) 모양의 행렬을 얻게 되는데, 바로 이 행렬이 gram matrix입니다. 

Style 정보를 인코딩하고자 하는 gram matrix의 핵심은 spatial information을 제외하고 feature map의 statistics만을 가져오는 것입니다. 위의 일련의 연산을 거쳐서 feature map의 공간적 특성은 사라지고, 채널과 채널 간 관계만이 살아남아 인코딩되게 됩니다. 바로 이 channel correlation이 이미지의 style 정보를 나타내게 되는 것입니다. 

Style target과 Transformed image의 gram matrix를 모두 구하면 두 gram matrix 사이의 L2 loss를 loss 값으로 사용합니다. 







<br>

## Various GAN applications

GAN을 이용한 사례들에는 무엇이 있는지 보겠습니다. 

**Deepfake**

![image-20220315153923845](https://user-images.githubusercontent.com/70505378/158335708-2a9659c8-ac6c-482d-97ce-91f6f4bc7af9.png)

**Face de-identification**

![image-20220315154002317](https://user-images.githubusercontent.com/70505378/158335711-3ec94def-940a-4b0a-b7c6-9bdfb9ac7814.png)

**Pose/Video translation**

![image-20220315154044004](https://user-images.githubusercontent.com/70505378/158335713-c080c979-bf0c-4e23-b001-196853f08e2d.png)

마치면서, GAN에는 상당히 다양한 응용 사례들이 있고 그 적용 범위가 넓지만, 동시에 윤리적 문제 또한 함께 중요하게 고려해야 한다는 이야기를 하고 싶습니다.  



<br>

## 실습) cGAN

이번 강의의 실습은 cGAN 논문을 베이스로 Generator와 Discriminator를 간단하게 구현해보고, GAN을 학습시킨 후 출력 결과를 확인해보는 과정으로 이어집니다. 

![image-20220316180528948](https://user-images.githubusercontent.com/70505378/158586420-8ab68fb6-de75-4ef7-8206-cb3666e14c00.png)

**Generator**

* `z`: random vector
* `y`: condition label
* return: G(concat(emb(z), emb(y)))
* 과정
  * z, y를 각각 임베딩
  * 임베딩 된 z, y를 concat
  * concat한 벡터를 포워딩하여 이미지(img_height * img_width) 차원에 mapping

```python
class Generator(nn.Module):
    # initializers
    def __init__(self):
        super(Generator, self).__init__()
        ## fill ##
        z_embed_dim = 200
        y_embed_dim = 1000
        out_dim = 1 * 28 * 28 # channels * height * width

        # self.z_embed = nn.Embedding(100, z_embed_dim) # z
        # self.y_embed = nn.Embedding(10, y_embed_dim) # y(condition)
        self.z_embed = nn.Linear(100, z_embed_dim) # z
        self.y_embed = nn.Linear(10, y_embed_dim) # y(condition)

        self.main = nn.Sequential(
            nn.Linear(z_embed_dim+y_embed_dim,128),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(1024, out_dim),
            nn.Tanh()
        )

        self.weight_init(0,1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # forward method
    def forward(self, input, label): # z, y(c)
 
        ## fill ##
        z = self.z_embed(input)
        y = self.y_embed(label)
        zy = torch.concat((z,y), dim=-1) # dim 0: batch

        g_z = self.main(zy)

        return g_z
```

<br>

**Discriminator**

* `input`: generated image(g_z) or real image(x)
* `label`: conditional label(y) or real label(x_y)
* return: D(concat(emb(input), emb(label)))
* 과정
  * input, label을 각각 임배딩(*nn.Linear*).
  * 임배딩 된 input, label을 concat.
  * concat한 벡터를 포워딩하여 1차원에 mapping(*nn.Linear*).
  * 마지막 layer에 sigmoid를 통해 확률 값으로 변환(real(1) or fake(0))

```python
class Discriminator(nn.Module):
    # initializers
    def __init__(self):
        super(Discriminator, self).__init__()
        ## fill ##
        input_embed_dim = 784
        label_embed_dim = 10
        out_dim = 1

        # self.input_embed = nn.Embedding(784, input_embed_dim)
        # self.label_embed = nn.Embedding(10, label_embed_dim)
        self.input_embed = nn.Linear(784, input_embed_dim)
        self.label_embed = nn.Linear(10, label_embed_dim)

        self.main = nn.Sequential(
            nn.Linear(input_embed_dim + label_embed_dim, 512),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(512, 256),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Linear(256, out_dim),
            nn.Sigmoid()
        )

        self.weight_init(0,1)

    # weight_init
    def weight_init(self, mean, std):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.normal_(m.weight, 1.0, 0.02)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                m.weight.data *= 0.1
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    # forward method
    def forward(self, input, label):
        ## fill ##
        input = self.x_embed(input)
        conditional = self.g_z_embed(label)
        contitional_input = torch.cat((x,conditional), dim=-1) # dim 0: batch
        
        out = self.main(conditional_input)

        return out
```

<br>

**Training**

Training 코드가 쉽지 않아서, line by line으로 주석을 달아 보았습니다.  대략적인 흐름은 다음과 같습니다. 

* generator 학습
  * generator는 만들어낸 가짜 이미지가 discriminator를 얼마나 잘 속이는지에 의해 학습된다. 
* discriminator 학습
  * discriminator는 두 가지 손실 값의 평균에 의해 학습된다. 
    * generator가 만들어낸 가짜 이미지를 얼마나 가짜라고 잘 구별하는지
    * 실제 real image를 얼마나 실제라고 잘 구별하는지

```python
# Train
discriminator.train()

g_loss = torch.Tensor([0])
d_loss = torch.Tensor([0])

for epoch in range(parser.n_epochs):
  for batch_idx, (x, y) in enumerate(train_loader):
    generator.train()
    # linear layer 통과를 위해 이미지 차원 resize
    x_flatten = x.view(x.shape[0], -1)
    # 라벨 one-hot encoding
    one_hot_label = torch.nn.functional.one_hot(y, num_classes=parser['n_classes'])
    # to GPU
    img_torch2vec = x_flatten.type(torch.FloatTensor).cuda()  
    label_torch = one_hot_label.type(torch.FloatTensor).cuda()

    # valid: generator가 만들어낸 이미지가 discriminator를 얼마나 잘 속일 수 있는지(generated image가 얼마나 real image라고 분류할 확률) 측정할 때 label로 사용. 1일 때 완벽하게 속인 것이므로 ones를 라벨로 사용.
    # fake: discriminator가 generated image를 얼마나 잘 구별 하는지(generated image를 real image가 아니라고 분류할 확률) 측정할 때 label로 사용. 0일 때 잘 구별한 것이므로 zeros를 라벨로 사용. 
    valid = torch.ones(parser.batch_size, 1).cuda()
    fake = torch.zeros(parser.batch_size, 1).cuda()

    # 실제 이미지, 실제 라벨 데이터
    real_imgs = img_torch2vec
    labels = label_torch

    # === generator 학습 ===
    optimizer_G.zero_grad()

    # generator 입력 생성: z(random vector)와 gen_labels(y, conditional label)
    z = torch.randn(parser.batch_size, parser.latent_dim).cuda()
    gen_labels = []
    for randpos in np.random.randint(0, parser.n_classes, parser.batch_size):
      gen_labels.append(torch.eye(parser.n_classes)[randpos])
    gen_labels = torch.stack(gen_labels).cuda()

    # fake images 생성
    gen_imgs = generator(z, gen_labels)
    
    # val_output: 각 fake images에 대해 real image일 확률을 반환, generator 손실값 계산
    val_output = discriminator(gen_imgs, gen_labels)
    # generator의 손실 함수 값은 fake image를 real image가 아니라고 예측되는 정도
    g_loss = cross_entropy(val_output, valid)
    g_loss.backward()
    optimizer_G.step()

    # === discriminator 학습 ===
    optimizer_D.zero_grad()
    
    # validity_real: real images를 입력으로 주고 real image일 확률을 반환(높을수록 잘 구별)
    validity_real = discriminator(real_imgs, labels)
    try:
        d_real_loss = cross_entropy(validity_real, valid)
    except:
        valid = torch.ones(validity_real.shape[0], 1).cuda()
        d_real_loss = cross_entropy(validity_real, valid)

    # validity_fake: fake image를 입력으로 주고 real image일 확률을 반환(낮을수록 잘 구별)
    validity_fake = discriminator(gen_imgs.detach(), gen_labels)
    d_fake_loss = cross_entropy(validity_fake, fake)
	
    # discriminator의 손실 함수 값은 d_real_loss와 d_fake_loss의 평균
    d_loss = (d_real_loss + d_fake_loss) / 2
    d_loss.backward()
    optimizer_D.step()
    
    if batch_idx % 500 == 0:
      print('{:<13s}{:<8s}{:<6s}{:<10s}{:<8s}{:<9.5f}{:<8s}{:<9.5f}'.format('Train Epoch: ', '[' + str(epoch) + '/' + str(parser['n_epochs']) + ']', 'Step: ', '[' + str(batch_idx) + '/' + str(len(train_loader)) + ']', 'G loss: ', g_loss.item(), 'D loss: ', d_loss.item()))

  if epoch % parser.sample_interval == 0:
    sample_image(n_row=10, epoch=epoch)
```

<br>

**Inference**

```python
def show_image(condition: int):
    generator.eval()

    z = torch.randn(1, parser.latent_dim).type(torch.FloatTensor).cuda()
    condition_vector = torch.eye(10)[condition].reshape(1,-1).cuda()
    gen_imgs = generator(z, condition_vector)
    plt.imshow(gen_imgs.view(1,1,28,28)[0][0].cpu().detach().numpy(), cmap='gray')
    
show_image(3)
```

![image-20220316181515298](https://user-images.githubusercontent.com/70505378/158586416-a734967c-13a8-4b92-b224-287baa7f485d.png)



















<br>

<br>

# 참고 자료


* Conditional generative model

  * Isola et al., Image-to-Image Translation with Conditional Adversarial Networks, CVPR 2017 
  * Kuleshov et al., Audio Super Resolution using Neural Networks, ICLR 2017 
  * Brown et al., Language Models are few shot learners, arXiv 2020 
  * Zhu et al., Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, ICCV 2017 
  * Ledig et al., Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network, CVPR 2017  
* Image translation GANs

  * Isola et al., Image-to-Image Translation with Conditional Adversarial Networks, CVPR 2017 
  * Zhu et al., Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks, ICCV 2017 
  * Johnson et al., Perceptual Losses for Real-Time Style Transfer and Super-Resolution, ECCV 2016  
* Various GAN applications

  * Gafni et al., Live Face De-Identification in Video, ICCV 2019 
  * Gu et al., Password-conditioned Anonymization and Deanonymization with Face Identity Transformers, ECCV
    2020 
  * Liu et al., Liquid Warping GAN: A Unified Framework for Human Motion Imitation, Appearance Transfer and Novel
    View Synthesis, ICCV 2019 
  * Wang et al., Video-to-Video Synthesis, NeurIPS 2018 
  * Gafni et al., Vid2Game: Controllable Characters Extracted from Real-World Videos, ICLR 2018  



<br>

