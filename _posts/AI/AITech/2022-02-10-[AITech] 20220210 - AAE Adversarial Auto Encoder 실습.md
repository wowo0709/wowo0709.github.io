---

layout: single
title: "[AITech] 20220210 - AAE Adversarial Auto Encoder 실습"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

# 학습 내용

이번 포스팅에서는 AAE(Adversarial Auto Encoder)가 무엇인지, 그리고 그 구조를 code level에서 살펴보도록 하겠습니다. 

## AAE

AAE에 대한 소개는 [Generative Models II]에서도 간단하게 했었는데요, 그 내용을 리뷰해보면 다음과 같습니다. 

**VAE(Variational Auto Encoder)**는 Variational inference(VI)라는 방식을 이용해 입력 데이터 x의 확률 분포인 posterior distribution `p`에 가장 잘 match하는 모델링 확률 분포 variational distribution `q`를 찾았었습니다. 그리고 이 과정에서 미리 알 수 없는 `p(z|x)` 와 비슷하게 만들어야 하는 `q(z|x)`를 찾기 위해 다음과 같이 식을 전개하여 **ELBO(Evidence Lower BOund)** term을 minimize하는 방식을 사용했었습니다. 

![image-20220210103609537](https://user-images.githubusercontent.com/70505378/153326506-14a03c82-4d6c-4ffa-b782-62420cc56706.png)

그리고 다시 ELBO term을 다음과 같이 전개하여 reconstruction term과 prior fitting term을 최소화하도록 학습했었습니다. 

![image-20220210103902153](https://user-images.githubusercontent.com/70505378/153326508-fe32d418-bafc-4bdf-ab13-0bbec29ba2ac.png)

그런데 여기서 VAE의 문제는 바로 저 **Prior Fitting Term이 미분가능해야만 한다**라는 점이었습니다. VAE는 이 식 자체를 loss function으로 사용하기 때문에, 학습을 위해서는 미분 가능한 form을 사용해야만 했었습니다. 

그리고 바로 그 미분가능한 확률 분포를 사용하기 위해 다음과 같이 **p(z) = N(0, I) (Isotropic Gaussian)**로 고정하여 prior fitting term을 사용했습니다. 

![image-20220210104605558](https://user-images.githubusercontent.com/70505378/153326509-f4ad401f-7567-4240-bd4f-e8f2fca70f20.png)

바로 이것이 문제죠. 실제 데이터가 N(0, I) 분포를 따르지 않는다면? 다른 확률 분포로 모델링을 시도하고 싶다면? 불가능했었습니다. 

<br>

바로 이런 VAE의 한계를 극복한 모델이 **AAE(Adversarial Auto Encoder)**입니다. 

AAE는 저 prior fitting term을 손실 함수 식 자체로 쓰는 것이 아니라, 이를 minimize하는 task를 GAN에게 맡겼습니다. GAN은 본래 실제 입력과 생성된 가짜 입력의 차이를 최소화하는 네트워크이기 때문에, **p(z)와 q(z|x) 간의 차이를 최소화하는 임무를 GAN에게 맡긴 것이죠.**

이렇게 함으로써 AAE는 sampling만 가능하다면 p(z)에 어떤 다른 확률 분포도 사용할 수 있게 되었고, 이는 **네트워크의 loss term에 있던 prior fitting term을 빼고 대신 GAN의 loss term을 추가함**으로써 이룰 수 있었습니다. 

AAE의 구조는 아래와 같습니다. 

![image-20220210184958477](https://user-images.githubusercontent.com/70505378/153408669-82836fd2-07fa-45ad-bfd9-b769e2df212a.png)

여기서 만들어야 할 네트워크는 세개입니다. 파란색 박스의 Generator (Encoder), 노란색 박스 Autoencoder에 있는 Decoder, 그리고 빨간색 박스의 Discriminator이죠. 

논문을 인용하자면 AAE는 **aggregated posterior** `q(z)`를 **arbitrary prior** `p(z)`에게 매칭하는 오토인코더입니다. 그렇게 하기 위해, **adversarial network**를 사용해 q(z)를 p(z)로 매칭하게 가이드를 하고, **autoencoder는** reconstruction error를 줄이게 됩니다. 

아래는 MNIST 데이터를 이용해 학습할 때 AAE와 VAE의 latent space의 차이를 보인 것입니다. 

![image-20220210212630506](https://user-images.githubusercontent.com/70505378/153408679-92a05276-3960-4211-9e73-e133eb468cd1.png)

AAE는 제약없이 훨씬 더 다양한 분포를 모델링할 수 있습니다. 

이제부터는 AAE의 각 부분의 구현 코드를 보도록 하겠습니다. 

<br>

### Reparametrization

Decoder에 들어가기 전, Encoder 아웃풋인 μ(mu)와 σ(sigma)가 나오게 됩니다. p(z)에서 샘플링을 할때, 데이터의 확률 분포와 같은 분포에서 샘플을 뽑아야하는데, backpropagation을 하기 위해선, reparametrization의 과정을 거칩니다. 즉, 정규분포에서 z을 샘플링 하는 것이죠.

더 자세히 들어가면, ϵ을 정규분포 (N(0, 1))에서 샘플링을 하고, 그 값을 분산 exp^(logvar/2)와 곱하고, 평균인 μ를 더해 샘플링된 `z1`과 인코더가 만든 μ와 σ로 reparameterized된 `z2`가 같은 분포를 가지게하도록 학습하는 것입니다. 

```python
def reparameterization(mu, logvar):
    std = torch.exp(logvar / 2)
    sampled_z = Variable(Tensor(np.random.normal(0, 1, (mu.size(0), latent_dim))))
    z = sampled_z * std + mu
    return z
```



### Encoder

```python
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(1024, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 512),  
            nn.Dropout(p=0.2),
            nn.ReLU()
        )

        self.mu = nn.Linear(512, latent_dim)
        self.logvar = nn.Linear(512, latent_dim)

    def forward(self, img):
        img_flat = img.view(img.shape[0], -1) 
        x = self.model(img_flat)
        mu = self.mu(x)
        logvar = self.logvar(x)

        z = reparameterization(mu, logvar)
        return z
```





### Decoder

```python
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.BatchNorm1d(512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.Tanh(),
        )

    def forward(self, z):
        q = self.model(z)
        q = q.view(q.shape[0], *img_shape)
        return q
```





### Discriminator

```python
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.Dropout(p=0.2),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        
    def forward(self, z):
        x = self.model(z)
        return x
```

### Overall Training 

```python
# Use binary cross-entropy loss
adversarial_loss = torch.nn.BCELoss()
pixelwise_loss = torch.nn.L1Loss()

# Initialization of three models
encoder = Encoder()
decoder = Decoder()
discriminator = Discriminator()

if torch.cuda.is_available():
    encoder.cuda()
    decoder.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()
    pixelwise_loss.cuda()

# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(encoder.parameters(), decoder.parameters()), lr=lr, betas=(b1, b2)
)
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

for epoch in range(n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths
        valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        # -----------------
        #  Train Generator
        # -----------------

        optimizer_G.zero_grad()

        encoded_imgs = encoder(real_imgs)
        decoded_imgs = decoder(encoded_imgs)

        # Loss measures generator's ability to fool the discriminator
        g_loss = 0.001 * adversarial_loss(discriminator(encoded_imgs), valid) + 0.999 * pixelwise_loss(
            decoded_imgs, real_imgs
        )

        g_loss.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Sample noise as discriminator ground truth
        z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

        # Measure discriminator's ability to classify real from generated samples
        real_loss = adversarial_loss(discriminator(z), valid)
        fake_loss = adversarial_loss(discriminator(encoded_imgs.detach()), fake)
        d_loss = 0.5 * (real_loss + fake_loss)

        d_loss.backward()
        optimizer_D.step()
```



<br>

## WAE

여기서 길게 언급하지는 않을 것이지만, **WAE(Wasserstein Auto Encoder)**라는 것도 있습니다. WAE는 AAE를 포함하는 조금 더 상위 개념의 네트워크라고 할 수 있습니다. 

WAE는 어떠한 cost function 'c'에 대한 Optimal transport (W<sub>c</sub>(P<sub>X</sub>, P<sub>Z</sub>))를 최소화하는 형태로, AAE처럼 objective function이 c-reconstruction cost와 regualarizer cost D<sub>Z</sub>(P<sub>Z</sub>,Q<sub>Z</sub>)로 구성되어 있습니다. 이 때 c가 squared cost이고 D<sub>Z</sub>가 GAN objective이면 WAE는 AAE와 일치합니다. 

WAE는 VAE나 AAE에서 발생할 수 있는 reconstruction 문제를 보완할 수 있다고 하는데, 자세히 알고싶으신 분들은 아래 참고자료의 WAE 논문을 읽어보시는 것을 추천드립니다. 











<br>

<br>

# 참고 자료

* [AAE paper](https://arxiv.org/pdf/1511.05644.pdf)
  * [VAE paper](https://arxiv.org/pdf/1312.6114.pdf)
  * [GAN paper](https://arxiv.org/pdf/1406.2661.pdf)
* [WAE paper](https://arxiv.org/pdf/1711.01558.pdf)

















<br>
