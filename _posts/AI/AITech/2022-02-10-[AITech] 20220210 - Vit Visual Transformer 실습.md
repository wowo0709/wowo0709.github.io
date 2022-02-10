---

layout: single
title: "[AITech] 20220210 - ViT Visual Transformer 실습"
categories: ['AI', 'AITech']
toc: true
toc_sticky: true
tag: []
---



<br>

# 학습 내용

이번 포스팅에서는 `ViT(Visual Transformer)`에 대해 code level에서 그 구조를 보도록 하겠습니다. 

본격적으로 들어가기 전에, 이 코드에서는 `einops`라는 pytorch supporting library을 사용하는데요, 이에 대한 설명은 아래에서 볼 수 있습니다. 

```python
# reference & tutorial : http://einops.rocks/pytorch-examples.html
%pip install einops
from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange, Reduce
```



## ViT란 무엇인가?

ViT(Visual Transformer)는 말 그대로 visual task를 풀기 위한 transformer입니다. 이전 포스팅에서 Transformer의 구조에 대해 천천히 뜯어보았는데요, Transformer는 원래 기계 번역을 위한 자연어 처리 모델인데, 이를 비전 분야로 가져온 것입니다. 

Transformer의 아이디어는 동시에 N개의 주변 정보를 이용하여 하나의 정보를 예측하는 것이 있었습니다. 이 아이디어가 ViT에도 그대로 사용되어, ViT에서는 하나의 이미지를 여러 개의 패치로 나눠 입력으로 주게 됩니다. 

ViT의 특징으로는 기존 Transformer의 Encoder 구조만 가져와서 사용한다는 것입니다. 아래는 ViT의 구조입니다. 

![image-20220210150804830](https://user-images.githubusercontent.com/70505378/153377506-e2e2e765-1813-453a-a13f-91dcba40e4b2.png)



위 구조를 보면 알 수 있듯이, ViT는 다음의 과정을 거칩니다. 

* Image를 patch로 뜯어내고 flatten하는 image embedding
  * 각각의 patch를 flatten하고 linear projection
* Linear projection이 적용된 tensor를 Encoder에 통과
  * MHA
  * MLP
* Encoder의 출력 값을 최종적으로 MLP Head를 거쳐 classfication

ViT는 CNN 구조를 대체할 거라는 전망이 있을 정도로 강력한 네트워크지만, 논문에 따르면 데이터의 개수가 매우 많아야만 그 성능이 아주 높아진다고 합니다. 기술된 바로는 약 1,200만 장의 ImageNet 데이터도 부족한 수준이고, Google 내부적으로 사용하는 약 6억 장의 이미지 데이터를 사용했을 때 기존의 다른 classification model들을 뛰어넘는 performance를 보여줬다고 합니다. 

이는 곧 우리가 ViT를 사용하려면 밑바닥부터 훈련시키는 것은 불가능하고, 어느 수준의 Fine-Tuning으로 pre-trained된 ViT 모델을 가져와서 사용해야 한다는 뜻이죠. 

여기까지 ViT의 전반에 대해 한 번 살펴봤고, 밑에서부터는 ViT의 각 과정/모듈과 그 과정/모듈에 해당하는 코드를 함께 보도록 하겠습니다. 







<br>

## Image Embedding

![image-20220210152031587](https://user-images.githubusercontent.com/70505378/153377510-5e0dac11-2031-4f86-9ea9-cc0e71a08761.png)

첫번째로 Image embedding입니다. Image embedding에서는 위에서 보다시피 입력 image를 각 patch로 뜯어내고, 이를 flatten 시킨 다음에 어떤 linear projection을 적용하는 과정이 수행됩니다. 

그리고 이 linear projection을 거치고 나면 각 image patch에는 **class token**이 붙고 **positional encodding**이 수행됩니다. Class token은 분류를 위해 각 이미지 패치에 라벨링을 해주는 것이고, 0~n번째까지 n+1개의 라벨링을 합니다. Positional encodding은 일반 Transformer와 마찬가지로 각 정보, 여기서는 각 픽셀에 순서를 매기는 것입니다. 

그래서 이 image embedding에서 수행되는 연산은 아래와 같이 표현될 수 있습니다. 

![image-20220210152409850](https://user-images.githubusercontent.com/70505378/153377511-31b3a89d-ff73-4d37-a129-acf99b58b568.png)

코드는 아래와 같습니다. 

```python
class image_embedding(nn.Module):
    def __init__(self, in_channels: int = 3, img_size: int = 224, patch_size: int = 16, emb_dim: int = 16*16*3):
        super().__init__()
        # [b, c, w, h] 크기의 이미지 배치를 [b, n_patch, d_patch] 로 뜯어낸다. : 28*28 크기의 이미지를 4*4 크기의 patch로 뜯어낸다. 
        self.rearrange = Rearrange('b c (num_w p1) (num_h p2) -> b (num_w num_h) (p1 p2 c) ', p1=patch_size, p2=patch_size)
        # [c, w_patch, h_patch] -> [d_embedding] : Linear projection of flatten patches
        self.linear = nn.Linear(in_channels * patch_size * patch_size, emb_dim)
        # [b, c, d_embedding] : 임베딩된 각각의 패치들 앞에 class token을 붙여준다. 
        self.cls_token = nn.Parameter(torch.randn(1, 1, emb_dim))
        # 패치의 개수 : (28*28) // (4*4) = (7*7)
        n_patches = img_size * img_size // patch_size**2
        # 0 + 1~n 개의 포지션 인코딩 (학습 가능)
        self.positions = nn.Parameter(torch.randn(n_patches + 1, emb_dim))

    def forward(self, x):
        batch, channel, width, height = x.shape
        print(f'Input x: batch={batch}, channel={channel}, width={width}, height={height}')

        x = self.rearrange(x) # flatten patches
        print(f'Rearrange: x.shape={x.shape} : [n_batch, n_patch, d_patch]')
        x = self.linear(x) # embedded patches 
        print(f'Linear: x.shape={x.shape} : [n_batch, n_patch, d_embedding]')

        # (1) Build [token; image embedding] by concatenating class token with image embedding
        c = repeat(self.cls_token, '() n d -> b n d', b=batch) # [batch, numbers, features]
        x = torch.cat((c, x), dim=1)
        print(f'Cat: x.shape={x.shape} : [n_batch, 1+n_patch, d_embedding]')
        print(f'self.positions: self.positions.shape={self.positions.shape} : [1+n_batch, d_embedding]')

        # (2) Add positional embedding to [token; image embedding]
        x += self.positions
        print(f'positional encodding: x.shape={x.shape}')

        return x

emb = image_embedding(1, 28, 4, 4*4)(x) # in_channels=1, img_size=28, patch_size=4, emb_dim=16

'''
Input x: batch=1, channel=1, width=28, height=28
Rearrange: x.shape=torch.Size([1, 49, 16]) : [n_batch, n_patch, d_patch]
Linear: x.shape=torch.Size([1, 49, 16]) : [n_batch, n_patch, d_embedding]
Cat: x.shape=torch.Size([1, 50, 16]) : [n_batch, 1+n_patch, d_embedding]
self.positions: self.positions.shape=torch.Size([50, 16]) : [1+n_batch, d_embedding]
positional encodding: x.shape=torch.Size([1, 50, 16])
'''
```





<br>

## Encoder

두번째로 Encoder 부분입니다. Encoder 부분은 Transformer와 약간 다릅니다. 

![image-20220210153017677](https://user-images.githubusercontent.com/70505378/153377515-1602002d-b80f-4028-af23-e07d1006b162.png)

먼저 Encoder에 필요한 MHA, MLP 모듈을 구현하고 이후 전체 Encoder를 구현하겠습니다. 

### MHA

앞선 과정에서 Embedded된 patch들을 Encoder의 입력으로 들어오고, 먼저 MHA(또는 MSA)를 통과합니다. 

![image-20220210153125766](https://user-images.githubusercontent.com/70505378/153377518-9b1848e7-5203-413a-b7c6-5153ba85dce5.png)

코드는 아래와 같습니다. 

```python
class multi_head_attention(nn.Module):
    def __init__(self, emb_dim: int = 16*16*3, num_heads: int = 8, dropout_ratio: float = 0.2, verbose = False, **kwargs):
        super().__init__()
        self.v = verbose

        self.emb_dim = emb_dim 
        self.num_heads = num_heads 
        # 임베딩된 단어의 차원 emb_dim은 num_heads 개의 attention head에 나눠서 처리
        self.scaling = (self.emb_dim // num_heads) ** -0.5
        
        self.value = nn.Linear(emb_dim, emb_dim)
        self.key = nn.Linear(emb_dim, emb_dim)
        self.query = nn.Linear(emb_dim, emb_dim)
        self.att_drop = nn.Dropout(dropout_ratio)

        self.linear = nn.Linear(emb_dim, emb_dim)
                
    def forward(self, x: Tensor) -> Tensor:
        # query, key, value
        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)
        if self.v: print(f'Q.shape={Q.size()}, K.shape={K.size()}, V.shape={V.size()}') 

        # q = k = v = patch_size**2 + 1 & h * d = emb_dim
        Q = rearrange(Q, 'b q (h d) -> b h q d', h=self.num_heads)
        K = rearrange(K, 'b k (h d) -> b h d k', h=self.num_heads)
        V = rearrange(V, 'b v (h d) -> b h v d', h=self.num_heads)
        if self.v: print(f'Q.shape={Q.size()}, K.shape={K.size()}, V.shape={V.size()}') 

        ## scaled dot-product
        weight = torch.matmul(Q, K) 
        weight = weight * self.scaling
        if self.v: print(f'weight.shape={weight.size()}') 
        
        attention = torch.softmax(weight, dim=-1)
        attention = self.att_drop(attention) 
        if self.v: print(f'attention.shape={attention.size()}')

        context = torch.matmul(attention, V) 
        context = rearrange(context, 'b h q d -> b q (h d)')
        if self.v: print(f'context.shape={context.size()}')

        x = self.linear(context)
        return x , attention
    
feat, att = multi_head_attention(4*4, 4, verbose=True)(emb)

'''
Q.shape=torch.Size([1, 50, 16]), K.shape=torch.Size([1, 50, 16]), V.shape=torch.Size([1, 50, 16])
Q.shape=torch.Size([1, 4, 50, 4]), K.shape=torch.Size([1, 4, 4, 50]), V.shape=torch.Size([1, 4, 50, 4])
weight.shape=torch.Size([1, 4, 50, 50])
attention.shape=torch.Size([1, 4, 50, 50])
context.shape=torch.Size([1, 50, 16])
'''
```



### MLP

MHA를 통과한 입력은 MLP의 입력으로 들어옵니다. 

```python
class mlp_block(nn.Module):
    def __init__(self, emb_dim: int = 16*16*3, forward_dim: int = 4, dropout_ratio: float = 0.2, **kwargs):
        super().__init__()
        self.linear_1 = nn.Linear(emb_dim, forward_dim * emb_dim)
        self.dropout = nn.Dropout(dropout_ratio)
        self.linear_2 = nn.Linear(forward_dim * emb_dim, emb_dim)
        
    def forward(self, x):
        x = self.linear_1(x)
        x = nn.functional.gelu(x)
        x = self.dropout(x) 
        x = self.linear_2(x)
        return x
```

### Overall Encoder

이제 위에서 구현한 모듈들을 이용해 전체 Encoder를 build합니다. 

위 Encoder 구조 그림에서 봤던 Layer Norm, Residual connection, dropout도 모두 포함합니다. 

```python
class encoder_block(nn.Sequential):
    def __init__(self, emb_dim:int = 16*16*3, num_heads:int = 8, forward_dim: int = 4, dropout_ratio:float = 0.2):
        super().__init__()

        self.norm_1 = nn.LayerNorm(emb_dim)
        self.mha = multi_head_attention(emb_dim, num_heads, dropout_ratio)

        self.norm_2 = nn.LayerNorm(emb_dim)
        self.mlp = mlp_block(emb_dim, forward_dim, dropout_ratio)

        self.residual_dropout = nn.Dropout(dropout_ratio)

    def forward(self, x):
        # x = normalize (input)
        norm_1_x = self.norm_1(x)
        # x', attention = multihead_attention (x)
        mha_x, attention = self.mha(norm_1_x)
        # x = x' + residual(x)
        x = mha_x + self.residual_dropout(x)

        # x' = normalize(x)
        norm_2_x = self.norm_2(x)
        # x' = mlp(x')
        mlp_x = self.mlp(norm_2_x)
        # x  = x' + residual(x)
        x = mlp_x + self.residual_dropout(x)

        return x, attention
```





<br>

## Overall Model

이제 마지막으로 위 과정들을 거친 출력을 Classification Head에 통과시켜 최종 분류를 수행합니다. 

논문에 따르면, classification head는 pre-training time에는 one hidden-layer가 있는 MLP이고, fine-tuning time에는 single linear layer라고 합니다. 

![image-20220210153959749](https://user-images.githubusercontent.com/70505378/153377498-439ed87e-3072-4683-a111-5c7d70a923b6.png)

코드는 아래와 같습니다. 

```python
class vision_transformer(nn.Module):
    """ Vision Transformer model
    classifying input images (x) into classes
    """
    def __init__(self, in_channel: int = 3, img_size:int = 224, 
                 patch_size: int = 16, emb_dim:int = 16*16*3, 
                 n_enc_layers:int = 15, num_heads:int = 3, 
                 forward_dim:int = 4, dropout_ratio: float = 0.2, 
                 n_classes:int = 1000):
        super().__init__()

        '''
        params : in_channels: int = 3, img_size: int = 224, patch_size: int = 16, emb_dim: int = 16*16*3
        return : x: [n_batch, 1+n_patch, d_embedding]
        '''
        self.img_emb = image_embedding(in_channel, img_size, patch_size, emb_dim)
        '''
        params : emb_dim:int = 16*16*3, num_heads:int = 8, forward_dim: int = 4, dropout_ratio:float = 0.2
        return : x: [n_batch, 1+n_patch, d_embedding], attention: [n_batch, n_head, 1+n_patch, 1+n_patch]
        '''
        # stacked encoders
        self.encoders = nn.ModuleList([encoder_block(emb_dim, num_heads, forward_dim, dropout_ratio) for _ in range(n_enc_layers)])    

        self.reduce_layer = Reduce('b n e -> b e', reduction='mean')
        self.normalization = nn.LayerNorm(emb_dim)
        self.classification_head = nn.Linear(emb_dim, n_classes) 

    def forward(self, x):
        # (1) image embedding
        x = self.img_emb(x)

        # (2) transformer_encoder
        attentions = []
        for encoder in self.encoders:
            x, attention = encoder(x)
            attentions.append(attention)
		
        # (3) classification head
        x = self.reduce_layer(x)
        x = self.normalization(x)
        x = self.classification_head(x)

        return x, attentions
    

y, att = vision_transformer(1, 28, 4, 4*4, 3, 2, 4, 0.2, 10)(x)
print(f'output y shape={y.shape}')
print(f'attention shape={att[0].shape}')

'''
Input x: batch=1, channel=1, width=28, height=28
Rearrange: x.shape=torch.Size([1, 49, 16]) : [n_batch, n_patch, d_patch]
Linear: x.shape=torch.Size([1, 49, 16]) : [n_batch, n_patch, d_embedding]
Cat: x.shape=torch.Size([1, 50, 16]) : [n_batch, 1+n_patch, d_embedding]
self.positions: self.positions.shape=torch.Size([50, 16]) : [1+n_batch, d_embedding]
positional encodding: x.shape=torch.Size([1, 50, 16])
output y shape=torch.Size([1, 10])
attention shape=torch.Size([1, 2, 50, 50])
'''
```







<br>

<br>

# 참고 자료

* [ViT 논문](https://arxiv.org/pdf/2010.11929.pdf)

















<br>
