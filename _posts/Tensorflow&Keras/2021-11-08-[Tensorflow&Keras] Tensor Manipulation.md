---
layout: single
title: "[Tensorflow&Keras] Tensor Manipulation"
categories: ['AI', 'TensorflowKeras']
toc: true
toc_sticky: true
tag: []

---

<br>

# Tensor Manipulation

- from Prof. Sunghun Kim's lecture lab


```python
# https://www.tensorflow.org/api_guides/python/array_ops
import tensorflow as tf
import numpy as np
import pprint
np.random.seed(17)  # for reproducibility

pp = pprint.PrettyPrinter(indent=4)
```

<br>

## Simple Array


```python
t = np.array([0., 1., 2., 3., 4., 5., 6.])
print(t)
print(t.ndim) # rank
print(t.shape) # shape
print(t[0], t[1], t[-1])
print(t[2:5], t[4:-1])
print(t[:2], t[3:])
```

    [0. 1. 2. 3. 4. 5. 6.]
    1
    (7,)
    0.0 1.0 6.0
    [2. 3. 4.] [4. 5.]
    [0. 1.] [3. 4. 5. 6.]



```python
# 2-dimensional array
t = np.array([[1., 2., 3.], 
              [4., 5., 6.], 
              [7., 8., 9.], 
              [10., 11., 12.]])
pp.pprint(t)
print(t.ndim) # rank
print(t.shape) # shape
```

    array([[ 1.,  2.,  3.],
           [ 4.,  5.,  6.],
           [ 7.,  8.,  9.],
           [10., 11., 12.]])
    2
    (4, 3)

<br>

## Shape, Rank, Axis


```python
t1 = tf.constant([1,2,3,4])
t2 = tf.constant([[1,2,3,4]])
print(t1, t1.numpy(), t1.dtype, t1.shape)
print(t2, t2.numpy(), t2.dtype, t2.shape)
```

    tf.Tensor([1 2 3 4], shape=(4,), dtype=int32) [1 2 3 4] <dtype: 'int32'> (4,)
    tf.Tensor([[1 2 3 4]], shape=(1, 4), dtype=int32) [[1 2 3 4]] <dtype: 'int32'> (1, 4)



```python
t = tf.constant([[1,2],
                 [3,4]])
t
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[1, 2],
           [3, 4]], dtype=int32)>




```python
t = tf.constant([[[1, 2, 3, 4], 
                   [5, 6, 7, 8], 
                   [9, 10, 11, 12]],
                  
                  [[13, 14, 15, 16], 
                   [17, 18, 19, 20], 
                   [21, 22, 23, 24]]])
print(tf.shape(t))
t
```

    tf.Tensor([2 3 4], shape=(3,), dtype=int32)





    <tf.Tensor: shape=(2, 3, 4), dtype=int32, numpy=
    array([[[ 1,  2,  3,  4],
            [ 5,  6,  7,  8],
            [ 9, 10, 11, 12]],
    
           [[13, 14, 15, 16],
            [17, 18, 19, 20],
            [21, 22, 23, 24]]], dtype=int32)>




```python
t = tf.constant( \
[
    [
        [
            [1,2,3,4], 
            [5,6,7,8],
            [9,10,11,12]
        ],
        [
            [13,14,15,16],
            [17,18,19,20], 
            [21,22,23,24]
        ]
    ]
])
t
```




    <tf.Tensor: shape=(1, 2, 3, 4), dtype=int32, numpy=
    array([[[[ 1,  2,  3,  4],
             [ 5,  6,  7,  8],
             [ 9, 10, 11, 12]],
    
            [[13, 14, 15, 16],
             [17, 18, 19, 20],
             [21, 22, 23, 24]]]], dtype=int32)>

<br>

## Matmul VS multiply


```python
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],
                       [2.]])
print(matrix1, matrix2)
tf.matmul(matrix1, matrix2)
```

    tf.Tensor([[3. 3.]], shape=(1, 2), dtype=float32) tf.Tensor(
    [[2.]
     [2.]], shape=(2, 1), dtype=float32)





    <tf.Tensor: shape=(1, 1), dtype=float32, numpy=array([[12.]], dtype=float32)>




```python
matrix1*matrix2    # broadcasting (be careful when using)
```




    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[6., 6.],
           [6., 6.]], dtype=float32)>

<br>

## Watch out broadcasting


```python
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
matrix1+matrix2
```




    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[5., 5.],
           [5., 5.]], dtype=float32)>




```python
matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2., 2.]])
matrix1+matrix2
```




    <tf.Tensor: shape=(1, 2), dtype=float32, numpy=array([[5., 5.]], dtype=float32)>

<br>

## Random values for variable initializations 


```python
tf.random.normal([3], mean=0., stddev=1.0)
```




    <tf.Tensor: shape=(3,), dtype=float32, numpy=array([-0.05744869, -0.39726454,  1.4566656 ], dtype=float32)>




```python
tf.random.uniform([2])  # For floats, the default range is [0, 1).  
                        # for ints, at least maxval must be specified explicitly.
```




    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([0.9564322, 0.5858873], dtype=float32)>




```python
tf.random.uniform(shape=[2, 3])   # or (2,3)
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[0.75047934, 0.44712412, 0.32288682],
           [0.6673713 , 0.28056622, 0.47909296]], dtype=float32)>




```python
np.random.uniform(size=[2,3])   # or (2,3)
```




    array([[0.294665  , 0.53058676, 0.19152079],
           [0.06790036, 0.78698546, 0.65633352]])




```python
tf.round(tf.random.uniform([2, 3]))  # nearest integer
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[1., 1., 1.],
           [1., 0., 1.]], dtype=float32)>




```python
np.round(tf.random.uniform([2, 3]), 3)
```




    array([[0.771, 0.564, 0.472],
           [0.188, 0.477, 0.808]], dtype=float32)

<br>

## Reduce Mean/Sum


```python
np.array([1,2,3]).sum()
```




    6




```python
tf.reduce_sum([1., 2.], axis=0)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=3.0>




```python
x = [[1., 2.],
     [3., 4.]]
print(np.mean(x))
print(tf.constant(x))
print(tf.reduce_mean(x))
print(tf.reduce_mean(x, axis=0))
print(tf.reduce_mean(x, axis=1))
```

    2.5
    tf.Tensor(
    [[1. 2.]
     [3. 4.]], shape=(2, 2), dtype=float32)
    tf.Tensor(2.5, shape=(), dtype=float32)
    tf.Tensor([2. 3.], shape=(2,), dtype=float32)
    tf.Tensor([1.5 3.5], shape=(2,), dtype=float32)



```python
tf.reduce_mean(x, axis=-1)  # 가장 안쪽 axis (마지막 축)
```




    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([1.5, 3.5], dtype=float32)>




```python
tf.reduce_sum(x) , tf.reduce_sum(x, axis=0), tf.reduce_sum(x, axis=-1), tf.reduce_mean(tf.reduce_sum(x, axis=-1))
```




    (<tf.Tensor: shape=(), dtype=float32, numpy=10.0>,
     <tf.Tensor: shape=(2,), dtype=float32, numpy=array([4., 6.], dtype=float32)>,
     <tf.Tensor: shape=(2,), dtype=float32, numpy=array([3., 7.], dtype=float32)>,
     <tf.Tensor: shape=(), dtype=float32, numpy=5.0>)

<br>

## Sorting


```python
mat = [[3,2,1],
       [2,1,3],
       [1,3,2]]
np.sort(mat, axis=0)
```




    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]])




```python
np.sort(mat, axis=1)
```




    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]])




```python
tf.sort(mat, axis=0)
```




    <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[1, 1, 1],
           [2, 2, 2],
           [3, 3, 3]], dtype=int32)>




```python
tf.sort(mat, axis=1)
```




    <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]], dtype=int32)>




```python
tf.sort(mat, axis=-1)
```




    <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[1, 2, 3],
           [1, 2, 3],
           [1, 2, 3]], dtype=int32)>




```python
tf.argsort(mat, 0)
```




    <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[2, 1, 0],
           [1, 0, 2],
           [0, 2, 1]], dtype=int32)>

<br>

## Argmax with axis


```python
x = [[0, 1, 2],
     [2, 1, 0]]
print(np.argmax(x, 0))
print(tf.argmax(x, axis=0))
```

    [1 0 0]
    tf.Tensor([1 0 0], shape=(3,), dtype=int64)



```python
tf.argmax(x, axis=1)
```




    <tf.Tensor: shape=(2,), dtype=int64, numpy=array([2, 0])>




```python
tf.argmax(x, axis=-1)
```




    <tf.Tensor: shape=(2,), dtype=int64, numpy=array([2, 0])>

<br>

## Reshape, squeeze, expand_dims


```python
t = np.array([[[0, 1, 2], 
               [3, 4, 5]],
              
              [[6, 7, 8], 
               [9, 10, 11]]])
t.shape
```




    (2, 2, 3)




```python
tf.reshape(t, shape=[-1, 3])
```




    <tf.Tensor: shape=(4, 3), dtype=int64, numpy=
    array([[ 0,  1,  2],
           [ 3,  4,  5],
           [ 6,  7,  8],
           [ 9, 10, 11]])>




```python
tf.reshape(t, shape=[-1, 1, 3])
```




    <tf.Tensor: shape=(4, 1, 3), dtype=int64, numpy=
    array([[[ 0,  1,  2]],
    
           [[ 3,  4,  5]],
    
           [[ 6,  7,  8]],
    
           [[ 9, 10, 11]]])>




```python
tf.squeeze([[0], [1], [2]])
```




    <tf.Tensor: shape=(3,), dtype=int32, numpy=array([0, 1, 2], dtype=int32)>




```python
tf.expand_dims([0, 1, 2], 0), tf.expand_dims([0, 1, 2], 1)  # axis 추가
```




    (<tf.Tensor: shape=(1, 3), dtype=int32, numpy=array([[0, 1, 2]], dtype=int32)>,
     <tf.Tensor: shape=(3, 1), dtype=int32, numpy=
     array([[0],
            [1],
            [2]], dtype=int32)>)

<br>

## One hot


```python
tf.one_hot([[0], [1], [2], [0]], depth=3) # rank 가 하나 늘어남
```




    <tf.Tensor: shape=(4, 1, 3), dtype=float32, numpy=
    array([[[1., 0., 0.]],
    
           [[0., 1., 0.]],
    
           [[0., 0., 1.]],
    
           [[1., 0., 0.]]], dtype=float32)>




```python
t = tf.one_hot([[0], [1], [2], [0]], depth=3)
tf.reshape(t, shape=[-1, 3])
```




    <tf.Tensor: shape=(4, 3), dtype=float32, numpy=
    array([[1., 0., 0.],
           [0., 1., 0.],
           [0., 0., 1.],
           [1., 0., 0.]], dtype=float32)>

<br>

## Casting


```python
tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32)   # type 변경
```




    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([1, 2, 3, 4], dtype=int32)>




```python
print([True, False, 1 == 1, 0 == 1])
tf.cast([True, False, 1 == 1, 0 == 1], tf.int32)  # True->1, False->0
```

    [True, False, True, False]





    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([1, 0, 1, 0], dtype=int32)>

<br>

## Stack


```python
x = [1, 4]
y = [2, 5]
z = [3, 6]

# Pack along first dim.
tf.stack([x, y, z], axis=0)
```




    <tf.Tensor: shape=(3, 2), dtype=int32, numpy=
    array([[1, 4],
           [2, 5],
           [3, 6]], dtype=int32)>




```python
tf.stack([x, y, z], axis=1)
```




    <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[1, 2, 3],
           [4, 5, 6]], dtype=int32)>

<br>

## Ones like and Zeros like


```python
x = [[0, 1, 2],
     [2, 1, 0]]

tf.ones_like(x)  # 모양이 같게
```




    <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[1, 1, 1],
           [1, 1, 1]], dtype=int32)>




```python
tf.zeros_like(x)
```




    <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[0, 0, 0],
           [0, 0, 0]], dtype=int32)>

<br>

## Zip



```python
for x, y in zip([1, 2, 3], [4, 5, 6]):   # zip is iterable
    print(x, y)
```

    1 4
    2 5
    3 6



```python
for x, y, z in zip([1, 2, 3], [4, 5, 6], [7, 8, 9]):
    print(x, y, z)
```

    1 4 7
    2 5 8
    3 6 9

<br>

## Transpose


```python
t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
pp.pprint(t.shape)
pp.pprint(t)
```

    (2, 2, 3)
    array([[[ 0,  1,  2],
            [ 3,  4,  5]],
    
           [[ 6,  7,  8],
            [ 9, 10, 11]]])



```python
t1 = tf.transpose(t, [1, 0, 2])
pp.pprint(t1.shape)
pp.pprint(t1)
```

    TensorShape([2, 2, 3])
    <tf.Tensor: shape=(2, 2, 3), dtype=int64, numpy=
    array([[[ 0,  1,  2],
            [ 6,  7,  8]],
    
           [[ 3,  4,  5],
            [ 9, 10, 11]]])>



```python
t = tf.transpose(t1, [1, 0, 2])
pp.pprint(t.shape)
pp.pprint(t)
```

    TensorShape([2, 2, 3])
    <tf.Tensor: shape=(2, 2, 3), dtype=int64, numpy=
    array([[[ 0,  1,  2],
            [ 3,  4,  5]],
    
           [[ 6,  7,  8],
            [ 9, 10, 11]]])>



```python
t2 = tf.transpose(t, [1, 2, 0])
pp.pprint(t2.shape)
pp.pprint(t2)
```

    TensorShape([2, 3, 2])
    <tf.Tensor: shape=(2, 3, 2), dtype=int64, numpy=
    array([[[ 0,  6],
            [ 1,  7],
            [ 2,  8]],
    
           [[ 3,  9],
            [ 4, 10],
            [ 5, 11]]])>



```python
t = tf.transpose(t2, [2, 0, 1])
pp.pprint(t.shape)
pp.pprint(t)
```

    TensorShape([2, 2, 3])
    <tf.Tensor: shape=(2, 2, 3), dtype=int64, numpy=
    array([[[ 0,  1,  2],
            [ 3,  4,  5]],
    
           [[ 6,  7,  8],
            [ 9, 10, 11]]])>

<br>

<br>

# Numpy and Tensorflow


```python
import numpy as np
import tensorflow as tf
```


```python
tf.__version__
```




    '2.6.0'




```python
np.random.normal(0, 1, [2,3])   # mean, std, size
```




    array([[ 0.32110636, -0.11374406,  1.35794504],
           [-0.11545032, -0.76371361, -2.75919108]])




```python
tf.random.normal([2,3], 0, 1)   # shape, mean, std
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[ 0.21684866, -0.33086377, -0.17857625],
           [-0.03199087, -0.43250433, -0.4005892 ]], dtype=float32)>




```python
print(tf.random.normal([2,5]).numpy())

print(np.random.normal([5]))
print(np.random.normal(1, 1, (2,5)))

```

    [[ 0.5514615  -0.01625128  1.2484167  -1.5482845  -1.7073648 ]
     [-0.20336466  0.04439849  0.8465315   0.97438097  0.25990826]]
    [4.84382764]
    [[1.84418205 0.9041666  0.95299747 1.27679941 0.8740016 ]
     [1.53984661 0.16875311 0.80037843 0.44693226 1.89621782]]



```python
np.random.seed(17)
tf.random.set_seed(17)
print(tf.random.normal([2,3]))
print(np.random.randn(2,3))
```

    tf.Tensor(
    [[ 0.01778085  2.3094206  -0.9550922 ]
     [-1.7634274   0.4548187  -0.1849394 ]], shape=(2, 3), dtype=float32)
    [[ 0.27626589 -1.85462808  0.62390111]
     [ 1.14531129  1.03719047  1.88663893]]



```python
np.ones((2,3))
```




    array([[1., 1., 1.],
           [1., 1., 1.]])




```python
tf.ones([2,3]), tf.ones((2,3))
```




    (<tf.Tensor: shape=(2, 3), dtype=float32, numpy=
     array([[1., 1., 1.],
            [1., 1., 1.]], dtype=float32)>,
     <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
     array([[1., 1., 1.],
            [1., 1., 1.]], dtype=float32)>)




```python
tf.ones((2)).numpy()
```




    array([1., 1.], dtype=float32)




```python
tf.ones((2,)).numpy(), tf.ones((1,2)).numpy()
```




    (array([1., 1.], dtype=float32), array([[1., 1.]], dtype=float32))




```python
tf.ones((1)).numpy(), np.ones(1)
```




    (array([1.], dtype=float32), array([1.]))




```python
tf.constant(3, shape=(2,3))
```




    <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[3, 3, 3],
           [3, 3, 3]], dtype=int32)>




```python
np.array([1,2,3,4]).reshape(2,2), np.reshape([1,2,3,4], (2,2))
```




    (array([[1, 2],
            [3, 4]]), array([[1, 2],
            [3, 4]]))




```python
tf.constant([1,2,3,4], shape=[2,2])
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[1, 2],
           [3, 4]], dtype=int32)>




```python
t = tf.constant([1,2,3,4])
tf.reshape(t, shape=[2,2])
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[1, 2],
           [3, 4]], dtype=int32)>




```python
x = tf.constant([1,2,3,4], shape=[2,2])
tf.zeros_like(x)
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[0, 0],
           [0, 0]], dtype=int32)>




```python
tf.fill([3,3], 6)
```




    <tf.Tensor: shape=(3, 3), dtype=int32, numpy=
    array([[6, 6, 6],
           [6, 6, 6],
           [6, 6, 6]], dtype=int32)>

<br>


## Define and initializing variables



```python
a0 = tf.Variable([1,2,3,4], dtype = tf.float32)
b = tf.constant(2, tf.float32)
a0*b
```




    <tf.Tensor: shape=(4,), dtype=float32, numpy=array([2., 4., 6., 8.], dtype=float32)>




```python
tf.multiply(a0, b) == a0 * b
```




    <tf.Tensor: shape=(4,), dtype=bool, numpy=array([ True,  True,  True,  True])>




```python
A2 = tf.constant([[1,2],[3,4]])
B2 = tf.constant([[5,6],[7,8]])
A2 + B2, A2 * B2, tf.add(A2, B2), tf.multiply(A2, B2)  # element-wise
```




    (<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
     array([[ 6,  8],
            [10, 12]], dtype=int32)>, <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
     array([[ 5, 12],
            [21, 32]], dtype=int32)>, <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
     array([[ 6,  8],
            [10, 12]], dtype=int32)>, <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
     array([[ 5, 12],
            [21, 32]], dtype=int32)>)




```python
tf.matmul(A2, B2)
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[19, 22],
           [43, 50]], dtype=int32)>




```python
tf.reduce_sum(A2), tf.reduce_sum(A2, axis=0), tf.reduce_sum(A2, axis=1)
```




    (<tf.Tensor: shape=(), dtype=int32, numpy=10>,
     <tf.Tensor: shape=(2,), dtype=int32, numpy=array([4, 6], dtype=int32)>,
     <tf.Tensor: shape=(2,), dtype=int32, numpy=array([3, 7], dtype=int32)>)

<br>

## advanced functions

- gradient(), reshape(), random()



```python
import tensorflow as tf

x = tf.Variable(-1.0)
with tf.GradientTape() as tape:
    tape.watch(x)
    y = tf.multiply(x, x)
    
g = tape.gradient(y, x)
print(g.numpy())
```

    -2.0



```python
gray1 = tf.random.uniform([2,2], maxval=255, dtype='int32')
gray2 = tf.reshape(gray1, [2*2, 1])
gray1, gray2
```




    (<tf.Tensor: shape=(2, 2), dtype=int32, numpy=
     array([[ 68, 184],
            [226, 168]], dtype=int32)>,
     <tf.Tensor: shape=(4, 1), dtype=int32, numpy=
     array([[ 68],
            [184],
            [226],
            [168]], dtype=int32)>)




```python
color = tf.random.uniform([2,2,3], maxval=255, dtype='int32') # color image
tf.reshape(color, [2*2, 3])
```




    <tf.Tensor: shape=(4, 3), dtype=int32, numpy=
    array([[131,  95, 245],
           [ 35, 138, 193],
           [144,  59,  43],
           [ 35,  71, 206]], dtype=int32)>

<br>
