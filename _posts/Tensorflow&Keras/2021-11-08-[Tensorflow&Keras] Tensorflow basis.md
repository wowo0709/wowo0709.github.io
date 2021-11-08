---
layout: single
title: "[Machine Learning] Tensorflow Basis"
categories: ['AI', 'Tensorflow&Keras']
toc: true
toc_sticky: true
tag: []
---

<br>

- from Chapter 12 of Hands-on-Machine-Learning-2nd

## Setup

First, let's import a few common modules, ensure MatplotLib plots figures inline and prepare a function to save the figures. We also check that Python 3.5 or later is installed (although Python 2.x may work, it is deprecated so we strongly recommend you use Python 3 instead), as well as Scikit-Learn ≥0.20 and TensorFlow ≥2.0.


```python
# Python ≥3.5 is required
import sys
assert sys.version_info >= (3, 5)

# Scikit-Learn ≥0.20 is required
import sklearn
assert sklearn.__version__ >= "0.20"

try:
    # %tensorflow_version only exists in Colab.
    %tensorflow_version 2.x
except Exception:
    pass

# TensorFlow ≥2.4 is required in this notebook
# Earlier 2.x versions will mostly work the same, but with a few bugs
import tensorflow as tf
from tensorflow import keras
assert tf.__version__ >= "2.4"

# Common imports
import numpy as np
import os

# to make this notebook's output stable across runs
np.random.seed(42)
tf.random.set_seed(42)

# To plot pretty figures
%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "deep"
IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID)
os.makedirs(IMAGES_PATH, exist_ok=True)

def save_fig(fig_id, tight_layout=True, fig_extension="png", resolution=300):
    path = os.path.join(IMAGES_PATH, fig_id + "." + fig_extension)
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format=fig_extension, dpi=resolution)
```

<br>

## Tensors and operations

### Tensors


```python
tf.constant([[1., 2., 3.], [4., 5., 6.]]) # matrix
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[1., 2., 3.],
           [4., 5., 6.]], dtype=float32)>




```python
tf.constant(42) # scalar
```




    <tf.Tensor: shape=(), dtype=int32, numpy=42>




```python
t = tf.constant([[1., 2., 3.], [4., 5., 6.]])
t
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[1., 2., 3.],
           [4., 5., 6.]], dtype=float32)>




```python
t.shape
```




    TensorShape([2, 3])




```python
t.dtype
```




    tf.float32

<br>

### Indexing


```python
# ellipsis(...) : 나머지, 생략된 부분
a = np.arange(12).reshape(3,4)
a[:], a[1,...], a[...,1]
```




    (array([[ 0,  1,  2,  3],
            [ 4,  5,  6,  7],
            [ 8,  9, 10, 11]]), array([4, 5, 6, 7]), array([1, 5, 9]))




```python
t[:, 1:]
```




    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[2., 3.],
           [5., 6.]], dtype=float32)>




```python
# in arrays
a = np.array([1,2,3,4])
print(a.shape, a[np.newaxis, :].shape, a[:, np.newaxis].shape, a[...].shape)
a[1], a[1, np.newaxis], a[..., np.newaxis], a[np.newaxis, ...]
```

    (4,) (1, 4) (4, 1) (4,)





    (2, array([2]), array([[1],
            [2],
            [3],
            [4]]), array([[1, 2, 3, 4]]))




```python
t
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[1., 2., 3.],
           [4., 5., 6.]], dtype=float32)>




```python
t[:, 1], t[..., 1], t[..., 1, tf.newaxis], t[:, 1, np.newaxis]
```




    (<tf.Tensor: shape=(2,), dtype=float32, numpy=array([2., 5.], dtype=float32)>,
     <tf.Tensor: shape=(2,), dtype=float32, numpy=array([2., 5.], dtype=float32)>,
     <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
     array([[2.],
            [5.]], dtype=float32)>,
     <tf.Tensor: shape=(2, 1), dtype=float32, numpy=
     array([[2.],
            [5.]], dtype=float32)>)

<br>

### Ops


```python
t
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[1., 2., 3.],
           [4., 5., 6.]], dtype=float32)>




```python
t + 10
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[11., 12., 13.],
           [14., 15., 16.]], dtype=float32)>




```python
tf.square(t)
```




    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[ 1.,  4.,  9.],
           [16., 25., 36.]], dtype=float32)>




```python
t @ tf.transpose(t)    # @: matrix mult (newly added in Python 3.5)
```




    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[14., 32.],
           [32., 77.]], dtype=float32)>




```python
tf.matmul(t, tf.transpose(t))
```




    <tf.Tensor: shape=(2, 2), dtype=float32, numpy=
    array([[14., 32.],
           [32., 77.]], dtype=float32)>

<br>

### Using `keras.backend`


```python
from tensorflow import keras
K = keras.backend
K.square(K.transpose(t)), tf.square(tf.transpose(t))
```




    (<tf.Tensor: shape=(3, 2), dtype=float32, numpy=
     array([[ 1., 16.],
            [ 4., 25.],
            [ 9., 36.]], dtype=float32)>,
     <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
     array([[ 1., 16.],
            [ 4., 25.],
            [ 9., 36.]], dtype=float32)>)

<br>

### From/To NumPy


```python
a = np.array([2., 4., 5.])
tf.constant(a)
```




    <tf.Tensor: shape=(3,), dtype=float64, numpy=array([2., 4., 5.])>




```python
t.numpy()
```




    array([[1., 2., 3.],
           [4., 5., 6.]], dtype=float32)




```python
np.array(t)
```




    array([[1., 2., 3.],
           [4., 5., 6.]], dtype=float32)




```python
tf.square(a)
```




    <tf.Tensor: shape=(3,), dtype=float64, numpy=array([ 4., 16., 25.])>




```python
np.square(t)
```




    array([[ 1.,  4.,  9.],
           [16., 25., 36.]], dtype=float32)

<br>

### Conflicting Types
- Tensorflow does not perform any type conversion automatically not to hurt performance


```python
np.array([2.0]).dtype   # numpy uses 64-bit floating
```




    dtype('float64')




```python
tf.constant([2.0]).dtype  # tensors use 32-bit floating
```




    tf.float32




```python
np.array([2.0]) + np.array([30])
```




    array([32.])




```python
tf.constant([2.0]) + tf.constant([30])
```


    ---------------------------------------------------------------------------
    
    InvalidArgumentError                      Traceback (most recent call last)
    
    <ipython-input-28-bb1d4d7dd0f5> in <module>()
    ----> 1 tf.constant([2.0]) + tf.constant([30])


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/math_ops.py in binary_op_wrapper(x, y)
       1365         #   r_binary_op_wrapper use different force_same_dtype values.
       1366         x, y = maybe_promote_tensors(x, y, force_same_dtype=False)
    -> 1367         return func(x, y, name=name)
       1368       except (TypeError, ValueError) as e:
       1369         # Even if dispatching the op failed, the RHS may be a tensor aware


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/util/dispatch.py in wrapper(*args, **kwargs)
        204     """Call target, and fall back on dispatchers if there is a TypeError."""
        205     try:
    --> 206       return target(*args, **kwargs)
        207     except (TypeError, ValueError):
        208       # Note: convert_to_eager_tensor currently raises a ValueError, not a


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/math_ops.py in _add_dispatch(x, y, name)
       1698     return gen_math_ops.add(x, y, name=name)
       1699   else:
    -> 1700     return gen_math_ops.add_v2(x, y, name=name)
       1701 
       1702 


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/ops/gen_math_ops.py in add_v2(x, y, name)
        453       return _result
        454     except _core._NotOkStatusException as e:
    --> 455       _ops.raise_from_not_ok_status(e, name)
        456     except _core._FallbackException:
        457       pass


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
       6939   message = e.message + (" name: " + name if name is not None else "")
       6940   # pylint: disable=protected-access
    -> 6941   six.raise_from(core._status_to_exception(e.code, message), None)
       6942   # pylint: enable=protected-access
       6943 


    /usr/local/lib/python3.7/dist-packages/six.py in raise_from(value, from_value)


    InvalidArgumentError: cannot compute AddV2 as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:AddV2]



```python
try:
    tf.constant(2.0) + tf.constant(40)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
```

    cannot compute AddV2 as input #1(zero-based) was expected to be a float tensor but is a int32 tensor [Op:AddV2]



```python
try:
    tf.constant(2.0) + tf.constant(40., dtype=tf.float64)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
```

    cannot compute AddV2 as input #1(zero-based) was expected to be a float tensor but is a double tensor [Op:AddV2]



```python
t2 = tf.constant(40., dtype=tf.float64)
tf.constant(2.0) + tf.cast(t2, tf.float32)
```




    <tf.Tensor: shape=(), dtype=float32, numpy=42.0>

<br>

### Strings


```python
tf.constant("hello world, 팀")   # b(byte) in only ascii literal characters
```




    <tf.Tensor: shape=(), dtype=string, numpy=b'hello world, \xed\x8c\x80'>




```python
tf.constant(b"hello world, 팀") 
```


      File "<ipython-input-37-be52ab402b39>", line 1
        tf.constant(b"hello world, 팀")
                   ^
    SyntaxError: bytes can only contain ASCII literal characters.




```python
tf.constant(["café", "파이썬"])  # unicode by default
```




    <tf.Tensor: shape=(2,), dtype=string, numpy=
    array([b'caf\xc3\xa9', b'\xed\x8c\x8c\xec\x9d\xb4\xec\x8d\xac'],
          dtype=object)>




```python
# exercise: 
# ord() <---> chr() : exchange between integer (for unicode point) and character
# ord() stands for “ordinal”. It is the number representing the position of c 
# in the sequence of Unicode codepoints. (문자와 아스키(유니)코드 변환)
ord(" "), ord("A"), ord('B'), ord("C"), ord("통"), chr(65), chr(233), chr(53685)
```




    (32, 65, 66, 67, 53685, 'A', 'é', '통')




```python
u_text = "A 쌍"
print("ascii (or unicode) numbers: \t", [ord(i) for i in u_text])
b_text = [i.encode() for i in u_text]
print("unicode encoding: \t\t", b_text)
print("decoding: \t\t\t", [i.decode() for i in b_text])
```

    ascii (or unicode) numbers: 	 [65, 32, 49933]
    unicode encoding: 		 [b'A', b' ', b'\xec\x8c\x8d']
    decoding: 			 ['A', ' ', '쌍']



```python
u = tf.constant([ord(c) for c in "caféx통"])
u
```




    <tf.Tensor: shape=(6,), dtype=int32, numpy=array([   99,    97,   102,   233,   120, 53685], dtype=int32)>

<br>

### String arrays


```python
p = tf.constant(["Café", "Coffee", "咖啡", "커피"])
p
```




    <tf.Tensor: shape=(4,), dtype=string, numpy=
    array([b'Caf\xc3\xa9', b'Coffee', b'\xe5\x92\x96\xe5\x95\xa1',
           b'\xec\xbb\xa4\xed\x94\xbc'], dtype=object)>




```python
tf.strings.length(p, unit="UTF8_CHAR")
```




    <tf.Tensor: shape=(4,), dtype=int32, numpy=array([4, 6, 2, 2], dtype=int32)>




```python
r = tf.strings.unicode_decode(p, "UTF8")
r
```




    <tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101], [21654, 21857], [52964, 54588]]>




```python
print(r)
```

    <tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101], [21654, 21857], [52964, 54588]]>

<br>

### Ragged tensors

- A RaggedTensor is a tensor with one or more ragged dimensions, which are dimensions whose slices may have different lengths. 
- For example, the inner (column) dimension of rt=[[3, 1, 4, 1], [], [5, 9, 2], [6], []] is ragged
- tensor arrays: list of tensors
- ragged tensors: static lists of lists of tensors




```python
r
```




    <tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101], [21654, 21857], [52964, 54588]]>




```python
print(r[1])
```

    tf.Tensor([ 67 111 102 102 101 101], shape=(6,), dtype=int32)



```python
print(r[1:3])
print(r[2:])
```

    <tf.RaggedTensor [[67, 111, 102, 102, 101, 101], [21654, 21857]]>
    <tf.RaggedTensor [[21654, 21857], [52964, 54588]]>



```python
try:
    x = tf.constant([[65, 66], [], [67]])
except:
    print("Error: Can't convert non-rectangular Python sequence to Tensor.")

```

    Error: Can't convert non-rectangular Python sequence to Tensor.



```python
r2 = tf.ragged.constant([[65, 66], [], [67]])
tf.concat([r, r2], axis=0)
```




    <tf.RaggedTensor [[67, 97, 102, 233], [67, 111, 102, 102, 101, 101], [21654, 21857], [52964, 54588], [65, 66], [], [67]]>




```python
tf.concat([r, r2], axis=0).to_tensor()  # tensor, not ragged_tensor
```




    <tf.Tensor: shape=(7, 6), dtype=int32, numpy=
    array([[   67,    97,   102,   233,     0,     0],
           [   67,   111,   102,   102,   101,   101],
           [21654, 21857,     0,     0,     0,     0],
           [52964, 54588,     0,     0,     0,     0],
           [   65,    66,     0,     0,     0,     0],
           [    0,     0,     0,     0,     0,     0],
           [   67,     0,     0,     0,     0,     0]], dtype=int32)>




```python
r3 = tf.ragged.constant([[68, 69, 70], [71], [], [72, 73]])
print(tf.concat([r, r3], axis=1))
```

    <tf.RaggedTensor [[67, 97, 102, 233, 68, 69, 70], [67, 111, 102, 102, 101, 101, 71], [21654, 21857], [52964, 54588, 72, 73]]>



```python
tf.strings.unicode_encode(r3, "UTF-8")
```




    <tf.Tensor: shape=(4,), dtype=string, numpy=array([b'DEF', b'G', b'', b'HI'], dtype=object)>




```python
r.to_tensor()
```




    <tf.Tensor: shape=(4, 6), dtype=int32, numpy=
    array([[   67,    97,   102,   233,     0,     0],
           [   67,   111,   102,   102,   101,   101],
           [21654, 21857,     0,     0,     0,     0],
           [52964, 54588,     0,     0,     0,     0]], dtype=int32)>

<br>

### Sparse tensors


```python
s = tf.SparseTensor(indices=[[0, 1], [1, 0], [2, 3]],
                    values=[1., 2., 3.],
                    dense_shape=[3, 4])
```


```python
print(s)
```

    SparseTensor(indices=tf.Tensor(
    [[0 1]
     [1 0]
     [2 3]], shape=(3, 2), dtype=int64), values=tf.Tensor([1. 2. 3.], shape=(3,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))



```python
tf.sparse.to_dense(s)
```




    <tf.Tensor: shape=(3, 4), dtype=float32, numpy=
    array([[0., 1., 0., 0.],
           [2., 0., 0., 0.],
           [0., 0., 0., 3.]], dtype=float32)>




```python
s2 = s * 2.0
```


```python
tf.sparse.to_dense(s2) + 1
```




    <tf.Tensor: shape=(3, 4), dtype=float32, numpy=
    array([[1., 3., 1., 1.],
           [5., 1., 1., 1.],
           [1., 1., 1., 7.]], dtype=float32)>




```python
try:
    s3 = s + 1.
except TypeError as ex:
    print(ex)
```

    unsupported operand type(s) for +: 'SparseTensor' and 'float'



```python
print(tf.sparse.to_dense(s))
```

    tf.Tensor(
    [[0. 1. 0. 0.]
     [2. 0. 0. 0.]
     [0. 0. 0. 3.]], shape=(3, 4), dtype=float32)



```python
s4 = tf.constant([[10., 20.], [30., 40.], [50., 60.], [70., 80.]])
tf.sparse.sparse_dense_matmul(s, s4)
```




    <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
    array([[ 30.,  40.],
           [ 20.,  40.],
           [210., 240.]], dtype=float32)>




```python
s4
```




    <tf.Tensor: shape=(4, 2), dtype=float32, numpy=
    array([[10., 20.],
           [30., 40.],
           [50., 60.],
           [70., 80.]], dtype=float32)>




```python
s5 = tf.SparseTensor(indices=[[0, 2], [0, 1]],
                     values=[1., 2.],
                     dense_shape=[3, 4])
print(s5)
```

    SparseTensor(indices=tf.Tensor(
    [[0 2]
     [0 1]], shape=(2, 2), dtype=int64), values=tf.Tensor([1. 2.], shape=(2,), dtype=float32), dense_shape=tf.Tensor([3 4], shape=(2,), dtype=int64))



```python
try:
    tf.sparse.to_dense(s5)
except tf.errors.InvalidArgumentError as ex:
    print(ex)
```

    indices[1] = [0,1] is out of order. Many sparse ops require sorted indices.
        Use `tf.sparse.reorder` to create a correctly ordered copy.
    
     [Op:SparseToDense]



```python
s6 = tf.sparse.reorder(s5)
tf.sparse.to_dense(s6)
```




    <tf.Tensor: shape=(3, 4), dtype=float32, numpy=
    array([[0., 2., 1., 0.],
           [0., 0., 0., 0.],
           [0., 0., 0., 0.]], dtype=float32)>

<br>

### Sets


```python
set1 = tf.constant([[2, 3, 5, 7], [7, 9, 0, 0]])
set2 = tf.constant([[4, 5, 6], [9, 10, 0]])
tf.sparse.to_dense(tf.sets.union(set1, set2))
```




    <tf.Tensor: shape=(2, 6), dtype=int32, numpy=
    array([[ 2,  3,  4,  5,  6,  7],
           [ 0,  7,  9, 10,  0,  0]], dtype=int32)>




```python
tf.sparse.to_dense(tf.sets.difference(set1, set2))
```




    <tf.Tensor: shape=(2, 3), dtype=int32, numpy=
    array([[2, 3, 7],
           [7, 0, 0]], dtype=int32)>




```python
tf.sparse.to_dense(tf.sets.intersection(set1, set2))
```




    <tf.Tensor: shape=(2, 2), dtype=int32, numpy=
    array([[5, 0],
           [0, 9]], dtype=int32)>

<br>

### Variables
- can be changable (mutable)


```python
t = tf.constant([1, 2, 3])
print (t)
try:
    t[0] = 7    # immutable
except:
    print ("does not support item assignment.")
```

    tf.Tensor([1 2 3], shape=(3,), dtype=int32)
    does not support item assignment.



```python
t2 = tf.Variable([1, 2, 3])   # almost the same as tf.Tensor
t2[1] = 7
```


    ---------------------------------------------------------------------------
    
    TypeError                                 Traceback (most recent call last)
    
    <ipython-input-93-17dc10ff56f4> in <module>()
          1 t2 = tf.Variable([1, 2, 3])   # almost the same as tf.Tensor
    ----> 2 t2[1] = 7


    TypeError: 'ResourceVariable' object does not support item assignment



```python
t2 = tf.Variable([1, 2, 3]) # mutable and can be assigned using assign() method
t2[2].assign(7)
```




    <tf.Variable 'UnreadVariable' shape=(3,) dtype=int32, numpy=array([1, 2, 7], dtype=int32)>




```python
t2.assign([7,8,9])
```




    <tf.Variable 'UnreadVariable' shape=(3,) dtype=int32, numpy=array([7, 8, 9], dtype=int32)>




```python
t2.assign_sub([1,1,1])
```




    <tf.Variable 'UnreadVariable' shape=(3,) dtype=int32, numpy=array([6, 7, 8], dtype=int32)>




```python
t2.assign_add([1,1,1])
```




    <tf.Variable 'UnreadVariable' shape=(3,) dtype=int32, numpy=array([7, 8, 9], dtype=int32)>




```python
v = tf.Variable([[1., 2., 3.], [4., 5., 6.]])
```


```python
v.assign(2 * v)
```




    <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
    array([[ 2.,  4.,  6.],
           [ 8., 10., 12.]], dtype=float32)>




```python
v[0, 1].assign(42)
```




    <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
    array([[ 2., 42.,  6.],
           [ 8., 10., 12.]], dtype=float32)>




```python
v[:, 2].assign([77., 88.])
```




    <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
    array([[ 2., 42., 77.],
           [ 8., 10., 88.]], dtype=float32)>




```python
try:
    v[1] = [7., 8., 9.]
except TypeError as ex:
    print(ex)
```

    'ResourceVariable' object does not support item assignment



```python
# scatter_nd_update(): Scatter updates into an existing tensor according to indices.
v.scatter_nd_update(indices=[[0, 0], [1, 2]],
                    updates=[100., 200.]) 
```




    <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
    array([[100.,  42.,  77.],
           [  8.,  10., 200.]], dtype=float32)>




```python
# tf.IndexedSlices(): A sparse representation of a set of tensor slices at given indices.
sparse_delta = tf.IndexedSlices(values=[[1., 2., 3.], [4., 5., 6.]],
                                indices=[1, 0])
v.scatter_update(sparse_delta)
```




    <tf.Variable 'UnreadVariable' shape=(2, 3) dtype=float32, numpy=
    array([[4., 5., 6.],
           [1., 2., 3.]], dtype=float32)>

<br>

### Tensor Arrays
- Class wrapping dynamic-sized, per-time-step, write-once Tensor arrays.


```python
array = tf.TensorArray(dtype=tf.float32, size=3)
array = array.write(0, tf.constant([1., 2.]))
array = array.write(1, tf.constant([3., 10.]))
array = array.write(2, tf.constant([5., 7.]))
```


```python
array
```




    <tensorflow.python.ops.tensor_array_ops.TensorArray at 0x7f2d856c92d0>




```python
array.stack()
```




    <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
    array([[ 1.,  2.],
           [ 3., 10.],
           [ 5.,  7.]], dtype=float32)>




```python
array.read(1)
```




    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([ 3., 10.], dtype=float32)>




```python
array.stack()
```




    <tf.Tensor: shape=(3, 2), dtype=float32, numpy=
    array([[1., 2.],
           [0., 0.],
           [5., 7.]], dtype=float32)>



- tf.nn (primitive neural net) operations


```python
mean, variance = tf.nn.moments(array.stack(), axes=0)
mean
```




    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([2., 3.], dtype=float32)>




```python
variance
```




    <tf.Tensor: shape=(2,), dtype=float32, numpy=array([4.6666665, 8.666667 ], dtype=float32)>

<br>

<br>

## Computing Gradients with Autodiff


```python
def f(w1, w2):
    return 3 * w1 ** 2 + 2 * w1 * w2
```


```python
w1, w2 = 5, 3
eps = 1e-6
(f(w1 + eps, w2) - f(w1, w2)) / eps
```




    36.000003007075065




```python
(f(w1, w2 + eps) - f(w1, w2)) / eps
```




    10.000000003174137




```python
w1, w2 = tf.Variable(5.), tf.Variable(3.)
with tf.GradientTape() as tape:
    z = f(w1, w2)

gradients = tape.gradient(z, [w1, w2])
```


```python
gradients
```




    [<tf.Tensor: shape=(), dtype=float32, numpy=36.0>,
     <tf.Tensor: shape=(), dtype=float32, numpy=10.0>]




```python
with tf.GradientTape() as tape:
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1)  # will be erased immediately after the call 
try:
    dz_dw2 = tape.gradient(z, w2)
except RuntimeError as ex:
    print(ex)
```

    A non-persistent GradientTape can only be used to compute one set of gradients (or jacobians)



```python
with tf.GradientTape(persistent=True) as tape:
    z = f(w1, w2)

dz_dw1 = tape.gradient(z, w1)
dz_dw2 = tape.gradient(z, w2) # works now!
del tape
```


```python
dz_dw1, dz_dw2
```




    (<tf.Tensor: shape=(), dtype=float32, numpy=36.0>,
     <tf.Tensor: shape=(), dtype=float32, numpy=10.0>)




```python
c1, c2 = tf.constant(5.), tf.constant(3.)
with tf.GradientTape() as tape:
    z = f(c1, c2)

gradients = tape.gradient(z, [c1, c2])
```


```python
gradients
```




    [None, None]




```python
with tf.GradientTape() as tape:
    tape.watch(c1)
    tape.watch(c2)
    z = f(c1, c2)

gradients = tape.gradient(z, [c1, c2])
```


    ---------------------------------------------------------------------------
    
    NameError                                 Traceback (most recent call last)
    
    <ipython-input-1-5ad75c746ad7> in <module>()
    ----> 1 with tf.GradientTape() as tape:
          2     tape.watch(c1)
          3     tape.watch(c2)
          4     z = f(c1, c2)
          5 


    NameError: name 'tf' is not defined



```python
gradients
```




    [<tf.Tensor: shape=(), dtype=float32, numpy=36.0>,
     <tf.Tensor: shape=(), dtype=float32, numpy=10.0>]




```python
with tf.GradientTape() as tape:
    z1 = f(w1, w2 + 2.)
    z2 = f(w1, w2 + 5.)
    z3 = f(w1, w2 + 7.)

tape.gradient([z1, z2, z3], [w1, w2])  # returns gradients of the vector's sum
# if you want individuals, use tape's jacobian() method
```




    [<tf.Tensor: shape=(), dtype=float32, numpy=136.0>,
     <tf.Tensor: shape=(), dtype=float32, numpy=30.0>]




```python
with tf.GradientTape() as tape:   # False 는 한 번만 호출, True 는 테이프 영구적
    z1 = f(w1, w2 + 2.)
    z2 = f(w1, w2 + 5.)
    z3 = f(w1, w2 + 7.)

tf.reduce_sum(tf.stack([tape.gradient(z, [w1, w2]) for z in (z1, z2, z3)]), axis=0)
del tape
```


    ---------------------------------------------------------------------------
    
    RuntimeError                              Traceback (most recent call last)
    
    <ipython-input-190-93dc17cf845c> in <module>()
          4     z3 = f(w1, w2 + 7.)
          5 
    ----> 6 tf.reduce_sum(tf.stack([tape.gradient(z, [w1, w2]) for z in (z1, z2, z3)]), axis=0)
          7 del tape


    <ipython-input-190-93dc17cf845c> in <listcomp>(.0)
          4     z3 = f(w1, w2 + 7.)
          5 
    ----> 6 tf.reduce_sum(tf.stack([tape.gradient(z, [w1, w2]) for z in (z1, z2, z3)]), axis=0)
          7 del tape


    /usr/local/lib/python3.7/dist-packages/tensorflow/python/eager/backprop.py in gradient(self, target, sources, output_gradients, unconnected_gradients)
       1030     """
       1031     if self._tape is None:
    -> 1032       raise RuntimeError("A non-persistent GradientTape can only be used to "
       1033                          "compute one set of gradients (or jacobians)")
       1034     if self._recording:


    RuntimeError: A non-persistent GradientTape can only be used to compute one set of gradients (or jacobians)



```python
with tf.GradientTape(persistent=True) as tape:   # False 는 한 번만 호출, True 는 테이프 영구적
    z1 = f(w1, w2 + 2.)
    z2 = f(w1, w2 + 5.)
    z3 = f(w1, w2 + 7.)

tf.reduce_sum(tf.stack([tape.gradient(z, [w1, w2]) for z in (z1, z2, z3)]), axis=0)
del tape
```

<br>
