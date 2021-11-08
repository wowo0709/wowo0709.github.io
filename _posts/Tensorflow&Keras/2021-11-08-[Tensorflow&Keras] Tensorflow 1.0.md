---
layout: single
title: "[Tensorflow&Keras] Tensorflow 1.0"
categories: ['AI', 'MachineLearning']
toc: true
toc_sticky: true
tag: []

---

<br>

## Tensors

- 텐서플로 프로그램은 tf.Tensor 객체 그래프를 만드는 것으로 먼저 시작하고, 각각의 텐서가 다른 텐서를 기반으로 어떤 식으로 계산될 수 있는지 구체화하고, 그 다음 그래프를 실행해서 원하는 결과를 얻게 됩니다.
- tf.Tensor는 다음과 같은 속성을 가지고 있습니다:

  - data type (예를 들어, float32 또는 int32, string)
  - 형태(shape)
- 텐서안의 각각 원소는 동일한 자료형이며 핵심 텐서는 다음과 같음. 
  - `constant`: values that don’t change.
  - `placeholder`: values that are unassigned (will be initialized by the session when you run it)
  - `variable`:  values that can change
  - tf.SparseTensor: 


```python
# colab has both versions
# %tensorflow_version 1.x
# import tensorflow as tf
# print(tf.__version__)
```

    TensorFlow 1.x selected.
    1.15.2



```python
# local desktop
import tensorflow.compat.v1 as tf
tf.compat.v1.disable_eager_execution()
tf.__version__
```




    '2.5.0'

<br>

<br>

## Basic exercise

텐서플로의 연산 메서드를 사용하면 값을 계산하는 것이 아니라, 해당 연산의 그래프를 생성한다. 


```python
# basic exercise
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])
result = tf.multiply(x1, x2) 
print(result)   # just define a model (not calculated) 
```

    Tensor("Mul:0", shape=(4,), dtype=int32)


연산 그래프의 결과를 출력하고 싶다면 `tf.Session`을 사용하여 graph를 execution해야 한다. 


```python
# if you want to see the result value?
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

result = tf.multiply(x1, x2)
print(result)

sess = tf.Session()  # Intialize the Session
print(sess.run(result)) # Print the result

sess.close()  # Close the session
```

    Tensor("Mul_1:0", shape=(4,), dtype=int32)
    [ 5 12 21 32]


그리고 **Session**을 사용하는 방식은 일반적으로 다음과 같다. 


```python
x1 = tf.constant([1,2,3,4])
x2 = tf.constant([5,6,7,8])

# Multiply
result = tf.multiply(x1, x2)

# Initialize Session and run `result`
with tf.Session() as sess:
  output = sess.run(result)
  print(output)
```

    [ 5 12 21 32]

<br>

## Tensor Basic variables

- 최소 한 개 이상의 열과 행으로 구성


```python
c0 = tf.constant(2.7)
c1 = tf.constant([1,2,3])
c0, type(c0), c1, type(c1)
```




    (<tf.Tensor 'Const_6:0' shape=() dtype=float32>,
     tensorflow.python.framework.ops.Tensor,
     <tf.Tensor 'Const_7:0' shape=(3,) dtype=int32>,
     tensorflow.python.framework.ops.Tensor)




```python
# all rank 0 (with initial values)
# 문자열은 문자 시퀀스가 아니고 단일 객체로 다루어짐.
mammal = tf.Variable("코끼리", tf.string)
ignition = tf.Variable(451, tf.int16)
floating = tf.Variable(3.14159265359, tf.float64)
its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)
```


```python
mammal, type(mammal), tf.rank(mammal)
```




    (<tf.Variable 'Variable:0' shape=() dtype=string>,
     tensorflow.python.ops.resource_variable_ops.ResourceVariable,
     <tf.Tensor 'Rank:0' shape=() dtype=int32>)




```python
# rank 1
mystr = tf.Variable(["안녕하세요"], tf.string)
f_numbers  = tf.Variable([3.14159, 2.71828], tf.float32)
primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
comp = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex128)
```


```python
comp.dtype, type(comp), mystr
```




    (tf.complex128,
     tensorflow.python.ops.resource_variable_ops.ResourceVariable,
     <tf.Variable 'Variable_4:0' shape=(1,) dtype=string>)




```python
# rank: higher degree 
data1 = tf.Variable([1, 2], tf.int32)
data2 = tf.Variable([[1., 2.]], tf.float32)
data3 = tf.Variable([[False, True],[True, False]], tf.bool)
data4 = tf.Variable([[4, 2], [9, 3], [16, 4], [25, 5]], tf.int32)
```


```python
data1, data2, data3, data4, type(data4)
```




    (<tf.Variable 'Variable_8:0' shape=(2,) dtype=int32>,
     <tf.Variable 'Variable_9:0' shape=(1, 2) dtype=float32>,
     <tf.Variable 'Variable_10:0' shape=(2, 2) dtype=bool>,
     <tf.Variable 'Variable_11:0' shape=(4, 2) dtype=int32>,
     tensorflow.python.ops.resource_variable_ops.ResourceVariable)




```python
data1.shape, data2.shape, data3.shape, data4.shape, type(data4.shape)
```




    (TensorShape([2]),
     TensorShape([1, 2]),
     TensorShape([2, 2]),
     TensorShape([4, 2]),
     tensorflow.python.framework.tensor_shape.TensorShape)




```python
my_image = tf.zeros([10, 299, 299, 3])  # 4 차원 텐서: 배치 x 높이 x 너비 x 색상
```


```python
my_image
```




    <tf.Tensor 'zeros:0' shape=(10, 299, 299, 3) dtype=float32>




```python
r = tf.rank(my_image)   # scalar value
print(r)
with tf.Session() as sess:
    print(r.eval())
```

    Tensor("Rank_1:0", shape=(), dtype=int32)
    4



```python
rank_3_tensor = tf.ones([3, 4, 5])
matrix = tf.reshape(rank_3_tensor, [6, 10]) 
matrixB = tf.reshape(matrix, [3, -1])
matrixAlt = tf.reshape(matrixB, [4, 3, -1])  # -1은 자동 결정하라는 뜻
# yet_another = tf.reshape(matrixAlt, [13, 2, -1])  # 에러!
```


```python
matrix, matrixB, matrixAlt
```




    (<tf.Tensor 'Reshape:0' shape=(6, 10) dtype=float32>,
     <tf.Tensor 'Reshape_1:0' shape=(3, 20) dtype=float32>,
     <tf.Tensor 'Reshape_2:0' shape=(4, 3, 5) dtype=float32>)




```python
data1.dtype
```




    tf.int32




```python
f_tensor = tf.cast(tf.constant([1, 2, 3]), dtype=tf.float32)
f_tensor.dtype
```




    tf.float32




```python
sp = tf.SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], dense_shape=[3, 4])
with tf.Session() as sess:
    print(sp.eval())
```

    SparseTensorValue(indices=array([[0, 0],
           [1, 2]], dtype=int64), values=array([1, 2]), dense_shape=array([3, 4], dtype=int64))

<br>

## Tensor evaluation

- useful when debugging
- use Tensor.eval method (note that eval() works only when session is activated


```python
constant = tf.constant([1, 2, 3])
v1 = constant * constant
try:
    print(v1.eval())
except:
    print("error")
```

    error



```python
with tf.Session() as sess:
    print(v1.eval())

```

    [1 4 9]



```python
# method 1
sess = tf.Session()
print(sess.run(v1))
sess.close()
# method 2
with tf.Session() as sess:
    print(sess.run(v1))
```

    [1 4 9]
    [1 4 9]



```python
with tf.Session() as sess:
    p = tf.placeholder(tf.float32)
    t1 = p + 1.0
    t2 = p + 2.0
    # t.eval()    # error
    print(t1.eval(feed_dict={p:2.0}))
    print(t2.eval(feed_dict={p:2.0}))
    print(sess.run([t1,t2], feed_dict={p: 2.0}))  # many tensor values
```

    3.0
    4.0
    [3.0, 4.0]

<br>

## Exercise

- Linear Regression with Tensorflow 1.x


```python
import numpy as np 
import tensorflow as tf 
import matplotlib.pyplot as plt
```


```python
np.random.seed(17) 
n = 100
x = np.random.randn(n)                # batch size
y = x*10 + 10                         # w=20, b=10 ???
y = y + np.random.randn(n) * 7       # add noise
  
n = len(x) # Number of data points

plt.scatter(x, y) 
plt.xlabel('x') 
plt.xlabel('y') 
plt.title("Training Data") 
plt.show() 
```


![output_34_0](https://user-images.githubusercontent.com/70505378/140709073-60d0ca2e-52a9-46f0-bb5a-107885343558.png)
    

<br>

### Numpy


```python
w=np.random.randn()
b=np.random.randn()

lr = 0.02                               
n_epoch = 300
lossHistory = []

for epoch in range(n_epoch):
    y_pred = w*x + b
    error = ((y_pred - y)**2).mean() 

    w = w - lr* ((y_pred - y)*x).mean()
    b = b - lr* (y_pred - y).mean()
    lossHistory.append(error)
        
print('epoch=', epoch, 'error=', error, 'w=', w, 'b=', b)

plt.plot(lossHistory)
```

    epoch= 299 error= 44.022319701911826 w= 10.268908970131976 b= 10.012253241317557




![output_36_2](https://user-images.githubusercontent.com/70505378/140709079-5f71fce0-033e-4331-8557-7361608ac9b8.png)
    

<br>

### using tensorflow 1.x (without tensorflow function)

- tf.disable_v2_behavior()
- tf.placeholder()
- tf.Session()
- tf.global_variables_initializer()
- feed_dict = {}


```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X = tf.placeholder("float") # X를 담을 공간
Y = tf.placeholder("float") # Y를 담을 공간
W = tf.Variable(np.random.randn(), name = "W") 
b = tf.Variable(np.random.randn(), name = "b") 

learning_rate = 0.02

# Graph
y_pred = X * W + b   # hypothesis
cost = tf.reduce_mean(tf.square(y_pred - Y)) # MSE

W_gradient = tf.reduce_mean((y_pred - Y) * X)
b_gradient = tf.reduce_mean(y_pred - Y)

W_update = W.assign_sub(learning_rate * W_gradient) # '-'
b_update = b.assign_sub(learning_rate * b_gradient) # '-'

# Execution
sess = tf.Session()
sess.run(tf.global_variables_initializer())
lossHistory = []

for i in range(300): 
    sess.run([W_update, b_update], feed_dict={X: x, Y: y})
    cost_val, W_val, b_val = sess.run([cost, W, b], feed_dict={X: x, Y: y})
    
    lossHistory.append(cost_val)

sess.close()
print(cost_val, W_val, b_val)

plt.plot(lossHistory)
```

    WARNING:tensorflow:From C:\Users\wjsdu\anaconda3\lib\site-packages\tensorflow\python\compat\v2_compat.py:96: disable_resource_variables (from tensorflow.python.ops.variable_scope) is deprecated and will be removed in a future version.
    Instructions for updating:
    non-resource variables are not supported in the long term
    44.022396 10.268426 10.009809




![output_38_2](https://user-images.githubusercontent.com/70505378/140709080-ff84996c-77ba-4955-867e-39900778b909.png)

<br>

### tensorflow 1.x (with tensorflow function)


```python
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

X = tf.placeholder("float") 
Y = tf.placeholder("float") 

W = tf.Variable(np.random.randn(), name = "W") 
b = tf.Variable(np.random.randn(), name = "b") 

y_pred = X * W + b   # hypothesis
cost = tf.reduce_mean(tf.square(y_pred - Y))   # mse

##### calculate gradient and make train graph
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)
#####

sess = tf.Session()
sess.run(tf.global_variables_initializer())
 
lossHistory = []
for epoch in range(300): 
    cost_val, W_val, b_val, _ = sess.run([cost, W, b, train],
                                         feed_dict={X: x, Y: y})
    lossHistory.append(cost_val)
 
print(cost_val, W_val, b_val)
sess.close()

plt.plot(lossHistory)
```

    44.022232 10.267602 10.01478


![output_40_2](https://user-images.githubusercontent.com/70505378/140709082-5ad50b92-e232-45fd-9285-231eb7740a93.png)
    


- We can see the same result.

<br>

<br>
