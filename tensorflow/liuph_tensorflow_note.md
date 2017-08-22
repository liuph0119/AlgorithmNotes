---
# Liuph's TensorFlow Note

---

-  Last update: 2017-08-22
- 作者：刘鹏华 (liuph3@mail2.sysu.edu.cn)
- 根据[TensorFlow官方教程](https://www.tensorflow.org/get_started/get_started)和以下文章整理：
- &nbsp; [TensorFlow从基础到实战：一步步教你创建交通标志分类神经网络](http://mp.weixin.qq.com/s?__biz=MjM5NTkwNzM0Mw==&mid=2651295222&idx=2&sn=0571635adf22abc27b3e72598e04e964&chksm=bd027f968a75f680b5170853fa0a2770a36958b6ceeba13ba526d4026e86210a9bf637a36b0e&mpshare=1)
- &nbsp; [Python 深度学习](https://www.datacamp.com/courses/deep-learning-in-python)
- &nbsp; [DataCamp 的 Keras 教程](https://www.datacamp.com/community/tutorials/deep-learning-python)
***
# 1. Introduction
## 1.1. Tensor(张量)
- 张量是一种带有幅度和多个方向的物理实体的一种数学表征。三维空间中的张量可以通过具有 3R 个数字的数组表示。这里的 R 表示张量的秩（rank）：比如在一个三维空间中，一个第二秩张量（second-rank tensor）可以用 3^2=9 个数字表示。在一个 N 维空间中，标量仍然只需要一个数字，而向量则需要 N 个数字，张量则需要 N^R 个数字。这就是为什么你常会听到人们说标量是 0 秩的张量：因为没有方向，你可以使用一个数字表示它。
- 标量可以用`单个数字`表示，向量是一个`有序的数字集合`，张量是一个`数字的阵列`。
***
- [更多关于张量](https://www.youtube.com/watchv=f5liqUk0ZTw)
## 1.2. 安装Tensorflow
- tensorflow目前支持windows版本，不过同大多数机器学习和深度学习的开源库一样，一般需要GPU支持和`64位`python.因此具体的安装过程很简单：下载python(64bit)，通过`pip install`来安装，或者前往[UC的python libs](http://www.lfd.uci.edu/~gohlke/pythonlibs/)下载对应的64 bit .whl文件安装。
- 测试tensorflow是否正确安装的方法：`import tensorflow as tf`，查看是否报错。
- 注意：py文件所在的目录最好不要跟库的名称一样，否则系统会将文件夹import而出错。
## 1.3. Get Start
### 1.3.1. Basics
- tensorflow支持一些基本的运算，但是每次执行都要`session`的`run`功能来实现。它只是定义了模型，但没有运行进程来计算结果。例如以下代码的输出不是两个数组的乘积，而是张量的描述：
```python
import tensorflow as tf
a1 = tf.constant([1,2,3,4])
a2 = tf.constant([5,6,7,8])
result = tf.multiply(a1, a2)
print(result)
```
输出结果为：`Tensor("Mul:0", shape=(4,), dtype=int32)`
### 1.3.2. Session
因此，需要先定义一个session, 调用session的run()方法来运行。
```python
import tensorflow as tf
sess = tf.Session() #initialize the session
print(sess.run(result))  #run session
sess.close() # remember to close the session
```
或者采用以下方式：
```python
with tf.Session() as sess:
    output = sess.run(result)
    print (output)
```

以上输出结果为：`[ 5 12 21 32]`

你可以指定 config 参数，然后使用 ConfigProto 协议缓冲来为你的 session 增加配置选项。比如说，如果你为你的 Session 增加了 `config=tf.ConfigProto(log_device_placement=True)` ，你就可以确保你录入了运算分配到的 GPU 和 CPU 设备。然后你就可以了解在该 session 中每个运算使用了哪个设备。比如当你为设备配置使用软约束时，你也可使用下面的配置 session: `config=tf.ConfigProto(allow_soft_placement=True)` .
***
### 1.3.3.Constant/Placeholder/Variable
以上`tf.Constant`表示定义常量，除此以外还有一些其他的形式，例如占位符（placeholder）和变量（variable）。  
占位符是指没有分配的值，在你开始运行该 session 时会被初始化。正如其名，它只是张量的一个占位符，在 session 运行时总是得到馈送。变量是指值可以变化的量。而常量的值则不会变化。  
变量定义后需要初始化，否则报错。比方说，以下代码执行会报错：`Attempting to use uninitialized value Variable`
```python
W = tf.Variable([.3], dtype=tf.float32) #variance
b = tf.Variable([-.3], dtype=tf.float32) #variance
print(sess.run(W))
```
因此，需要先将变量初始化。以下代码会初始化所有的全局变量：
```python
init = tf.global_variables_initializer()
sess.run(init)
```
变量初始化后还可以通过`tf.assign(var, value)`来更改其值。例如以上修改W和b的值可以通过以下语句来实现：
```python
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
```
placeholder的用途比较方便，例如:
```python
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

init = tf.global_variables_initializer()
sess.run(init)

print(sess.run(linear_model, {x: [1, 2, 3, 4]}))
```
因为`x`事先被定义为一个占位符，因此可以同时将`x`设置为多个值来评估。输出结果为`[ 0.          0.30000001  0.60000002  0.90000004]`
***
### 1.3.4. Squared Sum Loss
对于一般的分类和回归问题，一般对目标值和真实值的误差（又称loss）进行量化，例如以下例子中：
```python
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))
```
输出结果为`23.66`  
`y`作为一个与x相对应的placeholder，`tf.square(Vector)`方法可以计算向量的平方差，该结果同为向量。而`tf.reduce_sum(Vector)`则可以计算求和，输出结果为一个scalar。
***
### 1.3.5. Optimizer
tensorflow提供优化方案，最常用的就gradient descent。同时，tensorflow也提供其他的优化方案：  
- [tf.train.Optimizer](https://www.tensorflow.org/api_docs/python/tf/train/Optimizer)
- [tf.train.GradientDescentOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer)
- [tf.train.AdadeltaOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdadeltaOptimizer)
- [tf.train.AdagradOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdagradOptimizer)
- [tf.train.AdagradDAOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdagradDAOptimizer)
- [tf.train.MomentumOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/MomentumOptimizer)
- [tf.train.AdamOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/AdamOptimizer)
- [tf.train.FtrlOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/FtrlOptimizer)
- [tf.train.ProximalGradientDescentOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/ProximalGradientDescentOptimizer)
- [tf.train.ProximalAdagradOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/ProximalAdagradOptimizer)
- [tf.train.RMSPropOptimizer](https://www.tensorflow.org/api_docs/python/tf/train/RMSPropOptimizer)
```python
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
  sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})

print(sess.run([W, b]))
```
输出结果为：`[array([-0.9999969], dtype=float32), array([ 0.99999082],
 dtype=float32)]`  
 ### 1.3.6. Complete ML Model
 至此，一个机器学习算法已经完成。包括模型的建立、参数优化和精度评估。完整的代码如下：
 ```python
import tensorflow as tf
#Model parameters
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
#Model input and output
x = tf.placeholder(tf.float32)
linear_model = W * x + b
y = tf.placeholder(tf.float32)

#loss
loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
#optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

#training data
x_train = [1, 2, 3, 4]
y_train = [0, -1, -2, -3]
# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset values to wrong
for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})

#evaluate training accuracy
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
```
最后的输出形如：
```
W: [-0.21999997] b: [-0.456] loss: 4.01814
W: [-0.39679998] b: [-0.49552] loss: 1.81987
W: [-0.45961601] b: [-0.4965184] loss: 1.54482
...
W: [-0.99999678] b: [ 0.99999058] loss: 5.84635e-11
W: [-0.99999684] b: [ 0.9999907] loss: 5.77707e-11
W: [-0.9999969] b: [ 0.99999082] loss: 5.69997e-11
```
***
### 1.3.7.`tf.contrib.learn`(应该继承自`tf.estimator`)
`tf.contrib.learn`是一个高级接口，支持：
- running training loops
- running evaluation loops
- managing data sets
1. Basic Usage
```python
import tensorflow as tf
import numpy as np

#特征列
feature_columns = [tf.feature_column.numeric_column("x", shape=[1])]

#调用预定义的线性模型
estimator = tf.contrib.learn.LinearRegressor(feature_columns=feature_columns)


#训练集和测试集，定义batch size
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=None, shuffle=True)
train_input_fn = tf.contrib.learn.io.numpy_input_fn(
    {"x": x_train}, y_train, batch_size=4, num_epochs=1000, shuffle=False)
eval_input_fn = tf.contrib.learn.io.numpy_input_fn(
    {"x": x_eval}, y_eval, batch_size=4, num_epochs=1000, shuffle=False)

#迭代次数
estimator.fit(input_fn=input_fn, steps=1000)

#模型评估
train_metrics = estimator.evaluate(input_fn=train_input_fn)
eval_metrics = estimator.evaluate(input_fn=eval_input_fn)
print("train metrics: %r"% train_metrics)
print("eval metrics: %r"% eval_metrics)
```
输出结果：  
`train metrics: {'loss': 4.3091913e-07, 'global_step': 1000}`  
`eval metrics: {'loss': 0.0025798536, 'global_step': 1000}`

2. Custom Models  
给Estimator提供一个函数model_fn来告诉tf.contrib.learn如何评估预测，训练步骤和损失.
```python
import numpy as np
import tensorflow as tf
#Declare list of features, we only have one real-valued feature
def model(features, labels, mode):
  #Build a linear model and predict values
  W = tf.get_variable("W", [1], dtype=tf.float64)
  b = tf.get_variable("b", [1], dtype=tf.float64)
  y = W*features['x'] + b
  #Loss sub-graph
  loss = tf.reduce_sum(tf.square(y - labels))
  #Training sub-graph
  global_step = tf.train.get_global_step()
  optimizer = tf.train.GradientDescentOptimizer(0.01)
  train = tf.group(optimizer.minimize(loss),
                   tf.assign_add(global_step, 1))
  #ModelFnOps connects subgraphs we built to the
  #appropriate functionality.
  return tf.contrib.learn.ModelFnOps(
      mode=mode, predictions=y,
      loss=loss,
      train_op=train)

#把自定义的模型载入estimator 
estimator = tf.contrib.learn.Estimator(model_fn=model)
#define our data sets
x_train = np.array([1., 2., 3., 4.])
y_train = np.array([0., -1., -2., -3.])
x_eval = np.array([2., 5., 8., 1.])
y_eval = np.array([-1.01, -4.1, -7, 0.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": x_train}, y_train, 4, num_epochs=1000)


eval_input_fn = tf.contrib.learn.io.numpy_input_fn({"x":x_eval}, y_eval, batch_size=4, num_epochs=1000)
#train
estimator.fit(input_fn=input_fn, steps=1000)
#Here we evaluate how well our model did.
train_loss = estimator.evaluate(input_fn=input_fn)
eval_loss = estimator.evaluate(input_fn=eval_input_fn)
print("train loss: %r"% train_loss)
print("eval loss: %r"% eval_loss)
```
输出结果：
`train loss: {'loss': 1.2629069e-12, 'global_step': 1000}`  
`eval loss: {'loss': 0.010099798, 'global_step': 1000}`
***
# 2. MNIST for ML Beginners
## 2.1. Mnist 数据简介
Mnist 数据集是一个手写数字数据集，包括大量手写图片和标签样本，挂在 [Yann LeCun's website](http://yann.lecun.com/exdb/mnist/).  
在tensorflow中，通过以下两句代码可以下载和读取mnist数据集:
```python
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
```
mnist数据集包括三部分：55000 training data (`mnist.train`), 10000 test data (`mnist.test`), 5000 validation data (`mnist.validation`).每个mnist数据样本包括一幅手写数字图像(28 pix × 28 pix)和标签(0-9)，例如`mnist.train.images`表示训练集的图像X，`mnist.test.lables`表示测试集的标签。  
我们可以将图像表达为28×28的numpy array，并将其压平(`flatten`)为28×28=784的数值向量。事实上，计算机视觉中常采用flatten这种方式。因而，mnist.train.images为一个形状为[55000, 784]的张量。  
![images](https://www.tensorflow.org/images/mnist-train-xs.png)
一般来说，y值并非常规的0-9的标量，而是以"one hot vector"的形式存储，例如，3表示为[0,0,1,0,0,0,0,0,0,0]，即第i个位置为1，其他位置为0。因而，mnist.train.labels为一个形状为[55000, 10]的张量。  
![labels](https://www.tensorflow.org/images/mnist-train-ys.png)
## 2.2. SoftMax Regression
SoftMax Regression 是一个通用简单的模型，它能为一个目标属于多个类别分配概率，其输出为一个0-1的list，并且这个list的和为1。一般来说，对于成熟的模型，一般会将最后一层设置为softmax层。  
SoftMax Regression包括两步：first we add up the evidence of our input being in certain classes, and then we convert that evidence into probabilities. We do a weighted sum of the pixel intensities. The weight is negative if that pixel having a high intensity is evidence against the image being in that class, and positive if it is evidence in favor. 简单来说，softmax regression就是计算出各个位置的权重。例如下图，红色表示负权重，蓝色表示正权重。
![softmax weights](https://www.tensorflow.org/images/softmax-weights.png)
一般除了权重，我们也会加入增益(bias)。
因此，给定一个X，其对于类别 i 的evidence表示为：
![](http://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Ctext%7Bevidence%7D_i%20%3D%20%5Csum_j%20W_%7Bi%2C%7E%20j%7D%20x_j%20&plus;%20b_i)  
其中`Wi,j`表示 i 类的第 j 个特征的权重，`bi`表示 i 类的增益。  
接着采用`softmax`函数将其转化为概率![](http://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Cmathbf%7By%7D):  
`y = softmax(evidence)`  
此处`softmax`充当激活函数（`Activation`）的角色，此处输出10类概率。事实上，它的形式为`softmax(x) = normalize(exp(x))`，或者 ![](http://latex.codecogs.com/gif.latex?%5Cbg_white%20%5Ctext%7Bsoftmax%7D%28x%29_i%20%3D%20%5Cfrac%7B%5Cexp%28x_i%29%7D%7B%5Csum_j%20%5Cexp%28x_j%29%7D)。更多关于`softmax`函数的介绍可在[Softmax](http://neuralnetworksanddeeplearning.com/chap3.html#softmax)查看。  
因此，此处的softmax函数形式如下图：  
![Softmax regression scalargraph](https://www.tensorflow.org/images/softmax-regression-scalargraph.png)
或者可以表示为：![](http://latex.codecogs.com/gif.latex?%5Cbg_white%20y%20%3D%20%5Ctext%7Bsoftmax%7D%28Wx%20&plus;%20b%29)
## 2.3. Implemention using tf
```python
import tensorflow as tf
#None表示可能输入任意长度的图像数, 784表示将图像矩阵压扁
x = tf.placeholder(tf.float32, [None, 784])
#采用tf.zeros()来定义变量，并且已经初始化为0 
W = tf.Variable(tf.zeros([784, 10])) #weights
b = tf.Variable(tf.zeros([10])) #bias
#define softmax model
y = tf.nn.softmax(tf.matmul(x, W) + b)
```
损失函数(cost/loss)采用交叉熵(cross entropy, [more about cross entropy](http://colah.github.io/posts/2015-09-Visual-Information/)])来定义:  
![](http://latex.codecogs.com/gif.latex?%5Cbg_white%20H_%7By%27%7D%28y%29%20%3D%20-%5Csum_i%20y%27_i%20%5Clog%28y_i%29)  
其中，y为模型预测输出的概率分布，y'为真实的概率分布(实际上此处为ont hot vector[0,0,0,1,0,0,0,0,0,0])。因此，代码形式如下：
```python
#y--prediction  y_--truth
y_ = tf.placeholder(tf.float32, [None, 10])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
```
此处，`tf.reduce_sum()`将括号内的矩阵在第二维相加，因为`reduction_indices=[1]`。最后`tf.reduce_mean()`用于计算所有batch example的均值。  
不过，以上计算交叉熵的方式计算不稳定，因此我们经常对 `tf.matmul(x, W) + b` 采用 `softmax_cross_entropy_with_logits`方法。 
定义模型和损失函数后，tensorflow会自动采用[BP算法](http://colah.github.io/posts/2015-08-Backprop/)来优化参数。  
```python
#define softmax model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits=tf.matmul(x, W) + b))
#learning rate=0.5, target:minimize cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

#initial session and variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#run optimizer
for _ in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
  curr_loss = sess.run(cross_entropy, {x: batch_xs, y_: batch_ys})
  print("iteration--%3d loss: %s"%(_, curr_loss))
```
## 2.4. Evaluating model
模型的评估首先需要从概率分布中确定分类类别，`tf.argmax()`可以查找到张量中某个维度上的最大值的位置。例如，`tf.argmax(y,1)`和`tf.argmax(y_,1)`分别代表模型输出和真实的label。采用`tf.equal`对两者进行相等判别。  
```python
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```
其输出结果为`accuracy = 0.9183`，虽然精度达到了92%，但是因为采用的是简单的softmax模型，因此仍然有很大的提升空间，例如采用更成熟的模型。 [更多模型结果](https://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results)
- complete code
```python
import tensorflow as tf
#load data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

#None表示可能输入任意长度的图像数, 784表示将图像矩阵压扁
x = tf.placeholder(tf.float32, [None, 784])
#采用tf.zeros()来定义变量，并且已经初始化为0
W = tf.Variable(tf.zeros([784, 10])) #weights
b = tf.Variable(tf.zeros([10])) #bias
y_ = tf.placeholder(tf.float32,[None, 10]) #true

#define softmax model
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y_, logits=tf.matmul(x, W) + b))
#learning rate=0.5, target:minimize cross_entropy
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)
#accuracy evaluating
correct_prediction = tf.equal(tf.argmax(tf.matmul(x, W) + b, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#initial session and variables
sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

#run optimizer
for _ in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    curr_loss = sess.run(cross_entropy, {x: batch_xs, y_: batch_ys})
    print("iteration--%3d loss: %s"%(_, curr_loss))

print("accuracy = %s"%sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
```
# 3. Build a Multilayer Convolutional Network
## 3.1. Weight Initializition
初始化权重时，加入细微的噪声，避免0梯度。初始化增益时，因为采用纠正([ReLU](https://en.wikipedia.org/wiki/Rectifier_(neural_networks))，一般形式为![](https://wikimedia.org/api/rest_v1/media/math/render/svg/bb2c32931fad595832c8e66f2f73760ebcbc0096))神经元因此将其设置为比较小的正数，避免神经元坏死。因此，weights 和 biases 的初始化函数如下：
```python
def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)
```
## 3.2. Convolution and Pooling
```python
def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
```
卷积函数采用步长为1，填充为0，因此输入与输出形状相同。池化函数采用2×2的单元，因此最后输出的各维形状为输入的一半。
## 3.3. First Convolutional Layer
第一层卷积层包括卷积和池化。它将为每一个5×5的patch计算32个特征，因此它的 weight 张量的形状为[5, 5, 1, 32]，前两者为 patch 的 size ，第三个为输入通道的数目，最后一个为输出通道的数目。
```python
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
```
为此，首先我们需要将 x 的形状转为 4 维，其中第2和第3维分别表示图像的宽和高，第4维代表图像的颜色通道数目。
```python
x = tf.placeholder(tf.float32, [None, 784])
x_image = tf.reshape(x, [-1, 28, 28, 1])
```
接着，将图像与权重卷积，加上增益，并应用ReLU函数，最后池化。最后经过池化后，图像大小由 28×28 变为 14×14。
```python
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
```
## 3.4. Second Convolutional Layer
为了实现深度神经网络，再第一层后同样连接一层卷积层。该层为每个5×5的 patch 计算64个特征。最后经过池化后，图像大小由 14×14 变为 7×7。
```python
W_conv2 = weight_variable([5, 5, 1, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
```
## 3.5. Densely Connected Layer
图像大小已经变成了 7×7。再加入全连接层时，设置1024个神经元，并将图像压扁为一系列向量。乘以权重，加上增益，最后ReLU。
```python
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
```
## 3.6. Dropout
为了降低过拟合的概率，应用[dropout](https://www.cs.toronto.edu/~hinton/absps/JMLRdropout.pdf)，dropout过程根据概率来判断某个神经元的输出是否保留。
```python
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
```
## 3.7. Readout Layer
形如softmax regression 层。
```python
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_ = tf.placeholder(tf.float32,[None, 10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
```
## 3.8. Training & Evaluating 
与softmax的训练和评估相似，但是本模型与前者存在以下一些不同之处：
- 采用`ADAM Optimizer`代替`Gradient Descent Optimizer`
- 将`keep_prob`和`feed_dict`加入，用于控制`dropout`比例
- 训练过程中，每迭代100次日志都会记录一次
```python
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch[0], y_: batch[1], keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```
## 3.9. Complete Code
```python
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial)

def conv2d(x, W):
  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#reshape image data
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32,[None, 10])

x_image = tf.reshape(x, [-1, 28, 28, 1])

##first layer
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(tf.nn.bias_add(conv2d(x_image, W_conv1), b_conv1))
h_pool1 = max_pool_2x2(h_conv1)

##second layer
W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

##densely connected layer
W_fc1 = weight_variable([7 * 7 * 64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

##dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

##readout layer
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

##train
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

##session
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  for i in range(20000):
    batch_xs, batch_ys = mnist.train.next_batch(50)
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={
          x: batch_xs, y_: batch_ys, keep_prob: 1.0})
      print('step %d, training accuracy %g' % (i, train_accuracy))
    train_step.run(feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})

  print('test accuracy %g' % accuracy.eval(feed_dict={
      x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))
```
输出结果为（当然，如果迭代次数较多，可以达到99.2%的精度）：    
`step 0, training accuracy 0.1`  
`step 100, training accuracy 0.9`  
`test accuracy 0.9063`