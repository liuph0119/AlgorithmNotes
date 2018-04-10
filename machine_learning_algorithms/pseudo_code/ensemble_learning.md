# 集成学习
- 概述：从样本集D中有放回地生成k个训练集合{D_0, D_1,...,D_k-1}，采用不同的基础算法（决策树、贝叶斯等）学习k个学习器（分类器、回归模型）。最后通过这些学习器投票，来决定目标类别或者目标值。
- 注意：有放回地产生与D等大小的集合，会有d->∞ (1-1/d)^d ≈ 36.8% 的样本从未被采集到，被称为袋外样本。
 - - - 
## 1.Bootstrap Aggregation(Bagging)
- 算法概述：从样本集合D中有放回地选取k个与D等大小的训练集合{D_0, D_1,...,D_k-1}，分别训练分类器，之后通过投票的方式做分类。每个学习器的投票权重相等。
- **Algorithm Bagging.**
**Input:**
D: a set of *d* training tuples
k: the number of models in the ensemble
a classification learning scheme(decision tree algorithm, naive Bayesian, etc.)
**Output:** the ensemble--a composite model M*
**Method:**
(1) **for** i = 1 to *k* **do** //create *k* models
(2)      create bootstrap sample *D_i*, by sampling D with replacement
(3)      use *D_i* and the learning scheme to derive a model *M_i*
(4) **endfor** 

**To use the ensemble to classify a tuple X:**
    let each of the *k* models to classify **X** and return the majority vote 
- - -
## 2. Boost
- 算法概述：从样本集合D中有放回地选取k个与D等大小的训练集合{D_0, D_1,...,D_k-1}，分别训练分类器，之后通过投票的方式做分类。每个学习器的投票权重不相等。
- - -
### 2.1. Adaboost
- 算法概述：
	1. 从集合D中有放回地采样从大小为d的样本集合D中有放回地选取k个与D等大小的训练集合{D_0, D_1,...,D_k-1}，一开始每个样本被选取的概率为1/d，训练模型Mi；
	2. 计算模型Mi的训练误差error(Mi)。如果error(Mi)超过0.5，则舍弃Mi，重新采样训练Mi；否则调整每个样本的权重（正确分类的 样本权重更新 w = w*(error(Mi)/(1-error(Mi)))，之后再将权重标准化和为1）；
	3. 模型Mi的投票权重为log((1-error(Mi))/(error(Mi)));
	
- **Algorithm AdaBoost**
	- **Input:**
	D: a set of *d* training tuples
	k: the number of models in the ensemble
	a classification learning scheme(decision tree algorithm, naive Bayesian, etc.)
  - **Output:** a composited model
  - **Method:**
	(1) initial the weight of each tuple in D to 1/*d*
	(2) **for** i = 1 to *k* **do** //for each round
	(3) 	sample *D* with replacement according to the tuple weights to obtain *Di*;
	(4)  	use training set *Di* to derive a model *Mi*
	(5)  	compute *error* ( *Mi* ), the error rate of *Mi*
	(6)  	**if** *error* ( *Mi* ) > 0.5 **then**
	(7)      	abort Loop
	(8)  	**endif**
	(9)  	**for** each tuple in *D* that was corectly classified **do**
	(10)     	multiply the weight of the tuple by *error* ( *Mi* )/(1- *error* ( *Mi* ))//update weight
	(11) 	normalize the weight of each tuple
	(12) **endfor**

    **To use the ensemble to classify tuple X:**
    (1)  initial weight of each class to 0
    (2)  **for** i = 1 to *k* **do**//for each classifier generated do
    (3)      *wi* = *log((1-error(Mi))/(error(Mi)))*//weight of each classifier
    (4)      *c = Mi(X)*//get the class prediction for X from Mi
    (5)      add *wi *to class *c*
    (6)  **endfor**
    (7)  return the class with the largest weight


	