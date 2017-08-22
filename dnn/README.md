将DNN的代码写成了接口，不过存在的一点纰漏就是各个神经网络层的前向和反向传播没有隐藏起来，后续再说吧，目前是把各个神经网络层写成了类，最后用一个训练主函数和saveModel/loadModel函数来调用。
因此主函数中只需要按照如下方式声明数据层、全连接层的数目和参数，最后调用训练函数即可。

- 依赖于Qt库。编译的DLL为64位，包含DNN.h  DNN.lib  DNN.dll三个文件，配置方式参照普通库的配置方式即可。

具体的代码如下：  
```cpp
//导入训练数据
//文件中数据的存储方式形如（主要是我们常用的组织方式就是这样，所以没有转置）
//
// 	x1,x2,x3,x4,···,x288,x289,y
// 	x1,x2,x3,x4,···,x288,x289,y
// 	x1,x2,x3,x4,···,x288,x289,y
//
//但是导入数据后，为了便于处理，而需要转置成mMat_data = MatrixXd(_Ndim, _mcount)
//mMat_labels = MatrixXd(_mcount, 1);
NNDataLayer datalayer1("data/train.csv", 1024);
datalayer1.loadData(289, 40000, 26);
	
//导入验证数据
NNDataLayer datalayer2("data/validate.csv",10000);
datalayer2.loadData(289, 10000, 26);

	
//全连接层和Sigmoid层
NNFullyConnectLayer fullyconnectlayer1(17*17, 20);
NNFullyConnectLayer fullyconnectlayer2(20, 26);
NNSigmoidLayer sigmoidlayer1;
NNSigmoidLayer sigmoidlayer2;

QList<NNFullyConnectLayer> vFCLayer;
vFCLayer.append(fullyconnectlayer1);
vFCLayer.append(fullyconnectlayer2);

QList<NNSigmoidLayer> vSigLayer;
vSigLayer.append(sigmoidlayer1);
vSigLayer.append(sigmoidlayer2);


//损失层和精度层
NNQuadraticLoss losslayer;
Accuracy accuracy;

// 关于调参, 可以前期采用比较小的学习速率2.0，达到一个阈值0.95后保存模型
// 然后导入模型，换更大的学习速率10.0。
// 当然，导入train.csv之后精度提高得很快嘛.
//导入模型参数
//loadModel(vFCLayer, "data/model.pkl");
//训练的主函数
DNNTraining(datalayer1,datalayer2,vFCLayer,vSigLayer,losslayer,accuracy,5.0,100,0.95,true,"data/log_201707270904.csv");
//保存模型参数
//saveModel(vFCLayer, "data/model.pkl");
```
	
- 	关于应该如何进行数据的预测：  
	可以参照训练数据，将label随机设置，或者全部设置为0，设置batch_size = 测试数据集总数，形如验证数据层的参数设置。
```cpp
MatrixXd retx0, rety0;
int npos = datalayer_test.forward(retx0, rety0);
	
for (int i = 0; i < vFullyConnectList.size(); i ++)
{
	retx0 = vFullyConnectList[i].forward(retx0);
	retx0 = vSigmoidList[i].forward(retx0);
}
```	
	经过以上过程处理的retx0就是[Ndim, mCount]的概率矩阵，每一列为一条数据，每一行为一维特征，找出每一列的最大值，就是预测标签。
	与真实数据的精度对比（需要调整rety0的值）：
```cpp
accu = accuracy.forward(retx0, rety0);（当然，这个又是validation了）
```

- 目前已经将该lib去除Qt依赖而只剩下普通标准库,即将QList修改为vector, Qt的文件读写改为fstream的读写，并且编译了32位和64位库。
