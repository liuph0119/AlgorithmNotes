#include "dnn.h"
#include <vector>
int main(int argc, char *argv[])
{
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
	
	DNNTraining(datalayer1,datalayer2,vFCLayer,vSigLayer,losslayer,accuracy,5.0,100,0.95,true,"data/log_201707270904.csv");
	saveModel(vFCLayer, "data/model.pkl");
	
	//预测
	MatrixXd retx0, rety0;
	int npos = datalayer_test.forward(retx0, rety0);
	
	for (int i = 0; i < vFullyConnectList.size(); i ++)
	{
		retx0 = vFullyConnectList[i].forward(retx0);
		retx0 = vSigmoidList[i].forward(retx0);
	}
	accu = accuracy.forward(retx0, rety0);}
	printf("test accuracy = %.3f", accu);

	return 0;
}