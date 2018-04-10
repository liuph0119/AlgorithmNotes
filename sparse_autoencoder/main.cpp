
#include "preprocess.h"
#include "sae_activiation.h"
#include "sae.h"
#include "matrix.h"

#include <iostream>
#include <cmath>
#include <ctime>
#include <fstream>
#include <string>
#include <assert.h>
using namespace std;
#define SAE_TRAINING
#define ENCODER


int main(int argc, char* argv[])
{
	std::string data_dir = "../data/mnist/";
	srand(unsigned(time(0)));

#ifdef SAE_TRAINING
	// read data, filename, nFeatureNum, nSampleNum
	Matrix X = csv2mat(data_dir + "data.csv", 784, 60000);

	// normalize data
	cout << "normalizing...";
	normalizeData(X);
	cout << "success!\n";
	
	// training model
	SparseAutoEncoder sae;
	// X, , learning_rate, epochs, batch_size, lambda, rho, sparsity_weight
	double _r = rand() / (double)RAND_MAX * (-2);	
	cout << "learning rate = " << pow(10, _r) << endl;

	sae.trainSparseAutoEncoder2(X, 25, pow(10, _r), 100, 1024, 0.0001, 0.1, 2);
	
	// save sae weights (W1, W2, b1, b2)
  	sae.saveNetWeight(data_dir + "training_weight.csv");
	// save W1, each line(1*input_size) could be reshaped to sqrt(input_size)*sqrt(input_size) to visualize
 	mat2csv(sae.getW1(), data_dir + "W1.csv");

#endif

#ifdef ENCODER
	// before loading weights, net structure should be defined first
	//SparseAutoEncoder sae(180075, 32);
	//sae.loadNetWeight(data_dir + "training_weight.csv");


	// 随机挑选5个例子进行对比;
	//Matrix X = csv2mat(data_dir + "data.csv", 180075, 20);
	//normalizeData(X);
	ofstream fout(data_dir + "comparison.csv");
	if (!fout.good())
	{
		cerr << "open file fail!\n";
		return -1;
	}

	Matrix testMat(X.RowNo(), 1);
	Matrix decodevec, encodevec;
	for (int j = 0; j < 5; j++)
	{
 		int ind = rand() % X.ColNo();
 		for (int i = 0; i < X.RowNo(); i++)
 			//testMat(i, 0) = rand() / (double)RAND_MAX;
 			testMat(i, 0) = X(i, ind);

		// 编码再解码;
		encodevec = sigmoid(sae.getW1()*testMat + sae.getb1());

// 		if(j == 0)
//  			for (int i = 0; i < encodevec.RowNo(); i++)
//  				encodevec(i, 0) = rand() / (double)RAND_MAX;


		decodevec = sigmoid(sae.getW2()*encodevec + sae.getb2());

		// 计算余弦值;
		cout << "index = " << ind << "   cosin = " << cosin(testMat, decodevec) << endl;

		// 对比结果写入文件;
		for (int i = 0; i < testMat.RowNo() - 1; i++)
			fout << testMat(i, 0) << ",";
		fout << testMat(testMat.RowNo() - 1, 0) << "\n";

		// 编码值，即隐含层的激活值;
		for (int i = 0; i < encodevec.RowNo() - 1; i++)
			fout << encodevec(i, 0) << ",";
		fout << encodevec(encodevec.RowNo() - 1, 0) << "\n";

		// 解码后的值，即输出层的输出值;
		for (int i = 0; i < decodevec.RowNo() - 1; i++)
			fout << decodevec(i, 0) << ",";
		fout << decodevec(decodevec.RowNo() - 1, 0) << "\n";
		fout.flush();
	}
	fout.close();

	
#endif
	return 0;
}