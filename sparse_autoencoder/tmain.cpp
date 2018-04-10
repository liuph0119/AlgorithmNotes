
#include "sae_optimizer.h"
#include <ctime>
#include <iostream>

// y = WX+b
bool getGradient(Matrix X, Matrix y, Matrix W, Matrix b, Matrix &dW, Matrix &db)
{
	Matrix B = repmat(b, 1, X.ColNo());
	Matrix y_ = W*X+ B;
	//dW = ~(multiplyMat(meanMat(X, 0), repmat(meanMat((y_ - y), 0), X.RowNo(), 1)));
	dW = ~(meanMat(multiplyMat(X, repmat((y_ - y), X.RowNo(), 1)), 0));
	db = meanMat(y_ - y, 0);
	return true;
}


int main()
{
	srand(unsigned(time(0)));
	Matrix W(1, 4), b = Matrix(1, 1);

	for (int i = 0; i < W.RowNo(); i ++)
		for (int j = 0; j < W.ColNo(); j ++)
			W(i, j) = rand() / (double)RAND_MAX;


	for (int i = 0; i < b.RowNo(); i++)
		for (int j = 0; j < b.ColNo(); j++)
			b(i, j) = rand() / (double)RAND_MAX;


	Optimizer * opt = new_Optimizer("Adam");
	Matrix X(4, 4), y(1, 4);
	X(0, 0) = 1, X(1, 0) = 2, X(2, 0) =3, X(3, 0) = 4;
	y(0, 0)= 35;
	X(0, 1) = 2, X(1, 1) = 3, X(2, 1) = 4, X(3, 1) = 5;
	y(0, 1) = 46;
	X(0, 2) = 3, X(1, 2) = 4, X(2, 2) = 5, X(3, 2) = 6;
	y(0, 2) = 55;
	X(0, 3) = 4, X(1, 3) = 5, X(2, 3) = 6, X(3, 3) = 7;
	y(0, 3) = 65;
	
	Matrix dW = W, db = b;
	cout << "starting training...\n";
	for (int i = 0; i < 10000; i++)
	{
		getGradient(X, y, W, b, dW, db);
		cout << "updating weight...";
		//cout << W << b << dW << db;
		opt->update_weights(W, b, dW, db);
		Matrix y_ = W*X + repmat(b, 1, X.ColNo());
		//cout << W << b << endl;
		std::cout << "iteration " << i+1 <<"    cost = " << sumMat(powMat(y - y_, 2))/2.0 << endl;
	}

	std::cout << W;
	std::cout << b << endl;
	return 0;
}