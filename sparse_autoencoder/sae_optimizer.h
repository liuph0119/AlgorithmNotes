#ifndef SAE_OPTIMIZER_H
#define SAE_OPTIMIZER_H

#include "matrix.h"
#include "preprocess.h"
#include <math.h>
#include <algorithm>
#include <string>
#include <vector>


// SGD, Adagrad的初始权重需要较大;


class Optimizer {
public:
	virtual void name() {};
	Optimizer(double _lr = 0.01, double _decay = 0) { learning_rate = _lr; dDecay = _decay; nIterations = 0; }
	virtual ~Optimizer() {}
	virtual void update_weights(Matrix &W, Matrix &b, Matrix dW, Matrix db) {}

protected:
	double learning_rate;
	double dDecay;
	int nIterations;
};

class SGD :public Optimizer
{
public:
	static std::string name() { return "SGD"; }
	SGD(double _lr = 0.01, double _decay = 0)
	{
		learning_rate = _lr;
		dDecay = _decay;
		nIterations = 0;
	}
	~SGD(){}
	virtual void update_weights(Matrix &W, Matrix &b, const Matrix dW, const Matrix db)
	{
		double _lr = learning_rate;
		if (dDecay > 0)
			_lr = 1.0 / (1.0 + nIterations*dDecay)*learning_rate;

		W = W - _lr * dW;
		b = b - _lr * db;

		nIterations++;
	}
};


class Adagrad :public Optimizer
{
private:
	double dSigma_1, dSigma_2;
public:
	static std::string name() { return "Adagrad"; }
	Adagrad(double _lr = 0.5, double _decay = 0)
	{
		learning_rate = _lr;
		dDecay = _decay;
		nIterations = 0;
	}
	~Adagrad() {}
	virtual void update_weights(Matrix &W, Matrix &b, Matrix dW, Matrix db)
	{
		if (nIterations == 0)
		{
			dSigma_1 = 0;
			dSigma_2 = 0;
		}

		double _lr = learning_rate;
		if (dDecay > 0)
			_lr = 1.0 / (1.0 + nIterations*dDecay)*learning_rate;
		

		dSigma_1 += sumMat(powMat(dW, 2));
		dSigma_2 += sumMat(powMat(db, 2));

		W = W - _lr / std::sqrt(nIterations + 1.0) / std::sqrt(dSigma_1 / (nIterations + 1)) * dW;
		b = b - _lr / std::sqrt(nIterations + 1.0) / std::sqrt(dSigma_2 / (nIterations + 1)) * db;
		
		nIterations++ ;
	}
};

class Momentum :public Optimizer
{
private:
	double beta;
	int nT;
	Matrix vdW, vdb;
public:
	static std::string name() { return "Momentum"; }
	Momentum(double _lr = 0.05, double _beta = 0.9, int _nT = 100, double _decay = 0) {
		learning_rate = _lr; 
		beta = _beta; 
		nT = _nT; 
		dDecay = _decay;
		nIterations = 0;
	}
	~Momentum() {  }
	virtual void update_weights(Matrix &W, Matrix &b, const Matrix dW, const Matrix db)
	{
		if (nIterations == 0)
		{
			vdW = Matrix(dW.RowNo(), dW.ColNo());
			vdb = Matrix(db.RowNo(), db.ColNo());
			vdW.Null();
			vdb.Null();
		}

		double _lr = learning_rate;
		if (dDecay > 0)
			_lr = 1.0 / (1.0 + nIterations*dDecay)*learning_rate;

		vdW = beta*vdW + (1 - beta)*dW;
		vdb = beta*vdb + (1 - beta)*db;


		if (nIterations < nT)
		{
			W = W - _lr / (1 - beta*beta) * vdW;
			b = b - _lr / (1 - beta*beta) * vdb;
		}
		else
		{
			W = W - _lr * vdW;
			b = b - _lr * vdb;
		}
		nIterations++;
	}
};


class RMSprop :public Optimizer
{
private:
	double beta;
	double dEpsilon;
	Matrix SdW, Sdb;
public:
	static std::string name() { return "RMSprop";}
	RMSprop(double _lr = 0.1, double _beta = 0.999, double _decay = 0) :dEpsilon(10e-8)
	{
		learning_rate = _lr;
		beta = _beta;
		dDecay = _decay;
		nIterations = 0;
	}
	~RMSprop() {}
	virtual void update_weights(Matrix &W, Matrix &b, const Matrix dW, const Matrix db)
	{
		if (nIterations == 0)
		{
			SdW = Matrix(dW.RowNo(), dW.ColNo());
			Sdb = Matrix(db.RowNo(), db.ColNo());
			SdW.Null();
			Sdb.Null();
		}

		double _lr = learning_rate;
		if (dDecay > 0)
			_lr = 1.0 / (1.0 + nIterations*dDecay)*learning_rate;
		
		SdW = beta*SdW + (1.0 - beta)*(powMat(dW, 2.0));
		Sdb = beta*Sdb + (1.0 - beta)*(powMat(db, 2.0));


		W = W - _lr * (devisionMat(dW, plusMat(sqrtMat(SdW), dEpsilon)));
		b = b - _lr * (devisionMat(db, plusMat(sqrtMat(Sdb), dEpsilon)));

		nIterations++;
	}
};


class Adam: public Optimizer
{
private:
	const double beta_1, beta_2;
	double dEpsilon;
	Matrix vdW, vdb, SdW, Sdb;

public:
	static const std::string name() { return "Adam"; }
	Adam(double _lr = 0.05, double _decay = 0) : beta_1(0.9), beta_2(0.999), dEpsilon(10e-8), Optimizer() { learning_rate = _lr; 
		dDecay = _decay; 
		nIterations = 0;
	}
	virtual ~Adam() {}

	virtual void update_weights(Matrix &W, Matrix &b, const Matrix dW, const Matrix db)
	{
		if (nIterations == 0)
		{
			vdW = Matrix(dW.RowNo(), dW.ColNo());
			vdb = Matrix(db.RowNo(), db.ColNo());
			SdW = Matrix(dW.RowNo(), dW.ColNo());
			Sdb = Matrix(db.RowNo(), db.ColNo());
			vdW.Null();
			vdb.Null();
			SdW.Null();
			Sdb.Null();
		}

		double _lr = learning_rate;
		if (dDecay > 0)
			_lr = 1.0 / (1.0 + nIterations*dDecay)*learning_rate;

		vdW = beta_1*vdW + (1.0 - beta_1)*dW;
		vdb = beta_1*vdb + (1.0 - beta_1)*db;
		SdW = beta_2*SdW + (1.0 - beta_2)*(powMat(dW, 2.0));
		Sdb = beta_2*Sdb + (1.0 - beta_2)*(powMat(db, 2.0));

		Matrix vcdW =vdW / (1 - beta_1*beta_1),
			vcdb = vdb / (1 - beta_1*beta_1),
			ScdW = SdW / (1 - beta_2*beta_2),
			Scdb = Sdb /(1 - beta_2*beta_2);

		W = W - _lr * (devisionMat(vcdW, plusMat(sqrtMat(ScdW), dEpsilon)));
		b = b - _lr * (devisionMat(vcdb, plusMat(sqrtMat(Scdb), dEpsilon)));

		nIterations++;
	}
};


Optimizer* new_Optimizer(std::string sOptimizerType, double _lr = 0.01)
{
	if (sOptimizerType == "") return NULL;
	std::string act(sOptimizerType);
	if (act.compare(SGD::name()) == 0) return new SGD(_lr);
	if (act.compare(Momentum::name()) == 0) return new Momentum(_lr);
	if (act.compare(RMSprop::name()) == 0) return new RMSprop(_lr);
	if (act.compare(Adam::name()) == 0) return new Adam(_lr);
	if (act.compare(Adagrad::name()) == 0) return new Adagrad(_lr);
	
	return NULL;
}

#endif
