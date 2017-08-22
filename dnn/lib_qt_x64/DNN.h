// ���� ifdef ���Ǵ���ʹ�� DLL �������򵥵�
// ��ı�׼�������� DLL �е������ļ��������������϶���� DNN_EXPORTS
// ���ű���ġ���ʹ�ô� DLL ��
// �κ�������Ŀ�ϲ�Ӧ����˷��š�������Դ�ļ��а������ļ����κ�������Ŀ���Ὣ
// DNN_API ������Ϊ�Ǵ� DLL ����ģ����� DLL ���ô˺궨���
// ������Ϊ�Ǳ������ġ�
#ifdef DNN_EXPORTS
#define DNN_API __declspec(dllexport)
#else
#define DNN_API __declspec(dllimport)
#endif

#include <QtCore>
#include <Eigen\Dense>
using namespace Eigen;





//����;
class DNN_API BaseNNLayer
{
public:
	BaseNNLayer(){}
	virtual MatrixXd forward(){return MatrixXd::Zero(1,1);}
	virtual MatrixXd backward(){return MatrixXd::Zero(1,1);}
protected:
private:
};


//������;
// m*(N+1), N��������ά��, m��ʾ��¼��Ŀ 
class DNN_API NNDataLayer:public BaseNNLayer
{
public:
	NNDataLayer(const char *fn, int batch_size = 1024){ 
		msFn = QString(fn); 
		mn_batch_size = batch_size; 
		mn_pos = 0; 
	}
	bool loadData(int _Ndim, int _mcount, int _ny);
	int forward(MatrixXd& retx, MatrixXd& rety);
private:
	QString msFn;
	int mn_N_dim;
	int mn_m_count;
	int mn_n_ycount;

	MatrixXd mMat_data;
	MatrixXd mMat_labels;
	int mn_batch_size;
	int mn_length;
	int mn_pos;
};


//ȫ���Ӳ�;
class DNN_API NNFullyConnectLayer:public BaseNNLayer
{
public:
	NNFullyConnectLayer(int l_x, int l_y);
	void setLearningRate(double lr = 1.0);
	MatrixXd forward(MatrixXd _x);
	MatrixXd backward(MatrixXd d);

public:
	MatrixXd mMat_weights;
	MatrixXd mMat_bias;

private:
	double md_learning_rate;
	MatrixXd mMat_x;
	MatrixXd mMat_y;
	MatrixXd mMat_dw;
	MatrixXd mMat_db;
	MatrixXd mMat_dx;
};

//�������;
class DNN_API NNSigmoidLayer:public BaseNNLayer
{
public:
	NNSigmoidLayer(){}
	MatrixXd sigmoid(MatrixXd _x);
	MatrixXd forward(MatrixXd _mat);
	MatrixXd backward(MatrixXd d);
private:
	MatrixXd mMat_x; 
	MatrixXd mMat_y;
	MatrixXd mMat_dx;
};


class DNN_API NNQuadraticLoss:public BaseNNLayer{
public:
	NNQuadraticLoss(){};
	double forward(MatrixXd _x, MatrixXd _label);
	MatrixXd backward();
private:
	MatrixXd mMat_x;
	MatrixXd mMat_label;
	double md_loss;
	MatrixXd mMat_dx;
};

class DNN_API CrossEntropyLoss:public BaseNNLayer{
public:
	CrossEntropyLoss(){}

protected:
	double forward(MatrixXd _x, MatrixXd _label);
	MatrixXd backward();
	MatrixXd mMat_x;
	MatrixXd mMat_label;
	double md_loss;
	MatrixXd mMat_dx;
};


class DNN_API Accuracy:public BaseNNLayer
{
public:
	Accuracy(){};
	int findMaxofCol(MatrixXd mat);
	double forward(MatrixXd _x, MatrixXd _label);
private:
	double md_accu;
};



DNN_API bool DNNTraining(NNDataLayer &datalayer_training, NNDataLayer &datalayer_validation,
	QList<NNFullyConnectLayer> &vFullyConnectList, QList<NNSigmoidLayer> &vSigmoidList,
	NNQuadraticLoss &losslayer, Accuracy &accuracy,
	double dLearningRate = 1.0, double dEpoch = 100, double max_acc = 0.99, bool bEpochConstraint = true, QString sLogFn = "log.csv");

DNN_API bool saveModel(QList<NNFullyConnectLayer> &vFC, QString fn);

DNN_API bool loadModel(QList<NNFullyConnectLayer> &vFC, QString fn);


