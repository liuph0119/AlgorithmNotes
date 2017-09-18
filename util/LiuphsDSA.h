#ifndef LIUPHSDSA_H
#define LIUPHSDSA_H

#include <vector>
#include <iostream>
#include <ctime>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>


namespace LiuphsDSA{
	// �������;
	template<class TT>
	inline TT vector_sum(std::vector<TT> &v, int n)
	{
		return (n<1)?0:vector_sum(v, n-1) + v[n-1];
	}

	// ����ȥ��;
	template<class TT>
	void vector_reverse(std::vector<TT> &v, int nlow, int nhigh)
	{
		//if(nlow < nhigh){ std::swap(v[nlow], v[nhigh]); vector_reverse(v, nlow+1, nhigh-1); }
		while (nlow < nhigh) std::swap(v[nlow++], v[nhigh--]);
	}

	//��ӡ����;
	template<class TT>void vector_print(std::vector<TT> &v)
	{
		for (int i = 0; i <v.size(); i ++)
			std::cout << v[i] << " ";
		std::cout <<"\n";
	}

}

//��������;
namespace LiuphsDSA{

	//Ԫ��ȥ��;
	template<class TT>int vector_unique(std::vector<TT> &v)
	{
		int i=0, j=0;
		while (++j < v.size())
			if (v[i]!=v[j]) v[++i] = v[j];
		v.erase(v.begin()+i+1, v.end());
		return j-(i+1);
	}

	//���ֲ���;
	template<class TT>static int binSearch(std::vector<TT> &v, TT e, int nlow, int nhigh)
	{
		while(1 < nhigh-nlow)
		{
			int mi = (nlow+nhigh)>>1;		//half
			(e < v[mi])?nhigh=mi:nlow=mi;	//[low, mi) or [mi, high)
		}
		return (e==v[nlow])?nlow:-1;
	}
}

//������������;
namespace LiuphsDSA{
	/*********************************** ð������ ********************************************/
	//��¼һ���Ƿ�����;
	template<class TT> int bubble(std::vector<TT> &v, int nlow, int nhigh)
	{
		int nlast = nlow;						//���Ҳ������ʼ��Ϊ[nlow-1, low];
		while((++nlow) < nhigh)					//��������ɨ��;
			if (v[nlow-1] > v[nlow])			//��������;
			{
				nlast = nlow;					//�������Ҳ������λ�ü�¼;
				std::swap(v[nlow-1], v[nlow]);	//����;
			}
		return nlast;
	}
	//ð������; //ʱ�临�Ӷ� O(n)
	template<class TT> void bubbleSort(std::vector<TT> &v, int nlow, int nhigh)
	{
		while (nlow < (nhigh = bubble(v, nlow, nhigh)));
	}
	/*********************************** �鲢���� ********************************************/
	//��·�鲢; //��v�д�nlow-mi��mi-nhigh�ĺϲ�;
	//ÿ�αȽ�����vector�ĵ�һ�ȡС��һ��ŵ�Ŀ��������;
	template<class TT>void vector_merge(std::vector<TT> &v, int nlow, int mi, int nhigh)
	{
		int lb = mi - nlow, lc = nhigh - mi;
		std::vector<TT> B;
		for(int i = 0; i < lb; B.push_back(v[nlow+i++]));
		for (int i = 0, j = 0, k = 0; (j < lb) || (k < lc); )
		{
			if( (j<lb) && (lc<=k || B[j]<=v[k+mi]) ) v[nlow+(i++)] = B[j++]; 
			if( (k<lc) && (lb<=j || v[k+mi]< B[j]) ) v[nlow+(i++)] = v[mi+(k++)];
		}
		B.clear();
	}
	
	//�鲢����; //ʱ�临�Ӷ� O(nlog(n))
	template<class TT> void mergeSort(std::vector<TT> &v, int nlow, int nhigh)
	{
		if(nhigh-nlow < 2) return;
		int mi = (nlow+nhigh)>>1;				//���е�Ϊ��;
		mergeSort(v, nlow, mi);					//��ǰ�������;
		mergeSort(v, mi, nhigh);				//�Ժ�������;
		vector_merge(v, nlow, mi, nhigh);		//�鲢;
	}


}


//�б�;
namespace LiuphsDSA{
#define Posi(T) ListNode<T>*//�б�ڵ�λ��;
	template <typename TT>
	struct ListNode{
		TT data;
		Posi(TT) pred;//ǰ��;
		Posi(TT) succ;//���;
		ListNode(){}
		ListNode(TT e, Posi(TT) p=NULL, Posi(TT) s=NULL):data(e), pred(p),succ(s){} //Ĭ�Ϲ�����;
		Posi(TT) insertAsPred(TT const &e)//O(1)
		{
			Posi(TT) x = new ListNode(e, pred, this);
			pred->succ = x; this->pred = x; return x;
		}
		Posi(TT) insertAsSucc(TT const &e)//O(1)
		{
			Posi(TT) x = new ListNode(e, this, succ);
			succ->pred = x; this->succ = x; return x;
		}
	};

template<typename TT> class List{
private:
	int _size;//��ģ;
	Posi(TT) header;//ͷ�ڱ�;
	Posi(TT) tailer;//β�ڱ�;
protected: //�ڲ�����;
public:
	//��ʼ������;
	template <typename TT> 
	void init(){
		header = new ListNode<TT>;//�����ڵ�;
		tailer = new ListNode<TT>;
		header->succ = tailer; header->pred = NULL;	//����
		tailer->pred = header; tailer->succ = NULL;
		_size = 0;									//��ģ;
	}
	inline int size(){return _size;}
	//�����±������;
	TT operator[](int r) const { // O(r)��Ч�ʽϵͣ����˳���;
		Posi(TT) p = first();// first()
		while (0<r--) p = p->succ;
		return p->data;
	}
	//��ȡ��ʼ�ڵ�;

	inline Posi(TT) first(){
		return header->succ;
	}
	//��ȡĩβ�ڵ�;
	inline Posi(TT) last(){
		return tailer->pred;
	}
	//���ҽڵ�pǰ��n��ǰ���У�ֵΪe�Ľڵ�;
	Posi(TT) find(TT const &e, int n, Posi(TT) p) const {//O(n)
		while (0 < n--)//������ǰ����p��ǰ����data�Ƚ�;
			if(e == (p=p->pred)->data) return p;
		return NULL;//����ʧ��;
	}
	//�ڽڵ�pǰ����ڵ�;
	Posi(TT) insertBefore(Posi(TT)p, TT const& e){
		_size ++; return p->insertAsPred(e);
	}
	//�ڽڵ�p�����ڵ�;
	Posi(TT) insertAfter(Posi(TT)p, TT const& e){
		_size ++; return p->insertAsSucc(e);
	}
	//��ĩβ����;
	void insertAsLast(TT const& e)
	{
		_size ++; tailer->insertAsPred(e);
	}
	//��ͷ������;
	void insertAsFirst(TT const& e)
	{
		_size ++; header->insertAsSucc(e);
	}
	//���ƽڵ�;//O(n)
	void copyNodes(Posi(TT) p, int n)
	{
		init();
		while (n --){ insertAsLast(p->data); p = p->succ;}
	}
	//ɾ��ĳ���ڵ�;
	TT remove(Posi(TT) p)
	{
		TT e = p->data;
		p->pred->succ = p->succ;
		p->succ->pred = p->pred;
		delete p;
		_size --;
		return e;
	}
	//ȫ�����;//O(n)
	int clear()
	{
		int oldSize = _size;
		while(0 < _size)
			remove(header->succ);
		return oldSize;		
	}
	//�����б��ȥ��;
	int deduplicate(){
		if(_size < 2) return 0;
		int oldSize = _size;
		Posi(TT) p = first();
		int r = 1;
		while(tailer != (p = p->succ)){
			Posi(TT) q = find(p->data, r, p);//����pǰ��n��ǰ���Ƿ���q�ظ�;
			q?remove(q):r ++;
		}
		return oldSize - _size;
	}
	//�����б��Ψһ��; //O(n)
	int uniquify()
	{
		if(_size < 2) return 0;
		int oldSize = _size;
		Posi(TT) p = first(), q;		//pΪ��㣬qΪp�ĺ��;
		while(tailer != (q = p->succ))		//�����Ƚ����ڵ�p �� q;
			if (p->data != q.data) p = q;	//���p q ���죬p�ƶ���q;
			else remove(q);					//����ɾ����ͬ��q;
		return oldSize - _size;
	}
	//���������б��нڵ�p��n��ǰ���У����Ҳ�����e�������; //O(1) - O(n)
	Posi(TT) search(TT const& e, int n, Posi(TT) p) const{
		while ( 0 <= n--) if( (p = p->pred)->data <= e ) break;
		return p;
	}
	//�������;
	void traverse_print()
	{
		printf("list values(size = %d): head -> ", _size);
		Posi(TT) p = header;
		while(tailer != (p = p->succ))
			std::cout << p->data<<" -> ";
		printf("tail\n");
	}
	//ѡȡp��n���ڵ��е����ֵ;
	Posi(TT) selecMax(Posi(TT) p, int n)
	{
		Posi(TT) pmax = p;
		for (Posi(TT) cur = p; 1 < n; n --)
			if(! ((cur = cur->succ)->data < pmax->data))
				pmax = cur;
		return pmax;
	}
	//���б���ʼ��p����n��Ԫ����ѡ������;
	void selectionSort(Posi(TT) p, int n)
	{
		Posi(TT) head = p->pred;
		Posi(TT) tail = p;				//����������(head, tail)
		for (int i = 0; i< n; i ++) tail = tail->succ;
		while(1 < n){
			insertBefore(tail, remove(selecMax(head->succ, n)));
			tail = tail->pred; //����������/��������ķ�Χͬ������;
			n --;
		}
	}
	//���б���ʼ��p����n��Ԫ������������; // O(n~n^2)
	void insertSort(Posi(TT) p, int n)
	{
		for (int r = 0; r < n; r ++)
		{
			insertAfter(search(p->data, r, p), p->data);
			p = p->succ;
			remove(p->pred);
		}
	}

	//��list�����ݴ���vector;
	void data2vector(std::vector<TT> &v)
	{
		v.clear();
		Posi(TT) p = header;
		while(tailer != (p = p->succ))
			v.push_back(p->data);
	}
public:	//���� ����;
	//���캯��;
	List(){init<TT>();}
	//ͨ�����鹹��;
	List(TT *p, int n)
	{
		init<TT>();
		for (int i = 0; i<n; i++)
			insertAsLast(p[i]);
	}
	//��������;
	~List(){clear(); delete header; delete tailer;}
};
}

#endif