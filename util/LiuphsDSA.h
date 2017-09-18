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
	// 向量求和;
	template<class TT>
	inline TT vector_sum(std::vector<TT> &v, int n)
	{
		return (n<1)?0:vector_sum(v, n-1) + v[n-1];
	}

	// 向量去重;
	template<class TT>
	void vector_reverse(std::vector<TT> &v, int nlow, int nhigh)
	{
		//if(nlow < nhigh){ std::swap(v[nlow], v[nhigh]); vector_reverse(v, nlow+1, nhigh-1); }
		while (nlow < nhigh) std::swap(v[nlow++], v[nhigh--]);
	}

	//打印向量;
	template<class TT>void vector_print(std::vector<TT> &v)
	{
		for (int i = 0; i <v.size(); i ++)
			std::cout << v[i] << " ";
		std::cout <<"\n";
	}

}

//有序向量;
namespace LiuphsDSA{

	//元素去重;
	template<class TT>int vector_unique(std::vector<TT> &v)
	{
		int i=0, j=0;
		while (++j < v.size())
			if (v[i]!=v[j]) v[++i] = v[j];
		v.erase(v.begin()+i+1, v.end());
		return j-(i+1);
	}

	//二分查找;
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

//无序向量排序;
namespace LiuphsDSA{
	/*********************************** 冒泡排序 ********************************************/
	//记录一趟是否有序;
	template<class TT> int bubble(std::vector<TT> &v, int nlow, int nhigh)
	{
		int nlast = nlow;						//最右侧逆序初始化为[nlow-1, low];
		while((++nlow) < nhigh)					//自左向右扫描;
			if (v[nlow-1] > v[nlow])			//出现逆序;
			{
				nlast = nlow;					//更新最右侧逆序对位置记录;
				std::swap(v[nlow-1], v[nlow]);	//交换;
			}
		return nlast;
	}
	//冒泡排序; //时间复杂度 O(n)
	template<class TT> void bubbleSort(std::vector<TT> &v, int nlow, int nhigh)
	{
		while (nlow < (nhigh = bubble(v, nlow, nhigh)));
	}
	/*********************************** 归并排序 ********************************************/
	//二路归并; //将v中从nlow-mi和mi-nhigh的合并;
	//每次比较两个vector的第一项，取小的一项放到目标向量中;
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
	
	//归并排序; //时间复杂度 O(nlog(n))
	template<class TT> void mergeSort(std::vector<TT> &v, int nlow, int nhigh)
	{
		if(nhigh-nlow < 2) return;
		int mi = (nlow+nhigh)>>1;				//以中点为界;
		mergeSort(v, nlow, mi);					//对前半段排序;
		mergeSort(v, mi, nhigh);				//对后半段排序;
		vector_merge(v, nlow, mi, nhigh);		//归并;
	}


}


//列表;
namespace LiuphsDSA{
#define Posi(T) ListNode<T>*//列表节点位置;
	template <typename TT>
	struct ListNode{
		TT data;
		Posi(TT) pred;//前驱;
		Posi(TT) succ;//后继;
		ListNode(){}
		ListNode(TT e, Posi(TT) p=NULL, Posi(TT) s=NULL):data(e), pred(p),succ(s){} //默认构造器;
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
	int _size;//规模;
	Posi(TT) header;//头哨兵;
	Posi(TT) tailer;//尾哨兵;
protected: //内部函数;
public:
	//初始化函数;
	template <typename TT> 
	void init(){
		header = new ListNode<TT>;//创建节点;
		tailer = new ListNode<TT>;
		header->succ = tailer; header->pred = NULL;	//互联
		tailer->pred = header; tailer->succ = NULL;
		_size = 0;									//规模;
	}
	inline int size(){return _size;}
	//重载下标操作符;
	TT operator[](int r) const { // O(r)，效率较低，不宜常用;
		Posi(TT) p = first();// first()
		while (0<r--) p = p->succ;
		return p->data;
	}
	//获取起始节点;

	inline Posi(TT) first(){
		return header->succ;
	}
	//获取末尾节点;
	inline Posi(TT) last(){
		return tailer->pred;
	}
	//查找节点p前面n个前驱中，值为e的节点;
	Posi(TT) find(TT const &e, int n, Posi(TT) p) const {//O(n)
		while (0 < n--)//从右往前，将p的前驱与data比较;
			if(e == (p=p->pred)->data) return p;
		return NULL;//查找失败;
	}
	//在节点p前插入节点;
	Posi(TT) insertBefore(Posi(TT)p, TT const& e){
		_size ++; return p->insertAsPred(e);
	}
	//在节点p后插入节点;
	Posi(TT) insertAfter(Posi(TT)p, TT const& e){
		_size ++; return p->insertAsSucc(e);
	}
	//在末尾插入;
	void insertAsLast(TT const& e)
	{
		_size ++; tailer->insertAsPred(e);
	}
	//在头部插入;
	void insertAsFirst(TT const& e)
	{
		_size ++; header->insertAsSucc(e);
	}
	//复制节点;//O(n)
	void copyNodes(Posi(TT) p, int n)
	{
		init();
		while (n --){ insertAsLast(p->data); p = p->succ;}
	}
	//删除某个节点;
	TT remove(Posi(TT) p)
	{
		TT e = p->data;
		p->pred->succ = p->succ;
		p->succ->pred = p->pred;
		delete p;
		_size --;
		return e;
	}
	//全部清除;//O(n)
	int clear()
	{
		int oldSize = _size;
		while(0 < _size)
			remove(header->succ);
		return oldSize;		
	}
	//无序列表的去重;
	int deduplicate(){
		if(_size < 2) return 0;
		int oldSize = _size;
		Posi(TT) p = first();
		int r = 1;
		while(tailer != (p = p->succ)){
			Posi(TT) q = find(p->data, r, p);//查找p前面n个前驱是否与q重复;
			q?remove(q):r ++;
		}
		return oldSize - _size;
	}
	//有序列表的唯一化; //O(n)
	int uniquify()
	{
		if(_size < 2) return 0;
		int oldSize = _size;
		Posi(TT) p = first(), q;		//p为起点，q为p的后继;
		while(tailer != (q = p->succ))		//反复比较相邻的p 和 q;
			if (p->data != q.data) p = q;	//如果p q 互异，p移动到q;
			else remove(q);					//否则删除相同的q;
		return oldSize - _size;
	}
	//查找有序列表中节点p的n个前驱中，查找不大于e的最后者; //O(1) - O(n)
	Posi(TT) search(TT const& e, int n, Posi(TT) p) const{
		while ( 0 <= n--) if( (p = p->pred)->data <= e ) break;
		return p;
	}
	//遍历输出;
	void traverse_print()
	{
		printf("list values(size = %d): head -> ", _size);
		Posi(TT) p = header;
		while(tailer != (p = p->succ))
			std::cout << p->data<<" -> ";
		printf("tail\n");
	}
	//选取p后n个节点中的最大值;
	Posi(TT) selecMax(Posi(TT) p, int n)
	{
		Posi(TT) pmax = p;
		for (Posi(TT) cur = p; 1 < n; n --)
			if(! ((cur = cur->succ)->data < pmax->data))
				pmax = cur;
		return pmax;
	}
	//对列表中始于p的连n个元素做选择排序;
	void selectionSort(Posi(TT) p, int n)
	{
		Posi(TT) head = p->pred;
		Posi(TT) tail = p;				//待排序区间(head, tail)
		for (int i = 0; i< n; i ++) tail = tail->succ;
		while(1 < n){
			insertBefore(tail, remove(selecMax(head->succ, n)));
			tail = tail->pred; //待排序区间/有序区间的范围同步更新;
			n --;
		}
	}
	//对列表中始于p的连n个元素做插入排序; // O(n~n^2)
	void insertSort(Posi(TT) p, int n)
	{
		for (int r = 0; r < n; r ++)
		{
			insertAfter(search(p->data, r, p), p->data);
			p = p->succ;
			remove(p->pred);
		}
	}

	//将list的数据存入vector;
	void data2vector(std::vector<TT> &v)
	{
		v.clear();
		Posi(TT) p = header;
		while(tailer != (p = p->succ))
			v.push_back(p->data);
	}
public:	//构造 析构;
	//构造函数;
	List(){init<TT>();}
	//通过数组构造;
	List(TT *p, int n)
	{
		init<TT>();
		for (int i = 0; i<n; i++)
			insertAsLast(p[i]);
	}
	//析构函数;
	~List(){clear(); delete header; delete tailer;}
};
}

#endif