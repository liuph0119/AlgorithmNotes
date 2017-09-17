#ifndef LIUPHSUTILS_H
#define LIUPHSUTILS_H
#include <vector>
#include <iostream>
#include <ctime>
#include <string>
#include <map>
#include <fstream>
#include <sstream>
#include <cmath>
#include <omp.h>
namespace LiuphsUtils
{
	inline std::string namespace_name() {return std::string("LiuphsUtils");}

	// sum value of a vector
	template<class TT>
	TT vector_sum(std::vector<TT> &vValue)
	{
		TT val = 0;
		for (int i = 0; i < vValue.size(); ++i)
			val += vValue[i];
		return val;
	}


	// roulette wheel strategy selection
	// select the index of a value from an vector
	// should set random seed first!!!
	template <class TT>
	TT rouletteWheelSelection( std::vector<TT> &vValues, int &ind )
	{
		// 0 < r < sum_value
		double dSum = vector_sum(vValues);
		double r = ((double)rand()) / RAND_MAX * dSum;//(0, MAX]

		dSum = 0;
		for(ind = 0; ind < vValues.size(); ind ++)
		{
			dSum += vValues[ind];
			if(dSum > r) return vValues[ind];
		}
		// avoid growing like a '\' line
		ind = rand()%vValues.size();
		return vValues[ind];
	}

	// returns the highest value  and index (argmax)
	template<class TT>
	TT arg_max( std::vector<TT> &vValues, int &ind )
	{
		ind = 0;
		for (int i = 0; i < vValues.size(); i ++)
			if(vValues[ind] < vValues[i]) ind = i;
		return vValues[ind];
	}

	//replace from-string to to-string
	void replace_str(std::string& str, const std::string& from, const std::string& to) 
	{
		if (from.empty())
			return;
		size_t start_pos = 0;
		while ((start_pos = str.find(from, start_pos)) != std::string::npos) 
		{
			str.replace(start_pos, from.length(), to);
			start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
		}
	}

	// get length of an array, not pointer
	template <class T>
	int getArrayLen(T& array){return (sizeof(array) / sizeof(array[0]));}

	//trim function
	std::string trimmed( std::string &s )
	{
		s.erase(0, s.find_first_not_of(" \t\r\n"));
		s.erase(s.find_last_not_of(" \t\r\n")+1);
		return s;
	}
	
	//split a string by a certain pattern
	std::vector<std::string> split(std::string str,std::string pattern)
	{   
		std::string::size_type pos;
		std::vector<std::string> result;
		str += pattern;
		int size=str.size();
		for(int i=0; i<size; i++)
		{
			pos=str.find(pattern,i);
			if(pos<size)
			{
				std::string s=str.substr(i,pos-i);
				result.push_back(s);
				 i=pos+pattern.size()-1;
			}
		}
		return result;
	}

	// class to write logging
	class Logging
	{
	public:
		Logging(int max_len = 1000) {m_pfLogFile = NULL; cInfo = new char[max_len];}
		~Logging(void) { 
			if (NULL != m_pfLogFile)
			{
				fclose(m_pfLogFile);
				m_pfLogFile = NULL;
			}}

		//log file;
		FILE* m_pfLogFile;
		char* cInfo;
		std::string sInfo;

		// open a  log file by file-name
		int  SetLogFile(std::string cFileName)
		{
			if(NULL != m_pfLogFile)
			{
				fclose(m_pfLogFile);
			}
			m_pfLogFile = fopen(cFileName.c_str(),"a+"); 
			if(NULL == m_pfLogFile)
				return 0;
			return 1;
		}
		//write a new string
		int  WriteLogInfo(std::string pInfo)
		{
			time_t _t;
			time(&_t);
			std::string timestr (ctime(&_t));
			replace_str(timestr, "\n", "\t");
			// print to console while writing to log file
			printf( "> %s%s", timestr.c_str(), pInfo.c_str());

			if(NULL != m_pfLogFile)
			{
				fprintf(m_pfLogFile,"> %s\t%s", timestr.c_str(), pInfo.c_str());
				fflush(m_pfLogFile);
				return 1;
			}
			return 0;
		}

		//write a new char[]
		int  WriteLogInfo(char *cInfo)
		{
			time_t _t;
			time(&_t);
			std::string timestr (ctime(&_t));
			replace_str(timestr, "\n", "\t");
			printf( "> %s%s", timestr.c_str(), cInfo);

			if(NULL != m_pfLogFile)
			{
				fprintf(m_pfLogFile,"> %s\t%s", timestr.c_str(), cInfo);
				fflush(m_pfLogFile);
				return 1;
			}
			return 0;
		}
	};

	// class to handle timing and drawing text progress output
	// copied from mojo
	class progress
	{

	public:
		progress(int size=-1, const char *label=NULL ) {reset(size, label);}

#if (_MSC_VER  == 1600)
		unsigned int start_progress_time;
#else
		std::chrono::time_point<std::chrono::system_clock>  start_progress_time;
#endif
		unsigned int total_progress_items;
		std::string label_progress;
		// if default values used, the values won't be changed from last call
		void reset(int size=-1, const char *label=NULL ) 
		{
#if (_MSC_VER  == 1600)
			start_progress_time= clock();
#else
			start_progress_time= std::chrono::system_clock::now();
#endif
			if(size>0) total_progress_items=size; if(label!=NULL) label_progress=label;
		}
		float elapsed_seconds() 
		{	
#if (_MSC_VER  == 1600)
			float time_span = (float)(clock() - start_progress_time)/CLOCKS_PER_SEC;
			return time_span;
#else
			std::chrono::duration<double> time_span = std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::system_clock::now() - start_progress_time);
			return (float)time_span.count();
#endif
		}
		float remaining_seconds(int item_index)
		{
			float elapsed_dt = elapsed_seconds();
			float percent_complete = 100.f*item_index/total_progress_items;
			if(percent_complete>0) return ((elapsed_dt/percent_complete*100.f)-elapsed_dt);
			return 0.f;
		}
		// this doesn't work correctly with g++/Cygwin
		// the carriage return seems to delete the text... 
		void draw_progress(int item_index)
		{
			int time_remaining = (int)remaining_seconds(item_index);
			float percent_complete = 100.f*item_index/total_progress_items;
			if (percent_complete > 0)
			{
				std::cout << label_progress << (int)percent_complete << "% (" << (int)time_remaining << "sec remaining)              \r"<<std::flush;
			}
		}
		void draw_header(std::string name, bool _time=false)
		{
			std::string header = "==  " + name + "  ";

			int seconds = 0;
			std::string elapsed;
			int L = 79 - (int)header.length();
			if (_time)
			{
				seconds = (int)elapsed_seconds();
				int minutes = (int)(seconds / 60);
				int hours = (int)(minutes / 60);
				seconds = seconds - minutes * 60;
				minutes = minutes - hours * 60;
				std::string min_string = std::to_string((long long)minutes);
				if (min_string.length() < 2) min_string = "0" + min_string;
				std::string sec_string = std::to_string((long long)seconds);
				if (sec_string.length() < 2) sec_string = "0" + sec_string;
				elapsed = " " + std::to_string((long long)hours) + ":" + min_string + ":" + sec_string;
				L-= (int)elapsed.length();
			}
			for (int i = 0; i<L; i++) header += "=";
			if (_time)
				std::cout << header << elapsed << std::endl;
			else 
				std::cout << header << std::endl;
		}
	};
}


//Term Frequency - Inverse Document Frequency
namespace LiuphsUtils
{
	typedef struct
	{
		std::map<std::string, int > word_count;
		int nSubWordDict;					//number of words in dict of a doc
		int nTotalWords;
		std::map<std::string, std::vector<float>> word_tf_tfidf;
	}doc;

	typedef struct
	{
		int nTotalWords;
		int nDoc;
		std::vector<doc> docs;
		std::map<std::string, float> word_idf;
	}corpus;

	

	class TDIDF
	{
	public:
		TDIDF(std::string corpus_fn, std::string stopwords_fn){
			printf("If the corpus is Chinese, please use ASCII encoding.\n");
			mCorpusFn = corpus_fn;
			mStopWordsFn = stopwords_fn;
		}
		~TDIDF(){}
		void run()
		{
			if(!loadStopWords() || !loadCorpus() || !check_doc_corpus())
				printf("error!\n");
			if(!cal_tf_idf())
				printf("calculate tf-idf error!\n");
			searchword();
		}

		bool save_tfidf(std::string fn)
		{
			printf("save tf-idf file...");
			std::ofstream _out;
			_out.open(fn.c_str(), std::ios::binary);
			if(_out.bad() || _out.fail())
			{
				printf("error!\n");
				return false;
			}
			std::map<std::string, std::vector<float>>::iterator it;
			for (int i = 0; i < mCorpus.nDoc; i ++)
			{
				for (it = mCorpus.docs[i].word_tf_tfidf.begin(); it != mCorpus.docs[i].word_tf_tfidf.end(); it ++)
					_out << it->first.c_str() << "," << it->second[1]<<",";
				_out << "\n";
			}
			_out.close();
			printf("success!\n");
			return true;
		}
	protected:
		bool loadCorpus()
		{
			printf("load corpus...");
			std::ifstream _in(mCorpusFn.c_str());
			if(_in.bad() || _in.fail())
			{
				printf("fail!\n");
				return false;
			}
			std::string sline, sfield;
			while (getline(_in, sline))
			{
				doc _doc;
				std::istringstream _sin(sline);
				while (getline(_sin, sfield, ' '))
				{
					// stop words
					if (!notin(sfield, mvStopWords)) continue;
					// index of dict in a doc
					// here, only word_count of each doc is counted
					std::map<std::string, int>::iterator it = _doc.word_count.find(sfield);
					if(it == _doc.word_count.end())
						_doc.word_count.insert(std::pair<std::string, int>(sfield, 1));
					else
						it->second ++;
				}
				mCorpus.docs.push_back(_doc);
			}
			_in.close();
			printf("success!\n");
			return true;
		}
		bool loadStopWords()
		{
			printf("load stop-words...");
			std::ifstream _in(mStopWordsFn.c_str());
			if(_in.bad() || _in.fail())
			{
				printf("fail!\n");
				return false;
			}
			std::string sline;
			while(getline(_in, sline))
				mvStopWords.push_back(sline);
			_in.close();
			printf("success!\n");
			return true;
		}

		//judge if a word in vector
		bool notin(std::string s, std::vector<std::string> v)
		{
			bool flag = true;
			for (int i = 0; i < v.size(); i ++)
				flag = flag && s!=v[i];
			return flag;
		}
		//finish the structure of doc && corpus
		bool check_doc_corpus()
		{
			printf("checking corpus and docs...\n");
			mCorpus.nDoc = mCorpus.docs.size();
			mCorpus.nTotalWords = 0;
			printf("docs in corpus: %d\n", mCorpus.nDoc);
			//iteration in docs
			for (int i = 0; i < mCorpus.nDoc; i ++)
			{
				mCorpus.docs[i].nTotalWords = 0;
				mCorpus.docs[i].nSubWordDict = mCorpus.docs[i].word_count.size();
				printf("document %4d : ", i);
				//iteration in dict of doc
				for (std::map<std::string, int>::iterator it = mCorpus.docs[i].word_count.begin(); it != mCorpus.docs[i].word_count.end(); it ++)
				{
					printf("%10s ", it->first.c_str());
					mCorpus.docs[i].nTotalWords += it->second;
					mCorpus.nTotalWords += it->second;
					//the dict in corpus
					std::map<std::string, std::vector<int>>::iterator it_all = mWordOccurDocIds.find(it->first);
					if(it_all == mWordOccurDocIds.end())
					{
						mCorpus.word_idf.insert(std::pair<std::string, float>(it->first, (float)0.0));
						//words in which docs??
						std::vector<int> v_doc_ids;
						v_doc_ids.push_back(i);
						mWordOccurDocIds.insert(std::pair<std::string, std::vector<int>>(it->first, v_doc_ids));
					}
					else
						it_all->second.push_back(i);
				}
				printf("\n");
				//printf("words number in doc_%-4d and dict_%-4d: %d, %d\n", i, i, mCorpus.docs[i].nTotalWords, mCorpus.docs[i].nSubWordDict);
			}
			printf("total words in corpus: %d\n", mCorpus.nTotalWords);
			return true;
		}
		//calculate tf and idf and tf-idf;
		bool cal_tf_idf()
		{
			printf("calculating tf and tf-idf...");
			std::map<std::string, float>::iterator it;
			std::map<std::string, std::vector<int>>::iterator it_ids;
			for (it = mCorpus.word_idf.begin(); it != mCorpus.word_idf.end(); it++)
			{
				it_ids = mWordOccurDocIds.find(it->first);
				if(it_ids != mWordOccurDocIds.end())
					//inverse document frequent = log(|D|/(1+|d|))
					it->second = log((float)mCorpus.nDoc/(1.0 + it_ids->second.size()));
			}
			std::map<std::string, int>::iterator it_word;
			for (int i = 0; i < mCorpus.nDoc; i ++)
			{
				mCorpus.docs[i].word_tf_tfidf.clear();
				for (it_word = mCorpus.docs[i].word_count.begin(); it_word!=mCorpus.docs[i].word_count.end(); it_word++)
				{
					std::vector<float> v_tf_tfidf;
					//tf
					float _f_tf = (float)(it_word->second)/(float)mCorpus.docs[i].nTotalWords;
					float _f_tfidf = 0;
					it = mCorpus.word_idf.find(it_word->first);
					if(it!=mCorpus.word_idf.end())
						//idf
						_f_tfidf = (_f_tf*it->second);
					v_tf_tfidf.push_back(_f_tf);
					v_tf_tfidf.push_back(_f_tfidf);
					mCorpus.docs[i].word_tf_tfidf.insert(std::pair<std::string, std::vector<float>>(it_word->first, v_tf_tfidf));
				}
			}
			printf("success!\n");
			return true;
		}
		//query a certain word
		void searchword()
		{
			std::string word;
			int doc_id;
			char cword[1024];
			while (true)
			{
				printf("please input word and document_id(separated by \" \"; word='q' to quit):");
				scanf("%s %d", cword, &doc_id);
				word = cword;
				if(word == "q")
					break;
				if(doc_id > mCorpus.nDoc-1)
				{
					printf("document id is out of range!\n");
					continue;
				}
				std::map<std::string, std::vector<float>>::iterator it;
				std::map<std::string, float>::iterator it_idf;
				it = mCorpus.docs[doc_id].word_tf_tfidf.find(word);
				if(it != mCorpus.docs[doc_id].word_tf_tfidf.end())
					printf("%s(in doc_%d): %.6f\n", word.c_str(), doc_id, it->second[1]);
				else
					printf("%s is not in doc_%d\n", word.c_str(), doc_id);
			}
			

// 			for (it = mCorpus.docs[doc_id].word_tf_tfidf.begin(); it != mCorpus.docs[doc_id].word_tf_tfidf.end(); it++)
// 			{
// 				it_idf = mCorpus.word_idf.find(it->first);
// 				printf("%-15s: tf = %.6f      idf = %.6f      tf-idf = %.6f\n", it->first.c_str(), it->second[0], it_idf->second, it->second[1]);
// 			}
		}
		
	private:
		std::string mCorpusFn;
		std::string mStopWordsFn;
		std::vector<std::string> mvStopWords;
		std::map<std::string, std::vector<int>> mWordOccurDocIds;//word and which documents it appears
		corpus mCorpus;
	};
}


//plsa
namespace LiuphsUtils
{
	class pLSA
	{
	public:
		pLSA(std::string fn, int nTopic = 5, int nMaxIter = 100, double minDeltaLikelihood = 1e-6) { 
			printf("corpus format: 0 hello:2 world:6 plsa:1\n= = = = = = = = = = = = = = = = = = = =\n"); 
			mCorpusFn = fn; 
			mnTopic = nTopic;
			mnMaxIterNum = nMaxIter;
			mdDeltaMinLikelihood = minDeltaLikelihood;
			srand(unsigned(time(0)));
		}
		~pLSA(){mvWords.clear(); mvvWordCount.clear();}

		
		void run()
		{
			loadCorpus();
			randomize();
			plsa_algorithm();
		}

		bool save_doc_topic(std::string fn)
		{
			printf("save document-topic...");
			std::ofstream _out(fn.c_str());
			if (_out.bad() || _out.bad())
			{
				printf("error!\n");
				return false;
			}
			for (int i = 0; i < mnDoc; i ++)
			{
				for (int k = 0; k < mnTopic; k ++)
					_out << mpp_p_z_d[k][i] << ",";
				_out<<"\n";
			}
			_out.close();
			printf("success!\n");
			return true;
		}
		bool save_topic_word(std::string fn)
		{
			printf("save topic-word...");
			std::ofstream _out(fn.c_str());
			if (_out.bad() || _out.bad())
			{
				printf("error!\n");
				return false;
			}
			for (int i = 0; i < mnTopic; i ++)
			{
				for (int k = 0; k < mnWord; k ++)
					_out << mpp_p_w_z[k][i] << ",";
				_out<<"\n";
			}
			_out.close();
			printf("success!\n");
			return true;
		}

	protected:
		bool loadCorpus()
		{
			printf("load corpus...");
			std::ifstream _in(mCorpusFn.c_str());
			if (_in.bad() || _in.fail())
			{
				printf("error!\n");
				return false;
			}
			std::string sline;
			mnWord = 0, mnDoc = 0;
			while (getline(_in, sline))
			{
				mnDoc ++;
				std::vector<std::string> slist = split(sline, " ");
				for (int i = 1; i < slist.size(); i ++)
				{
					std::vector<std::string> ssublist = split(slist[i], ":");
					if (atoi(ssublist[0].c_str()) > mnWord) mnWord = atoi(ssublist[0].c_str());
				}
			}
			_in.close();
			_in.open(mCorpusFn.c_str());
			mvWords.resize(mnWord);
			mvvWordCount.resize(mnDoc);
			for (int i = 0; i < mnDoc; i ++)
				mvvWordCount[i].resize(mnWord);

			int _ndoc_id, _nword_id;
			while (getline(_in, sline))
			{
				std::vector<std::string> slist = split(sline, " ");
				_ndoc_id = atoi(slist[0].c_str());
				for (int i = 1; i < slist.size(); i ++)
				{
					std::vector<std::string> ssublist = split(slist[i], ":");
					_nword_id = atoi(ssublist[0].c_str());
					mvvWordCount[_ndoc_id][_nword_id] += atoi(ssublist[1].c_str());
					//printf("%d:%d ", _nword_id, mvvWordCount[_ndoc_id][_nword_id]);
				}
				//printf("\n");
			}
			_in.close();
			printf("success!\n");
			printf("doc num: %d   |   word num: %d\n", mnDoc, mnWord);
			return true;
		}
		bool randomize()
		{
			mdMaxLikelihood = 0;
			printf("topic num : %d\nrandomize parameters...", mnTopic);
			//…Í«Îƒ⁄¥Ê;
#pragma omp parallel sections 
			{
#pragma omp section
				{
					mppp_p_z_dw = new double**[mnTopic];
					for (int i = 0; i < mnTopic; i ++)
					{
						mppp_p_z_dw[i] = new double*[mnDoc];
						for (int j = 0; j < mnDoc; j ++)
						{
							mppp_p_z_dw[i][j] = new double[mnWord];
							memset(mppp_p_z_dw[i][j], 0, sizeof(double)*mnWord);
						}
					}
				}
#pragma omp section
				{
					mpp_p_w_z = new double*[mnWord];
					
					for (int i = 0; i < mnWord; i ++)
					{
						mpp_p_w_z[i] = new double[mnTopic];
						memset(mpp_p_w_z[i], 0, sizeof(double)*mnTopic);
					}
				}
#pragma omp section
				{
					mpp_p_z_d = new double*[mnTopic];
					for (int i = 0; i < mnTopic; i ++)
					{
						mpp_p_z_d[i] = new double[mnDoc];
						memset(mpp_p_z_d[i], 0, sizeof(double)*mnDoc);
					}
				}
			}
			

#pragma omp parallel sections
			{
#pragma omp section
#pragma omp parallel for schedule(dynamic)
				for (int i = 0; i < mnTopic; i ++)
					for (int j = 0; j < mnDoc; j ++)
						for (int k = 0; k < mnWord; k ++)
							mppp_p_z_dw[i][j][k] += (double)rand()/RAND_MAX;
#pragma omp section
#pragma omp parallel for schedule(dynamic)
				for (int i = 0; i < mnWord; i ++)
					for (int j = 0; j < mnTopic; j ++)
						mpp_p_w_z[i][j] += (double)rand()/RAND_MAX;
#pragma omp section
#pragma omp parallel for schedule(dynamic)
				for (int i = 0; i < mnTopic; i ++)
					for (int j = 0; j < mnDoc; j ++)
						mpp_p_z_d[i][j] += (double)rand()/RAND_MAX;
			}
			printf("success!\n");
// 			for (int i = 0; i < mnTopic; i ++)
// 			{
// 				for (int j = 0; j < mnDoc; j ++)
// 					printf("%.3f ", mpp_p_z_d[i][j]);
// 				printf("\n");
// 			}
			return true;
		}
		bool em1()
		{
			printf("expectation maximization step 1...");
#pragma omp parallel for schedule(dynamic)
			for (int k = 0; k < mnTopic; k ++)
			{
				for (int i = 0; i < mnDoc; i ++)
				{
					for (int j = 0; j < mnWord; j ++)
					{
						double _d_wz_zd = 0;
						for (int kk = 0; kk < mnTopic; kk ++)
						{
							_d_wz_zd += mpp_p_w_z[j][kk]*mpp_p_z_d[kk][i];
						}
						mppp_p_z_dw[k][i][j] = _d_wz_zd>0?(mpp_p_w_z[j][k]*mpp_p_z_d[k][i]/_d_wz_zd):(mpp_p_w_z[j][k]*mpp_p_z_d[k][i]/1e-6);
					}
				}
			}
			printf("success!\n");
			return true;
		}

		bool em2()
		{
			printf("update Pn(Wj|Zk)...");
#pragma omp parallel for schedule(dynamic)
			for (int j = 0; j < mnWord; j ++)
			{
				for (int k= 0; k < mnTopic; k ++)
				{
					double _d_dw_zdw = 0;
					for (int i = 0; i < mnDoc; i ++)
					{
						_d_dw_zdw += mvvWordCount[i][j]*mppp_p_z_dw[k][i][j];
					}

					double _d_dw_zdw_sum = 0;
					for (int jj = 0; jj < mnWord; jj ++)
					{
						double _d_dw_zdw_i = 0;
						for (int i = 0; i < mnDoc; i ++)
						{
							_d_dw_zdw_i += mvvWordCount[i][jj]*mppp_p_z_dw[k][i][jj];
						}
						_d_dw_zdw_sum += _d_dw_zdw_i;
					}

					mpp_p_w_z[j][k] = (_d_dw_zdw_sum) > 0?(_d_dw_zdw / _d_dw_zdw_sum) : (_d_dw_zdw/1e-6);
				}
			}
			printf("success!\r");

			printf("updating Pn(Zk|Di)...");
#pragma omp parallel for schedule(dynamic)
			for (int k = 0; k < mnTopic; k ++)
			{
				for (int i = 0; i < mnDoc; i ++)
				{
					double _d_dw_zdw = 0;
					for (int j = 0; j < mnWord; j ++)
						_d_dw_zdw += mvvWordCount[i][j]*mppp_p_z_dw[k][i][j];

					double _d_dw_zdw_sum = 0;
					for (int kk = 0; kk < mnTopic; kk ++)
					{
						double _d_dw_zdw_i = 0;
						for (int j = 0; j<mnWord; j ++)
						{
							_d_dw_zdw_i += mvvWordCount[i][j]*mppp_p_z_dw[kk][i][j];
						}

						_d_dw_zdw_sum += _d_dw_zdw_i;
					}
					mpp_p_z_d[k][i] = (_d_dw_zdw_sum)>0?(_d_dw_zdw/_d_dw_zdw_sum):(_d_dw_zdw/1e-6);
				}
			}
			printf("success!\r");

			printf("calculate maximum likelihood...");
			mdMaxLikelihood = 0;
#pragma omp parallel for schedule(dynamic)
			for (int i = 0; i < mnDoc; i ++)
			{
				for (int j = 0; j < mnWord; j ++)
				{
					double _dP_wjdi = 0;
					for (int k = 0; k < mnTopic; k ++)
					{
						_dP_wjdi += mpp_p_w_z[j][k]*mpp_p_z_d[k][i];
					}
					_dP_wjdi = 1.0/((double)mnDoc)*_dP_wjdi;		//P(di) = 1/mnDocs;??
					mdMaxLikelihood += mvvWordCount[i][j]*log(_dP_wjdi);
				}
			}
			printf("success!\rexpectation maximization step 2...success!\n");
			return true;
		}

		bool plsa_algorithm()
		{
			int _nLoopTime = 0;
			double _deltaMaxlikelihood, _maxlikelihood;
			_maxlikelihood = 0, mdMaxLikelihood = 0;
			do 
			{
				_maxlikelihood = mdMaxLikelihood;
				em1();
				em2();
				_nLoopTime ++;
				_deltaMaxlikelihood = fabs(mdMaxLikelihood-_maxlikelihood);
				printf("iteration %d\t\t", _nLoopTime);
				printf("maximum likelihood: %.4f\n\n", mdMaxLikelihood);
			} while (_nLoopTime  < mnMaxIterNum && _deltaMaxlikelihood > mdDeltaMinLikelihood);

			printf("result(doc-topic):\n");
			for (int i = 0; i < mnDoc; i ++)
			{
				for (int k = 0; k < mnTopic; k ++)
					printf("%.4f ", mpp_p_z_d[k][i]);
				printf("\n");
			}
			printf("\n");
			return true;
		}

	private:
		std::string mCorpusFn;
		int mnDoc;
		int mnWord;
		int mnTopic;
		std::vector<std::string> mvWords;
		std::vector<std::vector<int>> mvvWordCount;

		double*** mppp_p_z_dw;
		double** mpp_p_w_z;
		double** mpp_p_z_d;

		double mdMaxLikelihood;
		int mnMaxIterNum;
		double mdDeltaMinLikelihood;
	};
}


#endif
