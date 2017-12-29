#pragma once
#include "opencv2/core/core.hpp"
#include <iostream>
#include <ostream>
#include <fstream>
#include <list>
#define MAX_TASK_SIZE 6000
#define MAX_TASK_VOLUME 40000
#define JOINT_NUM 20
#define PIXEL_SHIFT 5
#define TRAINALL
//#define KINECT2 1
#define KINECT1 1
//#define USE_SORT
#define DISTANCE 200000000
#define LEAF_PIXEL_COUNT 50
#define TREE_HIGHT 10
using namespace cv;
using namespace std;

struct Pixel{
    Pixel() {}
    int pixel_index;
    int img_index;
    int m;
    int n;
    CvPoint3D32f point;
};

struct DepthImg{
    DepthImg() {}
    int index;
    Mat img;
    Mat fore;
    std::vector<CvPoint3D32f> joint_location;
};

struct IntIndex {
    int val;
    unsigned int index;
};

struct LeafNode {
    LeafNode() {}
    std::vector<std::vector<CvPoint3D32f> > offset;
    std::vector<std::vector<float> > weight;
};

struct AbsoluteLocation{
    AbsoluteLocation() {}
    std::vector<std::vector<CvPoint3D32f> > location;
    std::vector<std::vector<double> > weight;
};

class SlaveTask{
public:
    SlaveTask(){
        train_set_size = 0;
        node = -1;
        depth = -1;
        volume = 0;
        train_set_index = NULL;
    }

    int train_set_size;
    int node;
    int depth;
    int volume;
    int *train_set_index;

    bool operator < (SlaveTask& d) {
		return volume < d.volume;
	}

	bool operator <= (SlaveTask& d) {
		return volume <= d.volume;
	}

	bool operator > (SlaveTask& d) {
		return volume > d.volume;
	}

	bool operator >= (SlaveTask& d) {
		return volume >= d.volume;
	}

	void print(){
		printf("volume=%d size=%d node=%d depth=%d\n", volume, train_set_size, node, depth);
	}
};

class CRTree {
public:
    CRTree( char* filename);
    CRTree(int min_s, int max_d, int cj, CvRNG* pRNG) : min_samples(min_s), max_depth(max_d), num_leaf(0), num_joint(cj), cvRNG(pRNG) {
        num_nodes = (int)pow(2.0,int(max_depth+1))-1;
        treetable = new int[num_nodes * 6];
        for(unsigned int i=0; i<num_nodes * 6; ++i) treetable[i] = 0;
        leaf = new LeafNode[(int)pow(2.0,int(max_depth))];
    }
    ~CRTree() {delete[] leaf; delete[] treetable;}

    unsigned int GetDepth()  {return max_depth;}
    unsigned int GetNumJoints()  {return num_joint;}
    AbsoluteLocation regression(Mat Img,Mat fore);
    void growTree( std::vector< Pixel>& TrainSet,   std::vector<DepthImg> ImgSet,int samples);
    bool saveTree( char* filename);

    int mpi_rank;
    int mpi_size;
    int thread_rank;
    int thread_size;
    int openmp;
    int* rank_status;
    int debug_tasklist;
    int debug_output;
    int idle_slave_size;
    std::list<struct SlaveTask> task_list;
    void slave_work(std::vector<DepthImg> ImgSet,int samples);
    int  getIdleSlave();
    int* getIdleSlaves();

private:
    void grow( std::vector< Pixel>& TrainSet,  std::vector<DepthImg> ImgSet, int node, unsigned int depth, int samples);
    void parallel_grow( std::vector< Pixel>& TrainSet,  std::vector<DepthImg> ImgSet, int node, unsigned int depth, int samples);
    void makeLeaf( std::vector< Pixel>& TrainSet, std::vector<DepthImg> ImgSet, int node);
    bool optimizeTest(std::vector< Pixel>& SetA, std::vector< Pixel>& SetB,  std::vector< Pixel>& TrainSet,  std::vector< DepthImg> ImgSet,int* test, double* bestDict, unsigned int iter);
    bool parallel_optimizeTest(std::vector< Pixel>& SetA, std::vector< Pixel>& SetB,  std::vector< Pixel>& TrainSet,  std::vector< DepthImg> ImgSet,int* test, unsigned int iter);
    void generateTest(double* test, unsigned int max_w, unsigned int max_h);
    void split(std::vector< Pixel>& SetA, std::vector< Pixel>& SetB,  std::vector< Pixel>& TrainSet, std::vector<IntIndex>& valSet, int t);
    void evaluateTest(std::vector<IntIndex>& valSet,  double* test,  std::vector< Pixel>& TrainSet, std::vector< DepthImg> ImgSet);
    double measureSet(std::vector< Pixel>& SetA, std::vector< Pixel>& SetB, std::vector< DepthImg> ImgSet) { return -distMean(SetA,SetB,ImgSet);}
    double distMean( std::vector< Pixel>& SetA,  std::vector< Pixel>& SetB, std::vector< DepthImg> ImgSet);
    double depthCompareFeature(int dx,int dy,Mat depth,Mat fore,int u1,int v1,int u2, int v2);
    void sort(std::vector<IntIndex>& valSet);

    int* treetable;
    unsigned int min_samples;
    unsigned int max_depth;
    unsigned int num_nodes;
    unsigned int num_leaf;
    unsigned int num_joint;
    LeafNode* leaf;
    CvRNG *cvRNG;
};

inline void CRTree::generateTest(double* test, unsigned int max_w, unsigned int max_h)
{
    test[0] = cvRandInt( cvRNG ) % (max_h/2);
    test[1] = cvRandInt( cvRNG ) % (max_w/2);
    test[2] = cvRandInt( cvRNG ) % (max_h/2);
	test[3] = cvRandInt( cvRNG ) % (max_w/2);
}
