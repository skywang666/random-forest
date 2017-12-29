#include <vector>
#include <assert.h>
#include "opencv2/core/core.hpp"
using namespace cv;

void cluster(std::vector<std::vector<float> > data, std::vector<float> weights, float bandwidth, std::vector<std::vector<float> >& clusters, std::vector<int>& group);
void numCount(std::vector<int>& group,std::vector<int> &index_each_cluster,std::vector<int> &num_each_cluster,int numCluster);
void weightsCount(std::vector<int>& group,std::vector<float> weights,std::vector<int> &index_each_cluster,std::vector<float> &weights_each_cluster,int numCluster);
void sorting(int size,std::vector<int> &index,std::vector<int> &value);
void sorting(int size,std::vector<int> &index,std::vector<float> &value);
CvPoint3D32f depthPoint2cameraPoint(int m,int n,Mat depth);
