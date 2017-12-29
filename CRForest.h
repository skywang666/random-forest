#pragma once
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include "CRTree.h"
#include "meanshift.h"
#include <time.h>
#include <vector>
#include <iostream>

struct Joint
{
    Joint() {}
	std::vector<CvPoint3D32f> location;//各关节点的位置
	std::vector<double> weight;//各关节点的置信度
};

class CRForest {//Forest 类
    private:
        std::vector<CRTree*> vTrees;

    public:
        CRForest(int trees = 0)
        {
            vTrees.resize(trees);
        }

        ~CRForest()
        {
            for(std::vector<CRTree*>::iterator it = vTrees.begin(); it != vTrees.end(); ++it)
                delete *it;
            vTrees.clear();
        }

        void SetTrees(int n)
        {
            vTrees.resize(n);
        }

        int GetSize()
        {
            return vTrees.size();
        }

        unsigned int GetDepth()
        {
            return vTrees[0]->GetDepth();
        }

        Joint regression(Mat img,Mat fore) ;

        void trainForest(int min_s, int max_d, CvRNG* pRNG,  std::vector<Pixel>& TrainSet,  std::vector<DepthImg> ImgSet, int samples);

        void saveForest( const char* filename, unsigned int offset = 0);

        void loadForest( char* filename, int type = 0);
};

inline Joint CRForest::regression(Mat img,Mat fore)//Forest 类的回归函数，预测是要用
{
    Joint joint_set;
    std::vector<AbsoluteLocation> abs_location_set;

    for(int i=0; i<(int)vTrees.size(); ++i) {
		AbsoluteLocation abs_location = vTrees[i]->regression(img,fore);
		abs_location_set.push_back(abs_location);
	}

	for (int j=0;j<vTrees[0]->GetNumJoints();j++)
	{
	    CvPoint3D32f point;
		point.x=0;point.y=0;point.z=0;
		double totalweight = 0;
		int total=0;
		double weight;

		for (int i=0;i<abs_location_set.size();i++)
		{
		    for (int k=0;k<abs_location_set[i].location[j].size();k++)
		    {
		        point.x += abs_location_set[i].weight[j][k]*abs_location_set[i].location[j][k].x;
				point.y += abs_location_set[i].weight[j][k]*abs_location_set[i].location[j][k].y;
				point.z += abs_location_set[i].weight[j][k]*abs_location_set[i].location[j][k].z;
				totalweight += abs_location_set[i].weight[j][k];
				total++;
		    }
		}
		point.x = point.x/totalweight;
		point.y = point.y/totalweight;
		point.z = point.z/totalweight;
		weight = totalweight/total;
		joint_set.location.push_back(point);
		joint_set.weight.push_back(weight);
	}
	return joint_set;
}

inline void CRForest::trainForest(int min_s, int max_d, CvRNG* pRNG,  std::vector<Pixel>& TrainSet,  std::vector<DepthImg> ImgSet, int samples) {
	for(int i=0; i < (int)vTrees.size(); ++i) {
		vTrees[i] = new CRTree(min_s, max_d, ImgSet[0].joint_location.size(), pRNG);

        MPI_Comm_size(MPI_COMM_WORLD, &vTrees[i]->mpi_size);
        MPI_Comm_rank(MPI_COMM_WORLD, &vTrees[i]->mpi_rank);
		
        vTrees[i]->rank_status = (int*)malloc(sizeof(int)*vTrees[i]->mpi_size);
        for (int j=0; j < vTrees[i]->mpi_size; j++)
            vTrees[i]->rank_status[j] = 1;
		vTrees[i]->idle_slave_size = vTrees[i]->mpi_size - 1;
		
		char * env=NULL;
		env = getenv("FOREST_DEBUG_TASKLIST");
		if (env)
			vTrees[i]->debug_tasklist = 1;
		else
			vTrees[i]->debug_tasklist = 0;
		env = getenv("FOREST_DEBUG_OUTPUT");
		if (env)
			vTrees[i]->debug_output = 1;
		else
			vTrees[i]->debug_output = 0;
		
		if (vTrees[i]->mpi_rank == 0){
			vTrees[i]->growTree(TrainSet, ImgSet, samples);
			cout<<"End master work!"<<endl;
		}
		else{
			double t_start, t_end;
			t_start = MPI_Wtime();
			vTrees[i]->slave_work(ImgSet,samples);
			t_end = MPI_Wtime();
			cout<<"slave time is "<<t_end-t_start<<endl;
		}
	}
}

#define sprintf_s sprintf
inline void CRForest::saveForest(const char* filename, unsigned int offset) 
{
	char buffer[200];
	for(unsigned int i=0; i<vTrees.size(); ++i) 
	{
		sprintf_s(buffer,"%s%03d.txt",filename,i+offset);
		vTrees[i]->saveTree(buffer);
	}
}


inline void CRForest::loadForest( char* filename, int type)
{
	char buffer[200];
	for(unsigned int i=0; i<vTrees.size(); ++i) 
	{
		sprintf_s(buffer,"%s%03d.txt",filename,i);
		vTrees[i] = new CRTree(buffer);
	}
}
