#include "mpi.h"
#include <omp.h>
#include <time.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include "opencv/cxcore.h"
#include "opencv/cv.h"
#include "opencv/highgui.h"
#include "opencv2/core/core.hpp"
#include "CRTree.h"
#include "CRTree_mpi.h"
#include "meanshift.h"

using namespace std;
extern vector<Pixel> trainPixels;

// Read tree from file
CRTree::CRTree( char* filename) {
    cout << "Load Tree " << filename << endl;
    int dummy;
    ifstream in(filename);
    if(in.is_open())
    {
        in >> max_depth;
        num_nodes = (int)pow(2.0,int(max_depth+1))-1;
        treetable = new int[num_nodes * 6];

        in >> num_leaf;
        leaf = new LeafNode[num_leaf];

        in >> num_joint;

        int* ptT = &treetable[0];
        for(unsigned int n=0; n<num_nodes; ++n)
        {
            in >> dummy; in >> dummy;
            for(unsigned int i=0; i<6; ++i, ++ptT) {
                in >> *ptT;
            }
        }

        LeafNode* ptLN = &leaf[0];
        for(unsigned int l=1; l<=num_leaf; ++l, ++ptLN)
        {
            in >> dummy;
            in >> dummy;
            ptLN->offset.resize(dummy);
            ptLN->weight.resize(dummy);
            for(int i=0; i<dummy; ++i)
            {
                int K;
                in>>K;
                ptLN->offset[i].resize(K);
                ptLN->weight[i].resize(K);
                for(unsigned int k=0; k<K; ++k)
                {
                    in >> ptLN->offset[i][k].x;
                    in >> ptLN->offset[i][k].y;
                    in >> ptLN->offset[i][k].z;
					in >> ptLN->weight[i][k];
                }
            }
        }
    }else
        cout << "Could not read tree: " << filename << endl;
    in.close();
}

bool CRTree::saveTree( char* filename)  {
    cout << "Save Tree " << filename << endl;
    bool done = false;
    ofstream out(filename);
    if(out.is_open())
    {
        out << max_depth << " " << num_leaf << " " << num_joint << endl;
        int* ptT = &treetable[0];
        int depth = 0;
        unsigned int step = 2;
        for(unsigned int n=0; n<num_nodes; ++n)
        {
            if(n==step-1)
            {
                ++depth;
                step *= 2;
            }
            out << n << " " << depth << " ";
            for(unsigned int i=0; i<6; ++i, ++ptT)
            {
                out << *ptT << " ";
            }
            out << endl;
        }
        out << endl;

        LeafNode* ptLN = &leaf[0];
        for(unsigned int l=1; l<=num_leaf; ++l, ++ptLN)
        {
            out << l << " " <<ptLN->offset.size()<<" ";
            for(unsigned int i=0; i<ptLN->offset.size(); ++i)
            {
                out<<ptLN->offset[i].size()<<" ";
                for(unsigned int k=0; k<ptLN->offset[i].size(); ++k)
                {
                    out << ptLN->offset[i][k].x << " " << ptLN->offset[i][k].y << " " << ptLN->offset[i][k].z<<" ";
                    out << ptLN->weight[i][k]<<" ";
                }
            }
            out << endl;
        }
        out.close();
        done = true;
    }else
        cout << "Could not load tree: " << filename << endl;
    return done;
}

int CRTree::getIdleSlave() {
    int slave = 0;
    if (idle_slave_size > 0) {
        slave = mpi_size - 1;
        while ((rank_status[slave] == 2) && (slave > 0))
            slave --;
        assert(slave > 0);
        assert(rank_status[slave] == 1);
        rank_status[slave] = 2;
        idle_slave_size --;
        return slave;
    }else
        return slave;
}

int* CRTree::getIdleSlaves() {
#if 0
    int index=1;
    if (idle_slave_size > 0) {
        idle_slaves[0] = idle_slave_size;
        for (int i=1; i<mpi_size; i++) {
            if (rank_status[i] == 1) {
                rank_status[i] = 2;
				idle_slaves[index] = i;
				index ++;
            }
        }

        idle_slave_size = 0;
        return idle_slaves;
    }
    else{
        idle_slaves[0] = 0;
		return idle_slaves;
    }
#endif
    assert(0);
	return NULL;
}

void CRTree::makeLeaf( std::vector< Pixel>& TrainSet,  std::vector< DepthImg> ImgSet, int node)
{
    if (debug_output)
        cout<<"rank="<<mpi_rank<<" makeLeaf node="<<node<<" TrainSet Size= "<<TrainSet.size()<<endl;
    num_leaf ++ ;
    treetable[node*6] = num_leaf;
    LeafNode* ptL = &leaf[num_leaf - 1];
    std::vector<CvPoint3D32f> offset[JOINT_NUM];
	std::vector<float> weight[JOINT_NUM];
	assert(ImgSet[0].joint_location.size() == JOINT_NUM);

	float b = 0.05;//bandwidthin_set_index
	int C = 100;//采样数
	int j;
	#pragma omp parallel for shared(TrainSet, ImgSet, offset, weight)
	for (j=0;j<ImgSet[0].joint_location.size();j++)
	{
	    std::vector<std::vector<float> > offset_ij;//存储每个像素i位置与关节j的偏移
		std::vector<float> weights_ij;//权重
		std::vector<float> xyz;
		int sample_num = TrainSet.size()<C?TrainSet.size():C;
		if (TrainSet.size()<=C)
		{
		    for (int i=0;i<TrainSet.size();i++)
		    {
		        xyz.clear();
		        xyz.push_back(ImgSet[TrainSet[i].img_index].joint_location[j].x-TrainSet[i].point.x);
				xyz.push_back(ImgSet[TrainSet[i].img_index].joint_location[j].y-TrainSet[i].point.y);
				xyz.push_back(ImgSet[TrainSet[i].img_index].joint_location[j].z-TrainSet[i].point.z);
				offset_ij.push_back(xyz);
				weights_ij.push_back(1);
		    }
		}
		else
		{
		    std::vector<int> vi;
		    for (int i = 0; i < TrainSet.size(); i++)
				vi.push_back(i);
            srand ( unsigned ( time (NULL) ) );
            random_shuffle(vi.begin(), vi.end()); /* 打乱元素 */
            for (int i=0;i<C;i++)
			{
				xyz.clear();
				xyz.push_back(ImgSet[TrainSet[vi[i]].img_index].joint_location[j].x-TrainSet[vi[i]].point.x);
				xyz.push_back(ImgSet[TrainSet[vi[i]].img_index].joint_location[j].y-TrainSet[vi[i]].point.y);
				xyz.push_back(ImgSet[TrainSet[vi[i]].img_index].joint_location[j].z-TrainSet[vi[i]].point.z);
				offset_ij.push_back(xyz);
				weights_ij.push_back(1);
			}
		}
        std::vector<std::vector<float> > clusters;
        std::vector<int> group(sample_num, 0);
        cluster(offset_ij,weights_ij,b,clusters,group);
        vector<int> num_each_cluster;
		vector<int> index_each_cluster;

        numCount(group,index_each_cluster,num_each_cluster,clusters.size());
		sorting(clusters.size(),index_each_cluster,num_each_cluster);
        int K;
		K = clusters.size()<2?1:2;
		std::vector<CvPoint3D32f> offset_j;
		std::vector<float> weight_j;
		for (int i=0;i<K;i++)
		{
		    CvPoint3D32f temp;
			temp.x = clusters[index_each_cluster[i]][0];temp.y=clusters[index_each_cluster[i]][1];temp.z=clusters[index_each_cluster[i]][2];
			offset[j].push_back(temp);
			weight[j].push_back(num_each_cluster[i]);
		}
	}
	for (j=0;j<ImgSet[0].joint_location.size();j++)
	{
		ptL->offset.push_back(offset[j]);
		ptL->weight.push_back(weight[j]);
	}
}

void CRTree::evaluateTest(std::vector<IntIndex>& valSet,  double* test,  std::vector< Pixel>& TrainSet, std::vector< DepthImg> ImgSet)
{
    valSet.resize(TrainSet.size());
    #pragma omp parallel for if (openmp) shared(valSet)
    for(unsigned int i=0;i<TrainSet.size();++i)
    {
        double feat = depthCompareFeature(TrainSet[i].m,TrainSet[i].n,ImgSet[TrainSet[i].img_index].img,ImgSet[TrainSet[i].img_index].fore,test[0],test[1],test[2],test[3]);
        valSet[i].val = feat;
        valSet[i].index = i;
    }
#ifdef USE_SORT
    sort(valSet);
#endif
}

double CRTree::depthCompareFeature(int dm,int dn,Mat depth,Mat fore,int v1,int u1,int v2, int u2)
{
    int max = 10000;
    int width = depth.cols;
    int height = depth.rows;
    int m1,n1,m2,n2;
    double df1,df2;
    double d = 1.0*depth.at<ushort>(dm,dn)/1000;

    m1 = int(dm+1.0*v1/(1.0*d));
	n1 = int(dn+1.0*u1/(1.0*d));
	if (m1<0 || m1>=height || n1<0 || n1>=width || fore.at<uchar>(m1,n1) == 0)
		df1 = max;
    else
		df1 = depth.at<ushort>(m1,n1);

    m2 = int(dm+1.0*v2/(1.0*d));
	n2 = int(dn+1.0*u2/(1.0*d));
	if (m2<0 || m2>=height || n2<0 || n2>=width || fore.at<uchar>(m2,n2) == 0)
		df2 = max;
	else
		df2 = depth.at<ushort>(m2,n2);

    return df1-df2;
}

void CRTree::sort(std::vector<IntIndex>& valSet)
//升序排列
{
    int i,j,temp_val,temp_index;
    for(i = 0; i < valSet.size()-1; i++)
    {
        for(j = 0; j < valSet.size()-i-1; j++)
        {
            if(valSet[j].val > valSet[j+1].val)
            {
                temp_val = valSet[j].val;
				valSet[j].val = valSet[j+1].val;
				valSet[j+1].val = temp_val;

				temp_index = valSet[j].index;
				valSet[j].index = valSet[j+1].index;
				valSet[j+1].index = temp_index;
            }
        }
    }
}

#ifndef USE_SORT
void CRTree::split(vector< Pixel>& SetA, vector< Pixel>& SetB,  vector< Pixel>& TrainSet, vector<IntIndex>& valSet, int t)
{
    int sizeA = 0;
    int Aidx, Bidx;
    for (int i=0; i<valSet.size(); ++i)
        if (valSet[i].val < t)
            sizeA++;

    if (sizeA == 0) return;
    if ((TrainSet.size() - sizeA) == 0) return;
    SetA.resize(sizeA);
    SetB.resize(TrainSet.size()-SetA.size());
    Aidx = 0;
    Bidx = 0;
    for (int i=0; i<valSet.size(); ++i) {
        if (valSet[i].val < t)
        {
            SetA[Aidx] = TrainSet[valSet[i].index];
            Aidx ++ ;
        }
        else
        {
            SetB[Bidx] = TrainSet[valSet[i].index];
            Bidx ++ ;
        }
    }
}
#else
void CRTree::split(vector< Pixel>& SetA, vector< Pixel>& SetB,  vector< Pixel>& TrainSet, vector<IntIndex>& valSet, int t)
{
    vector<IntIndex>::iterator it = valSet.begin();
    while(it!=valSet.end() && it->val<t)
        ++it;

    SetA.resize(it-valSet.begin());
    SetB.resize(TrainSet.size()-SetA.size());

    it = valSet.begin();
    for(unsigned int i=0; i<SetA.size(); ++i, ++it)
        SetA[i] = TrainSet[it->index];
    it = valSet.begin()+SetA.size();
    for(unsigned int i=0; i<SetB.size(); ++i, ++it)
        SetB[i] = TrainSet[it->index];
}
#endif


double CRTree::distMean( std::vector< Pixel>& SetA,  std::vector< Pixel>& SetB, std::vector< DepthImg> ImgSet)
{
    float distance = DISTANCE;

    double errorA = 0;
    int sizeA = 0;
    for (int j=0;j<num_joint;j++)
    {
        vector<Pixel> Q;
        for (int i=0;i<SetA.size();i++)
        {
            double temp = pow(ImgSet[SetA[i].img_index].joint_location[j].x-SetA[i].point.x,2)
				+pow(ImgSet[SetA[i].img_index].joint_location[j].y-SetA[i].point.y,2)
				+pow(ImgSet[SetA[i].img_index].joint_location[j].z-SetA[i].point.z,2);
            if(temp<distance)
				Q.push_back(SetA[i]);
        }
		
        sizeA = sizeA + Q.size();
        CvPoint3D32f mean;
        mean.x=0;mean.y=0;mean.z=0;
        for (int i=0;i<Q.size();i++)
        {
            mean.x += ImgSet[Q[i].img_index].joint_location[j].x-Q[i].point.x;
			mean.y += ImgSet[Q[i].img_index].joint_location[j].y-Q[i].point.y;
			mean.z += ImgSet[Q[i].img_index].joint_location[j].z-Q[i].point.z;
        }
        if(Q.size()>0)
        {
			mean.x=mean.x/Q.size();
			mean.y=mean.y/Q.size();
			mean.z=mean.z/Q.size();
		}
		for (int i=0;i<Q.size();i++)
			errorA += pow((ImgSet[Q[i].img_index].joint_location[j].x-Q[i].point.x)-mean.x,2)
                +pow((ImgSet[Q[i].img_index].joint_location[j].y-Q[i].point.y)-mean.y,2)
                +pow((ImgSet[Q[i].img_index].joint_location[j].z-Q[i].point.z)-mean.z,2);
    }

    double errorB = 0;
	int sizeB = 0;
	for (int j=0;j<num_joint;j++)
	{
	    vector<Pixel> Q;
	    for (int i=0;i<SetB.size();i++)
		{
            double temp = pow(ImgSet[SetB[i].img_index].joint_location[j].x-SetB[i].point.x,2)
				+pow(ImgSet[SetB[i].img_index].joint_location[j].y-SetB[i].point.y,2)
				+pow(ImgSet[SetB[i].img_index].joint_location[j].z-SetB[i].point.z,2);
			if(temp<distance)
				Q.push_back(SetB[i]);
		}
		sizeB = sizeB + Q.size();
		CvPoint3D32f mean;
		mean.x=0;mean.y=0;mean.z=0;
		for (int i=0;i<Q.size();i++)
		{
			mean.x += ImgSet[Q[i].img_index].joint_location[j].x-Q[i].point.x;
			mean.y += ImgSet[Q[i].img_index].joint_location[j].y-Q[i].point.y;
			mean.z += ImgSet[Q[i].img_index].joint_location[j].z-Q[i].point.z;
		}
		if(Q.size()>0)
		{
			mean.x=mean.x/Q.size();
			mean.y=mean.y/Q.size();
			mean.z=mean.z/Q.size();
		}
		for (int i=0;i<Q.size();i++)
			errorB += pow((ImgSet[Q[i].img_index].joint_location[j].x-Q[i].point.x)-mean.x,2)
                +pow((ImgSet[Q[i].img_index].joint_location[j].y-Q[i].point.y)-mean.y,2)
                +pow((ImgSet[Q[i].img_index].joint_location[j].z-Q[i].point.z)-mean.z,2);
	}
	
    return (SetA.size()*errorA+SetB.size()*errorB)/(SetA.size()+SetB.size());
}

AbsoluteLocation CRTree::regression(Mat Img,Mat fore)
{
    double r = 0.5;
	AbsoluteLocation Z;
	std::vector<CvPoint3D32f> lc[JOINT_NUM];
	std::vector<double> wg[JOINT_NUM];
	for (int i=0;i<Img.rows;i=i+PIXEL_SHIFT)
	{
	    for (int j=0;j<Img.cols;j=j+PIXEL_SHIFT)
	    {
	        if (fore.at<uchar>(i,j)!=0)
	        {
	            CvPoint3D32f point;
	            point = depthPoint2cameraPoint(i,j,Img);
	            int* pnode = &treetable[0];
	            int node = 0;
	            while(pnode[0]==-1)
	            {
	                double val = depthCompareFeature(i,j,Img,fore,pnode[1],pnode[2],pnode[3],pnode[4]);
	                bool test = val >= pnode[5];
	                int incr = node+1+test;
	                node += incr;
	                pnode += incr*6;
	            }

	            assert(pnode[0] > 0);
	            for (int l=0;l<num_joint;l++)
	            {
	                for (int k=0;k < leaf[pnode[0] - 1].offset[l].size();k++)
	                {
	                    CvPoint3D32f p = leaf[pnode[0] - 1].offset[l][k];
	                    if (p.x*p.x+p.y*p.y+p.z*p.z<=4/*r*r*/)
						{
							CvPoint3D32f abs_p;
							abs_p.x = p.x+point.x;abs_p.y = p.y+point.y;abs_p.z = p.z+point.z;
							lc[l].push_back(abs_p);
							wg[l].push_back(leaf[pnode[0] - 1].weight[l][k]*point.z*point.z);
						}
	                }
	            }
	        }
	    }
	}
	for (int i=0;i<num_joint;i++)
	{
		Z.location.push_back(lc[i]);
		Z.weight.push_back(wg[i]);
	}

	return Z;
}

bool CRTree::optimizeTest(vector<Pixel>& SetA, vector<Pixel>& SetB,  vector<Pixel>& TrainSet,  std::vector<DepthImg> ImgSet, int* test, double* pbestDist, unsigned int iter)
{
    bool found = false;
    vector< Pixel> tmpA(TrainSet.size());
    vector< Pixel> tmpB(TrainSet.size());
    vector<IntIndex> valSet(TrainSet.size());
    double bestDist = -DBL_MAX;
	double tmpTest[5];
	openmp = 0;

#pragma omp parallel for if(!openmp) shared(bestDist, test, SetA, SetB) private(tmpA,tmpB,valSet,tmpTest)
	for(int i =0; i<iter; ++i)
	{
	    generateTest(&tmpTest[0], ImgSet[0].img.rows, ImgSet[0].img.cols);
	    evaluateTest(valSet, &tmpTest[0],TrainSet,ImgSet);
	    int vmin = INT_MAX;
		int vmax = INT_MIN;
		assert(valSet.size() > 0);
#ifndef USE_SORT
#pragma omp parallel for if(openmp) reduction(min:vmin) reduction(max:vmax) shared(valSet)
        for(int j=0;j<TrainSet.size();++j)
            {
                if (vmin > valSet[j].val) vmin = valSet[j].val;
                if (vmax < valSet[j].val) vmax = valSet[j].val;
            }
#else
        if(valSet.size()>0)
            {
                if(vmin>valSet.front().val)  vmin = valSet.front().val;
                if(vmax<valSet.back().val )  vmax = valSet.back().val;
            }
#endif

        int d = vmax-vmin;
        int upper_j = 0;
        if(d>0)
            upper_j = 50;
#pragma omp parallel for if(openmp) private(tmpA, tmpB) shared(d, vmin, TrainSet, valSet, ImgSet, found, bestDist, test, pbestDist, SetA, SetB)
        for(unsigned int j=0; j<upper_j; ++j)
        {
            double tmpDist;
            if (openmp) {
                tmpA.resize(TrainSet.size());
                tmpB.resize(TrainSet.size());
            }
            tmpA.clear();
            tmpB.clear();
            int tr = (cvRandInt( cvRNG ) % (d)) + vmin;
            split(tmpA, tmpB, TrainSet, valSet, tr);

            if( tmpA.size()+tmpA.size()>0 && tmpB.size()+tmpB.size()>0 )
            {
                tmpDist = measureSet(tmpA, tmpB, ImgSet);
                #pragma omp critical
                {
                    if(tmpDist>bestDist) {
                        found = true;
                        bestDist = tmpDist;
                        for(int t=0; t<4;++t) test[t] = tmpTest[t];
                        test[4] = tr;
                        *pbestDist = bestDist;
                        SetA = tmpA;
                        SetB = tmpB;
                    }
                }
            }
        }
    }
	return found;
}

bool CRTree::parallel_optimizeTest(vector<Pixel>& SetA, vector<Pixel>& SetB,  vector<Pixel>& TrainSet,  std::vector<DepthImg> ImgSet, int* test, unsigned int iter)
{
    bool found;
    vector< Pixel> tmpA(TrainSet.size());
	vector< Pixel> tmpB(TrainSet.size());
	vector<IntIndex> valSet(TrainSet.size());
	double bestDist = -DBL_MAX;
	double tmpTest[5];
	int new_iter;
	double t_start, t_end;
	if (debug_output)
        printf("rank=%d i need to offload optimize tasks TrainSet size=%d to compute nodes\n", mpi_rank, TrainSet.size());
    t_start = MPI_Wtime();
    int train_set_size = TrainSet.size();
    int *train_set_index;
    int cmd[2];
    new_iter = iter / mpi_size + 1;
    cmd[0] = train_set_size;
	cmd[1] = new_iter;
	MPI_Bcast(cmd, 2, MPI_INT, 0, MPI_COMM_WORLD);
	train_set_index = (int*)malloc(sizeof(int)*train_set_size);
	for (int i=0; i<train_set_size; i++)
        train_set_index[i] = trainPixels[TrainSet[i].pixel_index].pixel_index;
	//cout<<"rank "<<mpi_rank<<" size is "<<train_set_size<<endl;
    MPI_Bcast(train_set_index, train_set_size, MPI_INT, 0, MPI_COMM_WORLD);
    free(train_set_index);

    openmp = 0;
#pragma omp parallel for if (!openmp) shared(bestDist, test, SetA, SetB) private(tmpA,tmpB,valSet,tmpTest)
    for(int i =0; i<new_iter; ++i)
    {
        generateTest(&tmpTest[0], ImgSet[0].img.rows, ImgSet[0].img.cols);
        evaluateTest(valSet, &tmpTest[0],TrainSet,ImgSet);
        int vmin = INT_MAX;
		int vmax = INT_MIN;
#ifndef USE_SORT
#pragma omp parallel for if(openmp) reduction(min:vmin) reduction(max:vmax) shared(valSet)
        for(int j=0;j<TrainSet.size();++j)
		{
			if (vmin > valSet[j].val) vmin = valSet[j].val;
			if (vmax < valSet[j].val) vmax = valSet[j].val;
		}
#else
        if(valSet.size()>0)
		{
			if(vmin>valSet.front().val)  vmin = valSet.front().val;
			if(vmax<valSet.back().val )  vmax = valSet.back().val;
		}
#endif
        int d = vmax-vmin;
		int upper_j = 0;
		if(d>0)
            upper_j = 50;
#pragma omp parallel for if(openmp) private(tmpA, tmpB) shared(d, vmin, TrainSet, valSet, ImgSet, found, bestDist, test, SetA, SetB)
        for(unsigned int j=0; j<upper_j; ++j)
        {
            double tmpDist;
            if (openmp)
            {
				tmpA.resize(TrainSet.size());
				tmpB.resize(TrainSet.size());
			}
			tmpA.clear();
			tmpB.clear();
			int tr = (cvRandInt( cvRNG ) % (d)) + vmin;
			split(tmpA, tmpB, TrainSet, valSet, tr);
			if( tmpA.size()+tmpA.size()>0 && tmpB.size()+tmpB.size()>0 )
			{
			    tmpDist = measureSet(tmpA, tmpB, ImgSet);
			    #pragma omp critical
			    {
			        if(tmpDist>bestDist)
			        {
			            found = true;
						bestDist = tmpDist;
						for(int t=0; t<4;++t) test[t] = tmpTest[t];
						test[4] = tr;
						SetA = tmpA;
						SetB = tmpB;
			        }
			    }
			}
        }
    }

    int infound;
	int outfound;
	if (found)
		infound = 1;
	else
		infound = 0;
    MPI_Allreduce(&infound, &outfound, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
    if (outfound > 0)
    {
        found = true;
        struct doubleint { double val; int loc; } cinbuf, coutbuf;
        cinbuf.val = bestDist;
        cinbuf.loc = mpi_rank;
        MPI_Allreduce(&cinbuf, &coutbuf, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
        if (coutbuf.loc != 0)
        {
			if(debug_output)
				printf("rank=%d recv SetA SetB test to local_rank=%d\n", mpi_rank, coutbuf.loc);
            int train_set_size, i;
			int *train_set_index;
			MPI_Recv(&train_set_size, 1, MPI_INT, coutbuf.loc, 297, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			assert(train_set_size > 0);
			train_set_index = (int*)malloc(sizeof(int)*train_set_size);
			MPI_Recv(train_set_index, train_set_size, MPI_INT, coutbuf.loc, 298, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			SetA.clear();
			SetA.resize(train_set_size);
			for (i=0; i<train_set_size; i++)
            {
                assert(train_set_index[i] > -1);
				assert(train_set_index[i] < trainPixels.size());
                SetA[i] = trainPixels[train_set_index[i]];
            }
            free(train_set_index);

            MPI_Recv(&train_set_size, 1, MPI_INT, coutbuf.loc, 299, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            assert(train_set_size > 0);
            train_set_index = (int*)malloc(sizeof(int)*train_set_size);
			MPI_Recv(train_set_index, train_set_size, MPI_INT, coutbuf.loc, 300, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			SetB.clear();
			SetB.resize(train_set_size);
            for (i=0; i<train_set_size; i++)
            {
				assert(train_set_index[i] > -1);
				assert(train_set_index[i] < trainPixels.size());
                SetB[i] = trainPixels[train_set_index[i]];
            }
			free(train_set_index);

			MPI_Recv(test, 5, MPI_INT, coutbuf.loc, 301, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        else
			if(debug_output)
				printf("The best data is in root\n");
    }
    t_end = MPI_Wtime();
	if(debug_output)
		printf("rank=0, Time=%f, Task Size=%d, SetA Size=%d, SetB Size=%d\n", t_end-t_start, TrainSet.size(), SetA.size(), SetB.size());
    return found;
}

void CRTree::parallel_grow( vector<Pixel>& TrainSet,  std::vector<DepthImg> ImgSet,int node, unsigned int depth, int samples)
{
    int i;
    class SlaveTask task;
    struct Pixel pixel;

    if (debug_output)
        cout<<"rank="<<mpi_rank<<" parallel_grow node="<<node<<" depth="<<depth<<" TrainSet Size="<<TrainSet.size()<<endl;

    if(depth<max_depth && TrainSet.size()>0)
    {
        vector<Pixel> SetA;
        vector<Pixel> SetB;
        int test[6];

        if( parallel_optimizeTest(SetA, SetB, TrainSet, ImgSet,test, samples) )
        {
            int* ptT = &treetable[node*6];
            ptT[0] = -1; ++ptT;
            for(int t=0; t<5; ++t)
                ptT[t] = test[t];

            int task_volume = 0;
            if ((max_depth - depth + 1) > 0)
				task_volume = SetA.size()*(max_depth - depth + 1);
			else
				task_volume = SetA.size();
            if (SetA.size() > min_samples)
            {
                if ((SetA.size() < MAX_TASK_SIZE)&& (task_volume < MAX_TASK_VOLUME))
                {
                    task.train_set_size = SetA.size();
                    task.train_set_index = (int*)malloc(sizeof(int)*task.train_set_size);
                    for (i=0; i<task.train_set_size; i++)
                        task.train_set_index[i] = trainPixels[SetA[i].pixel_index].pixel_index;
                    task.node = 2*node+1;
                    task.depth = depth+1;
                    task.volume = task_volume;
                    task_list.push_back(task);
                    if (debug_output)
                        printf("Add task to task_list, train_set_size=%d, node=%d, depth=%d.\n", task.train_set_size, task.node, task.depth);
                }else
                    parallel_grow(SetA, ImgSet, 2*node+1, depth+1, samples);
            }else
                makeLeaf(SetA, ImgSet,2*node+1);

           if ((max_depth - depth + 1) > 0)
				task_volume = SetB.size()*(max_depth - depth + 1);
           else
				task_volume = SetB.size();
           if(SetB.size() > min_samples)
           {
               if ((SetB.size() < MAX_TASK_SIZE)&& (task_volume < MAX_TASK_VOLUME))
               {
                    task.train_set_size = SetB.size();
                    task.train_set_index = (int*)malloc(sizeof(int)*task.train_set_size);
                    for (i=0; i<task.train_set_size; i++)
                        task.train_set_index[i] = trainPixels[SetB[i].pixel_index].pixel_index;
                    task.node = 2*node+2;
                    task.depth = depth+1;
                    task.volume = task_volume;
                    task_list.push_back(task);
                    if (debug_output)
					printf("Add task to task_list, train_set_size=%d, node=%d, depth=%d.\n", task.train_set_size, task.node, task.depth);
               }else
                    parallel_grow(SetB, ImgSet,2*node+2, depth+1, samples);
           }else
                makeLeaf(SetB, ImgSet,2*node+2);
        }else
             makeLeaf(TrainSet, ImgSet,node);
    }else
         makeLeaf(TrainSet, ImgSet, node);
}

void CRTree::grow( vector<Pixel>& TrainSet,  std::vector<DepthImg> ImgSet,int node, unsigned int depth, int samples)
{
    assert(TrainSet.size() < MAX_TASK_SIZE);
    if (debug_output)
        cout<<"rank="<<mpi_rank<<" grow node="<<node<<" depth="<<depth<<" TrainSet Size="<<TrainSet.size()<<endl;
    if(depth<max_depth && TrainSet.size()>0)
    {
        vector<Pixel> SetA;
		vector<Pixel> SetB;
		int test[6];
		double bestDict;
		if( optimizeTest(SetA, SetB, TrainSet, ImgSet, test, &bestDict, samples) )
		{
            int* ptT = &treetable[node*6];
            ptT[0] = -1; ++ptT;
            for(int t=0; t<5; ++t)
                ptT[t] = test[t];
            if(SetA.size()>min_samples)
            {
                int cmd[2], node_depth[2];
                int slave,i,train_set_size;
                int* train_set_index;
				MPI_Status status;
                if (SetA.size()*(max_depth - depth + 1)>100)
                {
                    cmd[0] = 4;
                    cmd[1] = mpi_rank;
                    MPI_Send(cmd, 2, MPI_INT, 0, 3001, MPI_COMM_WORLD);
					MPI_Recv(cmd, 2, MPI_INT, 0, 291, MPI_COMM_WORLD, &status);
					assert(cmd[0] == 5);
					slave = cmd[1];

					if (slave > 0)
					{
					    train_set_size = SetA.size();
					    MPI_Send(&train_set_size, 1, MPI_INT, slave, 288, MPI_COMM_WORLD);
                        train_set_index = (int*)malloc(sizeof(int)*train_set_size);
                        for (i=0; i<train_set_size; i++)
                            train_set_index[i] = trainPixels[SetA[i].pixel_index].pixel_index;
                        MPI_Send(train_set_index, train_set_size, MPI_INT, slave, 289, MPI_COMM_WORLD);
                        node_depth[0] = 2*node+1;
                        node_depth[1] = depth+1;
                        MPI_Send(node_depth, 2, MPI_INT, slave, 290, MPI_COMM_WORLD);
						free(train_set_index);
					}
					else grow(SetA, ImgSet, 2*node+1, depth+1, samples);
                }
                else grow(SetA, ImgSet, 2*node+1, depth+1, samples);
            }
            else makeLeaf(SetA, ImgSet,2*node+1);

            if(SetB.size()>min_samples)
                grow(SetB, ImgSet,2*node+2, depth+1, samples);
            else
                makeLeaf(SetB, ImgSet,2*node+2);
		}
		else
            makeLeaf(TrainSet, ImgSet,node);
    }
    else
        makeLeaf(TrainSet, ImgSet, node);
}

void CRTree::growTree( std::vector<Pixel>& TrainSet,  std::vector<DepthImg> ImgSet, int samples)
{
    int slave;
    int*train_set_index;
    int i, j, k, train_set_size;
    int cmd[2], scmd[2], node_depth[2];
    MPI_Status status;
    struct Pixel pixel;
	class SlaveTask task;

	#pragma omp parallel
    {
        #pragma omp single
            {
                thread_size = omp_get_num_threads();
            }
        thread_rank = omp_get_thread_num();
    }

    parallel_grow(TrainSet, ImgSet, 0, 0, samples);
    if (mpi_size == 1) return;
    cmd[0]=cmd[1]=0;
    MPI_Bcast(cmd, 2, MPI_INT, 0, MPI_COMM_WORLD);

    task_list.sort();
    if (debug_tasklist)
    {
        list<class SlaveTask>::iterator it = task_list.begin();
        while(it!=task_list.end())
        {
            it->print();
            ++it;
		}
	}

	for (i=1; i<mpi_size; i++)
	{
	    if (!task_list.empty())
	    {
            task = task_list.back();
			assert(task.train_set_size > 0);
			assert(task.train_set_size < MAX_TASK_SIZE);
			assert(task.train_set_index);
			assert(task.node >=0 );
			assert(task.depth >=0 );
			task_list.pop_back();
			cmd[0] = 1;
			cmd[1] = 0;
			assert(rank_status[i] == 1);
			rank_status[i] = 2;
			idle_slave_size --;
			MPI_Send(cmd, 2, MPI_INT, i, 188, MPI_COMM_WORLD);
			MPI_Send(&task.train_set_size, 1, MPI_INT, i, 288, MPI_COMM_WORLD);
			MPI_Send(task.train_set_index, task.train_set_size, MPI_INT, i, 289, MPI_COMM_WORLD);
			node_depth[0] = task.node;
			node_depth[1] = task.depth;
			MPI_Send(node_depth, 2, MPI_INT, i, 290, MPI_COMM_WORLD);
			free(task.train_set_index);
		}
	}

	while (1)
	{
	    MPI_Recv(cmd, 2, MPI_INT, MPI_ANY_SOURCE, 3001, MPI_COMM_WORLD, &status);
	    if (cmd[0] == 3)
	    {
			if(debug_output)
				printf("rank=0 recv IAMIDLE command from rank=%d\n", cmd[1]);
	        assert(cmd[1] == status.MPI_SOURCE);
	        assert(rank_status[cmd[1]] == 2);
	        rank_status[cmd[1]] = 1;
	        idle_slave_size ++;
	        if (!task_list.empty())
	        {
	            task = task_list.back();
	            assert(task.train_set_size > 0);
				assert(task.train_set_size < MAX_TASK_SIZE);
				assert(task.train_set_index);
				assert(task.node >=0 );
				assert(task.depth >=0 );
				task_list.pop_back();
				scmd[0] = 1;
				scmd[1] = 0;

				assert(rank_status[cmd[1]] == 1);
				rank_status[cmd[1]] = 2;
				idle_slave_size --;
				MPI_Send(scmd, 2, MPI_INT, cmd[1], 188, MPI_COMM_WORLD);
				MPI_Send(&task.train_set_size, 1, MPI_INT, cmd[1], 288, MPI_COMM_WORLD);
				MPI_Send(task.train_set_index, task.train_set_size, MPI_INT, cmd[1], 289, MPI_COMM_WORLD);
				node_depth[0] = task.node;
				node_depth[1] = task.depth;
				MPI_Send(node_depth, 2, MPI_INT, cmd[1], 290, MPI_COMM_WORLD);
				free(task.train_set_index);
	        }
	        if (idle_slave_size == mpi_size - 1)
                break;
	    }
	    else
	    {
	        assert(cmd[1] == status.MPI_SOURCE);
			assert(rank_status[cmd[1]] == 2);
			slave = getIdleSlave();
			if (slave)
			{
                scmd[0] = 1;
				scmd[1] = cmd[1];
				MPI_Send(scmd, 2, MPI_INT, slave, 188, MPI_COMM_WORLD);
			}
			scmd[0] = 5;
			scmd[1] = slave;
			MPI_Send(scmd, 2, MPI_INT, cmd[1], 291, MPI_COMM_WORLD);
	    }
	}

	for (i=1; i<mpi_size; i++)
	{
	    cmd[0] = 2;
		cmd[1] = i;
		MPI_Send(cmd, 2, MPI_INT, i, 188, MPI_COMM_WORLD);
		cmd[0] = num_leaf;
		MPI_Send(cmd, 1, MPI_INT, i, 4000, MPI_COMM_WORLD);
		int ptT[7];
		MPI_Recv(ptT, 7, MPI_INT, i, 4001, MPI_COMM_WORLD, &status);

		while (ptT[0] != -1)
		{
			int node_idx = ptT[0];
			treetable[node_idx + 0] = ptT[1];
			treetable[node_idx + 1] = ptT[2];
			treetable[node_idx + 2] = ptT[3];
			treetable[node_idx + 3] = ptT[4];
			treetable[node_idx + 4] = ptT[5];
			treetable[node_idx + 5] = ptT[6];
			MPI_Recv(ptT, 7, MPI_INT, i, 4001, MPI_COMM_WORLD, &status);
		}
		
		int bufsize, position=0;
        char * buffer;
        struct LeafNodeLocation_struct mystruct;
		LeafNode* ptLN;

		MPI_Pack_size(1, LeafNodeLocation_type, MPI_COMM_WORLD, &bufsize);
        buffer = (char *) malloc(bufsize*JOINT_NUM);
        ptLN = &leaf[num_leaf];

        ptLN->offset.resize(JOINT_NUM);
        ptLN->weight.resize(JOINT_NUM);

        std::vector<CvPoint3D32f> offset;
        std::vector<float> weight;
		CvPoint3D32f point;

        MPI_Recv(buffer, bufsize*JOINT_NUM, MPI_CHAR, i, 4002, MPI_COMM_WORLD, &status);
        MPI_Unpack(buffer, bufsize, &position, &mystruct, 1, LeafNodeLocation_type, MPI_COMM_WORLD);
        while (mystruct.size > 0)
        {
            for (j=0; j<JOINT_NUM; j++)
            {
                position = 0;
				MPI_Unpack(buffer + j*bufsize, bufsize, &position, &mystruct, 1, LeafNodeLocation_type, MPI_COMM_WORLD);
				if (mystruct.size == -1)
				{
					assert(j == 0);
					goto endofloop;
				}
				assert((mystruct.size == 1) || (mystruct.size == 2));
                ptLN->offset[j].resize(mystruct.size);
                ptLN->weight[j].resize(mystruct.size);
				for (k=0; k<mystruct.size; k++)
				{
					ptLN->offset[j][k].x = mystruct.data[4*k + 0];
					ptLN->offset[j][k].y = mystruct.data[4*k + 1];
					ptLN->offset[j][k].z = mystruct.data[4*k + 2];
					ptLN->weight[j][k]   = mystruct.data[4*k + 3];
				}
			}
			num_leaf ++;
			ptLN = &leaf[num_leaf] ;
			ptLN->offset.resize(JOINT_NUM);
            ptLN->weight.resize(JOINT_NUM);
			MPI_Recv(buffer, bufsize*JOINT_NUM, MPI_CHAR, i, 4002, MPI_COMM_WORLD, &status);
		}
endofloop:
    free(buffer);
	}
}

void CRTree::slave_work(std::vector<DepthImg> ImgSet, int samples)
{
    int cmd[2];
    int train_set_size = 0;
    int* train_set_index;
    int i,j,k;
	vector<Pixel> TrainSet;
	extern vector<Pixel> trainPixels;
	int node_depth[2];
	int scmd;

    #pragma omp parallel
    {
        #pragma omp single
        {
            thread_size = omp_get_num_threads();
        }
        thread_rank = omp_get_thread_num();
    }

    MPI_Bcast(cmd, 2, MPI_INT, 0, MPI_COMM_WORLD);
    while (cmd[0] != 0)
    {
        train_set_size = cmd[0];
        int new_iter = cmd[1];
        train_set_index = (int*)malloc(sizeof(int)*train_set_size);
		//cout<<"rank "<<mpi_rank<<" size is "<<train_set_size<<endl;
        MPI_Bcast(train_set_index, train_set_size, MPI_INT, 0, MPI_COMM_WORLD);
        TrainSet.clear();
        TrainSet.resize(train_set_size);
        for (i=0; i<train_set_size; i++)
        {
            assert(train_set_index[i] > -1);
			assert(train_set_index[i] < trainPixels.size());
            TrainSet[i] = trainPixels[train_set_index[i]];
        }
        free(train_set_index);

        vector<Pixel> SetA;
        vector<Pixel> SetB;
		bool found;
		int test[5];
		double bestDict;
		found = optimizeTest(SetA, SetB, TrainSet, ImgSet,test,&bestDict, new_iter);
		int infound;
		int outfound;
		if (found)
			infound = 1;
		else
			infound = 0;
        MPI_Allreduce(&infound, &outfound, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
        if (outfound > 0)
        {
            struct doubleint { double val; int loc; } cinbuf, coutbuf;
            if (found)
				cinbuf.val = bestDict;
			else
				cinbuf.val = -DBL_MAX;
            cinbuf.loc = mpi_rank;
            MPI_Allreduce(&cinbuf, &coutbuf, 1, MPI_DOUBLE_INT, MPI_MAXLOC, MPI_COMM_WORLD);
            if (mpi_rank == coutbuf.loc )
            {
                assert(found == true);
				assert(SetA.size() > 0);
				assert(SetB.size() > 0);
				int train_set_size = SetA.size();
				int *train_set_index;
				MPI_Send(&train_set_size, 1, MPI_INT, 0, 297, MPI_COMM_WORLD);
				train_set_index = (int*)malloc(sizeof(int)*train_set_size);
				for (i=0; i<train_set_size; i++)
                    train_set_index[i] = trainPixels[SetA[i].pixel_index].pixel_index;
                MPI_Send(train_set_index, train_set_size, MPI_INT, 0, 298, MPI_COMM_WORLD);
                free(train_set_index);

                train_set_size = SetB.size();
                MPI_Send(&train_set_size, 1, MPI_INT, 0, 299, MPI_COMM_WORLD);
                train_set_index = (int*)malloc(sizeof(int)*train_set_size);
                for (i=0; i<train_set_size; i++)
                    train_set_index[i] = trainPixels[SetB[i].pixel_index].pixel_index;
                MPI_Send(train_set_index, train_set_size, MPI_INT, 0, 300, MPI_COMM_WORLD);
				free(train_set_index);

				MPI_Send(test, 5, MPI_INT, 0, 301, MPI_COMM_WORLD);
            }
        }
        MPI_Bcast(cmd, 2, MPI_INT, 0, MPI_COMM_WORLD);
    }

    while (1)
    {
        MPI_Recv(cmd, 2, MPI_INT, 0, 188, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        if (cmd[0] == 1)
        {
            double t_start, t_end;
            MPI_Recv(&train_set_size, 1, MPI_INT, cmd[1], 288, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            assert(train_set_size > 0);
            t_start = MPI_Wtime();
            train_set_index = (int*)malloc(sizeof(int)*train_set_size);
            MPI_Recv(train_set_index, train_set_size, MPI_INT, cmd[1], 289, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            TrainSet.clear();
            TrainSet.resize(train_set_size);
            for (i=0; i<train_set_size; i++)
            {
                assert(train_set_index[i] > -1);
				assert(train_set_index[i] < trainPixels.size());
                TrainSet[i] = trainPixels[train_set_index[i]];
            }
            free(train_set_index);
            MPI_Recv(node_depth, 2, MPI_INT, cmd[1], 290, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            assert(TrainSet.size() == train_set_size);
            grow(TrainSet, ImgSet, node_depth[0], node_depth[1], samples);
            t_end = MPI_Wtime();
			if(debug_output)
				printf("rank=%d compute task volume=%d source_rank=%d size=%d node=%d depth=%d, Time is %f\n",
					mpi_rank,
					train_set_size*(max_depth - node_depth[1] +1),
					cmd[1],
					train_set_size,
					node_depth[0],
					node_depth[1],
					t_end - t_start);
            cmd[0] = 3;
			cmd[1] = mpi_rank;
			MPI_Send(cmd, 2, MPI_INT, 0, 3001, MPI_COMM_WORLD);
        }
        else if (cmd[0] == 2)
        {
            int root_num_leaf = 0;
            MPI_Recv(&root_num_leaf, 1, MPI_INT, 0, 4000, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            int ptT[7];
			int node_count = 0;
			int leaf_count = 0;
			for (i=0; i<num_nodes; i++)
			{
			    if ((treetable[i*6] > 0) || (treetable[i*6] == -1))
			    {
			        if (treetable[i*6] > 0)
                        treetable[i*6] += root_num_leaf;
                    ptT[0] = i*6;
					ptT[1] = treetable[i*6 + 0];
					ptT[2] = treetable[i*6 + 1];
					ptT[3] = treetable[i*6 + 2];
					ptT[4] = treetable[i*6 + 3];
					ptT[5] = treetable[i*6 + 4];
					ptT[6] = treetable[i*6 + 5];
					MPI_Send(ptT, 7, MPI_INT, 0, 4001, MPI_COMM_WORLD);
					node_count ++;
			    }
			}
			ptT[0] = -1;
			MPI_Send(ptT, 7, MPI_INT, 0, 4001, MPI_COMM_WORLD);

			int bufsize, position=0;
			char * buffer;
			struct LeafNodeLocation_struct mystruct;
			MPI_Pack_size(1, LeafNodeLocation_type, MPI_COMM_WORLD, &bufsize);
			buffer = (char *) malloc(bufsize*JOINT_NUM);
			LeafNode* ptLN ;
			for (i=0; i<num_leaf; i++)
			{
			    ptLN = &leaf[i];
			    assert(JOINT_NUM == ptLN->offset.size());
			    for (j=0; j<JOINT_NUM; j++)
			    {
			        mystruct.size = ptLN->offset[j].size();
			        assert((mystruct.size == 1)||(mystruct.size == 2));
			        for(k=0; k<mystruct.size; ++k)
			        {
			            mystruct.data[ 4*k+0 ] = ptLN->offset[j][k].x;
                        mystruct.data[ 4*k+1 ] = ptLN->offset[j][k].y;
                        mystruct.data[ 4*k+2 ] = ptLN->offset[j][k].z;
                        mystruct.data[ 4*k+3 ] = ptLN->weight[j][k];
			        }
			        position = 0;
			        MPI_Pack(&mystruct, 1, LeafNodeLocation_type, buffer + j*bufsize, bufsize, &position, MPI_COMM_WORLD);
			    }
			    MPI_Send(buffer, bufsize*JOINT_NUM, MPI_CHAR, 0, 4002, MPI_COMM_WORLD);
			}
			mystruct.size = -1;
			position = 0;
			if (debug_output)
				printf("rank=%d finish, send node_count=%d, leaf_count=%d\n", mpi_rank, node_count, num_leaf);
			MPI_Pack(&mystruct, 1, LeafNodeLocation_type, buffer, bufsize, &position, MPI_COMM_WORLD);
			MPI_Send(buffer, bufsize*JOINT_NUM, MPI_CHAR, 0, 4002, MPI_COMM_WORLD);
			free(buffer);
			break;
        }
        else assert(0);
    }
}


