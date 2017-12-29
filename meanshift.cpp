#include "mpi.h"
#include "meanshift.h"
#include "CRForest.h"

float Gaussian(float dist, float bandwidth)
{
    return exp(-2*pow(dist/bandwidth,2));
}

float Distance(std::vector<float> a, std::vector<float> b)
{
    assert(a.size()==b.size());
    float dist = 0;
    for(int i=0; i<a.size(); ++i)
		dist += (a[i]-b[i])*(a[i]-b[i]);
    return sqrt(dist);
}

void shift(std::vector<float> oldMean, std::vector<std::vector<float> > data, std::vector<float> weights, float bandwidth,std::vector<float>& newMean)
{
    int dim = oldMean.size();
    float scale = 0;
    int move = 0;

    for(std::vector<std::vector<float> >::iterator it=data.begin(); it!=data.end(); ++it) {
        std::vector<float> pt = *it;
        float weight = weights[move++]*Gaussian(Distance(oldMean, pt), bandwidth);
        for(int i=0; i<dim; ++i)
			newMean[i] += pt[i]*weight;
        scale += weight;
    }
    for(int i=0; i<dim; ++i) {
        if(scale != 0)
			newMean[i] /= scale;
		else
			newMean[i] = oldMean[i];
    }
}

void cluster(std::vector<std::vector<float> > data, std::vector<float> weights, float bandwidth, std::vector<std::vector<float> >& clusters, std::vector<int>& group)
{
    for(int i=0; i<data.size(); ++i) {
        std::vector<float> oldMean = data[i];
        std::vector<float> newMean(oldMean.size(), 0);
		int num = 100;
        while(num != 0) {
            newMean.assign(oldMean.size(), 0);
			shift(oldMean, data, weights,bandwidth, newMean);
			if(Distance(oldMean, newMean) < 1e-3*bandwidth)
				break;
			oldMean = newMean;
			num--;
		}
		int j;
		for(j=0; j<clusters.size(); ++j)
			if(Distance(newMean, clusters[j]) < bandwidth/2)
				break;
        if(j==clusters.size())
			clusters.push_back(newMean);
        group[i] = j;
    }
}

void numCount(std::vector<int>& group,std::vector<int> &index_each_cluster,std::vector<int> &num_each_cluster,int numCluster)
{
    for (int i=0;i<numCluster;i++)
	{
		index_each_cluster.push_back(-1);
		num_each_cluster.push_back(0);
	}

	int move = 0;
	for (int i=0;i<group.size();i++)
	{
	    int tag = 0;
	    for (int j=0;j<=move;j++)
		{
			if (group[i]==index_each_cluster[j])
			{
				num_each_cluster[j]++;
				tag = 1;
				break;
			}
		}

		if (tag == 0)
		{
			index_each_cluster[move]=group[i];
			num_each_cluster[move]=1;
			move++;
		}
	}
}

void weightsCount(std::vector<int>& group,std::vector<float> weights,std::vector<int> &index_each_cluster,std::vector<float> &weights_each_cluster,int numCluster)
{
    for (int i=0;i<numCluster;i++)
	{
		index_each_cluster.push_back(-1);
		weights_each_cluster.push_back(0);
	}

	int move = 0;
	for (int i=0;i<group.size();i++)
	{
		int tag = 0;
		for (int j=0;j<=move;j++)
		{
			if (group[i]==index_each_cluster[j])
			{
				weights_each_cluster[j] = + weights[group[i]];
				tag = 1;
				break;
			}
		}
		if (tag == 0)
		{
			index_each_cluster[move]=group[i];
			weights_each_cluster[move]=weights[group[i]];
			move++;
		}
	}
}

void sorting(int size,std::vector<int> &index,std::vector<int> &value)
{
    int i,j,temp_val,temp_id;
    for(j=1;j<size;j++)
    {
        for(i=0;i<size-j;i++)
        {
            if(value[i]<value[i+1])
            {
                temp_val=value[i];
				value[i]=value[i+1];
				value[i+1]=temp_val;

				temp_id = index[i];
				index[i]=index[i+1];
				index[i+1]=temp_id;
            }
        }
    }
}

void sorting(int size,std::vector<int> &index,std::vector<float> &value)
{
    int i,j,temp_id;
    float temp_val;
    for(j=1;j<size;j++)
    {
        for(i=0;i<size-j;i++)
        {
            if(value[i]<value[i+1])
            {
                temp_val=value[i];
				value[i]=value[i+1];
				value[i+1]=temp_val;

				temp_id = index[i];
				index[i]=index[i+1];
				index[i+1]=temp_id;
            }
        }
    }
}

#ifdef KINECT1
CvPoint3D32f depthPoint2cameraPoint(int m,int n,Mat depth)
{
    CvPoint3D32f p;
    int s = 1000;
    float cx = 160, cy = 120;
    float fx = 285.6329, fy = 285.6306;

    p.z = 1.0*depth.at<ushort>(m,n)/(1.0*s);
    p.x = 1.0*(n-cx)*p.z/(1.0*fx);
    p.y = 1.0*(240-m-cy)*p.z/(1.0*fy);
    return p;
}
#endif

#ifdef KINECT2
CvPoint3D32f depthPoint2cameraPoint(int m,int n,Mat depth)
{
    CvPoint3D32f p;
    int s = 1000;
    float cx = 254.405, cy = 207.041;
    float fx = 366.948, fy = 366.948;

    p.z = 1.0*depth.at<ushort>(m,n)/(1.0*s);
    p.x = 1.0*(n-cx)*p.z/(1.0*fx);
    p.y = 1.0*(424-m-cy)*p.z/(1.0*fy);
    return p;
}
#endif
