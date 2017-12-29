#include "mpi.h"
#include <stdlib.h>
#include <fcntl.h>
#include <unistd.h>
#include <omp.h>
#include <iostream>
#include <fstream>
#include "CRForest.h"
#include "CRTree_mpi.h"
#ifndef OPENCV_HPP
#define OPENCV_HPP
#include "opencv2/opencv.hpp"
#endif

using namespace std;

#define DIR_COUNT 1

void train_tree();
void predict_skeleton();
Point cameraPoint2depthPoint(CvPoint3D32f point);

MPI_Datatype pixel_type;
MPI_Datatype intindex_type;
MPI_Datatype LeafNodeLocation_type;

vector<Pixel> trainPixels;
vector<DepthImg> trainDepthImgs;

int main(int argc, char* argv[])
{
    int size, rank;
	double t_start, t_end;

    MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	CRTREE_MPI_CREATE_PIXEL_TYPE();
	
	t_start = MPI_Wtime();
	char * env=NULL;
	env = getenv("FOREST_TRAIN");
	if (env)
		train_tree();
	t_end = MPI_Wtime();
	cout<<"Train time is "<<t_end-t_start<<endl;
	
    env = NULL;
	env = getenv("FOREST_PREDICT");
	if ((env)&&(rank == 0))
		predict_skeleton();

    MPI_Finalize();
	return 0;
}

void train_tree()
{
    int mpi_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

    int pic_start[DIR_COUNT] = {0};
	int pic_end[DIR_COUNT] =   {100};
	int count;
	int file_index=0;
	int pixel_index = 0;

	for (count=0; count<DIR_COUNT; count ++) {
        char depth_path[256];
		char foreground_path[256];
		char skeleton_path[256];
		char skeleton_abs_path[512];

		if (!mpi_rank) printf("Train Using KINECT1 !\n");
		sprintf(depth_path,"/HOME/nsccgz_xliao_1/WORKSPACE/data14000/%d/depth/", count+1);
		sprintf(foreground_path,"/HOME/nsccgz_xliao_1/WORKSPACE/data14000/%d/foreground/", count+1);
		sprintf(skeleton_path,"/HOME/nsccgz_xliao_1/WORKSPACE/data14000/%d/skeleton/", count+1);
		sprintf(skeleton_abs_path, "%s%s", skeleton_path, "skeleton.txt");
        int img_num = pic_end[count] - pic_start[count];
        ifstream infile;
        if (!infile)
		{
            cout<<"ERROR: No skeleton file have been read!"<<endl;
			cout<<skeleton_abs_path<<endl;
			return;
		}
        infile.open(skeleton_abs_path,ios::in);
        int index = pic_start[count];
        int pic_num;
        double x[JOINT_NUM],y[JOINT_NUM],z[JOINT_NUM];
		//cout<<"run here1 "<<mpi_rank<<" "<<" count "<<count<<endl;
        for (pic_num=0;pic_num<img_num;pic_num++)
        {
            infile>>index;
            for (int j=0;j<JOINT_NUM;j++)
			{
				infile>>x[j];infile>>y[j];infile>>z[j];
			}
			char c[8];
			memset(c, 0, 8);
			sprintf(c,"%d", index);
			char depth_abs_path[512];
			char foreground_abs_path[512];
			sprintf(depth_abs_path,"%s%s%s", depth_path ,c, "_do.png");
			sprintf(foreground_abs_path, "%s%s%s", foreground_path , c , "_fore.png");
			//cout<<"run here2 "<<mpi_rank<<" "<<" iterr "<<pic_num<<endl;
			DepthImg depthimg;
			depthimg.index = file_index;
			depthimg.img = imread(depth_abs_path,-1);
			if (!depthimg.img.data)
			{
				cout<<"ERROR: No depth image have been read!";
				return;
			}
			Mat fore_rgb = imread(foreground_abs_path,-1);;
			cvtColor(fore_rgb, depthimg.fore, CV_BGR2GRAY);
			//cout<<"run here3 "<<mpi_rank<<" "<<" iterr "<<pic_num<<endl;
			if (!depthimg.fore.data)
			{
				cout<<"ERROR: No foreground image have been read!";
				return;
			}
			CvPoint3D32f skeleton_point;
			for (int i=0;i<JOINT_NUM;i++)
			{
				skeleton_point.x = x[i];skeleton_point.y = y[i]; skeleton_point.z = z[i];
				depthimg.joint_location.push_back(skeleton_point);
			}
			trainDepthImgs.push_back(depthimg);
			//cout<<"run here4 "<<mpi_rank<<" "<<" iter "<<pic_num<<endl;
			Pixel pixel;
			for (int m=0;m<depthimg.img.rows;m=m+PIXEL_SHIFT)
			{
			    for (int n=0;n<depthimg.img.cols;n=n+PIXEL_SHIFT)
			    {
			        if (depthimg.fore.at<uchar>(m,n)!=0 && depthimg.img.at<ushort>(m,n)!=0)
			        {
                        pixel.pixel_index = pixel_index;
						pixel.img_index = depthimg.index;
						pixel.m = m;
						pixel.n = n;
						pixel.point = depthPoint2cameraPoint(m,n,depthimg.img);
						trainPixels.push_back(pixel);
						pixel_index ++ ;
			        }
			    }
			}
            file_index ++;
        }
        infile.close();
	}
    MPI_Barrier(MPI_COMM_WORLD);
    if (!mpi_rank)
		printf("picture count is %d, size of pixel_index is %d\n", file_index, trainPixels.size());

    CRForest crForest( 1 );
    time_t t = time(NULL);
	int seed = (int)t;
	CvRNG cvRNG(seed);
	crForest.trainForest(LEAF_PIXEL_COUNT, TREE_HIGHT, &cvRNG, trainPixels, trainDepthImgs, 2000);
	MPI_Barrier(MPI_COMM_WORLD);
	if (mpi_rank == 0)
		crForest.saveForest("/HOME/nsccgz_xliao_1/WORKSPACE/tree/skywangRF/", 0);
}

void predict_skeleton()
{
    string depth_path = "/HOME/nsccgz_xliao_1/WORKSPACE/skywang/data/2/depth/121_do.png";
	string foreground_path = "/HOME/nsccgz_xliao_1/WORKSPACE/skywang/data/2/foreground/121_fore.png";
	Mat depthimg = imread(depth_path,-1);
	Mat foregroundimg = imread(foreground_path,-1);

	CRForest crForest( 1 );
	crForest.loadForest("/home/sg/tree");
	Joint skeleton;
	skeleton = crForest.regression(depthimg,foregroundimg);

#ifdef KINECT1
	Mat back(240,320,CV_8UC3,Scalar(0,0,0));
#endif
#ifdef KINECT2
	Mat back(424,512,CV_8UC3,Scalar(0,0,0));
#endif

    for (int i=0;i<JOINT_NUM;i++)
    {
        Point drawpoint = cameraPoint2depthPoint(skeleton.location[i]);
        circle(back,drawpoint,3,Scalar(255,0,0),5);
    }

    namedWindow("fore");
	imshow("fore",foregroundimg);
	namedWindow("skeleton");
	imshow("skeleton",back);
	waitKey(0);
}

#ifdef KINECT2
Point cameraPoint2depthPoint(CvPoint3D32f point)
{
    Point p;
    int s = 1000;
    float cx = 254.405, cy = 207.041;
    float fx = 366.948, fy = 366.948;
    p.x = int(point.x*fx/point.z+cx);
    p.y = 424-int(point.y*fy/point.z+cy);
    return p;
}
#endif

#ifdef KINECT1
Point cameraPoint2depthPoint(CvPoint3D32f point)
{
    Point p;
    int s = 1000;
    float cx = 160, cy = 120;
    float fx = 285.6329, fy = 285.6306;
    p.x = int(point.x*fx/(point.z)+cx);
    p.y = 240-int(point.y*fy/(point.z)+cy);
    return p;
}
#endif
