#!/bin/bash
#OPT="-fomit-frame-pointer -O3 -funroll-loops -W -Wall -DARCH_INTEL64 -mavx"
OPT="-fomit-frame-pointer -O3 -funroll-loops -DARCH_INTEL64 -fopenmp -g"
#OPT="-O3 -fopenmp"
rm -rf *.o  ForestPose.exe
module load gcc/4.9.2 
/usr/local/mpi3/bin/mpic++ $OPT  -c meanshift.cpp `pkg-config --cflags --libs gtk+-2.0` -I/WORK/app/opencv/3.0.0/include
/usr/local/mpi3/bin/mpic++ $OPT  -c CRTree.cpp `pkg-config --cflags --libs gtk+-2.0` -I/WORK/app/opencv/3.0.0/include
/usr/local/mpi3/bin/mpic++ $OPT  -c ForestPose.cpp `pkg-config --cflags --libs gtk+-2.0` -I/WORK/app/opencv/3.0.0/include
/usr/local/mpi3/bin/mpic++ $OPT  -c CRTree_mpi.cpp 
/usr/local/mpi3/bin/mpic++ $OPT  *.o -o ForestPose.exe `pkg-config --cflags --libs gtk+-2.0`  -L/WORK/app/opencv/3.0.0/lib -lopencv_highgui -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -Wl,-rpath-link /WORK/app/libgphoto2/2.5.8/lib

