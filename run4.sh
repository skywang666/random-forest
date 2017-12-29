#!/bin/bash
source /WORK/app/osenv/ln1/set2.sh
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/WORK/app/opencv/3.0.0/lib
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/WORK/app/libgphoto2/2.5.8/lib
module load gcc/4.9.2
date
#FOREST_DEBUG_TASKLIST=1 FOREST_DEBUG_OUTPUT=1 FOREST_TRAIN=1 yhrun -N 2 -n 4 --ntasks-per-node=2 /HOME/nsccgz_xliao_1/skywang/skywangRF/ForestPose.exe
OMP_STACKSIZE=16384 FOREST_TRAIN=1 yhrun -N 2 -n 4 --ntasks-per-node=2 /HOME/nsccgz_xliao_1/skywang/skywangRF/ForestPose.exe
date
