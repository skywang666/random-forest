#include <stdlib.h>
#include <stdio.h>
#include "mpi.h"
#include "CRTree_mpi.h"

int CRTREE_MPI_CREATE_PIXEL_TYPE()
{
	MPI_Aint offsets[2], extent;
	MPI_Datatype oldtypes[2];
	int blockcounts[2];

	offsets[0] = 0;
	oldtypes[0] = MPI_INT;
	blockcounts[0] = 3;
	MPI_Type_extent(MPI_INT, &extent);
	offsets[1] = extent*blockcounts[0];
	oldtypes[1] = MPI_FLOAT;
	blockcounts[1] = 3;
	extern MPI_Datatype pixel_type;
	MPI_Type_struct(2, blockcounts, offsets, oldtypes, &pixel_type);
	MPI_Type_commit(&pixel_type);
	
    offsets[0] = 0;
    oldtypes[0] = MPI_INT;
    blockcounts[0] = 1;
	MPI_Type_extent(MPI_INT, &extent);
    offsets[1] = extent*blockcounts[0];
    oldtypes[1] = MPI_UNSIGNED;
    blockcounts[1] = 1;	
	extern MPI_Datatype intindex_type;
	MPI_Type_struct(2, blockcounts, offsets, oldtypes, &intindex_type);
	MPI_Type_commit(&intindex_type);

    offsets[0] = 0;
    oldtypes[0] = MPI_INT;
    blockcounts[0] = 1;
    MPI_Type_extent(MPI_INT, &extent);
    offsets[1] = extent*blockcounts[0];
    oldtypes[1] = MPI_FLOAT;
    blockcounts[1] = 8;
    extern MPI_Datatype LeafNodeLocation_type;
	MPI_Type_struct(2, blockcounts, offsets, oldtypes, &LeafNodeLocation_type);
	MPI_Type_commit(&LeafNodeLocation_type);

	return 0;
}
