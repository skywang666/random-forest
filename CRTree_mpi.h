#ifndef CRTREE_MPI
#define CRTREE_MPI

struct LeafNodeLocation_struct {
    int size;
    float data[8];
};

extern MPI_Datatype pixel_type;
extern MPI_Datatype intindex_type;
extern MPI_Datatype LeafNodeLocation_type;

int CRTREE_MPI_CREATE_PIXEL_TYPE();

#endif
