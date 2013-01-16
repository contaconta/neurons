#include "string.h"
#include "global.h"
#include "ksp_graph.h"
#include "ksp_computer.h"


#define MAXCHARS 80   /* max length of string contained in each field */

/*  the gateway routine.  */
void mexFunction( int nlhs, mxArray *plhs[],
        int nrhs, const mxArray *prhs[] ) {
    
    /* check proper input and output */
    if(nrhs != 6)
        mexErrMsgTxt("6 input required.");
    else if(nlhs > 1)
        mexErrMsgTxt("Too many output arguments.");
    else if(!mxIsStruct(prhs[0]))
        mexErrMsgTxt("First Input must be a structure.");
    else if(!mxIsCell(prhs[1]))
        mexErrMsgTxt("Second Input must be a cell.");
    /* get input arguments */
    const mxArray* Cells       = prhs[0];
    const mxArray* CellsList   = prhs[1];
    double  temporal_windows_sz= mxGetScalar(prhs[2]);
    double  spatial_windows_sz = mxGetScalar(prhs[3]);
    double *imagesize          = mxGetPr(prhs[4]);
    double  distanceToBoundary = mxGetScalar(prhs[5]);
    /* construct the graph */
    fflush(stdout);
    KShorthestPathGraph ksp_graph(Cells,
                                  CellsList,
                                  (unsigned int)temporal_windows_sz,
                                  spatial_windows_sz,
                                  imagesize, 
                                  distanceToBoundary);
    /* run the ksp optimization */
    mwSize dims[2] = {1, ksp_graph.GetNoOfVertices()};
    plhs[0] = mxCreateNumericArray(2, dims, mxUINT8_CLASS, mxREAL );
    unsigned char *labeled_objects = (unsigned char*) mxGetPr(plhs[0]);
    
    int nbr_paths = KShorthestPathComputer::ComputeKShorthestNodeDisjointPaths(ksp_graph,
                                                                               DEFAULT_MAX_TRAJ,
                                                                               MAX_PATH_LENGTH,
                                                                               labeled_objects);
    return;
}