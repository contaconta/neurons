/* ==========================================================================
 * phonebook.c 
 * example for illustrating how to manipulate structure and cell array
 *
 * takes a (MxN) structure matrix and returns a new structure (1x1)
 * containing corresponding fields: for string input, it will be (MxN)
 * cell array; and for numeric (noncomplex, scalar) input, it will be (MxN)
 * vector of numbers with the same classID as input, such as int, double
 * etc..
 *
 * This is a MEX-file for MATLAB.
 * Copyright 1984-2006 The MathWorks, Inc.
 *==========================================================================*/
/* $Revision: 1.6.6.2 $ */

#include "mex.h"
#include "string.h"
#include "enumerateLearners.h"

/*  the gateway routine.  */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    const char **fnames;       /* pointers to field names */
    mwSize dims;
    mxArray    *tmp, *fout;
    char       *pdata=NULL;
    int        ifield, nfields;
    mwIndex    idxCell;
    mwSize     nCells;
    mwSize     ndim;

    /* check proper input and output */
    if(nrhs!=2)
      mexErrMsgTxt("2 inputs required.");
    else if(nlhs > 1)
      mexErrMsgTxt("Too many output arguments.");
    else
      {
        if(!mxIsCell(prhs[0]))
          mexErrMsgTxt("First input must be a cell.");
        if(!mxIsUint32(prhs[1]) && !mxIsDouble(prhs[1]))
          mexErrMsgTxt("Second input must be a uint32/double.");
      }
    /* get input arguments */
    int width_detector = (int)mxGetPr(prhs[1])[0];
    int height_detector = (int)mxGetPr(prhs[1])[1];
    nCells = mxGetNumberOfElements(prhs[0]);

    int strLength;
    char *learner_type;
    /* copy data from input structure array */
    for (idxCell=0; idxCell<nCells; idxCell++) {
      tmp = mxGetCell(prhs[0],idxCell);

      strLength = mxGetN(tmp)+1;
      learner_type = (char*)mxCalloc(strLength, sizeof(char));
      mxGetString(tmp,learner_type,strLength);

      //mexPrintf("%s\n",learner_type);
      char** weak_learners;
      int nb_weak_learners = enumerate_learners(learner_type,width_detector,height_detector,weak_learners);

      plhs[0] = mxCreateCellMatrix(nb_weak_learners, 1);
      for(int line = 0; line < nb_weak_learners; line++)
        {
          mxSetCell(plhs[0], line, mxCreateString(weak_learners[line]));
          delete[] weak_learners[line];
        }

      mxFree(learner_type);
      delete[] weak_learners;
    }
}
    
