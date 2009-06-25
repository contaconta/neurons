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

#define MAXCHARS 80   /* max length of string contained in each field */

/*  the gateway routine.  */
void mexFunction( int nlhs, mxArray *plhs[],
                  int nrhs, const mxArray *prhs[] )
{
    const char **fnames;       /* pointers to field names */
    mwSize dims;
    mxArray    *tmp, *fout;
    char       *pdata=NULL;
    int        ifield, nfields;
    /*mxClassID  *classIDflags;*/
    mwIndex    jstruct;
    mwSize     NStructElems;
    mwSize     ndim;

    /* check proper input and output */
    if(nrhs!=1)
      mexErrMsgTxt("One input required.");
    else if(nlhs > 1)
      mexErrMsgTxt("Too many output arguments.");
    else if(!mxIsStruct(prhs[0]))
      mexErrMsgTxt("Input must be a structure.");
    /* get input arguments */
    nfields = mxGetNumberOfFields(prhs[0]);
    if(nfields!=1)
      mexErrMsgTxt("Input structure should contain only one field");

    NStructElems = mxGetNumberOfElements(prhs[0]);

    int strLength;
    char *learner_type;
    /* copy data from input structure array */
    for (jstruct=0; jstruct<NStructElems; jstruct++) {
      tmp = mxGetFieldByNumber(prhs[0],jstruct,0);

      strLength = mxGetN(tmp)+1;
      learner_type = (char*)mxCalloc(strLength, sizeof(char));
      mxGetString(tmp,learner_type,strLength);

      mexPrintf("%s\n",learner_type);
      char** weak_learners;
      int nb_weak_learners = enumerate_learners(learner_type,24,24,weak_learners);

      plhs[0] = mxCreateCellMatrix(nb_weak_learners, 1);
      for(int line = 0; line < nb_weak_learners; line++)
        mxSetCell(plhs[0], line, mxCreateString(weak_learners[line]));

      delete[] weak_learners;
    }
    return;
}
    
