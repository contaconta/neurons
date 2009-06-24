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
 
    /* create a 1x1 struct matrix for output  */
    /*plhs[0] = mxCreateStructMatrix(1, 1, nfields, fnames);*/

    mexPrintf("HERE %d\n", NStructElems);

    int strLength;
    char *learner_type;
    /* copy data from input structure array */
    for (jstruct=0; jstruct<NStructElems; jstruct++) {
      tmp = mxGetFieldByNumber(prhs[0],jstruct,0);
      /*
      if( mxIsChar(tmp)) {
        mxSetCell(fout, jstruct, mxDuplicateArray(tmp));
      }
      */

      strLength = mxGetN(tmp)+1;
      learner_type = (char*)mxCalloc(strLength, sizeof(char));
      mxGetString(tmp,learner_type,strLength);

      mexPrintf("%s\n",learner_type);
      enumerate_learners(learner_type,24,24);

      /* create output cell array */
      /*
      ndim = 1; //mxGetNumberOfDimensions(prhs[0]);
      dims = (mwSize)list_weak_learners.size(); //mxGetDimensions(prhs[0]);
      fout = mxCreateCellArray(ndim, &dims);
      */

      /*
      for( vector<string>::iterator iter = list_weak_learners.begin();
           iter != list_weak_learners.end(); ++iter ) {
        cout << *iter  << endl;
        mexPrintf("%s\n",iter->c_str());
        mxSetCell(fout, jstruct,iter->c_str());
      }
      */

      /* set each field in output structure */
      /*mxSetFieldByNumber(plhs[0], 0, 0, fout);*/
    }
    return;
}
    
