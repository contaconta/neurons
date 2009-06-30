/////////////////////////////////////////////////////////////////////////
// This program is free software; you can redistribute it and/or       //
// modify it under the terms of the GNU General Public License         //
// version 2 as published by the Free Software Foundation.             //
//                                                                     //
// This program is distributed in the hope that it will be useful, but //
// WITHOUT ANY WARRANTY; without even the implied warranty of          //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU   //
// General Public License for more details.                            //
//                                                                     //
// Written and (C) by Aurelien Lucchi and Kevin Smith                  //
// Contact aurelien.lucchi (at) gmail.com or kevin.smith (at) epfl.ch  // 
// for comments & bug reports                                          //
/////////////////////////////////////////////////////////////////////////

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
    mwSize     nLearnerTypes;
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
    nLearnerTypes = mxGetNumberOfElements(prhs[0]);

    int strLength;
    char **learner_type = (char**) mxCalloc(nLearnerTypes, sizeof(char*));
    
    /* copy data from input structure array */
    for (idxCell=0; idxCell<nLearnerTypes; idxCell++) {
      tmp = mxGetCell(prhs[0],idxCell);

      strLength = mxGetN(tmp)+1;
      learner_type[idxCell] = (char*)mxCalloc(strLength, sizeof(char));
      mxGetString(tmp,learner_type[idxCell],strLength);
    }

    char** weak_learners;
    int nb_weak_learners = enumerate_learners(learner_type,nLearnerTypes,
                                              width_detector,height_detector,weak_learners);

    plhs[0] = mxCreateCellMatrix(nb_weak_learners, 1);
    for(int line = 0; line < nb_weak_learners; line++)
      {
        mxSetCell(plhs[0], line, mxCreateString(weak_learners[line]));
        delete[] weak_learners[line];
      }

    for (idxCell=0; idxCell<nLearnerTypes; idxCell++) {
      mxFree(learner_type[idxCell]);
    }

    mxFree(learner_type);
    delete[] weak_learners;
}
    
