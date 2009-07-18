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

#include <stdio.h>
#include <string.h>
#include "mex.h"
#include "common.h"
#include "memClient.h"

//#define DEBUG_M

/*
  Arguments : format={'row','col'}, learner_index, type={'HA', 'RAY'}
 */
void mexFunction(int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    /* Check for proper number of input and output arguments */    
    if (nrhs != 3) {
	mexErrMsgTxt("3 input arguments required.");
    } 
    if (nlhs > 1){
	mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Check data type of input argument */
    if (!(mxIsChar(prhs[0]))) {
      mexErrMsgTxt("First argument must be of type string.");
    }
    if (!(mxIsUint8(prhs[1])) && !(mxIsDouble(prhs[1]))) {
      mexErrMsgTxt("Second argument must be of type uint8/double.");
    }
    if(mxGetNumberOfElements(prhs[1])!=1){
      mexErrMsgTxt("Second argument should contain 1 uint8/double.");
    }
    if (!(mxIsChar(prhs[2]))) {
      mexErrMsgTxt("Third argument must be of type string.");
    }

    /* Get weak learner index (-1 because C arrays start at 0) */
    int index = (int)mxGetScalar(prhs[1]) - 1;

    /* Copy string */
    int strLength = mxGetN(prhs[0])+1;
    char* sFormat = (char*)mxCalloc(strLength, sizeof(char));
    mxGetString(prhs[0],sFormat,strLength);
    eDataFormat eFormat;
    int data_size;
    int width;
    int height;
    getMemSize(width, height);
#ifdef DEBUG_M
    mexPrintf("width %d height %d\n",width,height);
#endif
    if(strcmp(sFormat,"row")==0)
      {
        eFormat = FORMAT_ROW;
        data_size = width;
      }
    else if(strncmp(sFormat,"col",3)==0)
      {
        eFormat = FORMAT_COLUMN;
        data_size = height;
      }
    else
      {
        // Memory allocated using mxCalloc (but not memory allocated with malloc or calloc)
        // is freed automatically by mexErrMsgTxt
        mexErrMsgTxt("Second argument should be 'row' or 'col'");
      }
    mxFree(sFormat);

    strLength = mxGetN(prhs[2])+1;
    char* sType = (char*)mxCalloc(strLength, sizeof(char));
    mxGetString(prhs[2],sType,strLength);
    eDataType eType;
    if(strcmp(sType,"HA")==0)
      eType = TYPE_HAAR;
    else if(strcmp(sType,"RA")==0)
      eType = TYPE_RAY;
    else
      {
        // Memory allocated using mxCalloc (but not memory allocated with malloc or calloc)
        // is freed automatically by mexErrMsgTxt
        mexErrMsgTxt("Fourth argument should be a learner type");
      }
    mxFree(sType);

    const mwSize dims[]={data_size};
    plhs[0] = mxCreateNumericArray(1,dims,mxDOUBLE_CLASS,mxREAL);
    double* pData = (double*)mxGetData(plhs[0]);

    getWeakLearnerResponses(pData, eFormat, eType, index);
}
