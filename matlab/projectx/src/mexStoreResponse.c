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
#include "memClient.h"


/*
  Arguments : data, format={'row','col'}, learner_index, type={'HA', 'RAY'}
 */
void mexFunction(int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    /* Check for proper number of input and output arguments */    
    if (nrhs != 4) {
	mexErrMsgTxt("4 input arguments required.");
    } 
    if (nlhs > 1){
	mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Check data type of input argument */
    if (!mxIsUint8(prhs[0]) && (!mxIsDouble(prhs[0]))
        && !mxIsUint16(prhs[0]) && !mxIsUint32(prhs[0])){
      mexErrMsgTxt("First argument must be of type uint/double.");
    }
    if (!(mxIsChar(prhs[1]))) {
      mexErrMsgTxt("Second argument must be of type string.");
    }
    if (!(mxIsUint8(prhs[2])) && !(mxIsDouble(prhs[2]))) {
      mexErrMsgTxt("Third argument must be of type uint8/double.");
    }
    if(mxGetNumberOfElements(prhs[2])!=1){
      mexErrMsgTxt("Third argument should contain 1 uint8/double.");
    }
    if (!(mxIsChar(prhs[3]))) {
      mexErrMsgTxt("Fourth argument must be of type string.");
    }
    

    /* Get the real data */
    unsigned int* pData=(unsigned int*)mxGetPr(prhs[0]);
    int nElements = mxGetNumberOfElements(prhs[0]);

    /* Get weak learner index (-1 because C arrays start at 0) */
    int index = (int)mxGetScalar(prhs[2]) - 1;

    /* Copy string */
    int strLength = mxGetN(prhs[1])+1;
    char* sFormat = (char*)mxCalloc(strLength, sizeof(char));
    mxGetString(prhs[1],sFormat,strLength);
    eDataFormat eFormat;
    if(strcmp(sFormat,"row")==0)
      eFormat = FORMAT_ROW;
    else if(strncmp(sFormat,"col",3)==0)
      eFormat = FORMAT_COLUMN;
    else
      {
        // Memory allocated using mxCalloc (but not memory allocated with malloc or calloc)
        // is freed automatically by mexErrMsgTxt
        mexErrMsgTxt("Second argument should be 'row' or 'col'");
      }
    mxFree(sFormat);

    strLength = mxGetN(prhs[3])+1;
    char* sType = (char*)mxCalloc(strLength, sizeof(char));
    mxGetString(prhs[3],sType,strLength);
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

    storeWeakLearnerResponses(pData, eFormat, eType, index, nElements);    
}
