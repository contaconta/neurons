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

void mexFunction(int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    /* Check for proper number of input and output arguments */    
    if (nrhs != 2) {
	mexErrMsgTxt("2 input arguments required.");
    } 
    if (nlhs > 1){
	mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Check data type of input argument */
    if (!(mxIsUint8(prhs[0])) && !(mxIsDouble(prhs[0]))) {
      mexErrMsgTxt("First argument must be of type uint8/double.");
    }
    if(mxGetNumberOfElements(prhs[0])<2){
      mexErrMsgTxt("First argument should contain 2 uint8/double.");
    }
    if (!(mxIsChar(prhs[1]))) {
      mexErrMsgTxt("Second argument must be of type string.");
    }
    
    int index_x = ((int *)mxGetPr(prhs[0]))[0];
    int index_y = ((int *)mxGetPr(prhs[0]))[0];

    /* Copy string */
    int strLength = mxGetN(prhs[1])+1;
    char* stype = (char*)mxCalloc(strLength, sizeof(char));
    mxGetString(prhs[1],stype,strLength);
    eDataType eType;
    if(strcmp(stype,"row")==0)
      eType = TYPE_ROW;
    else if(strncmp(stype,"col",3)==0)
      eType = TYPE_COLUMN;
    else
      {
        mexErrMsgTxt("Second argument should be 'row' or 'col'");
      }

    // TODO : this should be stored in matlab and passed as a parameter
    const int data_size = 10;

    const mwSize dims[]={data_size};
    plhs[0] = mxCreateNumericArray(1,dims,mxUINT32_CLASS,mxREAL);
    unsigned int* pData = (unsigned int*)mxGetData(plhs[0]);

    getWeakLearnerResponses(index_x, index_y, pData, eType);

    mxFree(stype);
}
