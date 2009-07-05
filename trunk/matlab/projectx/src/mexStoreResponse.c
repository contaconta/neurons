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
#include <stdio.h>
#include <sstream>
#include <iostream>
#include <fstream>

using namespace std;

void mexFunction(int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    mwSize nElements,j;
    mwSize number_of_dims;
    double *pIndices;
    unsigned char *pImage;
    unsigned int *pResult;
    const mwSize *dim_array;
    
    /* Check for proper number of input and output arguments */    
    if (nrhs != 3) {
	mexErrMsgTxt("3 input arguments required.");
    } 
    if (nlhs > 1){
	mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Check data type of input argument */
    if (!(mxIsUint8(prhs[0]))) {
      mexErrMsgTxt("First argument must be of type uint8.");
    }
    if (!(mxIsChar(prhs[1]))) {
      mexErrMsgTxt("Second argument must be of type string.");
    }
    
    int index = (int)*mxGetPtr(plhs[0]);

    /* Copy string */
    int strLength = mxGetN(tmp)+1;
    char* learner_type = (char*)mxCalloc(strLength, sizeof(char));
    mxGetString(prhs[1],learner_type,strLength);
   
    /* Get the real data */
    unsigned char* pData=(unsigned char *)mxGetPr(prhs[2]);
    int nElements = mxGetNumberOfElements(prhs[2]);
    
    /* Store data in a file */
    mexPrintf("nEl %d\n", nElements);

    stringstream out;
    out << learner_type << "_" << index;
    ofstream outputFile(out.str(),ios::out);
    //outputFile.write
    outputFile.close();

    mxFree(learner_type);
}
