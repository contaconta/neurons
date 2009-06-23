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
#include "loadImage.h"
#include <stdio.h>

void mexFunction(int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    mwSize nElements,j;
    mwSize number_of_dims;
    double *pIndices;
    unsigned int* pType;
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
      mexErrMsgTxt("Input array must be of type uint8.");
    }
    if (!(mxIsDouble(prhs[1]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
    
    /* Get the number of elements in the input argument */
    nElements=mxGetNumberOfElements(prhs[0]);
   
    /* Get the real data */
    pImage=(unsigned char *)mxGetPr(prhs[0]);
    pIndices=(double *)mxGetPr(prhs[1]);
    pType=(unsigned int*)mxGetPr(prhs[2]);
    
    /* Invert dimensions :
       Matlab : height, width
       OpenCV : width, hieght
    */
    const mwSize dims[]={1};
    plhs[0] = mxCreateNumericArray(1,dims,mxUINT32_CLASS,mxREAL);
    pResult = (unsigned int*)mxGetData(plhs[0]);
    dim_array=mxGetDimensions(prhs[0]);
    getRectangleFeature(pImage,dim_array[1],dim_array[0],pIndices,*pType,pResult);
}
