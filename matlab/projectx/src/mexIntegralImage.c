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
    unsigned char *pImage;
    int *pResult;
    const mwSize *dim_array;
    
    /* Check for proper number of input and output arguments */    
    if (nrhs != 2) {
	mexErrMsgTxt("Two input arguments required.");
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

    for(j=0;j<10;j++){
      printf("%d\n",pImage[j]);
      //pResult[j] = pIndices[j];
    }
    
    /* Invert dimensions :
       Matlab : height, width
       OpenCV : width, hieght
    */
    /* Create output matrix */
    number_of_dims=mxGetNumberOfDimensions(prhs[0]);
    dim_array=mxGetDimensions(prhs[0]);
    const mwSize dims[]={dim_array[0],dim_array[1]};
    plhs[0] = mxCreateNumericArray(2,dims,mxUINT32_CLASS,mxREAL);
    pResult = (int*)mxGetData(plhs[0]);
    copyIntegralImage(pImage,dim_array[1],dim_array[0],pIndices,pResult);
}
