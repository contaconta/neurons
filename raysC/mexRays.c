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
#include "rays.h"
#include "cv.h"

/* function [RAY1 RAY3 RAY4] = rays(E, G, angle, stride)
 * RAYS computes RAY features
 *   Example:
 *   -------------------------
 *   I = imread('cameraman.tif');
 *   SPEDGE = spedge_dist(I,30,2, 11);
 *   imagesc(SPEDGE);  axis image;
 *
 *
 *   FEATURE = spedge_dist(E, G, ANGLE, STRIDE)  computes a spedge 
 *   feature on a grayscale image I at angle ANGLE.  Each pixel in FEATURE 
 *   contains the distance to the nearest edge in direction ANGLE.  Edges 
 *   are computed using Laplacian of Gaussian zero-crossings (!!! in the 
 *   future we may add more methods for generating edges).  SIGMA specifies 
 *   the standard deviation of the edge filter.  
 */
void mexFunction(int nlhs,       mxArray *plhs[],
                      int nrhs, const mxArray *prhs[])
{
    mwSize nElements,j;
    mwSize number_of_dims;
    double sigma;
    double angle;
    int strLength;
    mxArray    *tmp;
    char *pImageName;
    
    /* Check for proper number of input and output arguments */    
    if (nrhs != 3) {
      mexErrMsgTxt("Three input arguments required.");
    } 
    if (nlhs > 1){
	mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Check data type of input argument */
    if (!(mxIsCell(prhs[0]))) {
      mexErrMsgTxt("Input array must be of type cell.");
    }
    if (!(mxIsDouble(prhs[1]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
    if (!(mxIsDouble(prhs[2]))) {
      mexErrMsgTxt("Input array must be of type double.");
    }
    
    mexPrintf("Loading input parameters\n");
  
    /* Get the real data */
    //pImage = (unsigned char *)mxGetPr(prhs[0]);
    //pGradient = (double *)mxGetPr(prhs[1]);
    //pImageName = (const char *)mxGetPr(prhs[0]);

    tmp = mxGetCell(prhs[0],0);
    strLength = mxGetN(tmp)+1;
    pImageName = (char*)mxCalloc(strLength, sizeof(char));
    mxGetString(tmp,pImageName,strLength);

    sigma=*((double *)mxGetPr(prhs[1]));
    angle=*((double *)mxGetPr(prhs[2]));

    /*
    for(j=0;j<10;j++){
      printf("%d\n",pImage[j]);
      //pResult[j] = pIndices[j];
    }
    */
    
    /* Invert dimensions :
       Matlab : height, width
       OpenCV : width, hieght
    */
    /* Create output matrix */
    /*
    number_of_dims=mxGetNumberOfDimensions(prhs[0]);
    dim_array=mxGetDimensions(prhs[0]);
    const mwSize dims[]={dim_array[0],dim_array[1]};
    plhs[0] = mxCreateNumericArray(2,dims,mxUINT32_CLASS,mxREAL);
    pResult = (int*)mxGetData(plhs[0]);
    copyIntegralImage(pImage,dim_array[1],dim_array[0],pResult);
    */

    mexPrintf("computeRays\n");
    IplImage* ray1;
    IplImage* ray3;
    IplImage* ray4;
    computeRays((const char*)pImageName, sigma, angle, &ray1,&ray3,&ray4,F_CANNY,true);

    // Release images
    mexPrintf("Cleaning\n");
    cvReleaseImage(&ray1);
    cvReleaseImage(&ray3);
    cvReleaseImage(&ray4);

    mxFree(pImageName);
}
