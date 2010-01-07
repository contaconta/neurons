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

char* usage = "image_name sigma angle edge_low_threshold edge_high_threshold";

/* mexRays computes the following RAY features ; distance (ray1), orientation (ray3), norm (ray4)
 * @param name of the input image.
 * @param sigma specifies the standard deviation of the edge filter.
 * @param angle specifies the angle at which the features should be computed.
 * @param edge_low_threshold low threshold used for the canny edge detection.
 * @param edge_high_threshold high threshold used for the canny edge detection.
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
    if ((nrhs != 3) && (nrhs != 5)) {
      mexErrMsgTxt("Three or four input arguments required.");
    } 
    if (nlhs > 1){
	mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Check data type of input argument */
    if (!(mxIsCell(prhs[0]))) {
      mexErrMsgTxt("First input argument must be of type cell.");
    }
    if (!(mxIsDouble(prhs[1]))) {
      mexErrMsgTxt("Second input argument must be of type double.");
    }
    if (!(mxIsDouble(prhs[2]))) {
      mexErrMsgTxt("Third input argument must be of type double.");
    }
    if ((nrhs > 3) && !(mxIsDouble(prhs[3]))) {
      mexErrMsgTxt("Fourth input argument must be of type double.");
    }
    if ((nrhs > 4) && !(mxIsDouble(prhs[4]))) {
      mexErrMsgTxt("Fifth input argument must be of type double.");
    }
    
    mexPrintf("Loading input parameters\n");
  
    // Get image name
    tmp = mxGetCell(prhs[0],0);
    strLength = mxGetN(tmp)+1;
    pImageName = (char*)mxCalloc(strLength, sizeof(char));
    mxGetString(tmp,pImageName,strLength);

    sigma=*((double *)mxGetPr(prhs[1]));
    angle=*((double *)mxGetPr(prhs[2]));

    mexPrintf("computeRays\n");
    IplImage* ray1;
    IplImage* ray3;
    IplImage* ray4;

    if(nrhs > 3)
      {
        double low_th = *((double *)mxGetPr(prhs[3]));
        double high_th = low_th + 10000;
        if(nrhs > 4)
          high_th = *((double *)mxGetPr(prhs[4]));
        computeRays((const char*)pImageName, sigma, angle, &ray1,&ray3,&ray4,F_CANNY,true,low_th,high_th);
      }
    else
      computeRays((const char*)pImageName, sigma, angle, &ray1,&ray3,&ray4,F_CANNY,true);

    // Release images
    mexPrintf("Cleaning\n");
    cvReleaseImage(&ray1);
    cvReleaseImage(&ray3);
    cvReleaseImage(&ray4);

    mxFree(pImageName);
}
