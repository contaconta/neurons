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

char* usage = "image_name sigma start_angle end_angle step_angle edge_low_threshold edge_high_threshold";

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
    double start_angle;
    double end_angle;
    double step_angle;
    int strLength;
    mxArray    *tmp;
    char *pImageName;
    
    /* Check for proper number of input and output arguments */    
    if ((nrhs != 5) && (nrhs != 7)) {
      mexErrMsgTxt("Five or seven input arguments required.");
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
    if (!(mxIsDouble(prhs[3]))) {
      mexErrMsgTxt("Third input argument must be of type double.");
    }
    if (!(mxIsDouble(prhs[4]))) {
      mexErrMsgTxt("Third input argument must be of type double.");
    }
    if ((nrhs > 5) && !(mxIsDouble(prhs[5]))) {
      mexErrMsgTxt("Fourth input argument must be of type double.");
    }
    if ((nrhs > 6) && !(mxIsDouble(prhs[6]))) {
      mexErrMsgTxt("Fifth input argument must be of type double.");
    }
    
    mexPrintf("Loading input parameters\n");
  
    // Get image name
    tmp = mxGetCell(prhs[0],0);
    strLength = mxGetN(tmp)+1;
    pImageName = (char*)mxCalloc(strLength, sizeof(char));
    mxGetString(tmp,pImageName,strLength);

    sigma=*((double *)mxGetPr(prhs[1]));
    start_angle=*((double *)mxGetPr(prhs[2]));
    end_angle=*((double *)mxGetPr(prhs[3]));
    step_angle=*((double *)mxGetPr(prhs[4]));

    mexPrintf("computeRays\n");

    computeDistanceDifferenceRays(pImageName,
                                  start_angle, end_angle, step_angle,
                                  0, 0);

    /*
    int nAngles = (end_angle-start_angle)/step_angle;
    IplImage** rays1 = new IplImage*[nAngles];
    IplImage** rays3 = 0;
    IplImage** rays4 = 0;
    //IplImage** rays3 = new IplImage*[nAngles];
    //IplImage** rays4 = new IplImage*[nAngles];

    double low_th = 15000;
    double high_th = 30000;
    if(nrhs > 5)
      {
        low_th = *((double *)mxGetPr(prhs[5]));
        high_th = low_th + 10000;
        if(nrhs > 6)
          high_th = *((double *)mxGetPr(prhs[6]));
      }

    double angle = 0;
    for(int a = 0;a<nAngles;a++)
      {
        computeRays((const char*)pImageName, sigma, angle, &rays1[a],&rays3[a],&rays4[a],F_CANNY,true,low_th,high_th);
        angle =+ step_angle;
      }


    // Release images
    mexPrintf("Cleaning\n");
    for(int a = 0;a<nAngles;a++)
      {
        cvReleaseImage(&rays1);
        cvReleaseImage(&rays3);
        cvReleaseImage(&rays4);
      }
    */

    mxFree(pImageName);
}
