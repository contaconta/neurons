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
#include "intensityFeature.h"
#include <stdio.h>
#include <vector>
#include <iostream>
#include "utils.h"

using namespace std;

// Caution : this number has to be chosen carefully to avoid any buffer
// overflow
#define MAX_WEAK_LEARNER_PARAM_LENGTH 500

/* Arguments : Cell containing image files,
   Cell containing parameters (ids for intensity features),
   Cell containing list of codebook images
*/
void mexFunction(int nlhs,       mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
    unsigned char *pImage;
    response_type *pResult;
    const mwSize *dim_array;
    mxArray *pCellParam;
    mxArray *pCellImage;
    mwSize nImages, nParams, nCodeBook;
    char pParam[MAX_WEAK_LEARNER_PARAM_LENGTH]; // weak learner parameters
    int strLength;
    
    /* Check for proper number of input and output arguments */    
    if (nrhs != 3) {
	mexErrMsgTxt("3 input arguments required.");
    } 
    if (nlhs > 1){
	mexErrMsgTxt("Too many output arguments.");
    }
    
    /* Check data type of input argument */
    if (!mxIsCell(prhs[0])) {
      mexErrMsgTxt("First argument must be of type cell.");
    }
    if (!mxIsCell(prhs[1])) {
      mexErrMsgTxt("Second argument must be of type cell.");
    }
    if (!mxIsCell(prhs[2])) {
      mexErrMsgTxt("Third argument must be of type cell.");
    }
    
    /* Get the real data */
    nImages = mxGetNumberOfElements(prhs[0]);
    nParams = mxGetNumberOfElements(prhs[1]);
    nCodeBook = mxGetNumberOfElements(prhs[2]);

    mwSize number_of_dims = 2;
    const mwSize dims[]={nImages, nParams};
    plhs[0] = mxCreateNumericArray(number_of_dims,dims,mxINT32_CLASS,mxREAL);
    //plhs[0] = mxCreateNumericArray(number_of_dims,dims,mxDOUBLE_CLASS,mxREAL);
    pResult = (response_type*)mxGetData(plhs[0]);

    /* Preload cloud and image classes */
    //vector<IplImage*> list_images;
    //vector<Cloud*> list_clouds;
    //string img_dir = "/localhome/aurelien/Documents/EM/raw_mitochondria2/originals/";
    //string img_dir = "/localhome/aurelien/usr/share/Data/LabelMe/Images/FIBSLICE/";
    //const int nbPointsPerCloud = 600;
    //string cloud_dir("/localhome/aurelien/Sources/EM/svm_test/intensity/Model-8-6000-3-i/");
    //string cloud_dir("//osshare/Work/neurons/matlab/projectx/temp/Model-8-6000-3-i/");
    //vector<string> cloud_files;
    //get_files_in_dir(cloud_dir,cloud_files,"cl");

    //mexPrintf("Loading files\n");
/*
    int iImage = 0;
    for(vector<string>::iterator itFiles = cloud_files.begin();
        itFiles != cloud_files.end(); itFiles++)
      {
        //mexPrintf("Loading file %s\n",itFiles->c_str());
        string filename = cloud_dir + *itFiles;
        // Cloud
        Cloud* point_set = new Cloud(filename);
        list_clouds.push_back(point_set);

        // Image
        string img_filename = img_dir + getNameFromPathWithoutExtension(*itFiles);
        img_filename += ".png";
        IplImage* img = cvLoadImage(img_filename.c_str(),1);

        if(img == 0)
          {
            mexErrMsgTxt("getIntensityFeature: Error while pre-loading files\n"); //img_filename.c_str());
          }
        else
          {
            list_images.push_back(img);
          }
      }
*/

    xImage* img = new xImage;
    pCellImage = mxGetCell(prhs[2],0);
    dim_array = mxGetDimensions(pCellImage);
    img->data = (unsigned char*)mxGetData(pCellImage);
    img->width = dim_array[1];
    img->height = dim_array[0];

    int iResult = 0;
    for(int iImage = 0;iImage<nImages;iImage++)
      {
        /* retrieve the image */
        pCellImage = mxGetCell(prhs[0],iImage);
        pImage = (unsigned char*)mxGetData(pCellImage);
        dim_array = mxGetDimensions(pCellImage);

        if(img->width != dim_array[1] || img->height != dim_array[0])
          {
            mexPrintf("img->width %d, dim_array[1] %d, img->height %d, dim_array[0] %d\n",img->width, dim_array[1], img->height, dim_array[0]);
            mexErrMsgTxt("getIntensityFeature: img->width != dim_array[1] || img->height != dim_array[0]\n");
          }

        //mexPrintf("Image %d\n",iImage);

        for(int iParam = 0;iParam<nParams;iParam++)
          {
            // retrieve cell content and transform it to a string
            pCellParam = mxGetCell(prhs[1],iParam);
            strLength = mxGetN(pCellParam)+1;
            mxGetString(pCellParam,pParam,strLength);

            pResult[iResult] = getIntensityFeature(pImage,
                                                   dim_array[1],dim_array[0],
                                                   pParam,
                                                   img);
            iResult++;
          }
      }

    delete img;
}
