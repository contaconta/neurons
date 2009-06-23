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

#include <iostream>
#include "utils.h"

using namespace std;

IplImage *getGray(IplImage *img)
{
  // Check we have been supplied a non-null img pointer
  if (!img)
    cout << "Unable to create grayscale image.  No image supplied";

  IplImage* gray8, * gray32;

  gray32 = cvCreateImage( cvGetSize(img), IPL_DEPTH_32F, 1 );

  if( img->nChannels == 1 )
    gray8 = (IplImage *) cvClone( img );
  else {
    gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
    cvCvtColor( img, gray8, CV_BGR2GRAY );
  }

  cvConvertScale( gray8, gray32, 1.0 / 255.0, 0 );

  cvReleaseImage( &gray8 );
  return gray32;
}

IplImage* getGray8(IplImage *img)
{
  IplImage *gray_image,*gray_img0;
  gray_image=cvCloneImage(img);
  
  //check image is it gray or not if nor convert it to the gray
  if(img->nChannels!=1){
    //convert original image to gray_scale image
    gray_img0 = cvCreateImage(cvSize( gray_image->width,
                                      gray_image->height), 8, 1 );
    cvCvtColor(gray_image, gray_img0, CV_RGB2GRAY );
    gray_image = cvCloneImage( gray_img0 );
    cvReleaseImage(&gray_img0);
  }
  return gray_image;
}

