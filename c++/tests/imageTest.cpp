
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
// Written and (C) by German Gonzalez                                  //
// Contact < german.gonzalez@epfl.ch > for comments & bug reports      //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>

#include "neseg.h"
#include "Image.h"

using namespace std;

int main(int argc, char **argv) {
  int x = 0;
  int y = 0;
  string filename = "GT001.png";
  Image<float>* imgT = new Image<float>(filename);
  printf("The uchar value is: %f\nThe float value is %f\n",
         (float)((uchar *)(imgT->img->imageData + y*imgT->img->widthStep))[x],
         imgT->at(x,y));

  //IplImage* img=imgT->img;
  // Access color image
  IplImage* img = cvLoadImage(filename.c_str(),CV_LOAD_IMAGE_COLOR);
  printf("nChannels: %d\n", img->nChannels);
  if(img->nChannels==3)
    {
      uchar* ptrColImage = &((uchar*)(img->imageData + img->widthStep*y))[x*3];
      printf("img(%d,%d): %d,%d,%d\n",x,y,*ptrColImage++,*ptrColImage++,*ptrColImage);
    }
  else
    {
      uchar* ptrColImage = &((uchar*)(img->imageData + img->widthStep*y))[x];
      printf("img(%d,%d): %d\n",x,y,*ptrColImage);
    }
}
