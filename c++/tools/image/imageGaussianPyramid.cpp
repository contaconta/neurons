
////////////////////////////////////////////////////////////////////////////////
// This program is free software; you can redistribute it and/or              //
// modify it under the terms of the GNU General Public License                //
// version 2 as published by the Free Software Foundation.                    //
//                                                                            //
// This program is distributed in the hope that it will be useful, but        //
// WITHOUT ANY WARRANTY; without even the implied warranty of                 //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU          //
// General Public License for more details.                                   //
//                                                                            //
// Written and (C) by German Gonzalez Serrano                                 //
// Contact < german.gonzalez@epfl.ch > for comments & bug reports             //
////////////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Image.h"

using namespace std;

int main(int argc, char **argv) {
  if(argc!=2){
    printf("Usage: imageGaussianPyramid image\n");
    exit(0);
  }

  string imageName(argv[1]);
  string directory = getDirectoryFromPath(imageName);
  string name      = getNameFromPathWithoutExtension(imageName);

  for(int i = 1; i <= 3; i++){
    printf("Loading image %s\n", imageName.c_str());
    char suffix[256];
    sprintf(suffix, "_%i.jpg", (int)pow(2.0,double(i)));
    Image<float>* img = new Image<float>(imageName);
    Image<float>* blur = img->calculate_derivative(0,0,2.0,directory + "blur" + suffix);
    blur->save();
    IplImage* imgDt = cvCreateImage(cvSize(img->width/2, img->height/2),IPL_DEPTH_8U, 3);
    string downName = directory + name + suffix;
    printf("Attempting to save in %s\n", downName.c_str());
    printf("%s\n", downName.c_str());
    cvSaveImage(downName.c_str(),imgDt);
    Image<float>* downSampled = new Image<float>(downName);
    for(int  y = 0; y < downSampled->height; y++)
      for(int x = 0; x < downSampled->width; x++)
        downSampled->put(x,y,blur->at(min(2*x,blur->width-1), min(2*y,blur->height-1)));
    downSampled->save();
    imageName = downName;
  }


}
