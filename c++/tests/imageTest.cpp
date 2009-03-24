
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
  Image<float>* imgT = new Image<float>("data/rk2.jpg");
  printf("The uchar value is: %f\nThe float value is %f\n",
         (float)((uchar *)(imgT->img->imageData + y*imgT->img->widthStep))[x],
         imgT->at(x,y));

}
