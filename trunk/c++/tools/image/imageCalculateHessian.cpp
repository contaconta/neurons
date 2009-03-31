
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
#include "Image.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc != 7){
    printf("Usage: imageCalculateHessian image sigma eigenValueFileH eigenValueFileL saveOrientation orientationFile\n");
    exit(0);
  }

  string imageName       = argv[1];
  float  sigma           = atof(argv[2]);
  string eigenValueH     = argv[3];
  string eigenValueL     = argv[4];
  bool   saveOrientation = atoi(argv[5]);
  string orientationFile = argv[6];

  Image<float>* orig = new Image<float>(imageName);
  orig->computeHessian(sigma, eigenValueH, eigenValueL, saveOrientation, orientationFile);

}
