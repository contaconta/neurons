
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
#include "Cube.h"
#include "Mask.h"

using namespace std;

int main(int argc, char **argv) {
  Cube<uchar, ulong>* cbu =
    new Cube<uchar, ulong>("/media/neurons/cutConv/cut.nfo");
  Cube<float, double>* tmp =
    new Cube<float, double>(cbu->cubeWidth,  cbu->cubeHeight, cbu->cubeDepth,
                            "/media/neurons/cutConv/tmp",
                            0.1,0.1,0.1);
  vector< float > mask = Mask::gaussian_mask(2, 4, true);
  printf("Doing the convolution with mask of size %i ...\n", mask.size());
  cbu->convolve_horizontally(mask, tmp);
  printf("Exiting ...\n");
}
