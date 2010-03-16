
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
#include "Timer.h"

using namespace std;

int main(int argc, char **argv) {

  Timer timer;
  unsigned long timeS, timeE;


  Cube<uchar, ulong>* cbu =
    new Cube<uchar, ulong>("/home/ggonzale/n1/3d/1/N1.nfo");


  timeS = timer.getMilliseconds();
  Cube<float, double>* tmp =
    new Cube<float, double>(cbu->cubeWidth,  cbu->cubeHeight, cbu->cubeDepth,
                            "/home/ggonzale/n1/3d/1/tmp",
                            0.1,0.1,0.1);
  timeE = timer.getMilliseconds();
  printf("Time to allocate the float:  %u\n", timeE-timeS);

  vector< float > mask = Mask::gaussian_mask(2, 4, true);
  printf("Doing the convolution with mask of size %i ...\n", mask.size());

  timeS = timer.getMilliseconds();
  cbu->convolve_horizontally(mask, tmp);
  timeE = timer.getMilliseconds();
  printf("Time to do the convolution:  %u\n", timeE-timeS);

  printf("Exiting ...\n");
}
