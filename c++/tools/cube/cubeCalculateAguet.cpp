
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
// Contact <ggonzale@atenea> for comments & bug reports                //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Cube.h"

using namespace std;

int main(int argc, char **argv) {
  int inverted;

  if(!(argc==4 || argc==5)){
    printf("Usage: cubeCalculateAguet cube sigma_xy sigma_z inverted=1\n");
    exit(0);
  }
  if(argc==4)
    inverted = 1;
  else
    inverted = atoi(argv[4]);

  printf("The last argument should be < 0 if the image is white over gray\n");

  Cube<uchar, ulong>* cube = new Cube<uchar, ulong>(argv[1]);
  cube->calculate_second_derivates(atof(argv[2]), atof(argv[3]));
  cube->calculate_aguet(atof(argv[2]), atof(argv[3]), inverted);
}
