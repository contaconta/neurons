
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
// Written and (C) by German Gonzalez Serrano                          //
// Contact < german.gonzalez@epfl.ch > for comments & bug reports      //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "Cube.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: cube-log10 cube output\n");
    exit(0);
  }

  Cube<float,double>* orig  = new Cube<float,double>(argv[1]);
  Cube<float,double>* flts = orig->create_blank_cube(argv[2]);

  printf("Finding the log  [");
#pragma omp parallel for
  for(int z=0; z < orig->cubeDepth; z++){
    for(int y = 0; y < orig->cubeHeight; y++)
      for(int x = 0; x < orig->cubeWidth; x++)
        flts->put(x,y,z,-log10(orig->at(x,y,z)) );
    printf("#");fflush(stdout);
  }
  printf("]\n");


}
