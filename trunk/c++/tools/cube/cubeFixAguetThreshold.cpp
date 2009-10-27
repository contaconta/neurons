
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
#include "Cube.h"
#include <omp.h>

using namespace std;

int main(int argc, char **argv) {

  if(argc != 3){
    printf("Usage: cubeFixAguetThreshold cube output\n");
    exit(0);
  }
  Cube<float, double>* orig = new Cube<float, double>(argv[1]);
  Cube<float, double>* dest = orig->duplicate_clean(argv[2]);

  float min, max;
  orig->min_max(&min, &max);

#pragma omp paralel for
  for(int z=0; z < orig->cubeDepth; z++)
    for(int y=0; y < orig->cubeHeight; y++)
      for(int x=0; x < orig->cubeWidth; x++)
        if(orig->at(x,y,z) == 0)
          dest->put(x,y,z,min);
        else
          dest->put(x,y,z,orig->at(x,y,z));
}
