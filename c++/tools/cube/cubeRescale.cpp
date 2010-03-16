
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
// Contact <german.gonzalez@epfl.ch> for comments & bug reports        //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Cube.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: cubeRescale cube out\n");
    exit(0);
  }

  Cube<float, double>* orig = new Cube<float, double>(argv[1]);
  Cube<float, double>* dest = orig->create_blank_cube(argv[2]);

  float min; float max;
  orig->min_max(&min, &max);

  for(int i = 0; i < orig->size(); i++)
    dest->voxels_origin[i] = (orig->voxels_origin[i] - min)/(max-min);
}
