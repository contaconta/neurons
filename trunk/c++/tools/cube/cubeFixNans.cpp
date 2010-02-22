
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
    printf("Usage cubeFixNans cube output\n");
    exit(0);
  }


  CubeF* orig = new CubeF(argv[1]);
  float minv, maxv;
  orig->min_max(&minv, &maxv);
  CubeF* dest = orig->create_blank_cube(argv[2]);
  for(int z = 0; z < orig->cubeDepth; z++){
    for(int y = 0; y < orig->cubeHeight; y++){
      for(int x = 0; x < orig->cubeWidth; x++){
        if(isnan(orig->at(x,y,z))){
          printf("Nan fixed at [%i,%i,%i]\n", x, y, z);
          dest->put(x,y,z, minv);
        }
        else
          dest->put(x,y,z,orig->at(x,y,z));
      }
    }
  }

}
