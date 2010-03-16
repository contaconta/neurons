
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

  if(argc!=5){
    printf("Usage: cubeaAverage cubeR cubeG cubeB out\n");
    exit(0);
  }

  Cube<uchar, ulong>* R = new Cube<uchar,ulong>(argv[1]);
  Cube<uchar, ulong>* G = new Cube<uchar,ulong>(argv[2]);
  Cube<uchar, ulong>* B = new Cube<uchar,ulong>(argv[3]);

  Cube<uchar, ulong>* out = R->create_blank_cube_uchar(argv[4]);

  for(int z = 0; z < R->cubeDepth; z++){
    for(int y = 0; y < R->cubeHeight; y++){
      for(int x = 0; x < R->cubeHeight; x++){
        out->put(x,y,z,
                 (uchar)((R->at(x,y,z) + G->at(x,y,z) + B->at(x,y,z))/3));
      }
    }
  }


}
