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
#include "Neuron.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: cubeConvertToUchar cube_orig.nfo output_cube\n");
    exit(0);
  }

  Cube<int, long>* orig  = new Cube<int, long>(argv[1]);
  Cube<uchar, ulong>* flts = orig->create_blank_cube_uchar(argv[2]);

  float min_value = 1e8;
  float max_value = -1e8;
  printf("Finding max and min values  [");
  for(int z=0; z < orig->cubeDepth; z++){
    for(int y = 0; y < orig->cubeHeight; y++){
      for(int x = 0; x < orig->cubeWidth; x++){
        if(orig->at(x,y,z) > max_value)
          max_value = orig->at(x,y,z);
        if(orig->at(x,y,z) < min_value)
          min_value = orig->at(x,y,z);
      }
    }
    printf("#");fflush(stdout);
  }
  printf("]\n");

  printf("Copying the values  [");
  for(int z=0; z < orig->cubeDepth; z++){
    for(int y = 0; y < orig->cubeHeight; y++)
      for(int x = 0; x < orig->cubeWidth; x++)
        flts->put(x,y,z,uchar(255*(orig->at(x,y,z)-min_value)/(max_value-min_value)));
    printf("#");fflush(stdout);
  }
  printf("]\n");
}
