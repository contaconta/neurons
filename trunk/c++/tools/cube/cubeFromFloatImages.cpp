
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

  if(argc!=8){
    printf("Usage: cubeFromFloatImages directory/format.jpg idx_b idx_e increment voxelWidth voxelHeight voxelDepth\n");
    exit(0);
  }

  Cube<float, double>* cube = new Cube<float,double>();
//   cube->create_cube_from_float_images
//     ( argv[1], atof(argv[2]), atof(argv[3]), atof(argv[4]),
//       atof(argv[5]), atof(argv[6]), atof(argv[7]));
}
