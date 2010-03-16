
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

  if(argc!=4){
    printf("Usage: cubeIntegralCube <volume.nfo> <output_filename.iv> layers{0,1}\n");
    exit(0);
  }

  Cube<uchar,ulong>* cube = new Cube<uchar,ulong>(argv[1]);
  if(cube->type != "uchar"){
    printf("Called with a type of cube %s, exiting\n", cube->type.c_str());
    exit(0);
  }

  if(atoi(argv[3]) == 1)
    cube->create_integral_cube_by_layers(argv[2]);
  else
    cube->create_integral_cube(argv[2]);


}
