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

  if(argc!=5)
    {
      printf("Usage: cubeAt volume_name x y z\n");
      exit(0);
    }

  string volume_str = argv[1];

  Cube<uchar,ulong>* cube_test = new Cube<uchar,ulong>(volume_str,false);
  if(cube_test->type == "uchar"){
    Cube<uchar,ulong>* cube = new Cube<uchar,ulong>(volume_str);
    printf("%s ->at(%i, %i,%i) = %u\n", argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]),
           cube->at(atoi(argv[2]), atoi(argv[3]), atoi(argv[4])));
  }
  if(cube_test->type == "float"){
    Cube<float,double>* cube = new Cube<float,double>(volume_str);
    printf("%s ->at(%i, %i,%i) = %f\n", argv[1], atoi(argv[2]), atoi(argv[3]), atoi(argv[4]),
           cube->at(atoi(argv[2]), atoi(argv[3]), atoi(argv[4])));

  }
}
