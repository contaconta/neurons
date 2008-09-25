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

  if(argc!=2)
    {
      printf("Usage: cubeStatistics volume_name\n");
      exit(0);
    }

  Cube_P* cube;

  string volume_str = argv[1];

  Cube<uchar,ulong>* cube_test = new Cube<uchar,ulong>(volume_str,false);
  if(cube_test->type == "uchar"){
    cube = new Cube<uchar,ulong>(volume_str);
  }
  if(cube_test->type == "float"){
    cube = new Cube<float,double>(volume_str);
  }

  cube->print_statistics();
  cube->histogram();





}
