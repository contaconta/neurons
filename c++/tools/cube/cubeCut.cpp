
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

  if(argc!=9){
    printf("Usage: cubeCut <volume.nfo> <x0> <y0> <z0> <x1> <y1> <z1> <output_name>\n");
    exit(0);
  }

  Cube<uchar, ulong>* test = new Cube<uchar,ulong>(argv[1]);
  if(test->type == "uchar"){
    Cube<uchar, ulong>* source = new Cube<uchar,ulong>(argv[1]);
    source->cut_cube
      (atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), 
       atoi(argv[6]), atoi(argv[7]), argv[8]);
  }
  else {
    Cube<float, double>* source = new Cube<float,double>(argv[1]);
    source->cut_cube
      (atoi(argv[2]), atoi(argv[3]), atoi(argv[4]), atoi(argv[5]), 
       atoi(argv[6]), atoi(argv[7]), argv[8]);
  }

}
