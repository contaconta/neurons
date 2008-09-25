
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

  if(argc!=3){
    printf("Usage: cubeMIP <volume.nfo> <outputImage_absolute_path>\n");
    exit(0);
  }

  Cube<uchar, ulong>* test = new Cube<uchar,ulong>(argv[1]);
  if(test->type == "uchar"){
    Cube<uchar, ulong>* source = new Cube<uchar,ulong>(argv[1]);
    source->createMIPImage(argv[2]);
  }
  else {
    Cube<float, double>* source = new Cube<float,double>(argv[1]);
    source->createMIPImage(argv[2]);
  }

}
