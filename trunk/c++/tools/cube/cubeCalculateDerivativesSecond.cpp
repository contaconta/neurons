
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

  if(argc!=5){
    printf("Usage: cubeCalculateDerivativesSecond cube sxy sz reflectToFile{1,0}\n");
    exit(0);
  }

  float sxy = atof(argv[2]);
  float sz = atof(argv[3]);
  int reflectToFile = atoi(argv[4]);


  Cube<uchar, ulong>* cube = new Cube<uchar, ulong>(argv[1]);

  if(reflectToFile)
    cube->calculate_second_derivates(sxy, sz);
  else
    cube->calculate_second_derivates_memory(sxy, sz);
}
