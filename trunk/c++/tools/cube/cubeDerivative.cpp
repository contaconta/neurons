
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

  if(argc!=7){
    printf("Usage: cubeDerivative cube nx ny nz sxy sz\n");
    exit(0);
  }

  int nx = atoi(argv[2]);
  int ny = atoi(argv[3]);
  int nz = atoi(argv[4]);
  float sxy = atof(argv[5]);
  float sz = atof(argv[6]);


  string cube_out = "g";
  for(int  i = 0; i < nx; i++)
    cube_out = cube_out + "x";
  for(int  i = 0; i < ny; i++)
    cube_out = cube_out + "y";
  for(int  i = 0; i < nz; i++)
    cube_out = cube_out + "z";


  Cube<uchar, ulong>* cube = new Cube<uchar, ulong>(argv[1]);
  char buff[1024];
  sprintf(buff, "%s_%0.2f_%0.2f", cube_out.c_str(), sxy, sz);

  printf("%s\n", buff);

  Cube<float, double>* tmp = cube->create_blank_cube("tmp");
  Cube<float, double>* output = cube->create_blank_cube(buff);

  cube->calculate_derivative(nx, ny, nz, sxy, sxy, sz, output, tmp);




}
