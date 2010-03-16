
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
    printf("Usage: ./cubeSteerSecond <volume.nfo> <sigma_xy> <sigma_z>\n");
    exit(0);
  }

  float sigma_xy = atof(argv[2]);
  float sigma_z  = atof(argv[3]);

  Cube<uchar, ulong>* source = new Cube<uchar,ulong>(argv[1]);
//   Cube<uchar, ulong>* source = new Cube<uchar,ulong>("/media/neurons/Filter/filter.nfo");
//   Cube<uchar, ulong>* source = new Cube<uchar,ulong>("/media/neurons/filter_anysotropic/filter.nfo");


  source->calculate_second_derivates(sigma_xy,sigma_z);
  source->calculate_eigen_values(sigma_xy,sigma_z,true);
  source->order_eigen_values(sigma_xy,sigma_z);
  source->calculate_f_measure(sigma_xy,sigma_z);
  source->calculate_aguet(sigma_xy,sigma_z);


//   source->substract_mean("cut_no_mean");

//   delete source;

//   Cube<float,double>* source2 = new
//     Cube<float,double>("/media/neurons/tests/cut_no_mean.nfo");

//   for(int z = 0; z < source2->cubeDepth; z++)
//     printf("%f\n", source2->at(125,125,z));

//   source2->calculate_second_derivates(sigma);
//   source2->calculate_eigen_values(sigma,true);
//   source2->calculate_f_measure(sigma);
}
