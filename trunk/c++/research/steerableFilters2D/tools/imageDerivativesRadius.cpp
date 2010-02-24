
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
// Contact <german.gonzalez@epfl.ch> for comments & bug reports        //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "SteerableFeatures2DRadius.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc != 3){
    printf("Usage imageRadius image radius\n");
    exit(0);
  }

  string imageName (argv[1]);
  float radius = atof(argv[2]);
  printf("imageRadius %s, %f\n", imageName.c_str(), radius);

  vector< RadialFunction* > rfunc;
  rfunc.push_back(new Po0(radius));
  rfunc.push_back(new Po1(radius));
  rfunc.push_back(new Po2(radius));
  rfunc.push_back(new Po3(radius));

  // rfunc.push_back(new Torus(2*radius, radius));

  SteerableFilter2DMR* stf = new SteerableFilter2DMR
    (imageName, 4, 1.0, "result.jpg", rfunc, true, true, false);

}
