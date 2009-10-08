
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
// Written and (C) by German Gonzalez Serrano                          //
// Contact < german.gonzalez@epfl.ch > for comments & bug reports      //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Graph.h"
#include <gsl/gsl_rng.h>

using namespace std;

int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: kMSTFileToGraph graphiOrig.gr kMSTFile graphDest.gr\n");
    exit(0);
  }

  const gsl_rng_type * T2;
  gsl_rng * r;
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r = gsl_rng_alloc (T2);


  Graph<Point3D, EdgeW<Point3D> >* grOrig
    = new Graph<Point3D, EdgeW<Point3D> >(argv[1]);

  Graph<Point3D, EdgeW<Point3D> >* grDest
    = new Graph<Point3D, EdgeW<Point3D> >(grOrig->cloud);

std::ifstream in(argv[2]);





}
