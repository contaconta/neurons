
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
// Contact < german.gonzalez@epfl.ch > for comments & bug reports      //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <argp.h>
#include "Graph.h"
#include "Cloud.h"
#include "CubeFactory.h"
#include "CloudFactory.h"
#include "Cloud_P.h"
#include "Point3Dt.h"
#include <gsl/gsl_rng.h>

using namespace std;

int main(int argc, char **argv) {

  if(argc!=4){
    printf("Use: cloudAddNoise cloud.cl percentage output.cl\n");
    exit(0);
  }

  // Random number generation
  const gsl_rng_type * T2;
  gsl_rng * r;
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r = gsl_rng_alloc (T2);

  Cloud_P* orig    = CloudFactory::load(argv[1]);
  float percentage = atof(argv[2]);
  Cloud_P* dest    = CloudFactory::newCloudWithType(orig);

  vector<double> spr = orig->spread();
  double wx = spr[1]-spr[0];
  double wy = spr[3]-spr[2];
  double wz = spr[5]-spr[4];


  // double idx = (gsl_rng_uniform(r));
  // printf("%f\n", idx);
  for(int i = 0; i < spr.size(); i++)
    printf("%f ", spr[i]);
  printf("\n");

  for(int i = 0; i < orig->points.size(); i++){
    dest->points.push_back(new Point3Dt
                           (orig->points[i]->coords[0],
                            orig->points[i]->coords[1],
                            orig->points[i]->coords[2],
                            1));
  }

  int nPointsToGenerate = percentage*orig->points.size();
  for(int i = 0; i < nPointsToGenerate; i++){
    dest->points.push_back(new Point3Dt
                           (spr[0] + wx*gsl_rng_uniform(r),
                            spr[2] + wy*gsl_rng_uniform(r),
                            spr[4] + wz*gsl_rng_uniform(r),
                            -1));
  }

  dest->saveToFile(argv[3]);
}
