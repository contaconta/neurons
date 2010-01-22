
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
#include "Cloud.h"
#include <gsl/gsl_rng.h>

using namespace std;

int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: cloudEliminateWidth cl1.cl cl2.cl\n");
    exit(0);
  }

  const gsl_rng_type * T2;
  gsl_rng * r;
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r = gsl_rng_alloc (T2);

  Cloud<Point2Dotw>* cl1 = new Cloud<Point2Dotw>(argv[1]);
  Cloud<Point2Dot>*  cl2 = new Cloud<Point2Dot>();

  int nPointsTotal = 1000;
  int nPointsAdded = 0;

  int nPos = 0;
  int nNeg = 0;

  while(nPointsAdded < nPointsTotal)
    {
      int idx = gsl_rng_uniform(r)*cl1->points.size();
      Point2Dotw* pt = dynamic_cast<Point2Dotw*>(cl1->points[idx]);
    if( (pt->type == 1 ) & (nPos < nPointsTotal/2) ){
      nPos++;
      nPointsAdded++;
      cl2->points.push_back
        (new Point2Dot(pt->coords[0],pt->coords[1], pt->theta, pt->type));
    }
    if( (pt->type == -1 ) & (nNeg < nPointsTotal/2) ){
      nNeg++;
      nPointsAdded++;
      cl2->points.push_back
        (new Point2Dot(pt->coords[0],pt->coords[1], pt->theta, pt->type));
    }
    }
  cl2->saveToFile(argv[2]);
}
