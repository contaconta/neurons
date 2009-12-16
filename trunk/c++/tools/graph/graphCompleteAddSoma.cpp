
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
#include "Graph.h"
#include "utils.h"
#include <float.h>
#include <omp.h>

using namespace std;

typedef Graph<Point3D, EdgeW<Point3D> > Graph3D;

int main(int argc, char **argv) {

  if(argc!=7){
    printf("Usage: graphCompleteAddSoma complete.gr X Y Z radious out.gr\n");
    exit(0);
  }

  string nameGraph (argv[1]);
  float  xS  = atof(argv[2]);
  float  yS  = atof(argv[3]);
  float  zS  = atof(argv[4]);
  float  R   = atof(argv[5]);
  string outName   (argv[6]);

  Graph3D* gr  = new Graph3D(nameGraph);
  Graph3D* out = new Graph3D(nameGraph);

  int   pointSoma = gr->cloud->findPointClosestTo(xS,yS,zS);
  vector< int > pointsInSoma = gr->cloud->findPointsCloserToThan(xS,yS,zS,R);

  //Removes all the edges between points in the soma
  for(int i = 0; i < pointsInSoma.size(); i++){
    for(int j = 0; j < pointsInSoma.size(); j++){
      int nE = out->eset.findEdgeBetween(pointsInSoma[i], pointsInSoma[j]);
      if( nE != -1)
        out->eset.edges.erase(out->eset.edges.begin() + nE);
    }
  }

  for(int i = 0; i < pointsInSoma.size(); i++)
    if( pointsInSoma[i] != pointSoma)
      out->eset.edges.push_back
        (new EdgeW<Point3D>(&out->cloud->points, pointSoma, pointsInSoma[i], 0));

  //Save the outGraph
  out->saveToFile(outName);


}
