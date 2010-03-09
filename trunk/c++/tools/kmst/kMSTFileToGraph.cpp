
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

  if(argc!=4){
    printf("Usage: kMSTFileToGraph graphiOrig.gr kMSTFile graphDest.gr\n");
    exit(0);
  }

  Graph<Point3D, EdgeW<Point3D> >* grOrig
    = new Graph<Point3D, EdgeW<Point3D> >(argv[1]);

  Graph<Point3D, EdgeW<Point3D> >* grDest
    = new Graph<Point3D, EdgeW<Point3D> >(grOrig->cloud);

  std::ifstream in(argv[2]);
  int nPoints, nEdges;
  int idx0, idx1; double weight;
  in >> nPoints; in >> nEdges;
  for(int i = 0; i < nEdges; i++){
    in >> idx0; in >> idx1; in >> weight;
    grDest->eset.edges.push_back
      (new EdgeW<Point3D>
       (&grDest->cloud->points, idx0-1, idx1-1, weight));
  }


  grDest->saveToFile(argv[3]);
}
