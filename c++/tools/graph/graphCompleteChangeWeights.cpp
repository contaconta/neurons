
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
#include "Cube.h"
#include <omp.h>

using namespace std;

typedef Graph<Point3D, EdgeW<Point3D> > Graph3D;

float computeCost(Graph3D* gr, CubeF* cube){
  int x,y,z;
  float cost;
  for(int i = 0; i < gr->cloud->points.size(); i++){
    cube->micrometersToIndexes3
      (gr->cloud->points[i]->coords[0], gr->cloud->points[i]->coords[1],
       gr->cloud->points[i]->coords[2], x, y, z);
    cost -= log10(cube->at(x,y,z));
  }
  return cost;
}

int main(int argc, char **argv) {

  if(argc!=5){
    printf("Usage: graphCompleteChangeWeights complete.gr volume odirectoryPaths out.gr\n");
    exit(0);
  }

  string nameGraph (argv[1]);
  string volumeName(argv[2]);
  string pathsDir  (argv[3]);
  string nameOut   (argv[4]);

  Graph3D* gr  = new Graph3D(nameGraph);
  Graph3D* out = new Graph3D(gr->cloud);
  CubeF* volume = new CubeF(volumeName);

  char pathName[256];
  int idx0, idx1;
  for(int nE = 0; nE < gr->eset.edges.size(); nE++){
    idx0 = gr->eset.edges[nE]->p0;
    idx1 = gr->eset.edges[nE]->p1;
    Graph3D* gr;
    sprintf(pathName, "%s/path_%04i_%04i.gr", pathsDir.c_str(), idx0, idx1);
    if(fileExists(pathName)){
      gr = new Graph3D(pathName);
    } else {
      sprintf(pathName, "%s/path_%04i_%04i.gr", pathsDir.c_str(), idx1, idx0);
      if(fileExists(pathName))
        gr = new Graph3D(pathName);
      else{
        printf("I am unable to find path %s, quit ... \n", pathName);
        exit(0);
      }
    }
    out->eset.edges.push_back
      (new EdgeW<Point3D>(&out->cloud->points, idx0, idx1,
                 computeCost(gr, volume)));

  }

  out->saveToFile(nameOut);


}
