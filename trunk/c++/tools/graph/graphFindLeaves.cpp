
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
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Graph.h"
#include "Cloud.h"
#include "CloudFactory.h"
#include "GraphFactory.h"


using namespace std;

int main(int argc, char **argv) {

  if(argc!=3){
    printf("graphFindLeaves graph.gr leaves.cl\n");
    exit(0);
  }

  Graph<Point3D, EdgeW<Point3D> >* gr =
    new Graph<Point3D, EdgeW<Point3D> >(argv[1]);

  Cloud<Point3D>* cl = new Cloud<Point3D>();

  vector<int> pointsToAdd = gr->findLeaves();

  for(int i = 0; i < pointsToAdd.size(); i++){
      cl->addPoint(gr->cloud->points[pointsToAdd[i]]->coords[0],
                   gr->cloud->points[pointsToAdd[i]]->coords[1],
                   gr->cloud->points[pointsToAdd[i]]->coords[2]);
  }

  //Leaves are green
  cl->v_r = 0;
  cl->v_g = 1;
  cl->v_b = 0;
  cl->v_radius = 0.8;

  cl->saveToFile(argv[2]);

}
