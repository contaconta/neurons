
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
#include "GraphFactory.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: graphToCloud graph.gr cloud.cl\n");
    exit(0);
  }
  printf("Heavy tailored for the diadem challenge\n");
  Graph_P* gr = GraphFactory::load(argv[1]);
  Cloud<Point3D>* cl = new Cloud<Point3D>();
  for(int i = 0; i <   gr->cloud->points.size(); i+=10){
    Point3Dw* pt = dynamic_cast<Point3Dw*>(gr->cloud->points[i]);
    cl->points.push_back
      (new Point3D(pt->coords[0], pt->coords[1], pt->coords[2]));
  }
  cl->saveToFile(argv[2]);
}
