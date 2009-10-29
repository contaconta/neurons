
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
#include "CloudFactory.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!=4){
    printf("Usage: graphCptFromCloud cloud.cl radius out.gr\n");
    exit(0);
  }

  Cloud_P* cloud   = CloudFactory::load(argv[1]);
  float    radious = atof(argv[2]);
  Graph<Point3D, Edge<Point3D> >* gr = new  Graph<Point3D, Edge<Point3D> >(cloud);

  for(int i = 0; i < cloud->points.size(); i++){
    Point* pt1 = cloud->points[i];
    for(int j = i+1; j < cloud->points.size(); j++){
      Point* pt2 = cloud->points[j];
      if(sqrt( (pt1->coords[0]-pt2->coords[0])*(pt1->coords[0]-pt2->coords[0]) +
               (pt1->coords[1]-pt2->coords[1])*(pt1->coords[1]-pt2->coords[1]) +
               (pt1->coords[2]-pt2->coords[2])*(pt1->coords[2]-pt2->coords[2]) )
         < radious)
        gr->eset.edges.push_back
          (new Edge<Point3D>(&gr->cloud->points,i, j));
    }
  }

  gr->saveToFile(argv[3]);
}
