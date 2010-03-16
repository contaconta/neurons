
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
#include "Cube.h"

using namespace std;


Graph<Point3Dw, EdgeW<Point3Dw> >*
addWidthToGraph
(Cube<float,double>*  detections,
 Graph<Point3D, EdgeW<Point3D> >* orig)
{
  Graph<Point3Dw, EdgeW<Point3Dw> >* dest =
    new Graph<Point3Dw, EdgeW<Point3Dw> >();

  //Add the points
  int ix, iy, iz;
  float scale;
  for(int nP = 0; nP < orig->cloud->points.size(); nP++){
    Point3D* pt = dynamic_cast<Point3D*>(orig->cloud->points[nP]);
    detections->micrometersToIndexes3
      (pt->coords[0], pt->coords[1], pt->coords[2], ix, iy, iz);
    scale = detections->at(ix, iy, iz);
    dest->cloud->points.push_back
      (new Point3Dw(pt->coords[0], pt->coords[1], pt->coords[2], scale));
  }

  //Adds the edges
  for(int i = 0; i < orig->eset.edges.size(); i++){
    EdgeW<Point3D>* e = dynamic_cast< EdgeW<Point3D>*>(orig->eset.edges[i]);
    dest->eset.edges.push_back
      (new EdgeW<Point3Dw>(&dest->cloud->points, e->p0, e->p1, e->w) );
  }

  return dest;
}


int main(int argc, char **argv) {

  if(argc!=4){
    printf("Usage: graphAddScale directory nPoints volume.nfo\n");
    exit(0);
  }


  string directory(argv[1]);
  int nPoints = atoi(argv[2]);
  CubeF* scales = new CubeF(argv[3]);

  char origName[2024];
  char destName[2024];

  for(int i = 0; i < nPoints; i++){
    for(int j = 0; j < nPoints; j++){
      sprintf(origName, "%s/path_%04i_%04i.gr",  directory.c_str(), i, j);
      if(fileExists(origName)){
        printf("%s\n", origName);
        sprintf(destName, "%s/path_%04i_%04i-w.gr", directory.c_str(), i,j);
        Graph<Point3D, EdgeW<Point3D> >* orig =
          new Graph<Point3D, EdgeW<Point3D> >(origName);
        Graph<Point3Dw, EdgeW<Point3Dw> >* dest =
          addWidthToGraph(scales, orig);
        dest->saveToFile(destName);
      }
    }
  }

}
