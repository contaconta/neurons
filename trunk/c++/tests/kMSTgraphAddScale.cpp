
////////////////////////////////////////////////////////////////////////////////
// This program is free software; you can redistribute it and/or              //
// modify it under the terms of the GNU General Public License                //
// version 2 as published by the Free Software Foundation.                    //
//                                                                            //
// This program is distributed in the hope that it will be useful, but        //
// WITHOUT ANY WARRANTY; without even the implied warranty of                 //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU          //
// General Public License for more details.                                   //
//                                                                            //
// Written and (C) by German Gonzalez Serrano                                 //
// Contact < german.gonzalez@epfl.ch > for comments & bug reports             //
////////////////////////////////////////////////////////////////////////////////

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
    printf("Usage: kMSTgraphAddScale graph directory\n");
    exit(0);
  }

  string directory(argv[2]);
  string graphName(argv[1]);

  Graph<Point3D, EdgeW<Point3D> >* orig =
    new Graph<Point3D, EdgeW<Point3D> >(graphName);
  vector<Cube<float, double>*> detections(4);
  detections[0] = new Cube<float, double>(directory + "/1/bf_2_3162_f.nfo");
  detections[1] = new Cube<float, double>(directory + "/2/bf_2_3162_f.nfo");
  detections[2] = new Cube<float, double>(directory + "/4/bf_2_3162_f.nfo");
  detections[3] = new Cube<float, double>(directory + "/8/bf_2_3162_f.nfo");

  vector<double> scales(4);
  scales[0] = 0.5;
  scales[1] = 1.0;
  scales[2] = 2.0;
  scales[3] = 3.0;

  string outGraph = getDirectoryFromPath(graphName) + "/" +
    getNameFromPathWithoutExtension(graphName) + "w.gr";
  Graph<Point3Dw, EdgeW<Point3Dw> >* dest =
    new Graph<Point3Dw, EdgeW<Point3Dw> >();

  //Add the points
  int ix, iy, iz;
  int maxScale;
  float maxValue = -1e8;
  for(int nP = 0; nP < orig->cloud->points.size(); nP++){
    Point3D* pt = dynamic_cast<Point3D*>(orig->cloud->points[nP]);
    maxScale = 0;
    maxValue = -1e8;
    for(int nC = 0; nC < 4; nC++){
      detections[nC]->micrometersToIndexes3
        (pt->coords[0], pt->coords[1], pt->coords[2], ix, iy, iz);
      if(detections[nC]->at(ix, iy, iz) > maxValue){
        maxValue = detections[nC]->at(ix, iy, iz);
        maxScale = nC;
      }
    }
    dest->cloud->points.push_back
      (new Point3Dw(pt->coords[0], pt->coords[1], pt->coords[2], scales[maxScale]));
  }

  //Adds the edges
  for(int i = 0; i < orig->eset.edges.size(); i++){
    EdgeW<Point3D>* e = dynamic_cast< EdgeW<Point3D>*>(orig->eset.edges[i]);
    dest->eset.edges.push_back
      (new EdgeW<Point3Dw>(&dest->cloud->points, e->p0, e->p1, e->w) );
  }

  dest->saveToFile(outGraph);

}
