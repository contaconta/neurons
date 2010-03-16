
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

Graph<Point3Dw, EdgeW<Point3Dw> >*
addWidthToGraph
(vector<Cube<float,double>* >& detections,
 Graph<Point3D, EdgeW<Point3D> >* orig)
{
  Graph<Point3Dw, EdgeW<Point3Dw> >* dest =
    new Graph<Point3Dw, EdgeW<Point3Dw> >();
  vector<double> scales(4);
  scales[0] = 0.5;
  scales[1] = 1.0;
  scales[2] = 1.5;
  scales[3] = 2.5;

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

  return dest;
}




int main(int argc, char **argv) {

  if(argc!=2){
    printf("Usage: kMSTgraphAddScale directory\n");
    exit(0);
  }

  string directory(argv[1]);

  vector<Cube<float, double>*> detections(4);
  detections[0] = new Cube<float, double>(directory + "/1/bf_2_3162_f.nfo");
  detections[1] = new Cube<float, double>(directory + "/2/bf_2_3162_f.nfo");
  detections[2] = new Cube<float, double>(directory + "/4/bf_2_3162_f.nfo");
  detections[3] = new Cube<float, double>(directory + "/8/bf_2_3162_f.nfo");
  char origName[2024];
  char destName[2024];


  //Tree the kmsts
  if(0){
    //#pragma omp parallel for
    for(int i = 1; i <= 969; i++){
      sprintf(origName, "%s/tree/kmsts/kmst%i.gr",  directory.c_str(), i);
      sprintf(destName, "%s/tree/kmsts/kmst%iw.gr", directory.c_str(), i);
      printf("%s\n", destName);
      Graph<Point3D, EdgeW<Point3D> >* orig =
        new Graph<Point3D, EdgeW<Point3D> >(origName);
      Graph<Point3Dw, EdgeW<Point3Dw> >* dest =
        addWidthToGraph(detections, orig);
      dest->saveToFile(destName);
    }
  }

  if(0){
    //#pragma omp parallel for
    for(int i = 1; i <= 969; i++){
      for(int j = 1; j <= 969; j++){
        sprintf(origName, "%s/tree/paths/path_%04i_%04i.gr",  directory.c_str(), i, j);
        if(fileExists(origName)){
          sprintf(destName, "%s/tree/paths/path_%04i_%04iw.gr", directory.c_str(), i,j);
          printf("%s\n", destName);
          Graph<Point3D, EdgeW<Point3D> >* orig =
            new Graph<Point3D, EdgeW<Point3D> >(origName);
          Graph<Point3Dw, EdgeW<Point3Dw> >* dest =
            addWidthToGraph(detections, orig);
          dest->saveToFile(destName);
        }
      }
    }
  }

  //And now for the MST
  if(1){
    sprintf(origName, "%s/tree/mstFromCptGraph.gr",  directory.c_str());
    sprintf(destName, "%s/tree/mstFromCptGraphw.gr",  directory.c_str());
    Graph<Point3D, EdgeW<Point3D> >* orig =
      new Graph<Point3D, EdgeW<Point3D> >(origName);
    Graph<Point3Dw, EdgeW<Point3Dw> >* dest =
      addWidthToGraph(detections, orig);
    dest->saveToFile(destName);
  }

  //And now for the pruned
  if(0){
    sprintf(origName, "%s/tree/mst/mstFromCptGraphPruned.gr",  directory.c_str());
    sprintf(destName, "%s/tree/mst/mstFromCptGraphPrunedw.gr",  directory.c_str());
    Graph<Point3D, EdgeW<Point3D> >* orig =
      new Graph<Point3D, EdgeW<Point3D> >(origName);
    Graph<Point3Dw, EdgeW<Point3Dw> >* dest =
      addWidthToGraph(detections, orig);
    dest->saveToFile(destName);
  }


}
