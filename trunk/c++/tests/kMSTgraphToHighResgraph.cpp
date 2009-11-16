
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
#include "Cube.h"
#include "Graph.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: kMSTgraphToHighResgraph directory output\n");
    exit(0);
  }

  string directory(argv[1]);
  string outGraph (argv[2]);

  Cube<float, double>* px = new Cube<float,double>(directory + "/2/merged_248p.nfo");
  Graph<Point3D, EdgeW<Point3D> >* highRes =
    new Graph<Point3D, EdgeW<Point3D> >();

  Graph<Point3D, EdgeW<Point3D> >* mst =
    new Graph<Point3D, EdgeW<Point3D> >(directory+"/tree/mstFromCptGraph.gr");


  Cube<int, double>* idx = new Cube<int, double>(px->cubeWidth, px->cubeHeight,
                                                 px->cubeDepth, 1, 1, 1);
  idx->put_all(-1);
  int x0, x1, y0, y1, z0, z1, a0, a1,e0, e1, l0, l1;
  float px0, px1, probEdge, dist;
  float ratioY =  px->voxelHeight/px->voxelWidth;
  float ratioZ =  px->voxelDepth /px->voxelWidth;


  //Puts the soma as the point 0
  highRes->cloud->points.push_back(mst->cloud->points[0]);
  px->micrometersToIndexes3(mst->cloud->points[0]->coords[0],
                            mst->cloud->points[0]->coords[1],
                            mst->cloud->points[0]->coords[2], x0, y0, z0);
  idx->put(x0,y0,z0,0);
  char buff[1024];
  Graph< Point3D, EdgeW< Point3D > >* path;

  for(int ne = 0; ne < mst->eset.edges.size(); ne++){
    a0 = mst->eset.edges[ne]->p0;
    a1 = mst->eset.edges[ne]->p1;
    //Find the path
    sprintf(buff, "%s/tree/paths_%04i_%04i.gr", a0, a1);
    if(fileExists(buff)){
      path = new Graph< Point3D, EdgeW< Point3D > >(buff);
      printf("processing %s\n", buff);
    } else {
      sprintf(buff, "%s/tree/paths_%04i_%04i.gr", a1, a0);
      if(fileExists(buff)){
        path = new Graph< Point3D, EdgeW< Point3D > >(buff);
        printf("processing %s\n", buff);
      }else continue;
    }

    // Here comes the fun
    for(int nep = 0; nep < path->eset.edges.size(); nep++){
      e0 = path->eset.edges[nep]->p0;
      e1 = path->eset.edges[nep]->p1;
      px->micrometersToIndexes3(path->cloud->points[e0]->coords[0],
                                path->cloud->points[e0]->coords[1],
                                path->cloud->points[e0]->coords[2],
                                x0, y0, z0);
      if(idx->at(x0,y0,z0)==-1){
        l0 = highRes->cloud->points.size();
        highRes->cloud->points.push_back
          (new Point3D
           (path->cloud->points[e0]->coords[0],
            path->cloud->points[e0]->coords[1],
            path->cloud->points[e0]->coords[2]) );
      } else {
        l0 = idx->at(x0,y0,z0);
      }
      px->micrometersToIndexes3(path->cloud->points[e1]->coords[0],
                                path->cloud->points[e1]->coords[1],
                                path->cloud->points[e1]->coords[2],
                                x1, y1, z1);
      if(idx->at(x1,y1,z1)==-1){
        l1 = highRes->cloud->points.size();
        highRes->cloud->points.push_back
          (new Point3D
           (path->cloud->points[e1]->coords[0],
            path->cloud->points[e1]->coords[1],
            path->cloud->points[e1]->coords[2]) );
      } else {
        l1 = idx->at(x1,y1,z1);
      }
      px0 = px->at(x0,y0,z0);
      px1 = px->at(x1,y1,z1);
      dist = sqrt((double) (x0-x1)*(x0-x1) +
                  ratioY*ratioY*(y0-y1)*(y0-y1) +
                  ratioZ*ratioZ*(z0-z1)*(z0-z1));

      if(fabs(px1-px0) < 1e-4) probEdge = -dist*log10(px1);
      else probEdge = fabs(dist*((log10(px1) * px1 - px1- log10(px0) * px0 + px0) /
                                 (-px0 + px1)));
      highRes->eset.edges.push_back
        (new EdgeW<Point3D>
         (&highRes->cloud->points, l0, l1,
          probEdge)
         );
    } // edges in the path



  } //All the edges
  highRes->saveToFile(outGraph);

}
