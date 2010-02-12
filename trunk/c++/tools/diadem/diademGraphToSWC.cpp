
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
#include "CubeFactory.h"
#include "SWC.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!=4){
    printf("Usage: graphtoSWC graph.gr cubeToTranslate.nfo out.swc\n");
    exit(0);
  }

  Graph<Point3D, EdgeW<Point3D> >* in = new Graph<Point3D, EdgeW<Point3D> >(argv[1]);
  Cube_P* cube        = CubeFactory::load(argv[2]);
  SWC* swc            = new SWC();
  Graph<Point3Dw, Edge<Point3Dw> >* gr = new Graph<Point3Dw, Edge<Point3Dw> >();

  for(int i = 0; i < in->cloud->points.size(); i++){
    //    printf("Adding point %i\n", i);
    int x, y, z;
    cube->micrometersToIndexes3
      (in->cloud->points[i]->coords[0],
       in->cloud->points[i]->coords[1],
       in->cloud->points[i]->coords[2],
       x, y, z);
    gr->cloud->points.push_back(new Point3Dw(x, y, z, 1));
  }

  for(int i = 0; i < in->eset.edges.size(); i++){
    //printf("Adding edge %i\n", i);
    if(in->eset.edges[i]->p0 != in->eset.edges[i]->p1)
      gr->eset.addEdge(in->eset.edges[i]->p0, in->eset.edges[i]->p1);
  }


  swc->gr = gr;
  swc->idxSoma = 0;
  swc->saveToFile(argv[3]);


}
