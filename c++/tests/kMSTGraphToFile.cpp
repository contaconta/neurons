
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


using namespace std;

int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: kMSTGraphToFile graph.gr file.txt\n");
    exit(0);
  }

  Graph<Point3D, EdgeW<Point3D> >* gr =
    new Graph<Point3D, EdgeW<Point3D> >(argv[1]);

  std::ofstream out(argv[2]);
  out << gr->cloud->points.size() << " " << gr->eset.edges.size() << std::endl;
  // I should add a one to the order of the points
  for(int i = 0; i < gr->eset.edges.size(); i++){
    EdgeW<Point3D>* ed = dynamic_cast<EdgeW<Point3D>*>(gr->eset.edges[i]);
    out << ed->p0 +1 << " " << ed->p1 +1 << " " << ed->w << std::endl;
  }
  out.close();

}
