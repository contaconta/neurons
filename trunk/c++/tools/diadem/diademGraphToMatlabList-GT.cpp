
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

using namespace std;

int main(int argc, char **argv) {

  if(argc!= 2){
    printf("Usage: diademGraphToMatlabList graph.gr\n");
    exit(0);
  }

  string graphName(argv[1]);
  string graphPrefix = getPathWithoutExtension(graphName);

  Graph<Point3Dw, Edge<Point3Dw> >* gr =
    new Graph<Point3Dw, Edge<Point3Dw> >(graphName);

  printf("Saving vertices\n");
  string vertexName = graphPrefix + "_vertices.txt";
  std::ofstream out(vertexName.c_str());
  for(int i = 0; i < gr->cloud->points.size(); i++){
    out << gr->cloud->points[i]->coords[0] << " "
        << gr->cloud->points[i]->coords[1] << " "
        << gr->cloud->points[i]->coords[2] << " "
        << std::endl;
    std::cout << gr->cloud->points[i]->coords[0] << " "
        << gr->cloud->points[i]->coords[1] << " "
        << gr->cloud->points[i]->coords[2] << " "
        << std::endl;

  }
  out.close();

  printf("Saving edges\n");
  string edgesName = graphPrefix + "_edges.txt";
  std::ofstream oute(edgesName.c_str());
  for(int i = 0; i < gr->eset.edges.size(); i++){
    Edge<Point3Dw>* ed = dynamic_cast<Edge<Point3Dw>*>(gr->eset.edges[i]);
    oute << ed->p0 << " " << ed->p1 << " " << 1.0 << std::endl;
  }
  oute.close();

}
