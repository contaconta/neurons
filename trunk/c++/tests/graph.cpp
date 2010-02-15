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
// Contact <ggonzale@atenea> for comments & bug reports                //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "CubeFactory.h"
#include "Graph.h"
#include "Point3D.h"
#include "GraphFactory.h"

using namespace std;


int main(int argc, char **argv) {

  Graph<Point3D,Edge<Point3D> >* gr = new Graph<Point3D,Edge<Point3D> >("data/graph.gr");
  gr->saveToFile("data/graph_copy.gr");
  printf("Check manually that the files data/graph.gr and data/graph_copy.gr represent the same graph\n");


  Graph_P* grp = GraphFactory::load("data/graph.gr");
  grp->saveToFile("data/graph_p.gr");
  grp = GraphFactory::load("data/graph_p.gr");
  grp->saveToFile("data/graph_p.gr");
  printf("Check manually that the files data/graph.gr and data/graph_p.gr represent the same graph\n");
  delete gr;
  // delete grp;
}
