
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
#include <gsl/gsl_rng.h>

using namespace std;

int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: kMSTFileToGraph kMSTFile graph.gr\n");
    exit(0);
  }

  const gsl_rng_type * T2;
  gsl_rng * r;
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r = gsl_rng_alloc (T2);


  Graph<Point3D, EdgeW<Point3D> >* gr
    = new Graph<Point3D, EdgeW<Point3D> >();

  gr->v_radius = 0.1;
  gr->cloud->v_radius = 0.1;
  gr->cloud->v_r = 0.0;
  gr->cloud->v_g = 0.0;
  gr->cloud->v_b = 1.0;

  int sizeGrid = 5;

  for(int y = 0; y < sizeGrid; y++)
    for(int x = 0; x < sizeGrid; x++)
      gr->cloud->points.push_back(new Point3D(x,y,0));
  gr->eset.setPointVector(&gr->cloud->points);


  // Create the edges.
  // Horizontal
  for(int y = 0; y < sizeGrid; y++)
    for(int x = 0; x < sizeGrid-1; x++){
      gr->eset.edges.push_back(new EdgeW<Point3D>(&gr->cloud->points,
                                                  y*sizeGrid+x, y*sizeGrid+x+1,
                                                  gsl_rng_uniform(r)) );
    }
  //Vertical
  for(int y = 0; y < sizeGrid-1; y++)
    for(int x = 0; x < sizeGrid; x++){

      gr->eset.edges.push_back(new EdgeW<Point3D>(&gr->cloud->points,
                                                  y*sizeGrid+x, (y+1)*sizeGrid+x,
                                                   gsl_rng_uniform(r)) );
    }

  gr->saveToFile(argv[2]);


  std::ofstream out(argv[1]);
  out << gr->cloud->points.size() << std::endl;
  out << gr->eset.edges.size() << std::endl;
  for(int i = 0; i < gr->eset.edges.size(); i++)
    out << gr->eset.edges[i]->p0 +1 << " " << gr->eset.edges[i]->p1 +1 << " " <<
      (int)(gr->eset.edges[i]->w*255) << std::endl;



}
