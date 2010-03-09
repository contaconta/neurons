
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
#include "Cloud.h"
#include "CubeFactory.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!= 5){
    printf("Usage: diademOutputToCloud outputFile.txt cube.nfo misses.cl extras.cl\n");
    exit(0);
  }


  std::ifstream in(argv[1]);
  Cloud<Point3Dw>* misses = new Cloud<Point3Dw>();
  Cloud<Point3Dw>* extras = new Cloud<Point3Dw>();
  Cube_P* reference = CubeFactory::load(argv[2]);

  char line[1024];

  float score, x0, y0, z0, w0;
  float x, y, z;
  in.getline(line, 1024);
  sscanf(line, "Score: %f\n", &score);
  printf("%f\n", score);

  in.getline(line, 1024); //blank line
  in.getline(line, 1024); //missing nodes

  while(in.getline(line, 1024)){
    if(line[0] != '(')
      break;
    sscanf(line, "(%f,%f,%f) %f\n", &x0, &y0, &z0, &w0);
    reference->indexesToMicrometers3
      (int(x0), int(y0), int(z0), x, y, z);
    misses->points.push_back
      (new Point3Dw(x, y, z, 1.0));
  }

  in.getline(line, 1024);
  while(  in.getline(line, 1024)){
    if(line[0] != '(')
      break;
    sscanf(line, "(%f,%f,%f) %f\n", &x0, &y0, &z0, &w0);
    reference->indexesToMicrometers3
      (int(x0), int(y0), int(z0), x, y, z);
    extras->points.push_back
      (new Point3Dw(x, y, z, 1.0));
  }

  misses->v_r = 1.0;
  misses->v_g = 0.5;
  misses->v_b = 0.0;
  extras->v_r = 1.0;
  extras->v_g = 0.0;
  extras->v_b = 0.5;


  extras->saveToFile(argv[3]);
  misses->saveToFile(argv[4]);

  in.close();
}
