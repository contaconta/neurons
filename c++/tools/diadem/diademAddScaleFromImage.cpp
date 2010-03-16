
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
#include "Image.h"
#include "Cube.h"

using namespace std;

int main(int argc, char **argv) {
  if(argc!= 5){
    printf("Usage: diademAddScaleFromImage cloud.cl image cube_to_translate out.cl\n");
    exit(0);
  }

  Cloud<Point3D>*  orig = new Cloud<Point3D>(argv[1]);
  Image<float>*  scales = new Image<float>  (argv[2]);
  CubeF*             cb = new CubeF        (argv[3]);
  Cloud<Point3Dw>* dest = new Cloud<Point3Dw>();

  int x, y, z, nP;
  for(nP = 0; nP < orig->points.size(); nP++){
    cb->micrometersToIndexes3
      (orig->points[nP]->coords[0], orig->points[nP]->coords[1],  orig->points[nP]->coords[2],
       x, y, z);
    dest->points.push_back
      (new Point3Dw(orig->points[nP]->coords[0],
                    orig->points[nP]->coords[1],
                    orig->points[nP]->coords[2],
                    scales->at(x,y)));

  }
  dest->saveToFile(argv[4]);

}
