
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
#include "Cube.h"
#include "Cloud.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!= 3){
    printf("Usage: checkNonMaxSup cube cloud\n");
    exit(0);
  }

  Cube<float, double>* cube = new Cube<float, double>(argv[1]);
  Cloud<Point3D>* cl = new Cloud<Point3D>(argv[2]);

  int px, py, pz;
  float wx, wy, wz;
  for(int i = 0; i < cl->points.size(); i++){
    cube->micrometersToIndexes3
      (cl->points[i]->coords[0], cl->points[i]->coords[1], cl->points[i]->coords[2],
       px, py, pz);
    printf("Testing for point %i ", i);
    double val = cube->at(px,py,pz);
    bool isNonMax = true;
    for(int z = max(pz-1, 0); z < min(pz+1, (int)cube->cubeDepth); z++){
        for(int y = max(py-1, 0); y < min(py+1,  (int)cube->cubeHeight); y++){
            for(int x = max(px-1, 0); x < min(px+1,  (int)cube->cubeWidth); x++){
                if(cube->at(x,y,z) >= val)
                  isNonMax = false;
              }
          }
      }
    printf("%i\n", isNonMax);
  }



}
