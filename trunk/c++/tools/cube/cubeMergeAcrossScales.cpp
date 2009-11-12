
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
#include "CubeFactory.h"
#ifdef WITH_OPENMP
#include <omp.h>
#endif


using namespace std;

int main(int argc, char **argv) {

  if(argc<4){
    printf("Usage: cubeMergeAcrossScales orig other1 other2 .... out\n");
    exit(0);
  }

  vector< Cube<float, double>* > cubes;
  for(int i = 2; i < argc-1; i++){
    cubes.push_back(new Cube<float, double>(argv[i]));
  }

  // We assume the first cube is the first one
  Cube<float, double>* orig = new Cube<float, double>(argv[1]);
  Cube<float, double>* merged = orig->duplicate(argv[argc-1]);

  printf("Now we are doing the business\n");
#pragma omp parallel for
  for(int z = 0; z < orig->cubeDepth; z++){
    float mx, my, mz;
    int   ix, iy, iz;
    for(int y = 0; y < orig->cubeHeight; y++){
      for(int x = 0; x < orig->cubeWidth; x++){
        orig->indexesToMicrometers3(x,y,z,mx,my,mz);
        for(int i = 0; i < cubes.size(); i++){
          cubes[i]->micrometersToIndexes3(mx,my,mz,ix,iy,iz);
          if( (ix>=0) && (iy>=0) && (iz>=0) &&
              (ix < cubes[i]->cubeWidth) &&
              (iy < cubes[i]->cubeHeight) &&
              (iz < cubes[i]->cubeDepth)){
            if( cubes[i]->at(ix,iy,iz) > merged->at(x,y,z))
              merged->put(x,y,z, cubes[i]->at(ix,iy,iz));
          }
        }//i
      }//X
    }//Y
    printf("#");
  }//Z


}
