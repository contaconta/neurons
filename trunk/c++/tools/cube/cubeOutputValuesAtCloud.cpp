#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <argp.h>
#include "CubeFactory.h"
#include "Cube.h"
#include "Cloud.h"

int main(int argc, char **argv) {

  printf("There is a lot of hardcoding in here\n");

  if(argc!=4){
    printf("Usage: cubeOutputValuesIncloud cube cloud output\n");
    exit(0);
  }

  Cube<float, double>* cb = new Cube<float, double>(argv[1]);
//   Cube<uchar, ulong >* cb = new Cube<uchar, ulong>(argv[1]);

  Cloud<Point3Dot>*     cl = new Cloud<Point3Dot>(argv[2]);
  std::ofstream out(argv[3]);
  
  int idx, idy, idz;
  for(int i = 0; i < cl->points.size(); i++){
    cb->micrometersToIndexes3
      (cl->points[i]->coords[0], cl->points[i]->coords[1], cl->points[i]->coords[2],
       idx, idy, idz);
    if( (idx > 0) && (idx < cb->cubeWidth) &&
        (idy > 0) && (idy < cb->cubeHeight) &&
        (idz > 0) && (idz < cb->cubeDepth) ){
      Point3Dot* pt = dynamic_cast<Point3Dot*>(cl->points[i]);
      out << pt->type << " "
          << (float)cb->at(idx, idy, idz) << std::endl;
    }
  }

  out.close();


}
