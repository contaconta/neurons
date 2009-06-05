
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
// Contact < german.gonzalez@epfl.ch > for comments & bug reports      //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <argp.h>
#include "Graph.h"
#include "Cloud.h"
#include "CubeFactory.h"
#include "CloudFactory.h"
#include "Cloud_P.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!=4){
    printf("Use: cloudAddNoise cloud.cl percentage output.cl\n");
    exit(0);
  }

  Cloud_P* orig    = CloudFactory::load(argv[1]);
  float percentage = atof(argv[2]);
  Cloud_P* dest    = CloudFactory::newCloudWithType(orig);

  for(int i = 0; i < orig->points.size(); i+=outOfHowMany){
    //This is wrong, but still will work
    dest->points.push_back(orig->points[i]);
  }

  dest->saveToFile(argv[3]);
}
