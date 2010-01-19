
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
#include "CloudFactory.h"
#include "Image.h"
using namespace std;

int main(int argc, char **argv) {

  if(argc != 4){
    printf("Usage: cloudEliminatePointsFromImage cloud image cloudOut\n");
    exit(0);
  }
  Cloud_P* cl = CloudFactory::load(argv[1]);
  Image<float>* img = new Image<float>(argv[2]);
  Cloud_P* dest = CloudFactory::newCloudSameClass(cl);

  for(int i = 0; i < cl->points.size(); i++){
    if(img->atm(cl->points[i]->coords[0], cl->points[i]->coords[1]) > 100)
      dest->points.push_back(cl->points[i]);
  }

  dest->saveToFile(argv[3]);
}
