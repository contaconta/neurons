
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
#include "Cloud_P.h"
#include "CloudFactory.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc<4){
    printf("Usage: cloudCombine cl1.cl cl2.cl .... clout.cl\n");
    exit(0);
  }

//   vector< Cloud_P* > clouds;
//   for(int i = 1; i < argc-1; i++){
//     clouds.push_back( CloudFactory::load(argv[i]));
//   }
//   Cloud_P* result = CloudFactory::newCloudSameClass(clouds[0]);

//   for(int i = 0; i < clouds.size(); i++){
//     for(int j = 0; j < clouds[i]->points.size(); j++){
//       result->points.push_back(clouds[i]->points[j]);
//     }
//   }


  printf("Heavy tuned for the results on the 2D imges, please read the code!\n");
  Cloud<Point2Dotw>*  pos = new Cloud<Point2Dotw>(argv[1]);
  Cloud<Point2Dotw>*  neg = new Cloud<Point2Dotw> (argv[2]);

  Cloud_P* result = new Cloud<Point2Dotw>();

  // for(int i = 0; i < pos->points.size(); i++){
    // Point3Dot* pt = dynamic_cast<Point3Dot*>(pos->points[i]);
    // if(pt->type == 1){
      // result->points.push_back
        // (new Point2Dot(pt->coords[0],
                       // pt->coords[1],
                       // pt->theta,
                       // pt->type));
    // }
  // }
  for(int i = 0; i < pos->points.size(); i++)
    result->points.push_back(pos->points[i]);

  for(int i = 0; i < neg->points.size(); i++)
    result->points.push_back(neg->points[i]);



  result->saveToFile(argv[argc-1]);

}
