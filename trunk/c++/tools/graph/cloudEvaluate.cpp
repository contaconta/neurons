
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
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <CubeFactory.h>
#include <CloudFactory.h>

using namespace std;

int main(int argc, char **argv) {

  if(argc != 6){
    printf("Usage: cloudEvaluate cloud volume threshold output max_min=1\n");
    exit(0);
  }

  Cloud_P* cl = CloudFactory::load(argv[1]);
  Cube_P*  cube = CubeFactory::load(argv[2]);
  float threshold = atof(argv[3]);
  int minMax      = atoi(argv[5]);

  string pointType = CloudFactory::inferPointType(cl);

  if( !((pointType == "Point3Dot") || (pointType == "Point3Dotw"))){
    printf("The cloud %s is not of Point3Dot or Point3Dotw, exiting\n", argv[1]);
    exit(0);
  }

  Cloud_P* output = CloudFactory::newCloudSameClass(cl);

  int x,y,z;
  for(int i = 0; i < cl->points.size(); i++){
    Point3Dot* pt = dynamic_cast<Point3Dot*>(cl->points[i]);
    cube->micrometersToIndexes3(pt->coords[0], pt->coords[1], pt->coords[2],
                                x, y, z);
    Point3Dot::Type type;

    if( ((pt->type == Point3Dot::TrainingPositive)  &&
         (cube->get(x,y,z) > threshold) &&
          minMax)
        ||
        ((pt->type == Point3Dot::TrainingPositive)  &&
         (cube->get(x,y,z) < threshold) &&
         !minMax)
        )
      type = Point3Dot::TruePositive;

    if( ((pt->type == Point3Dot::TrainingPositive)  &&
         (cube->get(x,y,z) < threshold) &&
         minMax)
        ||
        ((pt->type == Point3Dot::TrainingPositive)  &&
         (cube->get(x,y,z) > threshold) &&
         !minMax)
        )
      type = Point3Dot::FalseNegative;

    if( ((pt->type == Point3Dot::TrainingNegative)  &&
         (cube->get(x,y,z) > threshold) &&
         minMax)
        ||
        ((pt->type == Point3Dot::TrainingNegative)  &&
         (cube->get(x,y,z) < threshold) &&
         !minMax)
        )
      type = Point3Dot::FalsePositive;

    if( ((pt->type == Point3Dot::TrainingNegative)  &&
         (cube->get(x,y,z) < threshold) &&
         minMax)
        ||
        ((pt->type == Point3Dot::TrainingNegative)  &&
         (cube->get(x,y,z) > threshold) &&
         !minMax)
        )
      type = Point3Dot::TrueNegative;

    if(pointType == "Point3Dotw"){
      output->points.push_back
        (new Point3Dotw(pt->coords[0], pt->coords[1], pt->coords[2],
                        pt->theta, pt->phi, type));
    }
    if(pointType == "Point3Dot"){
      output->points.push_back
        (new Point3Dot(pt->coords[0], pt->coords[1], pt->coords[2],
                       pt->theta, pt->phi, type));
    }

  }


  output->saveToFile(argv[4]);
}
