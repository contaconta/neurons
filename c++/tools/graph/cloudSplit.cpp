

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

using namespace std;


int main(int argc, char **argv) {

  if(argc!=4){
    printf("Use: cloudSplit orig cloud1 cloud2\n");
    exit(0);
  }

  printf("Not general for any type of cloud, only for Point2Dot\n");

  Cloud<Point2Dot>* orig = new Cloud<Point2Dot>(argv[1]);
  Cloud<Point2Dot>* dest1 = new Cloud<Point2Dot>();
  Cloud<Point2Dot>* dest2 = new Cloud<Point2Dot>();

  for(int i = 0; i < orig->points.size(); i++){
    if(i%2==0)
      dest1->points.push_back(orig->points[i]);
    else
      dest2->points.push_back(orig->points[i]);
  }

  printf("The original cloud has %i points, dest1 has %i and dest2 has %i\n",
         orig->points.size(), dest1->points.size(), dest2->points.size());

  dest1->saveToFile(argv[2]);
  dest2->saveToFile(argv[3]);


}
