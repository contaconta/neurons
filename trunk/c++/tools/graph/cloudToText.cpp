
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
#include "Image.h"
#include "CloudFactory.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!=4){
    printf("Usage: cloudToText cloud.cl image.jpg out.txt\n");
    exit(0);
  }

  printf("Heavy tuned, read the code!!!!!\n");
  Cloud<Point2Dotw>* cl = new Cloud<Point2Dotw>(argv[1]);
  Image<float>* img = new Image<float>(argv[2], 0);
  std::ofstream out(argv[3]);

  vector< int > idxs(3);
  for(int i = 0; i < cl->points.size(); i++){
    Point2Dotw* pt = dynamic_cast<Point2Dotw*>(cl->points[i]);
    img->micrometersToIndexes(pt->coords, idxs);
    if( (idxs[0] < 0) || (idxs[0] >= img->width ) ||
        (idxs[1] < 0) || (idxs[1] >= img->height) )
      continue;
    out << idxs[0] << " " << idxs[1] << " " <<
      pt->theta << " " <<  pt->w 
        << " " << pt->type << std::endl;
  }

  out.close();


}
