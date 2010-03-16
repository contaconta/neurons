
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

using namespace std;

vector<vector< double > > circularMask(float radius)
{
  vector< vector< double> > mask = allocateMatrix(2*radius+1, 2*radius+1);
  int x0 = radius;
  int y0 = radius;
  for(int x = 0; x < 2*radius+1; x++)
    for(int y = 0; y < 2*radius+1; y++){
      float d = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
      if(d <= radius)
        mask[x][y] = 1;
    }
  return mask;
}


vector<vector< double > > torus(float rmax, float rmin)
{
  vector< vector< double> > mask = allocateMatrix(2*rmax+1, 2*rmax+1);
  int x0 = rmax;
  int y0 = rmax;
  for(int x = 0; x < 2*rmax+1; x++)
    for(int y = 0; y < 2*rmax+1; y++){
      float d = sqrt((x-x0)*(x-x0)+(y-y0)*(y-y0));
      if( (d <= rmax) && (d > rmin))
        mask[x][y] = 1;
        else
          mask[x][y] = -1;
    }
  return mask;
}

vector<vector< double > > dy(){
  vector< vector< double> > mask = allocateMatrix(3, 3);
  mask[0][2] = 1;
  mask[1][2] = 1;
  mask[2][2] = 1;
  mask[0][0] = -1;
  mask[1][0] = -1;
  mask[2][0] = -1;
  return mask;
}


int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: imageConvolution2D image dest\n");
    exit(0);
  }

  string nameImage(argv[1]);
  string nameOut  (argv[2]);

  // vector<vector< double > > mask = circularMask(5);
  vector<vector< double > > mask = torus(3,2);
  // vector<vector< double > > mask = dy();

  saveMatrix(mask, "test.txt");

  Image<float>* img  = new Image<float>(nameImage);
  Image<float>* dest = img->create_blank_image_float(nameOut);

  // img->convolve_2D(dest, mask);
  // dest->save();

}
