
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
#include "utils.h"

using namespace std;

void radius_function(Image<float>* img, float r)
{
  int x0 = img->width /2;
  int y0 = img->height/2;

  img->put_all(0);

  for(int x = 0; x < img->width; x++)
    for(int y = 0; y < img->height; y++)
      if( sqrt( (x-x0)*(x-x0) + (y-y0)*(y-y0) ) < r )
        img->put(x, y, 1);
}

void toroid(Image<float>* img, float rbig)
{
  int x0 = img->width /2;
  int y0 = img->height/2;

  float rsmall = rbig/2;
  img->put_all(0);

  float r;
  for(int x = 0; x < img->width; x++)
    for(int y = 0; y < img->height; y++){
      r = sqrt( (x-x0)*(x-x0) + (y-y0)*(y-y0) );
      if( (r > rsmall) && (r < rbig))
        img->put(x, y, 1);
    }
}

void linear_function(Image<float>* img, float rlimit)
{
  int x0 = img->width /2;
  int y0 = img->height/2;

  img->put_all(0);
  float r;

  for(int x = 0; x < img->width; x++)
    for(int y = 0; y < img->height; y++){
      r = sqrt( (x-x0)*(x-x0) + (y-y0)*(y-y0) );
      if( r < rlimit )
        img->put(x, y, r);
    }
}


void linear_function_inv(Image<float>* img, float rlimit)
{
  int x0 = img->width /2;
  int y0 = img->height/2;

  img->put_all(0);
  float r;

  for(int x = 0; x < img->width; x++)
    for(int y = 0; y < img->height; y++){
      r = sqrt( (x-x0)*(x-x0) + (y-y0)*(y-y0) );
      if( r < rlimit )
        img->put(x, y, rlimit -r);
    }
}

void squared_function(Image<float>* img, float rlimit)
{
  int x0 = img->width /2;
  int y0 = img->height/2;

  img->put_all(0);
  float r;

  for(int x = 0; x < img->width; x++)
    for(int y = 0; y < img->height; y++){
      r = sqrt( (x-x0)*(x-x0) + (y-y0)*(y-y0) );
      if( r < rlimit )
        img->put(x, y, r*r);
    }
}

void random_function(Image<float>* img, float rlimit)
{
  int x0 = img->width /2;
  int y0 = img->height/2;

  img->put_all(0);
  float r;

  for(int x = 0; x < img->width; x++)
    for(int y = 0; y < img->height; y++){
      r = sqrt( (x-x0)*(x-x0) + (y-y0)*(y-y0) );
      if( r < rlimit )
        img->put(x, y, -(rlimit - r + r*r/rlimit));
    }
}

void sine_function(Image<float>* img, float rlimit)
{
  int x0 = img->width /2;
  int y0 = img->height/2;

  img->put_all(0);
  float r;

  for(int x = 0; x < img->width; x++)
    for(int y = 0; y < img->height; y++){
      r = sqrt( (x-x0)*(x-x0) + (y-y0)*(y-y0) );
      if( r < rlimit )
        img->put(x, y, sin(M_PI*r/rlimit));
    }
}

void sine2_function(Image<float>* img, float rlimit)
{
  int x0 = img->width /2;
  int y0 = img->height/2;

  img->put_all(0);
  float r;

  for(int x = 0; x < img->width; x++)
    for(int y = 0; y < img->height; y++){
      r = sqrt( (x-x0)*(x-x0) + (y-y0)*(y-y0) );
      if( r < rlimit )
        img->put(x, y, sin(2*M_PI*r/rlimit));
    }
}

void gaussian_function(Image<float>* img, float rlimit)
{
  int x0 = img->width /2;
  int y0 = img->height/2;

  img->put_all(0);
  float r;
  float exponent = 0;

  for(int x = 0; x < img->width; x++)
    for(int y = 0; y < img->height; y++){
      exponent = (x-x0)*(x-x0) + (y-y0)*(y-y0); 
      r = sqrt( exponent );
      if( r < rlimit ){
        img->put(x, y, exp(-exponent/(4*rlimit)));
      }
    }
}



int main(int argc, char **argv) {

  if(argc != 3){
    printf("Usage imageRadius image radius\n");
    exit(0);
  }

  string imageName (argv[1]);
  float radious = atof(argv[2]);
  printf("imageRadius %s, %f\n", imageName.c_str(), radious);

  string outputDir = getDirectoryFromPath(imageName);


  Image<float>* img = new Image<float>(imageName);
  vector<string> outNames(9);
  outNames[0]    = "radius";
  outNames[1]    = "toroid";
  outNames[2]    = "linear";
  outNames[3]    = "linearInv";
  outNames[4]    = "squared";
  outNames[5]    = "random";
  outNames[6]    = "sine";
  outNames[7]    = "sine2";
  outNames[8]    = "gaussian";

  vector< Image< float >* > outputImages(outNames.size());
  vector< string > outputImagesNames(outNames.size());

  for(int i = 0; i < outNames.size(); i++){
    makeDirectory(outputDir + "/" + outNames[i]);
    outputImages[i] = img->create_blank_image_float
      (outNames[i] + "/" + outNames[i] + ".png");

    switch(i){
    case 0:
      radius_function(outputImages[i], radious);
      break;
    case 1:
      toroid(outputImages[i], radious);
      break;
    case 2:
      linear_function(outputImages[i], radious);
      break;
    case 3:
      linear_function_inv(outputImages[i], radious);
      break;
    case 4:
      squared_function(outputImages[i], radious);
      break;
    case 5:
      random_function(outputImages[i], radious);
      break;
    case 6:
      sine_function(outputImages[i], radious);
      break;
    case 7:
      sine2_function(outputImages[i], radious);
      break;
    case 8:
      gaussian_function(outputImages[i], radious);
      break;
    default:
      break;
    }
    outputImages[i]->save();

  }

  // Image<float>* out =
    // img->create_blank_image_float(outputName);

  // radius_function(out, 10.0);
  // toroid(out, 5.00, 10.0);
  // linear_function(out, 10.0, 1);
  // linear_function_inv(out, 10.0);
  // squared_function(out, 10.0);
  // random_function(out, 20.0);

  // out->save();


}
