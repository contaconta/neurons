
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
#include "IntegralImage.h"

using namespace std;

void changeComponent
(
 Image<float> * image,
 int componenetToChange,
 int newComponenet
 )
{
  for(int y = 0; y < image->height; y++)
    for(int x = 0; x < image->width; x++)
      if(image->at(x,y) == componenetToChange)
        image->put(x,y, newComponenet);
}



int main(int argc, char **argv) {
  if(argc!=3){
    printf("Usage: imageFindBorders image image_borders\n");
    exit(0);
  }

  string nameImg(argv[1]);
  string nameBorders(argv[2]);

  printf("Loading image\n");
  Image<float> * img  = new Image<float>(nameImg, true);
  Image<float> * res  = img->create_blank_image_float(nameBorders);
  // printf("Computing the integral image\n");
  // IntegralImage* iimg = new IntegralImage(img);

  int numberLabels = 0;
  int currLabel    = 0;
  int x = 0;
  int y = 0;
  int currVal = img->at(0,0);

  //First step, go through the lines re-using labels
  //first row
  for(x = 1; x < img->width; x++){
    if( fabs(currVal - img->at(x,y)) > 10){
      currVal = img->at(x,y);
      res->put(x,y, ++numberLabels);
      currLabel = numberLabels;
    }
    else {
      res->put(x,y,currLabel);
    }
  }

  //Rest of lines
  for(y = 1; y < img->height; y++){

    //x = 0 is a particular case, it does not have a predeccesor
    x = 0;
    currVal = img->at(x,y);
    if( fabs(currVal - img->at(x,y-1)) < 10){
      currLabel = res->at(x,y-1);
    } else {
      currLabel = ++numberLabels;
    }
    res->put(x,y,currLabel);

    // now for the rest
    for(x = 1; x < img->width; x++){
      // if it is the same as the pixel on the left, keep it
      if( fabs(currVal - img->at(x,y)) < 10){
        res->put(x,y, currLabel);
      } else {
        // if it is the same as the above pixel, take the label of the above
        if( fabs(img->at(x,y) - img->at(x,y-1)) < 10) {
          currLabel = res->at(x,y-1);
          currVal   = img->at(x,y-1);
          res->put(x,y,currLabel);
        } else { // new label
          currVal = img->at(x,y);
          res->put(x,y, ++numberLabels);
          currLabel = numberLabels;
        }
      }
    }
  }


  //And now we should merge the components
  for(int nC = 0; nC <= numberLabels; nC++){
    printf("Merging componenet %i\r", nC); fflush(stdout);
    for(int y = 1; y < img->height-1; y++){
      for(int x = 1; x < img->width-1; x++){
        if(res->at(x,y) == nC){
          //Check for the 4-neighbors if they are from other components
          // and have the same value
          //Left
          if( (res->at(x-1,y) != nC) &&
              (fabs(img->at(x-1,y)-img->at(x,y)) < 10)
              )
            changeComponent(res, res->at(x-1,y), nC);
          //Right
          if( (res->at(x+1,y) != nC) &&
              (fabs(img->at(x+1,y)-img->at(x,y)) < 10)
              )
            changeComponent(res, res->at(x+1,y), nC);
          //Top
          if( (res->at(x,y-1) != nC) &&
              (fabs(img->at(x,y-1)-img->at(x,y)) < 10)
              )
            changeComponent(res, res->at(x,y-1), nC);
          //Bottom
          if( (res->at(x,y+1) != nC) &&
              (fabs(img->at(x,y+1)-img->at(x,y)) < 10)
              )
            changeComponent(res, res->at(x,y+1), nC);

        }//nC
      }//x
    }//y
  }//nC





  res->save();

  printf("done ...\n");

}
