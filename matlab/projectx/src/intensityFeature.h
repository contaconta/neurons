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
// Written and (C) by Aurelien Lucchi and Kevin Smith                  //
// Contact aurelien.lucchi (at) gmail.com or kevin.smith (at) epfl.ch  // 
// for comments & bug reports                                          //
/////////////////////////////////////////////////////////////////////////

#ifndef INTENSITY_FEATURE_H
#define INTENSITY_FEATURE_H

#include <vector>
#include "Cloud.h"

using namespace std;

struct xImage
{
  unsigned int width;
  unsigned int height;
  unsigned char* data;
};

// Compute intensity feature from an image
// @param width and height are the width and the height of the image passed in parameter
// @param weak_learner_param : string that contains the id of the feature
// a_________     -> x
// |         |    |
// |         |    y
// |         |
// |         |
// |_________b
int getIntensityFeature(unsigned char *test_img,
                        int width, int height,
                        char* weak_learner_param,
                        vector<xImage*>& list_images,
                        vector<Cloud*>& list_clouds,
                        int nbPointsPerCloud);

#endif
