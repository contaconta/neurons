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

#ifndef INTEGRAL_H
#define INTEGRAL_H

// white = positive sign, black = negative sign
enum eFeatureType {TOP_BLACK=0,
                   BOTTOM_BLACK,
                   LEFT_BLACK,
                   RIGHT_BLACK};

// Computes the integral image of image img.  Assumes source image to be a
// gray level image (8 but per pixel)
unsigned int *Integral (unsigned char *img, int width, int height);

// Computes the sum of pixels within the rectangle specified by the top-left start
// co-ordinate and size
unsigned int BoxIntegral(unsigned int *img, int width, int height, int row, int col, int rows, int cols);

// Compute rectangle feature
unsigned int RectangleFeature(unsigned int *data, int width, int height, int row, int col, int rows, int cols, eFeatureType featureType);

#endif //INTEGRAL_H
