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
// @param row1, col1 : coordinates of the top left corner of the box
// @param row2, col2 : coordinates of the bottom right corner of the box
int BoxIntegral(unsigned int *data, int width, int height,
                         int row1, int col1, int row2, int col2);

double BoxIntegral(double *data, int width, int height,
                   int row1, int col1, int row2, int col2) ;


// Compute rectangle feature from the integral image
// @param width and height are the width and the height of the image passed in parameter
// @param weak_learner_param : string of the form "[WB]_ax%day%dbx%dby%d_B_ax%day%dbx%dby%d"
// a_________     -> x
// |         |    |
// |         |    y
// |         |
// |         |
// |_________b
int getRectangleFeature(unsigned int *intImg, int width, int height, char* weak_learner_param);

double getRectangleFeature(double *intImg, int width, int height, char* weak_learner_param);

// Compute rectangle feature from raw image
// @param width and height are the width and the height of the image passed in parameter
// @param weak_learner_param : string of the form "[WB]_ax%day%dbx%dby%d_B_ax%day%dbx%dby%d"
// a_________     -> x
// |         |    |
// |         |    y
// |         |
// |         |
// |_________b
int getRectangleFeatureFromRawImage(unsigned char *pImage, int width, int height, char* weak_learner_param);

#endif //INTEGRAL_H
