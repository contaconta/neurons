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

#include <algorithm>
#include <iostream>
#include "integral.h"
#include "utils.h"

using namespace std;

unsigned int *Integral(unsigned char *img, int width, int height)
{
  unsigned int *int_img = new unsigned int[width*height];
  int step = width;

  // first row
  int rs = 0;
  for(int j=0; j<width; j++) 
  {
    rs += img[j*height]; 
    int_img[j] = rs;
  }

  // remaining cells are sum above and to the left
  for(int i=1; i<height; ++i) 
  {
    rs = 0;
    for(int j=0; j<width; ++j) 
    {
      rs += img[j*height+i];
      int_img[i*step+j] = rs + int_img[(i-1)*step+j];
    }
  }

  // return the integral image
  return int_img;
}

unsigned int BoxIntegral(unsigned int *data, int width, int height, int row, int col, int rows, int cols) 
{
  // The subtraction by one for row/col is because row/col is inclusive.
  int r1 = std::min(row,          height) - 1;
  int c1 = std::min(col,          width)  - 1;
  int r2 = std::min(row + rows,   height) - 1;
  int c2 = std::min(col + cols,   width)  - 1;

  float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
  if (r1 >= 0 && c1 >= 0) A = data[r1 * width + c1];
  if (r1 >= 0 && c2 >= 0) B = data[r1 * width + c2];
  if (r2 >= 0 && c1 >= 0) C = data[r2 * width + c1];
  if (r2 >= 0 && c2 >= 0) D = data[r2 * width + c2];

  return std::max(0.f, A - B - C + D);
}

unsigned int RectangleFeature(unsigned int *data, int width, int height, int row, int col, int rows, int cols, eFeatureType featureType)
{
  unsigned int val = 0;
  switch(featureType)
    {
    case TOP_BLACK:
      val = BoxIntegral(data, width, height, row, col+cols/2, rows, cols/2) - BoxIntegral(data, width, height, row, col, rows, cols/2);
      break;
    case BOTTOM_BLACK:
      val =  BoxIntegral(data, width, height, row, col, rows, cols/2) - BoxIntegral(data, width, height, row, col+cols/2, rows, cols/2);
      break;
    }
  return val;
}

/*
unsigned int BoxIntegral(unsigned int *data, int width, int height, int row1, int col1, int row2, int col2) 
{
  // The subtraction by one for row/col is because row/col is inclusive.
  int r1 = std::min(row1,   height) - 1;
  int c1 = std::min(col1,   width)  - 1;
  int r2 = std::min(row2,   height) - 1;
  int c2 = std::min(col2,   width)  - 1;

  float A(0.0f), B(0.0f), C(0.0f), D(0.0f);
  if (r1 >= 0 && c1 >= 0) A = data[r1 * width + c1];
  if (r1 >= 0 && c2 >= 0) B = data[r1 * width + c2];
  if (r2 >= 0 && c1 >= 0) C = data[r2 * width + c1];
  if (r2 >= 0 && c2 >= 0) D = data[r2 * width + c2];

  return std::max(0.f, A - B - C + D);
}

unsigned int RectangleFeature(unsigned int *data, int width, int height, char* weak_learner_param)
{
  int row1, col1, row2, col2;
  unsigned int val = 0;
  switch(featureType)
    {
    case TOP_BLACK:
      val = BoxIntegral(data, width, height, row, col+cols/2, rows, cols/2) - BoxIntegral(data, width, height, row, col, rows, cols/2);
      break;
    case BOTTOM_BLACK:
      val =  BoxIntegral(data, width, height, row, col, rows, cols/2) - BoxIntegral(data, width, height, row, col+cols/2, rows, cols/2);
      break;
    }
  return val;
}
*/
