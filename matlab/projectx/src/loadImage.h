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

// Compute and copy the integral image to pResult
// @param size_x and size_y are the width and the height of the image
void copyIntegralImage(unsigned char *pImage, int size_x, int size_y, int *pResult);

// Compute the integral image and copy the box integral whose indices are passed as parameter to pResult
// @param size_x and size_y are the width and the height of the image
// @param pIndices : array whose first 2 elements are the (x,y) coordinates of the top-left corner.
// 	The next 2 elements (pIndices[2] and pIndices[3]) are the number of columns and rows.
void getBoxIntegral(unsigned char *pImage, int size_x, int size_y, double *pIndices, unsigned int *pResult);

// Compute the integral image and copy the box integral whose indices are passed as parameter to pResult
// @param size_x and size_y are the width and the height of the image
// @param pIndices : array whose first 2 elements are the (x,y) coordinates of the top-left corner.
// 	The next 2 elements (pIndices[2] and pIndices[3]) are the number of columns and rows.
void getRectangleFeature(unsigned char *pImage, int size_x, int size_y, double *pIndices, unsigned int type, unsigned int *pResult);
