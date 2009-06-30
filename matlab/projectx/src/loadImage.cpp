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
#include <ctime>
#include <iostream>
#include "integral.h"

using namespace std;

void copyIntegralImage(unsigned char *pImage, int size_x, int size_y, int *pResult)
{
  unsigned int* imgi = Integral(pImage,size_x,size_y);

  if(pResult != 0)
    {
      int n = 0;
      for(int i=0;i<size_x;i++)
        for(int j=0;j<size_y;j++)      
          {
            pResult[n] = imgi[j*size_x+i];
            n++;
          }
    }

  delete[] imgi;
}

void getBoxIntegral(unsigned char *pImage, int size_x, int size_y, double *pIndices, unsigned int *pResult)
{
  unsigned int* imgi = Integral(pImage,size_x,size_y);  

  *pResult = BoxIntegral(imgi, size_x, size_y, pIndices[0], pIndices[1], pIndices[2], pIndices[3]);
  delete[] imgi;
}
