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
#include <string>
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

int getRectangleFeature(unsigned char *pImage, int width, int height, int max_width, int max_height, char* weak_learner_param)
{
  int val = 0;
  int row1, col1, row2, col2;
  // TODO : check returned type
  unsigned int* intImg = Integral(pImage,width,height);

  string params(weak_learner_param);
  int start_idx = params.find_first_of("_")+1;
  int end_idx = params.find_first_of("_",start_idx);
 
  //cout << "start_idx: " << start_idx << endl;
  //cout << "end_idx: " << end_idx << endl;

  bool parseString = true;
  while(parseString)
    {
      if(end_idx == string::npos)
        {
          // Process end of the string
          end_idx = params.length();
          parseString = false;
        }

      string sub_params = params.substr(start_idx,end_idx-start_idx);
      //cout << "sub_params: " << &sub_params.c_str()[1] << endl;

      // Extract row and column numbers
      sscanf(&sub_params.c_str()[1],"ax%day%dbx%dby%d",&col1,&row1,&col2,&row2);

      cout << "row1: " << row1 << " col1:" << col1 << endl;
      cout << "row2: " << row2 << " col2:" << col2 << endl;

      if(sub_params[0] == 'W')
        {
          val += BoxIntegral(intImg, width, height, row1, col1, row2, col2);
        }
      else
        {
          val -= BoxIntegral(intImg, width, height, row1, col1, row2, col2);
        }

      cout << "value: " << val << endl;

      if(parseString)
        {
          start_idx = end_idx+1;
          //params[end_idx] = '*';
          end_idx = params.find_first_of("_",end_idx+1);
        }
    }

  delete[] intImg;

  return val;
}
