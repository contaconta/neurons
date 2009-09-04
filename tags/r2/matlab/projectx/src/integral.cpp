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

inline int BoxIntegral(unsigned int *data, int width, int height, int row1, int col1, int row2, int col2) 
{
  // -1 to adjust rows/cols reflecting origin change from (1,1) to (0,0), and -1 to shift from coordinates to indexes.
  int r1 = row1 - 2;
  int c1 = col1 - 2;
  int r2 = row2 - 2;
  int c2 = col2 - 2;

  int A(0), B(0), C(0), D(0);
  if (r1 >= 0 && c1 >= 0) A = data[r1 * width + c1];
  if (r1 >= 0 && c2 >= 0) B = data[r1 * width + c2];
  if (r2 >= 0 && c1 >= 0) C = data[r2 * width + c1];
  if (r2 >= 0 && c2 >= 0) D = data[r2 * width + c2];

  //cout << "r1: " << r1 << " c1:" << c1 << endl;
  //cout << "r2: " << r2 << " c2:" << c2 << endl;
  //cout << "data: " << A << " " << B << " " << C << " " << D << endl;

  return A - B - C + D;
}

inline double BoxIntegral(double *data, int width, int height, int row1, int col1, int row2, int col2) 
{
  // -1 to adjust rows/cols reflecting origin change from (1,1) to (0,0), and -1 to shift from coordinates to indexes.
  int r1 = row1 - 2;
  int c1 = col1 - 2;
  int r2 = row2 - 2;
  int c2 = col2 - 2;

  double A(0.0), B(0.0), C(0.0), D(0.0);
  if (r1 >= 0 && c1 >= 0) A = data[r1 * width + c1];
  if (r1 >= 0 && c2 >= 0) B = data[r1 * width + c2];
  if (r2 >= 0 && c1 >= 0) C = data[r2 * width + c1];
  if (r2 >= 0 && c2 >= 0) D = data[r2 * width + c2];

  //cout << "r1: " << r1 << " c1:" << c1 << endl;
  //cout << "r2: " << r2 << " c2:" << c2 << endl;
  //cout << "data: " << A << " " << B << " " << C << " " << D << endl;

  return A - B - C + D;
}

int getRectangleFeature(unsigned int *intImg, int width, int height, char* weak_learner_param)
{
  int val = 0;
  int row1, col1, row2, col2;

  string params(weak_learner_param);
  int start_idx = params.find_first_of("_")+1;
  int end_idx = params.find_first_of("_",start_idx);
 
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

      if(sub_params[0] == 'W')
        {
          val += BoxIntegral(intImg, width, height, row1, col1, row2, col2);
        }
      else
        {
          val -= BoxIntegral(intImg, width, height, row1, col1, row2, col2);
        }

      if(parseString)
        {
          start_idx = end_idx+1;
          end_idx = params.find_first_of("_",end_idx+1);
        }
    }

  return val;
}

double getRectangleFeature(double *intImg, int width, int height, char* weak_learner_param)
{
  double val = 0;
  int row1, col1, row2, col2;

  string params(weak_learner_param);
  int start_idx = params.find_first_of("_")+1;
  int end_idx = params.find_first_of("_",start_idx);
 
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

      if(sub_params[0] == 'W')
        {
          val += BoxIntegral(intImg, width, height, row1, col1, row2, col2);
        }
      else
        {
          val -= BoxIntegral(intImg, width, height, row1, col1, row2, col2);
        }

      if(parseString)
        {
          start_idx = end_idx+1;
          end_idx = params.find_first_of("_",end_idx+1);
        }
    }

  return val;
}

int getRectangleFeatureFromRawImage(unsigned char *pImage, int width, int height, char* weak_learner_param)
{
  int val = 0;
  int row1, col1, row2, col2;
  unsigned int* intImg = Integral(pImage,width,height);

  string params(weak_learner_param);
  int start_idx = params.find_first_of("_")+1;
  int end_idx = params.find_first_of("_",start_idx);
 
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

      //cout << "row1: " << row1 << " col1:" << col1 << endl;
      //cout << "row2: " << row2 << " col2:" << col2 << endl;

      if(sub_params[0] == 'W')
        {
          val += BoxIntegral(intImg, width, height, row1, col1, row2, col2);
        }
      else
        {
          val -= BoxIntegral(intImg, width, height, row1, col1, row2, col2);
        }

      if(parseString)
        {
          start_idx = end_idx+1;
          end_idx = params.find_first_of("_",end_idx+1);
        }
    }

  delete[] intImg;

  return val;
}
