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


#include <ctime>
#include <iostream>
#include "cv.h"
#include "highgui.h"
#include "integral.h"
#include "utils.h"

using namespace std;

//-------------------------------------------------------

int main(void) 
{
  string filename("Images/img1.jpg");
  IplImage *img=cvLoadImage(filename.c_str());
  if(!img)
    cout << "Error while loading " << filename.c_str() << endl;

  // Create a window 
  cvNamedWindow("result", CV_WINDOW_AUTOSIZE );

  IplImage* imgi = 0; //Integral(img);

  // Display the result
  //IplImage* imgi = getGray(imgi2);
  IplImage* gray8 = cvCreateImage( cvGetSize(img), IPL_DEPTH_8U, 1 );
  cvConvertScale( imgi, gray8, 1/255.0, 0 );
  cvShowImage("result", gray8);

  cout << "imgi " << imgi->nChannels << "\n";
  cout << "imgi2 " << gray8->nChannels << "\n";

  uchar* ptrimgi;
  for(int i=0;i<imgi->width;i++)
    {
      for(int j=0;j<imgi->height;j++)
        {
          ptrimgi = &((uchar*)(imgi->imageData + imgi->widthStep*j))[i*imgi->nChannels];
          //cout << "img " << i << " " << j << " " << (int)*ptrimgi << endl;
        }
    }

  while(1)
    {
      // If ESC key pressed exit loop
      if( (cvWaitKey(10) & 255) == 27 )
        break;
    }

  cvDestroyWindow("result");
  cvReleaseImage(&imgi);
  return 0;
}
