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

#include "highgui.h"
#include <stdio.h>
#include "rays.h"

#define PI 3.1415926536

/* function [RAY1 RAY3 RAY4] = rays(E, G, angle, stride)
 * RAYS computes RAY features
 *   Example:
 *   -------------------------
 *   I = imread('cameraman.tif');
 *   SPEDGE = spedge_dist(I,30,2, 11);
 *   imagesc(SPEDGE);  axis image;
 *
 *
 *   FEATURE = spedge_dist(E, G, ANGLE, STRIDE)  computes a spedge 
 *   feature on a grayscale image I at angle ANGLE.  Each pixel in FEATURE 
 *   contains the distance to the nearest edge in direction ANGLE.  Edges 
 *   are computed using Laplacian of Gaussian zero-crossings (!!! in the 
 *   future we may add more methods for generating edges).  SIGMA specifies 
 *   the standard deviation of the edge filter.  
 */
void computeRays(const char *pImageName, double sigma, double angle, IplImage** ray1, int filterType)
{
    /*
    for(j=0;j<10;j++){
      printf("%d\n",pImage[j]);
      //pResult[j] = pIndices[j];
    }
    */
    
    /* Invert dimensions :
       Matlab : height, width
       OpenCV : width, hieght
    */
    /* Create output matrix */
    /*
    number_of_dims=mxGetNumberOfDimensions(prhs[0]);
    dim_array=mxGetDimensions(prhs[0]);
    const mwSize dims[]={dim_array[0],dim_array[1]};
    plhs[0] = mxCreateNumericArray(2,dims,mxUINT32_CLASS,mxREAL);
    pResult = (int*)mxGetData(plhs[0]);
    copyIntegralImage(pImage,dim_array[1],dim_array[0],pResult);
    */

  IplImage* img = cvLoadImage(pImageName,0);
  if(!img)
    {
      printf("Error while loading %s\n",pImageName);
      return;
    }


  //printf("%d %d\n",img->width, img->nChannels);
  
  uchar* ptrImg;
  IplImage* gx = 0;
  IplImage* gy = 0;
  IplImage* g = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U, 1);
  if(filterType == F_SOBEL)
    {
      gx = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_16S, 1);
      gy = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_16S, 1);
      //IplImage* g = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_32F, 1);
      cvSobel(img, gx, 1, 0, 3);
      cvSobel(img, gy, 0, 1, 3);
      //cvSaveImage("gx.png",gx);
      //cvSaveImage("gy.png",gy);

      // compute the gradient norm GN
      int nx = 0;
      int ny = 0;
      int i = 0;
      for(int y=0;y<img->height;y++)
        for(int x=0;x<img->width;x++)
          {
            nx = ((short*)(gx->imageData + gx->widthStep*y))[x*gx->nChannels];
            nx *= nx;
            ny = ((short*)(gy->imageData + gy->widthStep*y))[x*gy->nChannels];
            ny *= ny;
            //ptrImg = (short*)&((short*)(g->imageData + g->widthStep*y))[x*g->nChannels];
            //ptrImg = &(((((float*)g->imageData) + g->widthStep*y)))[x*g->nChannels];
            //*ptrImg = 'a';
            ptrImg = ((uchar*)(g->imageData + g->widthStep*y)) + x; //x*g->nChannels;
            //printf("%d %x\n",i++,ptrImg);
            //*ptrImg = 1;
            //*ptrImg = (uchar)(abs(nx)/4.0);
            *ptrImg = (uchar)(sqrt(nx+ny)/4.0f/1.5f);
          }
    }
  else
    {
      //cvSmooth( gray, edge, CV_BLUR, 3, 3, 0, 0 );
      //cvNot( gray, edge );

      // Run the edge detector on grayscale
      cvCanny(img, g, edge_low_thresh, edge_up_thresh, apertureSize);
    }

  cvSaveImage("g.png",g);

  // ensure good angles
  angle = fmod(angle,360.0);
  if (angle < 0)
    angle += 360;

  // convert to radians
  angle = angle * (PI/180.0);
  
  // get a scamline in direction angle
  list<int> xs;
  list<int> ys;
  linepoints(img->width,img->height,angle,xs,ys);

  //printf("size %d %d\n",xs.size(),ys.size());

  //for(int i = 0;i< xs.size(); i++)
  list<int>::iterator ix = xs.begin();
  list<int>::iterator iy = ys.begin();
  for(;ix != xs.end(); ix++,iy++)
    {
      if((*ix < 0) || (*iy < 0) || (*ix >= img->width) || (*iy >= img->height))
        printf("Warning : %d %d\n",*ix,*iy);
      cvSet2D(img,*iy,*ix,cvScalar(0));
    }
  cvSaveImage("img.png",img);

  printf("ray1\n");

  // initialize the output matrices
  *ray1 = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U, 1);
  cvSet(*ray1, cvScalar(0)); // TODO : DEBUG ONLY !!!
  //IplImage* ray3 = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U, 1);
  //IplImage* ray4 = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U, 1);

  // determine the unit vector in the direction of the Ray
  //rayVector = unitvector(angle);

  list<int> xj;
  list<int> yj;
  int x,y;
  int t;
  int steps_since_edge = 0;  // the border of the image serves as an edge
  uchar* ptrImgRay1;
  // if S touches the top & bottom of the image
  //if (((angle >= 45) && (angle <= 135))  || ((angle >= 225) && (angle <= 315)))
  double m_angle = angle;
  bool scan_left_right = false;
  if( (((float)img->height)/(float)img->width) < tan(m_angle))
    scan_left_right = true;
  if(m_angle > PI/2.0f)
    {
      
      //if(m_angle > 0.75*PI)
      //  m_angle = PI-m_angle;
      //else
        m_angle = m_angle-(PI/2.0f);
      
      if( (((float)img->height)/(float)img->width) < tan(m_angle))
        scan_left_right = false;
      else
        scan_left_right = true;
      /*
      if(m_angle > 0.75*PI)
        m_angle = m_angle-(PI/2.0f);
      else
        m_angle = PI-m_angle;
      */
    }
  printf("tan %f %f %f %f %d\n",m_angle,m_angle*180/PI,tan(m_angle),(((float)img->height)/img->width),
         (float)((float)img->height/(float)img->width) < (float)tan(m_angle));

  //if( (((float)img->height)/(float)img->width) < tan(m_angle))
  if(scan_left_right)
    {
      // scan to the left
      int x_ofs = 0;

      printf("scan to the left\n");
      do
        {
          //for(int i = 0;i < ys.size();i++)
          //for(;ix != xs.end(); ix++,iy++)

          xj.clear();
          yj.clear();
          ix = xs.begin();
          iy = ys.begin();
          for(;ix != xs.end(); ix++,iy++)
            {
              t = *ix + x_ofs;
              //if(t < 0)
              //  break;
              if(t>=0)
                {
                  yj.push_back(*iy);
                  xj.push_back(t);
                }
            }

          //for(int i = 0;i < yj.size();i++)
          ix = xj.begin();
          iy = yj.begin();
          for(;ix != xj.end(); ix++,iy++)
            {
              x = *ix;
              y = *iy;

              //if(x < 0 || y < 0)
              if((*ix < 0) || (*iy < 0) || (*ix >= img->width) || (*iy >= img->height))
                printf("l %d %d\n",x,y);

              ptrImg = ((uchar*)(g->imageData + g->widthStep*y)) + x;
              //if(*ptrImg != 0)
              if(*ptrImg > 30) // threshold edge map
                steps_since_edge = 0;

              ptrImgRay1 = ((uchar*)((*ray1)->imageData + (*ray1)->widthStep*y)) + x;
              *ptrImgRay1 = (uchar)steps_since_edge;

              steps_since_edge++;
            }

          x_ofs--;
        }
      while(yj.size() > 0);

      // scan to the right
      x_ofs = 1;

      printf("scan to the right\n");
      do
        {

          //printf("x_ofs %d",x_ofs);

          //for(int i = 0;i < ys.size();i++)
          //for(;ix != xs.end(); ix++,iy++)
          xj.clear();
          yj.clear();
          ix = xs.begin();
          iy = ys.begin();
          for(;ix != xs.end(); ix++,iy++)
            {
              t = *ix + x_ofs;
              if(t < img->width)
                {
                  yj.push_back(*iy);
                  xj.push_back(t);
                }
            }

          //for(int i = 0;i < yj.size();i++)
          ix = xj.begin();
          iy = yj.begin();
          for(;ix != xj.end(); ix++,iy++)
            {
              x = *ix;
              y = *iy;

              //if(x < 0 || y < 0)
              if((*ix < 0) || (*iy < 0) || (*ix >= img->width) || (*iy >= img->height))
                continue;
                //printf("r %d %d\n",x,y);

              ptrImg = ((uchar*)(g->imageData + g->widthStep*y)) + x;
              if(*ptrImg != 0)
              //if(*ptrImg > 10) // threshold edge map
                steps_since_edge = 0;

              ptrImgRay1 = ((uchar*)((*ray1)->imageData + (*ray1)->widthStep*y)) + x;
              *ptrImgRay1 = (uchar)steps_since_edge;

              steps_since_edge++;
            }

          x_ofs++;
        }
      while(yj.size() > 0);
    }
  else
    {
      // scan to the bottom
      int y_ofs = 0;

      printf("scan to the bottom\n");

      do
        {
          //for(int i = 0;i < ys.size();i++)
          //for(;ix != xs.end(); ix++,iy++)

          //ix = xs.end();
          //ix--;
          //printf("xs %d\n",*ix);
          //iy = ys.end();
          //iy--;
          //printf("ys %d\n",*iy);

          xj.clear();
          yj.clear();
          ix = xs.begin();
          iy = ys.begin();
          //for(;iy != ys.end(); ix++,iy++)
          for(;ix != xs.end(); ix++,iy++)
            {
              t = *iy + y_ofs;
              //if(t < 0)
              //  break;
              if(t>=0)
                {
                  yj.push_back(t);
                  xj.push_back(*ix);
                }
            }

          //for(int i = 0;i < yj.size();i++)
          ix = xj.begin();
          iy = yj.begin();
          for(;ix != xj.end(); ix++,iy++)
            {
              x = *ix;
              y = *iy;

              //if(x < 0 || y < 0)
              if((*ix < 0) || (*iy < 0) || (*ix >= img->width) || (*iy >= img->height))
                printf("b %d %d\n",x,y);

              ptrImg = ((uchar*)(g->imageData + g->widthStep*y)) + x;
              //if(*ptrImg != 0)
              if(*ptrImg > 10) // threshold edge map
                steps_since_edge = 0;

              ptrImgRay1 = ((uchar*)((*ray1)->imageData + (*ray1)->widthStep*y)) + x;
              *ptrImgRay1 = (uchar)steps_since_edge;

              steps_since_edge++;
            }

          y_ofs--;
        }
      while(yj.size() > 0);

      // scan to the top
      y_ofs = 1;

      do
        {
          //for(int i = 0;i < ys.size();i++)
          //for(;ix != xs.end(); ix++,iy++)
          xj.clear();
          yj.clear();
          ix = xs.begin();
          iy = ys.begin();
          for(;iy != ys.end(); ix++,iy++)
            {
              t = *iy + y_ofs;
              //if(t >= img->height)
                //break;
              if(t < img->height)
                {
                  yj.push_back(t);
                  xj.push_back(*ix);
                }
            }

          //for(int i = 0;i < yj.size();i++)
          ix = xj.begin();
          iy = yj.begin();
          for(;ix != xj.end(); ix++,iy++)
            {
              x = *ix;
              y = *iy;

              //if(x < 0 || y < 0)
              if((*ix < 0) || (*iy < 0) || (*ix >= img->width) || (*iy >= img->height))
                printf("t %d %d\n",x,y);

              ptrImg = ((uchar*)(g->imageData + g->widthStep*y)) + x;
              //if(*ptrImg != 0)
              if(*ptrImg > 10) // threshold edge map
                steps_since_edge = 0;

              ptrImgRay1 = ((uchar*)((*ray1)->imageData + (*ray1)->widthStep*y)) + x;
              *ptrImgRay1 = (uchar)steps_since_edge;

              steps_since_edge++;
            }

          y_ofs++;
        }
      while(yj.size() > 0);
    }

  printf("Saving ray1\n");
  cvSaveImage("ray1.png",*ray1);

  printf("Releasing\n");
  //cvReleaseImage(&ray1);
  cvReleaseImage(&img);
  if(gx)
    cvReleaseImage(&gx);
  if(gy)
    cvReleaseImage(&gy);
  cvReleaseImage(&g);
  printf("End release\n");
}

/*
 * defines the points in a line in an image at an arbitrary angle
 */
void linepoints(int img_width, int img_height, double angle, list<int>& xs, list<int>& ys)
{
  /*
  // flip the sign of the angle (matlab y axis points down for images) and
  // convert to radians
if Angle ~= 0
    %angle = deg2rad(Angle);
    angle = deg2rad(360 - Angle);
else
    angle = Angle;
end
  */

  // format the angle so it is between 0 and less than pi/2
  const float EPS = 0.001f;
  if((angle > PI-EPS) && (angle < PI+EPS))
    angle = 0;
  else
    if (angle > PI)
      angle -= PI;

  // find where the line intercepts the edge of the image.  draw a line to
  // this point from (1,1) if 0<=angle<=pi/2.  otherwise pi/2>angle>pi draw 
  // from the upper left corner down.  linex and liney contain the points of 
  // the line
  int start_x;
  int start_y;
  int end_x;
  int end_y;

  if((angle > PI/2-EPS) && (angle < PI/2+EPS))
    {
      // straight line on y axis
      start_x = 0;
      start_y = 0;
      end_x = 0;
      end_y = img_width;

      intline(start_x, end_x, start_y, end_y, xs, ys, img_width, img_height);
    }
  else if ((angle >= 0 ) && (angle <= PI/2.0))
    {
      start_x = 0;
      start_y = 0;
      end_x = img_width-1;
      end_y = img_width*tan(angle);
      if(end_y > 0)
        end_y--;

      intline(start_x, end_x, start_y, end_y, xs, ys, img_width, img_height);
    }
  else
    {
      start_x = 0;
      start_y = img_height-1;
      end_x = img_width-1;
      end_y = img_height-img_height*tan(PI - angle);
      if(end_y > 0)
        end_y--;

      intline(start_x, end_x, start_y, end_y, xs, ys, img_width, img_height);
    }

  printf("xy %f %d %d %d %d\n",angle,start_x, end_x, start_y, end_y);

  // TODO : Ask Kevin
  // if the angle points to quadrant 2 or 3, we need to re-sort the elements 
  // of Sr and Sc so they increase in the direction of the angle
  /*
if (270 <= Angle) || (Angle < 90)
    reverse_inds = length(Sr):-1:1;
    Sr = Sr(reverse_inds);
    Sc = Sc(reverse_inds);
end
  */
}


/*
% intline creates a line between two points
%INTLINE Integer-coordinate line drawing algorithm.
%   [X, Y] = INTLINE(X1, X2, Y1, Y2) computes an
%   approximation to the line segment joining (X1, Y1) and
%   (X2, Y2) with integer coordinates.  X1, X2, Y1, and Y2
%   should be integers.  INTLINE is reversible; that is,
%   INTLINE(X1, X2, Y1, Y2) produces the same results as
%   FLIPUD(INTLINE(X2, X1, Y2, Y1)).
 */
void intline(int x1, int x2, int y1, int y2, list<int>& xs, list<int>& ys,int img_width, int img_height)
{
  int x,y,t;
  int dx = abs(x2 - x1);
  int dy = abs(y2 - y1);

  // Check for degenerate case.
  if ((dx == 0) && (dy == 0))
    {
      xs.push_back(x1);
      ys.push_back(y1);
      return;
    }

  bool flip = false;
  if (dx >= dy)
    {
      if (x1 > x2)
        {
          // Always "draw" from left to right.
          t = x1; x1 = x2; x2 = t;
          t = y1; y1 = y2; y2 = t;
          flip = true;
        }

      double m = (y2 - y1)/(double)(x2 - x1);
      printf("m1 %f %d %d\n",m,x1,x2);
      int y;
      for(int x = x1;x<=x2;x++)
        {
          y = round(y1 + m*(x - x1));
          if(x< 0 || x>=img_width || y < 0 || y >= img_height)
            {
              printf("w %d %d\n",x,y);
              continue; //break;
            }
          xs.push_back(x);
          ys.push_back(y);
        }
    }
  else
    {
      if (y1 > y2)
        {
          // Always "draw" from bottom to top.
          t = x1; x1 = x2; x2 = t;
          t = y1; y1 = y2; y2 = t;
          flip = true;
        }
      double m = (x2 - x1)/(double)(y2 - y1);
      printf("m2 %f %d %d\n",m,y1,y2);
      int x;
      for(int y = y1;y<=y2;y++)
        {
          x = round(x1 + m*(y - y1));
          if(x< 0 || x>=img_width || y < 0 || y >= img_height)
            continue; //break;
          xs.push_back(x);
          ys.push_back(y);
        }
    }
  
  if (flip)
    {
      xs.reverse();
      ys.reverse();
    }
}
