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
#include <fstream>
#include <sstream>
#include <vector>
#include "combnk.h"

#define PI 3.1415926536

// DEBUG ONLY
string getNameFromPathWithoutExtension(string path){
  string nameWith =  path.substr(path.find_last_of("/\\")+1);
  string nameWithout = nameWith.substr(0,nameWith.find_last_of("."));
  return nameWithout;
}

/* Computes RAY features 1,3 and 4 on an image pImageName at a specified angle.
 * @param sigma specifies the standard deviation of the edge filter.
 * @param ray1 is a pointer to the ray responses for the first type of feature (Distance feature)
 * @param ray3 is a pointer to the ray responses for the first type of feature (Norm feature)
 * @param ray4 is a pointer to the ray responses for the first type of feature (Orientation feature)
 * @param filterType specifies the type of edge filter to be use (F_SOBEL or F_CANNY)
 * @param saveImages specifies if the images should be saved (DEBUG mode only)
 * @param edge_low_threshold low threshold used for the canny edge detection
 * @param edge_high_threshold high threshold used for the canny edge detection
 */
void computeDistanceDifferenceRay(const char *pImageName,
                                  int start_angle, int end_angle, int step_angle,
                                  IplImage** ray1, IplImage** ray2)
{
  vector<int> ca;
  for(int a = start_angle;a<=end_angle;a+=step_angle)
    {
      ca.push_back(a);
    }
  vector<int> cb;
  cb.push_back (1);
  cb.push_back (2);

  vector<int> pairs;
  recursive_combination(ca.begin(),ca.end(),0,
                        cb.begin(),cb.end(),0,6-4,pairs);

}

/* Computes RAY features 1,3 and 4 on an image pImageName at a specified angle.
 * @param sigma specifies the standard deviation of the edge filter.
 * @param ray1 is a pointer to the ray responses for the first type of feature (Distance feature)
 * @param ray3 is a pointer to the ray responses for the first type of feature (Norm feature)
 * @param ray4 is a pointer to the ray responses for the first type of feature (Orientation feature)
 * @param filterType specifies the type of edge filter to be use (F_SOBEL or F_CANNY)
 * @param saveImages specifies if the images should be saved (DEBUG mode only)
 * @param edge_low_threshold low threshold used for the canny edge detection
 * @param edge_high_threshold high threshold used for the canny edge detection
 */
void computeRays(const char *pImageName, double sigma, double angle,
                 IplImage** ray1, IplImage** ray3, IplImage** ray4,
                 int filterType, bool saveImages, int edge_low_threshold, int edge_high_threshold)
{
  // control sensitivity to edge detection.
  // 0 means that the ray will stop after hitting the first edge.
  static const uchar threshold_edge_map = 0;

  IplImage* img = cvLoadImage(pImageName,0);
  if(!img)
    {
      printf("Error while loading %s\n",pImageName);
      return;
    }

  uint* ptrGN;
  int lastGN;
  int lastGA;

  // Compute gradient images
  IplImage* edge = 0;
  IplImage* gn = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_32S, 1);
  IplImage* gx = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_16S, 1);
  IplImage* gy = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_16S, 1);
  cvSobel(img, gx, 1, 0, 3);
  cvSobel(img, gy, 0, 1, 3);

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
        ptrGN = ((uint*)(gn->imageData + gn->widthStep*y)) + x; //x*g->nChannels;
        //printf("%d %x\n",i++,ptrImg);
        //*ptrImg = (uchar)(sqrt(nx+ny)/4.0f/1.5f);
        *ptrGN = (uint)(sqrt(nx+ny));
      }

  //if(filterType == F_SOBEL)
  //  edge = gn; //TODO : convert type
  //else
    {
      edge = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_8U, 1);
      //cvSmooth( gray, edge, CV_BLUR, 3, 3, 0, 0 );
      //cvNot( gray, edge );

      // Run the edge detector on grayscale
      cvCanny(img, edge, edge_low_threshold, edge_high_threshold, apertureSize);
    }

    //if(saveImages)
    stringstream sout;
    string s(pImageName);
    sout << "/tmp/" << getNameFromPathWithoutExtension(s) << "edge_" << edge_low_threshold << "_" << edge_high_threshold << ".png";
    cvSaveImage(sout.str().c_str(),edge);

  // ensure good angles
  angle = fmod(angle,360.0);
  if (angle < 0)
    angle += 360;

  // convert to radians
  angle = angle * (PI/180.0);
  
  // get a scanline in direction angle
  list<int> xs;
  list<int> ys;
  list<int>::iterator ix;
  list<int>::iterator iy;
  linepoints(img->width,img->height,angle,xs,ys);

  /*
  // Draw ray on the image
  list<int>::iterator ix = xs.begin();
  list<int>::iterator iy = ys.begin();
  for(;ix != xs.end(); ix++,iy++)
    {
      if((*ix < 0) || (*iy < 0) || (*ix >= img->width) || (*iy >= img->height))
        printf("Warning : %d %d\n",*ix,*iy);
      cvSet2D(img,*iy,*ix,cvScalar(0));
    }
  cvSaveImage("img.png",img);
  */

  // initialize the output matrices
  if(ray1 != 0)
    *ray1 = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_32S, 1);
  if(ray3 != 0)
    *ray3 = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_32S, 1); // TODO : should be a 32 bits image !!!
  if(ray4 != 0)
    *ray4 = cvCreateImage(cvSize(img->width,img->height), IPL_DEPTH_32S, 1); // TODO : should be a 32 bits image !!!

  //cvSet(*ray1, cvScalar(0)); // TODO : DEBUG ONLY !!!
  //cvSet(*ray3, cvScalar(0)); // TODO : DEBUG ONLY !!!
  //cvSet(*ray4, cvScalar(0)); // TODO : DEBUG ONLY !!!

  // determine the unit vector in the direction of the Ray
  float ray_x = sin(angle);
  float ray_y = cos(angle);
  float nxf;
  float nyf;
  float n;

  list<int> xj;
  list<int> yj;
  int x,y;
  int t;
  int steps_since_edge = 0;  // the border of the image serves as an edge
  uint* ptrImgRay1;
  uint* ptrImgRay3;
  uint* ptrImgRay4;
  uchar* ptrEdge;

  if(angle > PI/2.0f)
    angle = PI-angle;

  // if ray touches the top & bottom of the image
  if( (((float)img->height)/(float)img->width) < fabs(tan(angle)))
    {
      // scan to the left
      int x_ofs = 0;

      //printf("scan to the left\n");
      do
        {
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

          ix = xj.begin();
          iy = yj.begin();
          lastGN = 0;
          lastGA = 1;
          steps_since_edge = 0;
          for(;ix != xj.end(); ix++,iy++)
            {
              x = *ix;
              y = *iy;

              if((*ix < 0) || (*iy < 0) || (*ix >= img->width) || (*iy >= img->height))
                printf("warning l %d %d\n",x,y);

              ptrEdge = ((uchar*)(edge->imageData + edge->widthStep*y)) + x;
              if(*ptrEdge > threshold_edge_map) // threshold edge map
                {
                  // reset ray1
                  steps_since_edge = 0;

                  // ray3
                  ptrGN = ((uint*)(gn->imageData + gn->widthStep*y)) + x;
                  lastGN = *ptrGN;
                  
                  // ray 4
                  nx = ((short*)(gx->imageData + gx->widthStep*y))[x*gx->nChannels];
                  ny = ((short*)(gy->imageData + gy->widthStep*y))[x*gy->nChannels];
                  n = nx*nx + ny*ny;
                  nxf = nx/n;
                  nyf = ny/n;
                  lastGA = (nxf * ray_x + nyf * ray_y) * 65536;
                }

              //ptrImgRay1 = ((uchar*)((*ray1)->imageData + (*ray1)->widthStep*y)) + x;
              //*ptrImgRay1 = (uchar)steps_since_edge;

              if(ray1 != 0)
                {
                  ptrImgRay1 = ((uint*)((*ray1)->imageData + (*ray1)->widthStep*y)) + x;
                  *ptrImgRay1 = (uint)steps_since_edge;
                }

              if(ray3 != 0)
                {
                  ptrImgRay3 = ((uint*)((*ray3)->imageData + (*ray3)->widthStep*y)) + x;
                  *ptrImgRay3 = (uint)lastGN;
                }

              if(ray4 != 0)
                {
                  ptrImgRay4 = ((uint*)((*ray4)->imageData + (*ray4)->widthStep*y)) + x;
                  *ptrImgRay4 = (uint)lastGA;
                }

              steps_since_edge++;
            }

          x_ofs--;
        }
      while(yj.size() > 0);

      // scan to the right
      x_ofs = 1;

      //printf("scan to the right\n");
      do
        {
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

          ix = xj.begin();
          iy = yj.begin();
          lastGN = 0;
          lastGA = 1;
          steps_since_edge = 0;
          for(;ix != xj.end(); ix++,iy++)
            {
              x = *ix;
              y = *iy;

              if((*ix < 0) || (*iy < 0) || (*ix >= img->width) || (*iy >= img->height))
                //continue;
                printf("warning r %d %d\n",x,y);

              ptrEdge = ((uchar*)(edge->imageData + edge->widthStep*y)) + x;
              if(*ptrEdge > threshold_edge_map) // threshold edge map
                {
                  // reset ray1
                  steps_since_edge = 0;

                  // ray3
                  ptrGN = ((uint*)(gn->imageData + gn->widthStep*y)) + x;
                  lastGN = *ptrGN;
                  
                  // ray 4
                  nx = ((short*)(gx->imageData + gx->widthStep*y))[x*gx->nChannels];
                  ny = ((short*)(gy->imageData + gy->widthStep*y))[x*gy->nChannels];
                  n = nx*nx + ny*ny;
                  nxf = nx/n;
                  nyf = ny/n;
                  lastGA = (nxf * ray_x + nyf * ray_y) * 65536;
                }

              if(ray1 != 0)
                {
                  ptrImgRay1 = ((uint*)((*ray1)->imageData + (*ray1)->widthStep*y)) + x;
                  *ptrImgRay1 = (uint)steps_since_edge;
                }

              if(ray3 != 0)
                {
                  ptrImgRay3 = ((uint*)((*ray3)->imageData + (*ray3)->widthStep*y)) + x;
                  *ptrImgRay3 = (uint)lastGN;
                }

              if(ray4 != 0)
                {
                  ptrImgRay4 = ((uint*)((*ray4)->imageData + (*ray4)->widthStep*y)) + x;
                  *ptrImgRay4 = (uint)lastGA;
                }

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

      //printf("scan to the bottom\n");

      do
        {
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

          ix = xj.begin();
          iy = yj.begin();
          lastGN = 0;
          lastGA = 1;
          steps_since_edge = 0;
          for(;ix != xj.end(); ix++,iy++)
            {
              x = *ix;
              y = *iy;

              if((*ix < 0) || (*iy < 0) || (*ix >= img->width) || (*iy >= img->height))
                printf("warning b %d %d\n",x,y);

              ptrEdge = ((uchar*)(edge->imageData + edge->widthStep*y)) + x;
              if(*ptrEdge > threshold_edge_map) // threshold edge map
                {
                  // reset ray1
                  steps_since_edge = 0;

                  // ray3
                  ptrGN = ((uint*)(gn->imageData + gn->widthStep*y)) + x;
                  lastGN = *ptrGN;
                  
                  // ray 4
                  nx = ((short*)(gx->imageData + gx->widthStep*y))[x*gx->nChannels];
                  ny = ((short*)(gy->imageData + gy->widthStep*y))[x*gy->nChannels];
                  n = nx*nx + ny*ny;
                  nxf = nx/n;
                  nyf = ny/n;
                  lastGA = (nxf * ray_x + nyf * ray_y) * 65536;
                }

              if(ray1 != 0)
                {
                  ptrImgRay1 = ((uint*)((*ray1)->imageData + (*ray1)->widthStep*y)) + x;
                  *ptrImgRay1 = (uint)steps_since_edge;
                }

              if(ray3 != 0)
                {
                  ptrImgRay3 = ((uint*)((*ray3)->imageData + (*ray3)->widthStep*y)) + x;
                  *ptrImgRay3 = (uint)lastGN;
                }

              if(ray4 != 0)
                {
                  ptrImgRay4 = ((uint*)((*ray4)->imageData + (*ray4)->widthStep*y)) + x;
                  *ptrImgRay4 = (uint)lastGA;
                }

              steps_since_edge++;
            }

          y_ofs--;
        }
      while(yj.size() > 0);

      // scan to the top
      y_ofs = 1;

      do
        {
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
          lastGN = 0;
          lastGA = 1;
          steps_since_edge = 0;
          for(;ix != xj.end(); ix++,iy++)
            {
              x = *ix;
              y = *iy;

              if((*ix < 0) || (*iy < 0) || (*ix >= img->width) || (*iy >= img->height))
                printf("warning t %d %d\n",x,y);

              ptrEdge = ((uchar*)(edge->imageData + edge->widthStep*y)) + x;
              if(*ptrEdge > threshold_edge_map) // threshold edge map
                {
                  // reset ray1
                  steps_since_edge = 0;

                  // ray3
                  ptrGN = ((uint*)(gn->imageData + gn->widthStep*y)) + x;
                  lastGN = *ptrGN;
                  
                  // ray 4
                  nx = ((short*)(gx->imageData + gx->widthStep*y))[x*gx->nChannels];
                  ny = ((short*)(gy->imageData + gy->widthStep*y))[x*gy->nChannels];
                  n = nx*nx + ny*ny;
                  nxf = nx/n;
                  nyf = ny/n;
                  lastGA = (nxf * ray_x + nyf * ray_y) * 65536;
                }

              if(ray1 != 0)
                {
                  ptrImgRay1 = ((uint*)((*ray1)->imageData + (*ray1)->widthStep*y)) + x;
                  *ptrImgRay1 = (uint)steps_since_edge;
                }

              if(ray3 != 0)
                {
                  ptrImgRay3 = ((uint*)((*ray3)->imageData + (*ray3)->widthStep*y)) + x;
                  *ptrImgRay3 = (uint)lastGN;
                }

              if(ray4 != 0)
                {
                  ptrImgRay4 = ((uint*)((*ray4)->imageData + (*ray4)->widthStep*y)) + x;
                  *ptrImgRay4 = (uint)lastGA;
                }

              steps_since_edge++;
            }

          y_ofs++;
        }
      while(yj.size() > 0);
    }

  if(saveImages)
    {
      printf("Saving ray1\n");
      save32bitsimage("ray1.ppm",*ray1);
      printf("Saving ray3\n");
      save32bitsimage("ray3.ppm",*ray3);
      printf("Saving ray4\n");
      save32bitsimage("ray4.ppm",*ray4);
    }

  //printf("Releasing\n");
  //cvReleaseImage(&ray1);
  cvReleaseImage(&img);
  if(gx)
    cvReleaseImage(&gx);
  if(gy)
    cvReleaseImage(&gy);
  cvReleaseImage(&gn);
  //if(filterType == F_CANNY)
  cvReleaseImage(&edge);
  //printf("End release\n");
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

  bool flip = false;

  // format the angle so it is between 0 and less than pi/2
  const float EPS = 0.001f;
  /*
  if((angle > PI-EPS) && (angle < PI+EPS))
    {
      angle = 0;
      flip = true;
    }
  else
  */
    if (angle > PI)
      {
        angle -= PI;
        flip = true;
      }

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
      end_y = img_height-img_width*tan(PI - angle);
      //end_y = img_height-img_height*tan(angle);
      if(end_y > 0)
        end_y--;

      intline(start_x, end_x, start_y, end_y, xs, ys, img_width, img_height);
      flip = !flip;
    }

  if(flip)
    {
      xs.reverse();
      ys.reverse();
    }

  //printf("xy %f %d %d %d %d\n",angle,start_x, end_x, start_y, end_y);

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
      //printf("m1 %f %d %d\n",m,x1,x2);
      int y;
      for(int x = x1;x<=x2;x++)
        {
          y = round(y1 + m*(x - x1));
          if(x< 0 || x>=img_width || y < 0 || y >= img_height)
            {
              //printf("w %d %d\n",x,y);
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
      //printf("m2 %f %d %d\n",m,y1,y2);
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

/*
 * Save a 32 bits image to a binary file
 */
void save32bitsimage(char* filename, IplImage* img)
{
  ofstream ofs(filename, ios::out | ios::binary);

  uint* ptrImg;
  for(int y=0;y<img->height;y++)
    for(int x=0;x<img->width;x++)
      {
        ptrImg = ((uint*)(img->imageData + img->widthStep*y)) + x*(img)->nChannels;
        ofs.write((char*)ptrImg,sizeof(int));
      }
  ofs.close();
}
