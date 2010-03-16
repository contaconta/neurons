
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
// Written and (C) by German Gonzalez                                  //
// Contact <german.gonzalez@epfl.ch> for comments & bug reports        //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "SWC.h"
#include "Image.h"

//Let's do it in opencv format (easier)

using namespace std;

void renderSphereInImage
(IplImage* img, int xs, int ys, float radious, int r, int g, int b)
{
  for(int x = max((float)0.0, (float)xs-radious);
      x < min((float)img->width, (float)xs+radious);
      x++){
    for(int y = max((float)0.0, (float)ys-radious);
        y < min((float)img->height, (float)ys+radious);
        y++){
      if( sqrt((x-xs)*(x-xs)+(y-ys)*(y-ys)) < radious ){
        CvScalar s;
        s=cvGet2D(img,y,x); // get the (i,j) pixel value
        s.val[0]=r;
        s.val[1]=g;
        s.val[2]=b;
        cvSet2D(img,y,x,s); // set the (i,j) pixel value
      }//if
    }//y
  }//x
}//render

void renderLineInImage
(IplImage* img, int x0, int y0, int x1, int y1, int r, int g, int b)
{
  if(abs(x1-x0) >= abs(y1-y0)){
    float my = ((float)(y1-y0))/(x1-x0);
    float y  = 0;
    int   xt, yt;
    if(min(x0,x1) == x0)
      y = y0;
    else
      y = y1;
    for(int x = min(x0,x1); x <= max(x0,x1); x++)
      {
        if(x < 0) xt = 0; else if(x >= img->width ) xt = img->width -1; else xt=x;
        if(y < 0) yt = 0; else if(y >= img->height) yt = img->height-1; else yt=y;
        CvScalar s;
        s=cvGet2D(img,yt,xt); // get the (i,j) pixel value
        s.val[0]=r;
        s.val[1]=g;
        s.val[2]=b;
        cvSet2D(img,yt,xt,s); // set the (i,j) pixel value
        y += my;
      }
  } //go in the x
  else {
    float mx = ((float)(x1-x0))/(y1-y0);
    float x  = 0;
    int   yt,xt;
    if(min(y0,y1) == y0)
      x = x0;
    else
      x = x1;
    for(int y = min(y0,y1); y <= max(y0,y1); y++)
      {
        if(x < 0) xt = 0; else if(x >= img->width ) xt = img->width -1; else xt=x;
        if(y < 0) yt = 0; else if(y >= img->height) yt = img->height-1; else yt=y;
        CvScalar s;
        s=cvGet2D(img,yt,xt); // get the (i,j) pixel value
        s.val[0]=r;
        s.val[1]=g;
        s.val[2]=b;
        cvSet2D(img,yt,xt,s); // set the (i,j) pixel value
        x += mx;
      }
  }//go in the y
}


int main(int argc, char **argv) {

  if(argc!=4){
    printf("Usage: imageRenderSWC image swc imageOut\n");
    exit(0);
  }

  string imageName(argv[1]);
  string swcName(argv[2]);
  string outName(argv[3]);

  IplImage* original = cvLoadImage(imageName.c_str(),0);
  SWC*      swc      = new SWC(swcName);
  IplImage* dest     = cvLoadImage(imageName.c_str(),1);

  for(int i = 0; i < swc->allPoints->points.size(); i++){
    renderSphereInImage(dest,
                        swc->allPoints->points[i]->coords[0],
                        swc->allPoints->points[i]->coords[1],
                        4.0,
                        255,0,0);
  }

  //Render the edges
  for(int i = 0; i < swc->gr->eset.edges.size(); i++){
    int p0 = swc->gr->eset.edges[i]->p0;
    int p1 = swc->gr->eset.edges[i]->p1;
    renderLineInImage(dest,
                      swc->allPoints->points[p0]->coords[0],
                      swc->allPoints->points[p0]->coords[1],
                      swc->allPoints->points[p1]->coords[0],
                      swc->allPoints->points[p1]->coords[1],
                      0,0,255);
  }



  for(int i = 0; i < swc->soma->points.size(); i++){
    renderSphereInImage(dest,
                        swc->soma->points[i]->coords[0],
                        swc->soma->points[i]->coords[1],
                        8.0,
                        0,255,255);
  }



  printf("dest has %i channels\n", dest->nChannels);

  cvSaveImage(outName.c_str(), dest);
}
