
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
#include "IntegralImage.h"
#include "Graph.h"

using namespace std;

int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: imageFindSoma img soma.gr\n");
    exit(0);
  }

  string nameImg(argv[1]);
  string nameGraph(argv[2]);

  Image<float> * img  = new Image<float>(nameImg);
  IntegralImage* iimg = new IntegralImage(img);

  // Image<float>* res1 = img->create_blank_image_float("/media/neurons/findSoma/n7s1.png");
  // Image<float>* res2 = res1->create_blank_image_float("/media/neurons/findSoma/n7s2.png");
  // Image<float>* res  = res2->create_blank_image_float("/media/neurons/findSoma/n7s.png");

  //And now finds the soma (just a simple integral over 50x50 px
  int step = 30;
  // res1->put_all(255);
  // res2->put_all(255);
  // res ->put_all(255);
  int x1, y1, xS, yS;
  float value;
  float minValue = 255;

  printf("Doing the first pass\n");
  //Now the displacement will be included into the computation
  for(int disp = 0; disp < step; disp+=5){
    printf("disp = %i\n", disp);
    //Initial search
    for(int y0 = disp; y0 < img->height - step; y0+=step){
      for(int x0 = disp; x0 < img->width - step; x0+=step){
        value = iimg->integral(x0,y0,x0+step,y0+step)/(step*step);
        if(value < minValue){
          minValue = value;
          xS = x0;
          yS = y0;
        }
        // for(int i = x0; i <= x0+step; i++)
          // for(int j = y0; j <= y0+step; j++)
            // res1->put(i,j,value);
      }
    }
  }

  // printf("Saving the result\n");
  // for(int x = xS; x < xS+step; x++)
    // for(int y = yS; y < yS+step; y++){
      // res->put(x,y,0);
    // }

  printf("Saving the images\n");
  // iimg->save();
  // res1->save();
  // res2->save();
  // res->save();

  //And now the graph with the contour will be saved
  printf("Saving the graph\n");
  Graph<Point2Do, EdgeW<Point2Do> >* gr =
    new Graph<Point2Do, EdgeW<Point2Do> >();
  int x, y;
  y = yS;
  vector<float> smicrom(2);
  vector<int> sindex(2);
  int stepContour = 3;
  sindex[0] = xS; sindex[1]=yS;
  img->indexesToMicrometers(sindex, smicrom);
  for(x = 0; x <= step; x+=stepContour){
    gr->cloud->points.push_back
      (new Point2Do(smicrom[0]+x, smicrom[1], M_PI/2));
  }
  for(y = 1; y <= step-1; y+=stepContour){
    gr->cloud->points.push_back
      (new Point2Do(smicrom[0]+step, smicrom[1]-y, 0));
  }
  for(x = step; x>=0; x-=stepContour){
    gr->cloud->points.push_back
      (new Point2Do(smicrom[0]+x, smicrom[1]-step, M_PI/2));
  }
  for(y = step-1; y>=1; y-=stepContour){
    gr->cloud->points.push_back
      (new Point2Do(smicrom[0], smicrom[1]-y, 0));
  }
  for(int i = 0; i < gr->cloud->points.size()-1; i++)
    gr->eset.edges.push_back
      (new EdgeW<Point2Do>(&gr->cloud->points, i, i+1, 1));
  gr->eset.edges.push_back
    (new EdgeW<Point2Do>(&gr->cloud->points, 0, gr->cloud->points.size()-1, 1));
  gr->v_radius = 0.2;
  gr->cloud->v_radius = 0.2;
  gr->saveToFile(nameGraph);

}
