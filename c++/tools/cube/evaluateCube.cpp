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
// Contact <ggonzale@atenea> for comments & bug reports                //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "Cube.h"
#include <float.h>

using namespace std;

int main(int argc, char **argv) {

  if(argc!=5){
    printf("Usage: cubeEvaluate cube groundTruth file {linear,log}\n");
    exit(0);
  }

  string file = argv[3];

  Cube<uchar, ulong>* test = new Cube<uchar, ulong>(argv[1]);

  if(test->type == "uchar"){

    Cube<uchar, ulong>* cube = new Cube<uchar, ulong>(argv[1]);
    Cube<uchar, ulong>* gt = new Cube<uchar, ulong>(argv[2]);

    //First we will do it with just 10 points to make it fast to compute.
    float max_value, min_value;
    cube->min_max(&min_value, &max_value);
    printf("%f %f\n", min_value,max_value);

    float step = (max_value - min_value)/100;

    std::ofstream out(argv[3]);

    for(float thres = -1; thres < 257; thres = thres + 1){
      int tp = 0;
      int fp = 0;
      int tn = 0;
      int fn = 0;

      for(int z = 8; z < cube->cubeDepth-8; z++){
        for(int y = 20; y < cube->cubeHeight-20; y++){
          for(int x = 20; x < cube->cubeWidth-20; x++){
            if (cube->at(x,y,z) < thres){
              if(gt->at(x,y,z) < 0.5) tp++;
              else fp ++;
            }
            else{
              if(gt->at(x,y,z) < 0.5) fn++;
              else tn ++;
            }
          }
        }
      }
      printf("Thr: %f tpr = %f fpr = %f\n", thres, (float)tp/(tp + fn), float(fp)/(fp+tn) );
      out << (float)tp/(tp + fn) << " " << float(fp)/(fp+tn) << std::endl;
    }
    out.close();
  }
  if(test->type == "float"){

    Cube<float,double>* cube = new Cube<float,double>(argv[1]);
    Cube<uchar, ulong>* gt = new Cube<uchar, ulong>(argv[2]);

    //First we will do it with just 10 points to make it fast to compute.
    float max_value_cube, min_value_cube;
    cube->min_max(&min_value_cube, &max_value_cube);
    printf("%f %f\n", min_value_cube,max_value_cube);

    std::ofstream out(argv[3]);

    float min_value, max_value, step, thres;

    if(argv[4] == "linear") {
      min_value = min_value_cube;
      max_value = max_value_cube;
      step = (max_value - min_value)/100;
    } else {
      min_value = min_value_cube + FLT_MIN;
      max_value = max_value_cube;
      step = (log(max_value) - log(min_value))/100;
      printf("Step = %f\n", step);
    }

    for(int i = 0; i <= 100; i++){
      if(argv[4] == "linear") {
        thres = min_value + i*step;
      } else{
        if(i==0) thres = 0;
        else thres = exp(log(min_value) + i*step);
      }
      int tp = 0;
      int fp = 0;
      int tn = 0;
      int fn = 0;

      for(int z = 8; z < cube->cubeDepth-8; z++){
        for(int y = 20; y < cube->cubeHeight-20; y++){
          for(int x = 20; x < cube->cubeWidth-20; x++){
            if (cube->at(x,y,z) >= thres){
              if(gt->at(x,y,z) < 0.5) tp++;
              else fp ++;
            }
            else{
              if(gt->at(x,y,z) < 0.5) fn++;
              else tn ++;
            }
          }
        }
      }
      printf("Thr: %f tpr = %f fpr = %f\n", thres, (float)tp/(tp + fn), float(fp)/(fp+tn) );
      out << (float)tp/(tp + fn) << " " << float(fp)/(fp+tn) << std::endl;
    }
    out.close();
  }

}
