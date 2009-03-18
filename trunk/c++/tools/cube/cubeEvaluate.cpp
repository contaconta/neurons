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


void getStatisticsForThreshold
(Cube<float, double>* cube,
 Cube<uchar, ulong> * gt  ,
 float thres,
 float& TPR, float& FPR)
{
  int tp = 0;
  int fp = 0;
  int tn = 0;
  int fn = 0;
  for(int z = 2; z < cube->cubeDepth -2; z++){
    for(int y = 2; y < cube->cubeHeight-2; y++){
      for(int x = 2; x < cube->cubeWidth-2; x++){
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
  TPR = (float)tp/(tp + fn);
  FPR = float(fp)/(fp+tn);
}

void getROCLinear
(Cube<float, double>* cube,
 Cube<uchar, ulong> * gt  ,
 vector< float >& TPR,
 vector< float >& FPR,
 int nSteps,
 vector< float >& thresholds
 )
{
  float cubeMin, cubeMax;
  cube->min_max(&cubeMin, &cubeMax);
  float step = (cubeMax - cubeMin)/nSteps;
  float tpr, fpr;

  for(float threshold = cubeMin;
      threshold <= cubeMax + step;
      threshold += step)
    {
      getStatisticsForThreshold(cube, gt, threshold, 
                                tpr, fpr);
      TPR.push_back(tpr);
      FPR.push_back(fpr);
      thresholds.push_back(threshold);
    }
}

void getThresholdAndTPRForFPR
(Cube<float, double>* cube,
 Cube<uchar, ulong> * gt  ,
 float  targetFPR,
 float  max_search,
 float  min_search,
 float& TPR,
 float& threshold,
 float  precission = 1e-4,
 int    maxIterations = 20)
{
  float HIGH = max_search;
  float LOW  = min_search;
  float tpr, fpr;
  float THRES = (HIGH+LOW)/2;
  int   iterations = 0;

  while( (fabs(HIGH-LOW) > precission) &&
         (iterations < maxIterations) )
    {
      THRES = (HIGH+LOW)/2;
      getStatisticsForThreshold(cube, gt, THRES, tpr, fpr);
      if( fpr > targetFPR)
        LOW = THRES;
      else
        HIGH = THRES;
      iterations++;
      printf(" ->iteration %02i: HIGH = %f, LOW = %f, TPR = %f, FPR = %f\n",
             iterations, HIGH, LOW, tpr, fpr);
    }
  threshold = (HIGH+LOW)/2;
  TPR       = tpr;
}

void getROCEvenlyLinear
(Cube<float, double>* cube,
 Cube<uchar, ulong> * gt  ,
 vector< float >& TPR,
 vector< float >& FPR,
 int nSteps,
 vector< float >& thresholds
 )
{
  float cubeMin, cubeMax, thr, tpr;
  cube->min_max(&cubeMin, &cubeMax);
  TPR.push_back(0);
  FPR.push_back(0);
  thresholds.push_back(cubeMax);
  for(float fpr = 1.0/nSteps; fpr < 1; fpr+= 1.0/nSteps)
    {
      getThresholdAndTPRForFPR( cube, gt, fpr, thresholds[thresholds.size() -1],
                                cubeMin, tpr, thr);
      FPR.push_back(fpr);
      TPR.push_back(tpr);
      thresholds.push_back(thr);
    }
  TPR.push_back(1.0);
  FPR.push_back(1.0);
  thresholds.push_back(cubeMin);
}

void getROCEvenlyLog
(Cube<float, double>* cube,
 Cube<uchar, ulong> * gt  ,
 vector< float >& TPR,
 vector< float >& FPR,
 int nSteps,
 vector< float >& thresholds,
 float loglimitlow
 )
{
  float cubeMin, cubeMax, thr, tpr;
  cube->min_max(&cubeMin, &cubeMax);
  TPR.push_back(0);
  FPR.push_back(0);
  thresholds.push_back(cubeMax);
  for(float fprl = loglimitlow; fprl < 0; fprl+= (-loglimitlow)/nSteps)
    {
      printf("fprl: %f  fpr: %f\n", fprl, pow(10,fprl));
      getThresholdAndTPRForFPR( cube, gt, pow(10,fprl), thresholds[thresholds.size() -1],
                                cubeMin, tpr, thr);
      FPR.push_back(pow(10,fprl));
      TPR.push_back(tpr);
      thresholds.push_back(thr);
    }
  TPR.push_back(1.0);
  FPR.push_back(1.0);
  thresholds.push_back(cubeMin);
}




int main(int argc, char **argv) {

  if(argc!=3){
    printf("Usage: cubeEvaluate cube groundTruth\n");
    exit(0);
  }

  Cube<float, double>* cube        = new Cube<float, double>(argv[1]);
  Cube<uchar, ulong>*  groundTruth = new Cube<uchar, ulong> (argv[2]);

  /** To get the TPR for a FPR */
  double fpr = 0.001;
  float cubeMin, cubeMax, threshold, tpr;
  cube->min_max(&cubeMin, &cubeMax);
  getThresholdAndTPRForFPR( cube, groundTruth, fpr, cubeMax, cubeMin,  tpr, threshold);
  printf("The FPR id %f, tpr found is  %f and threshold %f %f\n",
         fpr, tpr, threshold);




  /** Test of getROCEvenlyLog ******/
//   vector< float > TPR;
//   vector< float > FPR;
//   vector< float > thresholds;

//   getROCEvenlyLog(cube, groundTruth, TPR, FPR, 9, thresholds, -3);
//   for(int i = 0; i < TPR.size(); i++)
//     printf("Thres: %f, TPR: %f, FPR = %f\n", thresholds[i], TPR[i], FPR[i]);




  /******* Test of getROCEvenlyLinear ******/
//   vector< float > TPR;
//   vector< float > FPR;
//   vector< float > thresholds;

//   getROCEvenlyLinear(cube, groundTruth, TPR, FPR, 10, thresholds);
//   for(int i = 0; i < TPR.size(); i++)
//     printf("Thres: %f, TPR: %f, FPR = %f\n", thresholds[i], TPR[i], FPR[i]);



  //******Test of the getROCLinear **********//
//   vector< float > TPR;
//   vector< float > FPR;
//   vector< float > thresholds;
//   float cubeMin, cubeMax, threshold, tpr;

//   getROCLinear(cube, groundTruth, TPR, FPR, 10, thresholds);

//   for(int i = 0; i < TPR.size(); i++)
//     printf("Thres: %f, TPR: %f, FPR = %f\n", thresholds[i], TPR[i], FPR[i]);

//   //And continues for the getThresholdAndTPRForFPR

//   cube->min_max(&cubeMin, &cubeMax);
//   int idx = 5;
//   getThresholdAndTPRForFPR( cube, groundTruth, FPR[idx], cubeMax, cubeMin,  tpr, threshold);
//   printf("The FPR was %f, with TPR %f and found %f\n"
//          "The threshold was %f and found %f\n",
//          FPR[idx], TPR[idx], tpr,
//          thresholds[idx], threshold);


  //***** Test of the getStatisticsForThreshold *********//
//   float FPR, TPR, cubeMin, cubeMax;
//   cube->min_max(&cubeMin, &cubeMax);
//   float threshold = 5.25;
//   getStatisticsForThreshold(cube, groundTruth, threshold, TPR, FPR);

//   printf("The TPR is %f and the FPR is %f\n", TPR, FPR);
//   printf("The cube has as min %f and as max %f\n", cubeMin, cubeMax);



}
