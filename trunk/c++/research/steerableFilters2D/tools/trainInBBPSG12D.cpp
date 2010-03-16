
////////////////////////////////////////////////////////////////////////////////
// This program is free software; you can redistribute it and/or              //
// modify it under the terms of the GNU General Public License                //
// version 2 as published by the Free Software Foundation.                    //
//                                                                            //
// This program is distributed in the hope that it will be useful, but        //
// WITHOUT ANY WARRANTY; without even the implied warranty of                 //
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU          //
// General Public License for more details.                                   //
//                                                                            //
// Written and (C) by German Gonzalez Serrano                                 //
// Contact < german.gonzalez@epfl.ch > for comments & bug reports             //
////////////////////////////////////////////////////////////////////////////////

#include "Configuration.h"
#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "SteerableFilter2DM.h"
#include "SVMSteerableFeatures3D.h"
#include "utils.h"
#include "Neuron.h"
#include <iomanip>
#include <vector>
#include <gsl/gsl_rng.h>


using namespace std;

int main(int argc, char **argv) {

  
  // Training parameters for the wonderful drive dataset
  bool   includeOddOrders  = false;
  bool   includeEvenOrders = true;
  bool   includeOrder0     = false;
  float  sigmaImages       = 2.0;
  string trainingVolumeS        =
    "/home/ggonzale/mount/cvlabfiler/drive/d21/1/21_training_green.jpg";
  string testingVolumeS         =
    "/home/ggonzale/mount/cvlabfiler/drive/d22/1/22_training_green.jpg";
  string outputDirSimulation    =
    "/home/ggonzale/mount/cvlabfiler/drive/training/";
  string trainingCloudString    =
    "/home/ggonzale/mount/cvlabfiler/drive/d21/21_2500_2500.cl";
  string testingCloudString     =
    "/home/ggonzale/mount/cvlabfiler/drive/d22/22_2500_2500.cl";

  /*
  // Training parameters for roads
  bool   includeOddOrders  = true;
  bool   includeEvenOrders = true;
  bool   includeOrder0     = false;
  float  sigmaImages       = 3.0;
  string trainingVolumeS        =
    "/home/ggonzale/mount/cvlabfiler/roads/m1/1/miami-1.jpg";
  string testingVolumeS         =
    "/home/ggonzale/mount/cvlabfiler/roads/m6/1/miami-6.jpg";
  string outputDirSimulation    =
    "/home/ggonzale/mount/cvlabfiler/roads/training/";
  string trainingCloudString    =
    "/home/ggonzale/mount/cvlabfiler/roads/m1/m1_2500_2500_0.cl";
    //    "/home/ggonzale/mount/cvlabfiler/roads/m1/training.cl";
  string testingCloudString     =
    //    "/home/ggonzale/mount/cvlabfiler/roads/m6/testing.cl";
    "/home/ggonzale/mount/cvlabfiler/roads/m6/m6_2500_2500_0.cl";
  */


  string trainingCoordinatesS   = outputDirSimulation + "/training.txt";
  string testingCoordinatesS    = outputDirSimulation + "/testing.txt";

  // Output the coordinates for the training and test data
  printf("Obtaining the coordinates\n");
  //  printf("The training cloud has %i points\n", trainingCloud->points.size());
  SteerableFilter2DM* stf1
    = new SteerableFilter2DM(trainingVolumeS,
                             4, sigmaImages, "",
                             includeOddOrders, includeEvenOrders, includeOrder0);
  stf1->outputCoordinates(trainingCloudString,
                          trainingCoordinatesS);

  SteerableFilter2DM* stf2
    = new SteerableFilter2DM(testingVolumeS,
                             4,  sigmaImages, "",
                             includeOddOrders, includeEvenOrders, includeOrder0);
  stf2->outputCoordinates(testingCloudString,
                          testingCoordinatesS);

  // Do the training
  SVMSteerableFeatures3D* svmstf3 =
    new SVMSteerableFeatures3D(trainingCoordinatesS, testingCoordinatesS,
                               outputDirSimulation, true);
  double C_result, sk_result, error;
  float C_init  = 1;
  float C_end   = 2;
  float C_step  = 1;
  float sk_init = -4;
  float sk_end  = 6;
  float sk_step = 1.0;

  svmstf3->findCandSGrid(C_result, sk_result, error,
                         C_init, C_end, C_step,
                         sk_init, sk_end, sk_step);

}




