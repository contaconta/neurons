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
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "cv.h"
#include "highgui.h"
#include "Cube.h"
#include "polynomial.h"
#include "utils.h"
#include "SteerableFilter2DM.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include "Configuration.h"
#include "Image.h"

// #include "Torch3.h"
#include "MatDataSet.h"
#include "TwoClassFormat.h"
#include "ClassMeasurer.h"
#include "MSEMeasurer.h"
#include "QCTrainer.h"
#include "CmdLine.h"
#include "Random.h"
#include "SVMRegression.h"
#include "SVMClassification.h"
#include "KFold.h"
#include "DiskXFile.h"
#include "ClassFormatDataSet.h"
#include "MeanVarNorm.h"
#include "Timer.h"


#ifdef WITH_OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace Torch;


void printStuffFromSvm(SVM* svm)
{
  printf("The SVM has %i support vectors and %i in the bound \n",
         svm->n_support_vectors,
         svm->n_support_vectors_bound);

  int nalph = 0;
  //  printf("A %f\n", (float)svm->n_alpha);
  for(int i = 0; i < svm->n_support_vectors; i++)
    {
      if(abs(svm->sv_alpha[i]) > 1e-3)
        nalph++;
    }

  printf("The number of alphas != 0 is %i\n", nalph);

}

double getResponseSVMSF
(SVM* svm,
 vector<float>& coords)
{
  // First is to get the derivative coordinates rotated
  //  vector< double > coords = stf->getDerivativeCoordinatesRotated(x,y,theta);

  //Creates a sequence with it
  int sz = coords.size();
  float** frames;
  frames = (float**)malloc(sizeof(float*));
  frames[0] = (float*)malloc(sz*sizeof(float));
  for(int i = 0; i < coords.size(); i++)
    frames[0][i] = coords[i];
  Sequence* seq = new Sequence(frames, 1, coords.size());
  // delete &coords;

  // Calculates the response
  svm->forward(seq);

  // Returns the value
  return svm->outputs->frames[0][0];
}



int main(int argc, char **argv) {

  string nameImage(argv[1]);

  
  //Parameters of the simulation for the drive dataset 
  bool   includeOddOrders  = false;
  bool   includeEvenOrders = true;
  bool   includeOrder0     = false;
  float  sigmaImgs  = 2.0;
  string directory = getDirectoryFromPath(nameImage);
  string xFile =
    "/home/ggonzale/mount/cvlabfiler/drive/training/svm_1.000e+01_1.000e-02.svm";
  double sk     = 1e-02;
  double C      = 10;
  


  /*
  //Paramters for road images
  bool   includeOddOrders  = true;
  bool   includeEvenOrders = true;
  bool   includeOrder0     = false;
  string directory = getDirectoryFromPath(nameImage);
  float  sigmaImgs  = 3.0;
  string xFile =
    "/home/ggonzale/mount/cvlabfiler/roads/training/svm_1.000e+01_1.000e+01.svm";

  double sk     = 1e+01;
  double C      = 10;
  */


  string imageName  =
    nameImage;
  string outputName  =
    directory + "/out.jpg";
  string imageThetaN =
    directory + "/theta.jpg";
  if(!fileExists(imageThetaN)){
    Image<float>* image = new Image<float>(imageName);
    // image->computeHessian(sigmaImgs, directory+"/l1.jpg", directory + "/l2.jpg",
                          // true, directory + "theta.jpg");
  }
  string imageHessianN =
    directory + "/l1.jpg";

  double sigma  = 1.0/(sk*sk);

  SteerableFilter2DM* stf =
    new SteerableFilter2DM(imageName, 4, sigmaImgs, outputName,
                           includeOddOrders, includeEvenOrders, includeOrder0);
  stf->result->put_all(0.0);

  //  int xInit = 575;
  //int yInit = 525;
  //int xEnd  = 650;
  //int yEnd  = 549;

  //  int xInit = 489;
  //  int yInit = 509;
  //  int xEnd  = 769;
  //  int yEnd  = 765;


  int xInit = 0;
  int yInit = 0;
  int xEnd  = stf->image->width;
  int yEnd  = stf->image->height;


  Image< float >* theta   = new Image<float>(imageThetaN);
  Image< float >* hessian = new Image<float>(imageHessianN);

  int nSupportVectors = 0;
  int dimensionOfSupportVectors = 0;

  Allocator *allocator = new Allocator;
  SVM *svm = NULL;
  Kernel *kernel = NULL;
  kernel = new(allocator) GaussianKernel((double)sigma);
  svm = new(allocator) SVMClassification(kernel);
  DiskXFile* model = new(allocator) DiskXFile(xFile.c_str(),"r");
  svm->loadXFile(model);
  svm->setROption("C", C);
  svm->setROption("cache size", 100);

  nSupportVectors           = svm->n_support_vectors;
  dimensionOfSupportVectors = svm->sv_sequences[0]->frame_size;
  vector< vector< double > > svectors =
    allocateMatrix(nSupportVectors, dimensionOfSupportVectors);
  vector< double > alphas(nSupportVectors);
  for(int i = 0; i < nSupportVectors; i++){
    alphas[i] = svm->sv_alpha[i];
    for(int j = 0; j < dimensionOfSupportVectors; j++){
      svectors[i][j] = svm->sv_sequences[i]->frames[0][j];
    }
  }

  //saveMatrix(svectors, "supportVectors.txt");
  //saveVectorDouble(alphas, "alphas.txt");

  //  svectors.resize(1);
  //alphas.resize(1);
  //for(int i = 0; i < svectors[0].size(); i++)
  //  svectors[0][i] = 1;
  //alphas[0] = 1;

  saveMatrix(svectors, "supportVectors.txt");
  saveVectorDouble(alphas, "alphas.txt");



  printf("CubeName = %s\nxFile = %s\nsigma = %f\n"
         "C = %f\nstdv = %f\nCubeTheta = %s\n"
         "cubeAguet = %s\noutputName = %s\n",
         imageName.c_str(), xFile.c_str(), sigmaImgs, C, sigma, imageThetaN.c_str(),
         imageHessianN.c_str(), outputName.c_str());
   printf("Computing between [%i,%i] and [%i,%i]\n",
         xInit, yInit, xEnd, yEnd);

#pragma omp parallel for
  for(int y  = yInit; y < yEnd; y++){
    printf("#"); fflush(stdout);
      for(int x = xInit; x < xEnd; x++){
        // printf("[%i,%i,%i] %f\n", x, y, z, cubeAguet->at(x,y,z));
        double res = 0;
        double expn = 0;
        //       if(hessian->at(x,y) > 0){
        vector< float > coords =
          stf->getDerivativeCoordinatesRotated
          (x,y, theta->at(x,y));
        if(0){
          res = 0;
          expn = 0;
          for(int i = 0; i < alphas.size(); i++){
            expn = 0;
            for(int j = 0; j < svectors[i].size(); j++)
              expn -= (svectors[i][j] - coords[j])* (svectors[i][j] - coords[j]);
            res += alphas[i]*exp(expn*sigma);
          }
        }
        else
          {
            res = getResponseSVMSF(svm, coords);
          }
        stf->result->put(x,y, res);
          //} //if > 0
      } //X
    }//Y
  printf("]\n");
  stf->result->save();
  printf("And out!\n");
}
