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
#include "Image.h"
#include "polynomial.h"
#include "SteerableFilter2D.h"
#include "SteerableFilter2DMultiScale.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>

#include "Torch3.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace Torch;


void printStuffFromSvm(SVM* svm)
{
  printf("The SVM has %i support vectors and %i in the bound \n",
         svm->n_support_vectors,
         svm->n_support_vectors_bound);
  // svm->checkSupportVectors();

  int nalph = 0;
  printf("A %f\n", svm->n_alpha);
  for(int i = 0; i < svm->n_alpha; i++)
    {
      if(abs(svm->sv_alpha[i]) > 1e-3)
        nalph++;
    }

  // printf("The SVM has %i alphas \n",
         // svm->n_support_vectors,
         // svm->n_support_vectors_bound);

  
  printf("The number of alphas != 0 is %i\n", nalph);

}


double getResponseSVMSF
(SVM* svm,
 SteerableFilter2DMultiScale* stf,
 int x, int y, float theta)
{
  // First is to get the derivative coordinates rotated
  vector< double > coords = stf->getDerivativeCoordinatesRotated(x,y,theta);

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


  if(argc!=10){
    printf("Usage: steerImageMultiScaleTorch image xFile sigma_start sigma_end sigma_step mask C sigma_kernel order\n");
    exit(0);
  }


  Allocator *allocator = new Allocator;
  SVM *svm = NULL;
  Kernel *kernel = NULL;
  double stdv = atof(argv[8]);
  kernel = new(allocator) GaussianKernel((double)1.0/(stdv*stdv));
  svm = new(allocator) SVMClassification(kernel);

  DiskXFile* model = new(allocator) DiskXFile(argv[2],"r");

  svm->loadXFile(model);
  svm->setROption("C", atof(argv[7]));
  svm->setROption("cache size", 200);

  printStuffFromSvm(svm);

  SteerableFilter2DMultiScale* stf =
    new SteerableFilter2DMultiScale(argv[1], atoi(argv[9]), atof(argv[3]),
                                  atof(argv[4]),atof(argv[5]));

  vector< Image<float>* > responses;
  string filename = argv[1];

  string directory = filename.substr(0,filename.find_last_of("/\\")+1);
  stf->result->put_all(0.0);
  stf->orientation->put_all(0.0);

  char buff[512];

  for(int angle = 0; angle < 180; angle+=20){
    printf("Angle = %i ", angle);
    int idx= angle/20;
    sprintf(buff, "%s/resp_%i.jpg", directory.c_str(), angle);
    Image<float>* new_img = stf->result->create_blank_image_float(buff);
    responses.push_back(new_img);

    responses[idx]->put_all(0.0);
    printf("[");
#ifdef _OPENMP
#pragma omp parallel for
#endif

    // for(int x = stf->result->width/2; x < stf->result->width; x+=2){
      // for(int y =  stf->result->height/2; y < stf->result->height; y++){
      // printf("[%i,%i] = %f\n",x,0,responses[idx]->at(x,0));
      // for(int y =  stf->result->height/2; y < stf->result->height/2+8; y+=2){
    for(int x = 0; x < stf->result->width; x++){
      printf("#"); fflush(stdout);
      for(int y  = 0; y < stf->result->height; y++){
        // responses[idx]->put(x,y, stf->response(float(angle*M_PI)/180, x, y));
        responses[idx]->put(x,y,
                            getResponseSVMSF(svm, stf, x, y,  float(angle*M_PI)/180 ));
                            // stf->response(float(angle*M_PI)/180, x, y));
        if( responses[idx]->at(x,y) > stf->result->at(x,y) ){
          stf->result->put(x,y,responses[idx]->at(x,y));
          stf->orientation->put(x,y,angle*M_PI/180);
        }
      }
    }
    printf("]\n");
    responses[idx]->save();
  }
  stf->result->save();
  stf->orientation->save();
  for(int i = 0; i < responses.size(); i++)
    responses[i]->save();
}
