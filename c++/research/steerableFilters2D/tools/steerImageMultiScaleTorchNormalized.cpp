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
#include <argp.h>

#include "SteerableFilter2DMultiScaleNormalized.h"

#include "Torch3.h"

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace std;
using namespace Torch;


/* Parse a single option. */
const char *argp_program_version =
  "steerImageMultiScaleTorchNormalized 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "given an image and an SVM trained, produces the output image";

/* A description of the arguments we accept. */
static char args_doc[] = "image svmFile response_format output_image output_image_orientation";

/* The options we understand. */
static struct argp_option options[] = {
  {"sigma_start",   's',  "float", 0, "sigma start"},
  {"sigma_end",     'e',  "float", 0, "sigma end"},
  {"sigma_step",    't',  "float", 0, "sigma step"},
  {"C",             'C',  "float", 0, "the C of the svm"},
  {"s_k",           'k',  "float", 0, "the s_k of the svm"},
  {"order",         'o',  "float", 0, "the order of the filter"},
  {"mask",          'm',  "image", 0, "if defined, only evaluate points whose mask is > 100"},
  {"angle",         'a',  "int",   0, "the angular resolution (in degrees)"},
  { 0 }
};

struct arguments
{
  float sigma_start;
  float sigma_end;
  float sigma_step;
  float C;
  float s_k;
  int order;
  int angle;
  string image;
  string svmFile;
  string response_format;
  string output_image;
  string output_image_orientation;
  string mask_name;
};

/* Parse a single option. */
static error_t
parse_opt (int key, char *arg, struct argp_state *state)
{
  /* Get the input argument from argp_parse, which we
     know is a pointer to our arguments structure. */
  struct arguments *argments = (arguments*)state->input;

  switch (key)
    {
    case 's':
      argments->sigma_start = atof(arg);
      break;
    case 'e':
      argments->sigma_end = atof(arg);
      break;
    case 't':
      argments->sigma_step = atof(arg);
      break;
    case 'C':
      argments->C = atof(arg);
      break;
    case 'k':
      argments->s_k = atof(arg);
      break;
    case 'o':
      argments->order = atoi(arg);
      break;
    case 'a':
      argments->angle = atoi(arg);
      break;
    case 'm':
      argments->mask_name = arg;
      break;
    case ARGP_KEY_ARG:
      if(state->arg_num == 0)
        argments->image = arg;
      if(state->arg_num == 1)
        argments->svmFile = arg;
      if(state->arg_num == 2)
        argments->response_format = arg;
      if(state->arg_num == 3)
        argments->output_image = arg;
      if(state->arg_num == 4)
        argments->output_image_orientation = arg;
      if (state->arg_num >= 5)
      /* Too many arguments. */
        argp_usage (state);
      // argments->args[state->arg_num] = arg;
      break;

    case ARGP_KEY_END:
      /* Not enough arguments. */
      if (state->arg_num < 5)
        argp_usage (state);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}



void printStuffFromSvm(SVM* svm)
{
  printf("b is %i\n", svm->b);

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
 SteerableFilter2DMultiScaleNormalized* stf,
 int x, int y, float theta)
{
  // First is to get the derivative coordinates rotated
  vector< double > coords = stf->getDerivativeCoordinatesRotated(x,y,theta);

  //Creates a sequence with it
  int sz = coords.size();
  float** frames;
  frames = (float**)malloc(sizeof(float*));
  frames[0] = (float*)malloc(sz*sizeof(float));
  for(int i = 0; i < sz; i++)
    frames[0][i] = coords[i];
  Sequence* seq = new Sequence(frames, 1, coords.size());

  // Calculates the response
  svm->forward(seq);
  free(frames[0]);
  free(frames);
  // delete &coords;
  // Returns the value
  return svm->outputs->frames[0][0];
}


/* Our argp parser. */
static struct argp argp = { options, parse_opt, args_doc, doc };


int main(int argc, char **argv) {

  struct arguments a;
  a.sigma_start = 2;
  a.sigma_end = 8;
  a.sigma_step = 3;
  a.C = 10;
  a.s_k = 1000;
  a.order = 4;
  a.angle = 30;

  argp_parse (&argp, argc, argv, 0, 0, &a);

  // if(argc!=13){
    // printf("Usage: steerImageMultiScaleTorchHermite image xFile sigma_start sigma_end sigma_step mask C sigma_kernel order s/response_format_i.jpg output_image.jpg output_image_orientation.jpg\n");
    // exit(0);
  // }

  string response_format = a.response_format;

  Allocator *allocator = new Allocator;
  
  SteerableFilter2DMultiScaleNormalized* stf =
    new SteerableFilter2DMultiScaleNormalized(a.image, a.order, a.sigma_start,
                                           a.sigma_end, a.sigma_step,
                                           a.output_image, a.output_image_orientation);

  vector< Image<float>* > responses;
  string filename = a.image;

  string directory = filename.substr(0,filename.find_last_of("/\\")+1);
  stf->result->put_all(0.0);
  stf->orientation->put_all(0.0);

  char buff[512];

  // Image<float>* mask = NULL;
  // if(a.mask_name != "" )
    // mask = new Image<float>(a.mask_name);


  //Initialization of the svm's and openmp
  int nthreads = omp_get_max_threads();
  omp_set_num_threads(nthreads);
  vector< SVM* > svms;
  for(int i = 0; i < nthreads; i++){
    SVM *svm = NULL;
    Kernel *kernel = NULL;
    double stdv = a.s_k;
    kernel = new(allocator) GaussianKernel((double)1.0/(stdv*stdv)); 
    svm = new(allocator) SVMClassification(kernel);
    // svm->b = 0;
    //print("Main: The SVM has a b of %f\n", svm->b);
    //print("Main: loading the model from %s\n", a.svmFile.c_str());
    DiskXFile* model = new(allocator) DiskXFile(a.svmFile.c_str(),"r");
    svm->loadXFile(model);
    svm->setROption("C", a.C);
    svm->setROption("cache size", 200);
    //print("Main2: The SVM has a b of %f\n", svm->b);
    //printStuffFromSvm(svm);
    svms.push_back(svm);
  }



  //Creation of the image results
  for(int angle = 0; angle < 180; angle+=a.angle){
    int idx= angle/a.angle;
    sprintf(buff, response_format.c_str(), directory.c_str(), angle);
    Image<float>* new_img = stf->result->create_blank_image_float(buff);
    responses.push_back(new_img);
    responses[idx]->put_all(0.0);
  }

  for(int angle = 0; angle < 180; angle+=a.angle){
    int idx= angle/a.angle;
#ifdef _OPENMP
#pragma omp parallel for
#endif
    for(int y  = 0; y < stf->result->height; y++){
      printf("Angle: %i :[%i ]\r", angle, (y+1)*100/stf->result->height); fflush(stdout);
      for(int x = 0; x < stf->result->width; x++){
        float result = getResponseSVMSF(svms[omp_get_thread_num()],
                                         stf, x, y,  float(angle*M_PI)/180 );
        responses[idx]->put(x,y,result);
        if(angle == 0){
          stf->result->put(x,y,responses[0]->at(x,y));
        }
        else if( responses[idx]->at(x,y) > stf->result->at(x,y) ){
          stf->result->put(x,y,responses[idx]->at(x,y));
          stf->orientation->put(x,y,angle*M_PI/180);
        }
      }
    }
    responses[idx]->save();
  }
  printf("\n");


  stf->result->save();
  stf->orientation->save();
}
