
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
#include <sstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "cv.h"
#include "highgui.h"
#include "Image.h"
#include "polynomial.h"
#include "SteerableFilter2D.h"
#include <gsl/gsl_rng.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_blas.h>
#include <argp.h>

#include "SteerableFilter2DMultiScale.h"
#include "SteerableFilter2DMultiScaleHermite.h"
#include "SteerableFilter2DMultiScaleOrthogonal.h"
#include "SteerableFilter2DMultiScaleNormalized.h"

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "generatePoints 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "From an image and a cloud of points, outputs either the 'steerable coefficients' or the 'hermite coefficients'";

/* A description of the arguments we accept. */
static char args_doc[] = "<image> <order> <sigma_start> <sigma_end> <sigma_step> <pointCloud> <output_coords>";

/* The options we understand. */
static struct argp_option options[] = {
  {"hermite",   'h',  0, 0, "if defined, outputs the hermite coeficients"},
  {"orthogonal",'o',  0, 0, "if defined, outputs the orthogonal coeficients"},
  {"normalized",'n',  0, 0, "if defined, outputs the orthogonal coeficients"},
  { 0 }
};

struct arguments
{
  string image;
  int order;
  float sigma_start;
  float sigma_end;
  float sigma_step;
  string pointCloud;
  string output_filename;
  bool hermite;
  bool orthogonal;
  bool normalized;
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
    case 'h':
      argments->hermite = true;
      break;
    case 'o':
      argments->orthogonal = true;
      break;
    case 'n':
      argments->normalized = true;
      break;

    case ARGP_KEY_ARG:
      if(state->arg_num == 0){
        argments->image = arg;
      }
      if(state->arg_num == 1){
        argments->order = atoi(arg);
      }
      if(state->arg_num == 2){
        argments->sigma_start = atof(arg);
      }
      if(state->arg_num == 3){
        argments->sigma_end = atof(arg);
      }
      if(state->arg_num == 4){
        argments->sigma_step = atof(arg);
      }
      if(state->arg_num == 5){
        argments->pointCloud = arg;
      }
      if(state->arg_num == 6){
        argments->output_filename = arg;
      }
      if (state->arg_num >= 7)
      /* Too many arguments. */
        argp_usage (state);
      break;

    case ARGP_KEY_END:
      /* Not enough arguments. */
      if (state->arg_num < 1)
        argp_usage (state);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

/* Our argp parser. */
static struct argp argp = { options, parse_opt, args_doc, doc };

int main(int argc, char **argv) {

  struct arguments arguments;
  arguments.hermite = false;
  arguments.orthogonal = false;

  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  printf("Arguments: \n  image: %s\n  order: %i\n  sigma_start: %f\n"
         "  sigma_end: %f\n  sigma_step: %f\n  pointCloud: %s\n  outputFile: %s\n",
         arguments.image.c_str(), arguments.order, arguments.sigma_start,
         arguments.sigma_end, arguments.sigma_step, arguments.pointCloud.c_str(),
         arguments.output_filename.c_str());


  SteerableFilter2DMultiScale* stf;

  if(arguments.hermite){
      stf = new SteerableFilter2DMultiScaleHermite
        (arguments.image, arguments.order, arguments.sigma_start,
         arguments.sigma_end, arguments.sigma_step);
      stf->outputCoordinates(arguments.pointCloud,arguments.output_filename);
      exit(0);
  }
  if(arguments.orthogonal){
    stf = new SteerableFilter2DMultiScaleOrthogonal
      (arguments.image, arguments.order, arguments.sigma_start,
       arguments.sigma_end, arguments.sigma_step);
    stf->outputCoordinates(arguments.pointCloud,arguments.output_filename);
    exit(0);
  }
  if(arguments.normalized){
    stf = new SteerableFilter2DMultiScaleNormalized
      (arguments.image, arguments.order, arguments.sigma_start,
       arguments.sigma_end, arguments.sigma_step);
    stf->outputCoordinates(arguments.pointCloud,arguments.output_filename);
    exit(0);
  }

  stf = new SteerableFilter2DMultiScale
    (arguments.image, arguments.order, arguments.sigma_start,
     arguments.sigma_end, arguments.sigma_step);
  stf->outputCoordinates(arguments.pointCloud,arguments.output_filename);

  // if(atoi(argv[8]) >= 1){
  // }
  // else {
    // stf->outputCoordinatesAllOrientations(argv[6],argv[7]);
  // }
}

