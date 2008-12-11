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
#include "CubeFactory.h"
#include "Neuron.h"
#include "Cloud.h"
#include <argp.h>
#include <gsl/gsl_rng.h>


using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "imageToCloud 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "imageToCloud creates a cloud with the pixels that are not cero on the image";

/* A description of the arguments we accept. */
static char args_doc[] = "image cloud";

/* The options we understand. */
static struct argp_option options[] = {
  {"theta", 't',  "cube with theta", 0, "if defined, saves the theta orientation in the cloud"},
  {"phi",   'p',  "cube with phi",   0, "if defined, saves the phi orientation in the cloud"},
  {"mask",          'm',  "mask_image",        0, "if true, do not take any point whose mask is 0"},
  {"type",          'y',  0,                   0, "if true saves the type of the points"},
  {"numberPositive", 'N', "positive_points",-  0, "if defined, the number of positive points to get"},
  {"numberNegative", 'M', "negative_points",-  0, "if defined, the number of negative points to get"},
  { 0 }
};

struct arguments
{
  string name_theta;
  string name_phi;
  string name_cube;
  string name_cloud;
  string name_mask;
  int    number_points;
  int    number_negative_points;
  bool   save_type;
  bool   save_negative;
  bool   save_orientation;
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
    case 't':
      argments->name_theta = arg;
      argments->save_orientation = true;
      break;
    case 'p':
      argments->name_phi = arg;
      argments->save_orientation = true;;
      break;
    case 'm':
      argments->name_mask = arg;
      break;
    case 'y':
      argments->save_type = true;
      break;
    case 'N':
      argments->number_points = atoi(arg);
      break;
    case 'M':
      argments->number_negative_points = atoi(arg);
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 2)
      /* Too many arguments. */
        argp_usage (state);
      if (state->arg_num == 0)
        argments->name_cube = arg;
      if (state->arg_num == 1)
        argments->name_cloud = arg;
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

  struct arguments args;
  /* Default values. */
  args.name_cube = "";
  args.name_theta = "";
  args.name_phi = "";
  args.name_mask = "";
  args.name_cloud = "cloud.cl";
  args.save_negative = false;
  args.save_type     = false;
  args.number_points   = 0;
  args.number_negative_points = 0;

  argp_parse (&argp, argc, argv, 0, 0, &args);

  printf("Cube: %s\nTheta: %s\nPhi: %s\nOut: %s\nMsk: %s\nSave_negative: %i\nSave_type: %i\nSave_orientation: %i\nNumber_points: %i\nNumber_negative_points: %i\n",
         args.name_cube.c_str(),args.name_theta.c_str(),
         args.name_phi.c_str(), args.name_cloud.c_str(),
         args.name_mask.c_str(),
         args.save_negative, args.save_type,
         args.save_orientation, args.number_points,
         args.number_negative_points);

  Cube_P* cborig = CubeFactory::load(args.name_cube);
  if (cborig->type == "uchar")
    cborig = dynamic_cast<Cube<uchar,ulong>*>(cborig);
  else if (cborig->type == "float")
    cborig = dynamic_cast<Cube<float,double>*>(cborig);
  else{
    printf("cubeToCloud: the cube is not of floats or uchars, exiting...\n");
    exit(0);
  }

  vector< int > indexes(3);
  vector< float > micrometers(3);
  indexes[2] = 0;

  Cloud_P* cloud;
  if(args.save_orientation && args.save_type)
    cloud = new Cloud< Point3Dot >();
  if(args.save_orientation && !args.save_type)
    cloud = new Cloud< Point3Do >();
  if(!args.save_orientation && args.save_type)
    cloud = new Cloud< Point3Dt >();
  if(!args.save_orientation && !args.save_type)
    cloud = new Cloud< Point3D >();

  // Random number generation
  const gsl_rng_type * T2;
  gsl_rng * r;
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r = gsl_rng_alloc (T2);





}
