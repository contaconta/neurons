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
#include <argp.h>
#include "CubeFactory.h"
#include "Cube.h"

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "cubeThreshold 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "cubeThreshold ouputs the thresholded version of the cube";

/* A description of the arguments we accept. */
static char args_doc[] = "volume.nfo outputvolume";

/* The options we understand. */
static struct argp_option options[] = {
  {"threshold",  't' , "float",    0, "Threshold the cube" },
  {"lowValue" ,  'l',  "value",    0, "The value to put points below the threshold"},
  {"highValue" , 'h',  "value",    0, "The value to put points above the threshold"},
  { 0 }
};

struct arguments
{
  char* args[2];                /* arg1 & arg2 */
  bool  putHigherValuesTo, putLowerValuesTo;
  float lowValue;
  float highValue;
  float threshold;
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
      argments->threshold = atof(arg);
      break;
    case 'l':
      argments->lowValue = atof(arg);
      argments->putLowerValuesTo = true;
      break;
    case 'h':
      argments->highValue = atof(arg);
      argments->putHigherValuesTo = true;
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 2)
      /* Too many arguments. */
        argp_usage (state);
      argments->args[state->arg_num] = arg;
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
  /* Default values. */
  arguments.threshold = 0;
  arguments.putLowerValuesTo  = false;
  arguments.putHigherValuesTo = false;
  arguments.highValue   = 1;
  arguments.lowValue    = 0;
  arguments.args[1] = "output";

  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  printf ("Volume = %s\nOutput = %s\n"
          "putHigherValuesTo = %s\n"
          "putLowerValuesTo = %s\n"
          "threshold = %f \n"
          "highValue = %f\n"
          "lowValue = %f\n",
          arguments.args[0], arguments.args[1],
          arguments.putHigherValuesTo ? "yes" : "no",
          arguments.putLowerValuesTo ? "yes" : "no",
          arguments.threshold,
          arguments.highValue,
          arguments.lowValue
          );

  Cube_P* cube = CubeFactory::load(arguments.args[0]);

  cube->threshold(arguments.threshold, arguments.args[1],
                  arguments.putHigherValuesTo, arguments.putLowerValuesTo,
                  arguments.highValue, arguments.lowValue);
}
