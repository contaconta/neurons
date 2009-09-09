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
#include <argp.h>

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "cubeStatistics 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "cubeStatistics get some useful statistics of the cube";

/* A description of the arguments we accept. */
static char args_doc[] = "volume.nfo outputvolume";

/* The options we understand. */
static struct argp_option options[] = {
  {"histogram",  'h' , "file",     0, "where to store the histogram" },
  { 0 }
};

struct arguments
{
  string histogramFile;
  string cubeName;
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
      argments->histogramFile = arg;
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num == 0)
        argments->cubeName = arg;
      break;
      if (state->arg_num > 1)
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
  /* Default values. */
  arguments.histogramFile = "";
  arguments.cubeName  = "";
  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  Cube_P* cube;

  string volume_str = arguments.cubeName;

  Cube<uchar,ulong>* cube_test = new Cube<uchar,ulong>(volume_str,false);
  if(cube_test->type == "uchar"){
    cube = new Cube<uchar,ulong>(volume_str);
  }
  if(cube_test->type == "float"){
    cube = new Cube<float,double>(volume_str);
  }

  cube->print_statistics();
  cube->histogram(arguments.histogramFile);





}
