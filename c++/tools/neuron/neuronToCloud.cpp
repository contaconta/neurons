
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
// Contact <german.gonzalez@epfl.ch> for comments & bug reports        //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include "Neuron.h"
#include "Cube.h"
#include "Cloud.h"
#include <argp.h>

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "neuronToCloud 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "converts an asc neuronlucida file into a neseg cloud. The points are in micrometers unless otherwise stated";

/* A description of the arguments we accept. */
static char args_doc[] = "neuron.asc cloud.cl";

/* The options we understand. */
static struct argp_option options[] = {
  {"verbose"  ,    'v', 0,           0,  "Produce verbose output" },
  {"min_width"  ,  'w', "min_width",           0,  "Produce verbose output" },
  { 0 }
};

struct arguments
{
  string name_neuron;
  string name_cloud;
  int verbose;
  double width;
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
    case 'v':
      argments->verbose = 1;
      break;
    case 'w':
      argments->width = atof(arg);
      break;


    case ARGP_KEY_ARG:
      if (state->arg_num >= 2)
      /* Too many arguments. */
        argp_usage (state);
      if(state->arg_num == 0)
        argments->name_neuron = arg;
      if(state->arg_num == 1)
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




int main(int argc, char** argv)
{

  printf("This code needs to be checked and redone!\n");
  exit(0);

  // struct arguments args;
  // args.name_neuron = "";
  // args.name_cloud = "";
  // args.width = 0;

  // argp_parse (&argp, argc, argv, 0, 0, &args);

  // Neuron* neuronita = new Neuron(args.neuron_name);
  // double min_width = args.width;

  // neuronita->toCloud(args.name_neuron, args.name_neuron, min_width, args.name_neuron);

  return 0;
}
