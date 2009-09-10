
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
// Contact < german.gonzalez@epfl.ch > for comments & bug reports      //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include <argp.h>
#include "Neuron.h"

using namespace std;

/* Parse a single option. */
const char *argp_program_version =
  "0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "removes the z component of the points of a neuron";

/* A description of the arguments we accept. */
static char args_doc[] = "neuron_original.asc output.asc ";

/* The options we understand. */
static struct argp_option options[] = {
  // {"value",   't',  "default_value", 0, "blah"},
  // {"invert",  'i',  0 , 0, "blah"},
  { 0 }
};

struct arguments
{
  float value;
  bool flag;
  char* args[2];
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
    // case 't':
      // argments->value = atof(arg);
      // break;
    // case 'i':
      // argments->flag = false;
      // break;

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


void flattenNeuronSegment(NeuronSegment* np)
{
  for(int i = 0; i < np->points.size(); i++){
    np->points[i].coords[2]=0;
  }
  np->markers.resize(0);
  np->spines.resize(0);

  for(int i = 0; i < np->childs.size(); i++)
    flattenNeuronSegment(np->childs[i]);

}



int main(int argc, char **argv) {

  struct arguments args;
  /* Default values. */
  // arguments.flag = true;
  // arguments.value = 0;

  argp_parse (&argp, argc, argv, 0, 0, &args);

  Neuron* neuronita = new Neuron(args.args[0]);

  for(int i = 0; i < neuronita->axon.size(); i++)
    flattenNeuronSegment(neuronita->axon[i]);
  for(int i = 0; i < neuronita->dendrites.size(); i++)
    flattenNeuronSegment(neuronita->dendrites[i]);

  neuronita->save(args.args[1]);
}
