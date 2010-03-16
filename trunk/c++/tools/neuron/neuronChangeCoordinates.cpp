
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
#include <vector>

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
static char args_doc[] = "neuron_original.asc matrix.txt output.asc ";

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
  char* args[3];
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
      if (state->arg_num >= 3)
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


void applyTransformationToSegment(NeuronSegment* np, vector< vector< float > > &matrix)
{

  vector< float > newCoords(4);
  for(int i = 0; i < np->points.size(); i++){
    //Here do the matrix multiplication
    for(int j = 0; j < 3; j++)
      newCoords[j] =
        matrix[j][0]*np->points[i].coords[0] + 
        matrix[j][1]*np->points[i].coords[1] +
        matrix[j][2]*np->points[i].coords[2] +
        matrix[j][3];
    newCoords[3] = np->points[i].coords[3];
    np->points[i].coords = newCoords;
  }
  np->markers.resize(0);
  np->spines.resize(0);

  for(int i = 0; i < np->childs.size(); i++)
    applyTransformationToSegment(np->childs[i], matrix);
}



int main(int argc, char **argv) {

  struct arguments args;
  /* Default values. */
  // arguments.flag = true;
  // arguments.value = 0;

  argp_parse (&argp, argc, argv, 0, 0, &args);

  Neuron* neuronita = new Neuron(args.args[0]);

  vector< vector< float > > matrix;

  std::ifstream mat (args.args[1]);
  for(int i = 0; i < 4; i++){
    vector< float > row(4);
    for(int j = 0; j < 4; j++)
      mat >> row[j];
    matrix.push_back(row);
  }

  for(int i = 0; i < neuronita->axon.size(); i++)
    applyTransformationToSegment(neuronita->axon[i], matrix);
  for(int i = 0; i < neuronita->dendrites.size(); i++)
    applyTransformationToSegment(neuronita->dendrites[i], matrix);

  neuronita->save(args.args[2]);
}
