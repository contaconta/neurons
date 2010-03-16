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
#include "Neuron.h"
#include <argp.h>

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "cubeRenderNeuron 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "renders an asc file into volume in a pixel per pixel basis";

/* A description of the arguments we accept. */
static char args_doc[] = "cubeRenderNeuron cubeModel.nfo neuron.asc";

/* The options we understand. */
static struct argp_option options[] = {
  {"phi",    'p', "phi", 0,   "saves the phi angle in that volume file"},
  {"theta",  't', "theta", 0, "saves the theta angle in that volume file"},
  {"output", 'o', "output", 0, "name of the rendered mask"},
  {"scale",  's', "scale", 0, "saves the scale of the dendrite in that file"},
  {"renderScale",  'r', "scale", 0, "multiplyes the width of the neuron by that factor"},
  {"min_width", 'w', "0",         0, "minimum width in micrometers of the dendrites to be drawn"},
  { 0 }
};

struct arguments
{
  string theta_file;
  string phi_file;
  string scale_file;
  string cube_name;
  string output_name;
  string neuron_name;
  float  min_width;
  float renderScale;
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
      argments->theta_file = arg;
      break;
    case 'p':
      argments->phi_file = arg;
      break;
    case 'o':
      argments->output_name = arg;
      break;
    case 's':
      argments->scale_file = arg;
      break;
    case 'w':
      argments->min_width = atof(arg);
      break;
    case 'r':
      argments->renderScale = atof(arg);
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 2)
      /* Too many arguments. */
        argp_usage (state);
      if (state->arg_num == 0)
        argments->cube_name = arg;
      if (state->arg_num == 1)
        argments->neuron_name = arg;
      break;
    case ARGP_KEY_END:
      /* Not enough arguments. */
      if (state->arg_num < 2)
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
  args.theta_file = "";
  args.phi_file = "";
  args.scale_file = "";
  args.cube_name = "";
  args.neuron_name = "";
  args.output_name = "output";
  args.min_width = 0;
  args.renderScale = 1;

  argp_parse(&argp, argc, argv, 0, 0, &args);

  printf("The arguments are:\n  cube_name: %s\n  neuron_name: %s\n"
         "  output_name: %s\n  theta_file: %s\n  phi_file: %s\n  scale_name: %s\n"
         "  min_width: %f\n renderScale = %f\n",
         args.cube_name.c_str(), args.neuron_name.c_str(), args.output_name.c_str(),
         args.theta_file.c_str(), args.phi_file.c_str(), args.scale_file.c_str(),
         args.min_width, args.renderScale);
  // exit(0);

  Neuron* neuron = new Neuron(args.neuron_name);

  Cube<uchar, ulong>* orig = new Cube<uchar,ulong>(args.cube_name);
  Cube<uchar,ulong>* rendered = orig->create_blank_cube_uchar(args.output_name);

  Cube<float, double>* theta = NULL;
  Cube<float, double>* phi   = NULL;
  Cube<float, double>* scale = NULL;

  if(args.theta_file != "")
    theta = orig->create_blank_cube(args.theta_file);
  if(args.phi_file != "")
    phi = orig->create_blank_cube(args.phi_file);
  if(args.scale_file != "")
    scale = orig->create_blank_cube(args.scale_file);

  neuron->renderInCube(rendered, theta, phi, scale, args.min_width, args.renderScale);
  // rendered->put_all(255);
  // neuron->renderSegmentInCube(neuron->dendrites[0],rendered, theta, phi, scale, args.min_width, args.renderScale);
}
