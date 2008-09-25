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
#include "Image.h"
#include "Neuron.h"
#include <argp.h>

using namespace std;


/* Parse a single option. */
const char *argp_program_version =
  "imageDerivative 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "imageDerivative calculates the derivative of an image of a certain order and sigma";

/* A description of the arguments we accept. */
static char args_doc[] = "imageDerivative image";

/* The options we understand. */
static struct argp_option options[] = {
  {"orthogonal",   't',  0, 0, "if the flag is tagged, it calculates the 'orthogonal' masks (i.e. go_)"},
  {"orthogonal",   'n',  0, 0, "if the flag is tagged, it calculates the 'normalized' gaussian (i.e. gn_)"},
  {"order_x",      'x',  "2", 0, "the order of the derivative in the x direction"},
  {"order_y",      'y',  "0", 0, "the order of the derivative in the y direction"},
  {"file",         'f',  "out.jpg", 0, "output filename"},
  {"sigma",        's',  "2.0",    0, "the sigma where the derivatives should be calculated"},
  { 0 }
};

struct arguments
{
  float sigma;
  bool orthogonal;
  bool normalized;
  string output_file;
  string filename_image;
  int o_x;
  int o_y;
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
    case 'x':
      argments->o_x = atoi(arg);
      break;
    case 'y':
      argments->o_y = atoi(arg);
      break;
    case 't':
      argments->orthogonal = true;
      break;
    case 'n':
      argments->normalized = true;
      break;
    case 'f':
      argments->output_file = arg;
      break;
    case 's':
      argments->sigma = atof(arg);
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 1)
      /* Too many arguments. */
        argp_usage (state);
      if (state->arg_num == 0)
        argments->filename_image = arg;
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
  arguments.sigma = 1.0;
  arguments.orthogonal = false;
  arguments.normalized = true;
  arguments.output_file = "out.jpg";
  arguments.o_x = 2;
  arguments.o_y = 0;
  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  printf("The parameters are:\n -> image: %s\n ->orthogonal: %i\n"
         " ->o_x: %i\n ->o_y: %i\n ->sigma: %f\n ->output_file:%s\n",
         arguments.filename_image.c_str(),  arguments.orthogonal,
         arguments.o_x, arguments.o_y, arguments.sigma, arguments.output_file.c_str());

  Image<float>* img = new Image<float>(arguments.filename_image);

  if(arguments.orthogonal){
    img->calculate_gaussian_orthogonal(arguments.o_x, arguments.o_y,
                                       arguments.sigma, arguments.output_file);
    exit(0);
  }
  if(arguments.normalized){
    img->calculate_gaussian_normalized(arguments.o_x, arguments.o_y,
                                       arguments.sigma, arguments.output_file);
    exit(0);
  }
  img->calculate_derivative(arguments.o_x, arguments.o_y,
                            arguments.sigma, arguments.output_file);


}
