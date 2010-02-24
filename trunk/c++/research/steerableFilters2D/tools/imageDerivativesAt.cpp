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
#include "SteerableFilter2DOrthogonal.h"

using namespace std;


/* Parse a single option. */
const char *argp_program_version =
  "imageDerivativesAt 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "imageDerivativesAt outputs the derivatives coefficients at a given location";

/* A description of the arguments we accept. */
static char args_doc[] = "imageDerivativesAt image x y";

/* The options we understand. */
static struct argp_option options[] = {
  {"orthogonal",   't',  0, 0, "if the flag is tagged, it outputs the 'orthogonal' masks (i.e. go_)"},
  {"order",        'o',  "order", 0, "the order of the derivatives"},
  {"file",         'f',  "out.txt", 0, "output filename"},
  {"sigma",        's',  "sigma",    0, "the sigma where the derivatives should be calculated"},
  { 0 }
};

struct arguments
{
  float sigma;
  bool orthogonal;
  string output_file;
  string filename_image;
  int order;
  int x;
  int y;
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
    case 'o':
      argments->order = atoi(arg);
      break;
    case 't':
      argments->orthogonal = true;
      break;
    case 'f':
      argments->output_file = arg;
      break;
    case 's':
      argments->sigma = atof(arg);
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 3)
      /* Too many arguments. */
        argp_usage (state);
      if (state->arg_num == 0)
        argments->filename_image = arg;
      if (state->arg_num == 1)
        argments->x = atoi(arg);
      if (state->arg_num == 2)
        argments->y = atoi(arg);
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
  arguments.output_file = "out.txt";
  arguments.order = 2;
  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  printf("The parameters are:\n -> image: %s\n ->orthogonal: %i\n"
         " ->order: %i\n ->sigma: %f\n ->output_file:%s\n",
         arguments.filename_image.c_str(),  arguments.orthogonal,
         arguments.order, arguments.sigma, arguments.output_file.c_str());

  SteerableFilter2D* stf;
  if(arguments.orthogonal)
    stf = new SteerableFilter2DOrthogonal(arguments.filename_image, arguments.order, arguments.sigma);
  else
    stf = new SteerableFilter2D(arguments.filename_image, arguments.order, arguments.sigma);

  std::ofstream out(arguments.output_file.c_str());
  for(int i = 0; i < stf->derivatives.size(); i++){
    out << stf->derivatives[i]->at(arguments.x, arguments.y) << std::endl;
  }
  out.close();

}
