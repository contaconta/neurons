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
#include "utils_neseg.h"

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "cubeGaussianMask 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "outputs the coefficients of a gaussian mask of a given sigman and a given derivative order";

/* A description of the arguments we accept. */
static char args_doc[] = "sigma order";

/* The options we understand. */
static struct argp_option options[] = {
  {"normalized",   'n',  0, 0, "if the flag is tagged, it calculates the 'normalized' mask"},
  { 0 }
};

struct arguments
{
  bool normalized;
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
    case 'n':
      argments->normalized = true;
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 2)
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
  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  double sigma = atof(arguments.args[0]);
  int    order = atoi(arguments.args[1]);

  // printf("Gaussian mask of sigma = %f(%s) and order = %i (%s)\n", sigma,
         // arguments.args[0], order, arguments.args[1]);

  double en = 1;
  if(arguments.normalized){
    en = Mask::energy1DGaussianMask(order, sigma);
  }
  vector< float > mask = Mask::gaussian_mask(order, sigma, true);
  for(int i = 0; i < mask.size(); i++){
    std::cout << mask[i]/en << " ";
  }
  std::cout << std::endl;
}
