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
#include "Cloud.h"
#include <argp.h>
#include <gsl/gsl_rng.h>

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "imageHistogram 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "returns the histogram of an image";

/* A description of the arguments we accept. */
static char args_doc[] = "imageHistogram cloud";

/* The options we understand. */
static struct argp_option options[] = {
  {"float",         'f',  0,   0, "if the image is of type float (in the 'raw' file)"},
  {"nbins",         'n',  "5", 0, "number of bins used to calculate the histogram"},
  {"ignore-lower-value",  'l',  0,   0, "if set, the lower value will not be included in the histogram"},
  // {"output", 'N', "positive_points",-  0, "if defined, the number of positive points to get"},
  { 0 }
};

struct arguments
{
  int nbins;
  bool uchar_image;
  bool ignore_lowe_value;
  string imageName;
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
    case 'f':
      argments->uchar_image = false;
      break;
    case 'n':
      argments->nbins = atoi(arg);
      break;
    case 'l':
      argments->ignore_lowe_value = true;
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 1)
      /* Too many arguments. */
        argp_usage (state);
      if (state->arg_num == 0)
        argments->imageName = arg;
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
  arguments.nbins = 5;
  arguments.uchar_image = true;
  arguments.ignore_lowe_value = false;

  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  vector<int> hist;
  vector<float> range;

  if(arguments.uchar_image == true){
    Image<uchar>* img = new Image<uchar>(arguments.imageName);
    img->histogram(arguments.nbins, hist, range, arguments.ignore_lowe_value);
  } else{

    Image<float>* img = new Image<float>(arguments.imageName);
    img->histogram(arguments.nbins, hist, range, arguments.ignore_lowe_value);
  }

  for(int i = 0; i < hist.size(); i++){
    printf("%f : %i\n", range[i], hist[i]);
  }

}

