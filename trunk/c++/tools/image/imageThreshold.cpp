
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
#include "Image.h"

using namespace std;

/* Parse a single option. */
const char *argp_program_version =
  "0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  " thresholds an image ";

/* A description of the arguments we accept. */
static char args_doc[] = "image threshold";

/* The options we understand. */
static struct argp_option options[] = {
  {"valueUp",   'u',  "0", 0,   "lower value to put in the thresholded pixels"},
  {"valueDown", 'd',  "255", 0, "upper value to put in the thresholded pixels"},
  {"float",   'f',   0,  0, "if indicated, the image is of float type, if not, uchar"},
  {"output",  'o',  "output.jpg", 0, "if defined, the thresholded image is saved in output"},
  { 0 }
};

struct arguments
{
  string image_name;
  string output;
  float  valueDown;
  float  valueUp;
  float  threshold;
  bool floatImage;
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
    case 'u':
      argments->valueUp = atof(arg);
      break;
    case 'd':
      argments->valueDown = atof(arg);
      break;
    case 'f':
      argments->floatImage = true;
      break;
    case 'o':
      argments->output = arg;
      break;


    case ARGP_KEY_ARG:
      if (state->arg_num >= 2)
      /* Too many arguments. */
        argp_usage (state);
      if(state->arg_num == 0)
        argments->image_name = arg;
      if(state->arg_num == 1)
        argments->threshold = atof(arg);
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

  struct arguments args;
  /* Default values. */
  args.output = "";
  args.valueDown  = 0;
  args.valueUp  = 255;
  args.floatImage = false;

  argp_parse (&argp, argc, argv, 0, 0, &args);

  if(args.floatImage){
    Image<float>* img = new Image<float>(args.image_name);
    img->threshold(args.threshold, args.output, args.valueDown, args.valueUp);
    img->save();
  } else {
    Image<uchar>* img = new Image<uchar>(args.image_name);
    img->threshold(args.threshold, args.output, args.valueDown, args.valueUp);
    img->save();
  }

}
