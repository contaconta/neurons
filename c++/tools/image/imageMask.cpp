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
  "imageMask 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "imageMask masks an image with another one";

/* A description of the arguments we accept. */
static char args_doc[] = "image mask output";

/* The options we understand. */
static struct argp_option options[] = {
  {"value",   't',  "float", 0, "value to put the points outside the mask"},
  {"alpha",   'a',  "float", 0, "value to put the points outside the mask"},
  {"invert",  'i',  0 , 0, "wether the important points in the mask are the dark ones"},
  { 0 }
};

struct arguments
{
  float value;
  float alpha;
  bool flag_max;
  bool flag_alpha_blending;
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
    case 't':
      argments->value = atof(arg);
      break;
    case 'a':
      argments->flag_alpha_blending = true;
      argments->alpha = atof(arg);
      break;
    case 'i':
      argments->flag_max = 0;
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 3)
      /* Too many arguments. */
        argp_usage (state);
      argments->args[state->arg_num] = arg;
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

  struct arguments arguments;
  /* Default values. */
  arguments.flag_max = 1;
  arguments.value = 0;

  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  printf("Img: %s\n, Msk: %s\n, Flg: %i\n, Val: %f\n", 
         arguments.args[0],         arguments.args[1],
         arguments.flag_max, arguments.value);


  Image< float >* img  = new Image< float >(arguments.args[0]);
  Image< float >* mask = new Image< float >(arguments.args[1]);
  Image< float >* out  = img->copy(arguments.args[2]);

  if(arguments.flag_alpha_blending)
    out->applyMaskAlphaBlending(mask, arguments.value, arguments.flag_max, arguments.alpha);
  else
    out->applyMask(mask, arguments.value, arguments.flag_max);
  out->save();
}
