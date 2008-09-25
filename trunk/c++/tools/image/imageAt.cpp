
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
#include "Image.h"
#include  <argp.h>

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "imageAt 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "imageMask masks an image with another one";

/* A description of the arguments we accept. */
static char args_doc[] = "image mask";

/* The options we understand. */
static struct argp_option options[] = {
  {"uchar",   'c',  0, 0, "if the image is of uchar type"},
  { 0 }
};

struct arguments
{
  string filename;
  int x;
  int y;
  bool uchar_image;
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
    case 'c':
      argments->uchar_image = true;
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 3)
      /* Too many arguments. */
        argp_usage (state);
      if (state->arg_num == 0)
        argments->filename = arg;
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

  arguments.uchar_image = false;
  arguments.x = 0;
  arguments.y = 0;
  arguments.filename = "";

  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  if(arguments.uchar_image == false){
    Image< float >* img = new Image<float>(arguments.filename);
    printf("%f\n", img->at(arguments.x, arguments.y));
  }
  else{
    Image< uchar >* img = new Image<uchar>(arguments.filename);
    printf("%i\n", img->at(arguments.x, arguments.y));
  }



}
