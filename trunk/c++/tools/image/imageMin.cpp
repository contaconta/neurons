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
  "imageMax 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "imageMin outputs the minimum of some images";

/* A description of the arguments we accept. */
static char args_doc[] = "imageMax image1 image2 image3 ...";

/* The options we understand. */
static struct argp_option options[] = {
  {"output",       'o',  "out.jpg", 0, "output filename"},
  { 0 }
};

struct arguments
{
  string output_file;
  vector< string > images;
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
      argments->output_file = arg;
      break;

    case ARGP_KEY_ARG:
      argments->images.push_back(arg);
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
  arguments.output_file = "out.jpg";
  arguments.images.resize(0);
  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  printf("Output file: %s\n", arguments.output_file.c_str());

  if(arguments.images.size() == 0){
    printf("There is no image from whom to get the max... exiting\n");
    exit(0);
  }

  Image<float>* img = new Image<float>(arguments.images[0]);
  string directory = getDirectoryFromPath(arguments.images[0]);
  string output = directory + "/" + arguments.output_file;
  Image< float >* result = img->copy(arguments.output_file);

  // for(int x = 0; x < img->width; x++)
    // for(int y = 0; y < img->height; y++)
      // result->put(x,y,img->at(x,y));

  for(int i = 1; i < arguments.images.size(); i++){
    Image< float >* img2 = new Image< float>(arguments.images[i]);
    for(int x = 0; x < img->width; x++)
      for(int y = 0; y < img->height; y++)
        if( img2->at(x,y) < result->at(x,y))
          result->put(x,y, img2->at(x,y));
  }

  result->save();

}
