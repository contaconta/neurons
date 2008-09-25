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

/** Variables for the arguments.*/
const char *argp_program_version =
  "imageRenderNeuron 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "renders an asc file into an image";

/* A description of the arguments we accept. */
static char args_doc[] = "imageRenderNeuron image_model neuron.asc";

/* The options we understand. */
static struct argp_option options[] = {
  {"orientation", 't', "orientation.jpg", 0, "saves the orientation of the neuron in that file"},
  {"output", 'o', "output.jpg", 0, "name of the rendered image mask"},
  {"scale", 's', "scale.jpg", 0, "saves the scale of the dendrite in that file"},
  {"min_width", 'w', "0",         0, "minimum width in micrometers of the dendrites to be drawn"},
  {"neuron_displacement", 'd', 0, 0, "in the case the asc file is created for volumes (centered in (0,0)), as in the case of neurons"},
  {"neuron_scale", 'c', "scale", 0, "in the case the asc file needs to be scaled (as in the case of the neurons)"},
  { 0 }
};

struct arguments
{
  string orientation_file;
  string scale_file;
  string image_name;
  string output_name;
  string neuron_name;
  float  min_width;
  bool   displace_neuron;
  float    neuron_scale;
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
      argments->orientation_file = arg;
      break;
    case 'd':
      argments->displace_neuron = true;
      break;
    case 'c':
      argments->neuron_scale = atof(arg);
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

    case ARGP_KEY_ARG:
      if (state->arg_num >= 2)
      /* Too many arguments. */
        argp_usage (state);
      if (state->arg_num == 0)
        argments->image_name = arg;
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

  struct arguments arguments;
  arguments.orientation_file = "";
  arguments.scale_file = "";
  arguments.image_name = "";
  arguments.neuron_name = "";
  arguments.output_name = "output.jpg";
  arguments.min_width = 0;
  arguments.displace_neuron = false;
  arguments.neuron_scale = 0;

  argp_parse(&argp, argc, argv, 0, 0, &arguments);

  //Laziness to not modify it according to the argp parser
  int save_orientation = (arguments.orientation_file == "");
  int save_scale       = (arguments.scale_file == "");
  float min_width = arguments.min_width;

  Image<float>* orig = new Image<float>(arguments.image_name);
  Neuron* neuron = new Neuron(arguments.neuron_name);
  if(arguments.neuron_scale != 0){
    neuron->projectionMatrix[0] *= arguments.neuron_scale;
    neuron->projectionMatrix[1] *= arguments.neuron_scale;
    neuron->projectionMatrix[2] *= arguments.neuron_scale;
    neuron->projectionMatrix[4] *= arguments.neuron_scale;
    neuron->projectionMatrix[5] *= arguments.neuron_scale;
    neuron->projectionMatrix[6] *= arguments.neuron_scale;
    neuron->projectionMatrix[8] *= arguments.neuron_scale;
    neuron->projectionMatrix[9] *= arguments.neuron_scale;
    neuron->projectionMatrix[10] *= arguments.neuron_scale;
  }

  if(arguments.displace_neuron == true){
    neuron->projectionMatrix[12] += float(orig->width)/2;
    neuron->projectionMatrix[13] += float(orig->height)/2;
  }


  string theta_str = arguments.orientation_file;
  string image_name = arguments.image_name;

  Image<float>* rendered = orig->create_blank_image_float(arguments.output_name);
  Image<float>* orientation = NULL;
  Image<float>* scale = NULL;
  if(arguments.orientation_file != "")
    orientation = orig->create_blank_image_float(arguments.orientation_file);
  if(arguments.scale_file != "")
    scale = orig->create_blank_image_float(arguments.scale_file);

  neuron->renderInImage(rendered, orientation, scale, min_width);

  rendered->save();
  if(arguments.orientation_file != "")
    orientation->save();
  if(arguments.scale_file != "")
    scale->save();
}
