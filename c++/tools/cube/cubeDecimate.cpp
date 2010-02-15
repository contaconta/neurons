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
#include "CubeFactory.h"
#include "Cube.h"
#include "Cloud_P.h"
#include "Cloud.h"
#include "Point3D.h"

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "cubeDecimate 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "cubedecimate produces a list of local maxima of the volume";

/* A description of the arguments we accept. */
static char args_doc[] = "volume.nfo outputfile";

/* The options we understand. */
static struct argp_option options[] = {
  {"verbose"  ,  'v', 0,           0,  "Produce verbose output" },
  {"threshold",  't' , "float",      0,
   "Threshold above(bellow) the points" },
  {"min"      ,  'm',  0, 0,"The important points are the low ones"},
  {"layer"    ,  'l',  0, 0, "Do nonmax layer by layer"},
  {"windowxy" ,  'w',  "int", 0,"width and height of the window"},
  {"windowz" ,   'z',  "int", 0,"depth of the window"},
  {"logscale",   'o',  0, 0, "Do the decimation in a log scale"},
  {"not-cloud",  'c',  0, 0, "save the points as a list of indexes instead of a cloud"},
  {"somaX"   ,   'X',  "float", 0, "X coordinate of the soma"},
  {"somaX"   ,   'Y',  "float", 0, "Y coordinate of the soma"},
  {"somaX"   ,   'Z',  "float", 0, "Z coordinate of the soma"},
  { 0 }
};

struct arguments
{
  char *args[2];                /* arg1 & arg2 */
  int flag_min, verbose, windowxy, windowz, flag_layer, flag_log;
  float threshold, somaX, somaY, somaZ;
  bool saveAsCloud;
  bool somaDefined;
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
    case 'v':
      argments->verbose = 1;
      break;
    case 'm':
      argments->flag_min = 1;
      break;
    case 't':
      argments->threshold = atof(arg);
      break;
    case 'l':
      argments->flag_layer = 1;
      break;
    case 'o':
      argments->flag_log = 1;
      break;
    case 'w':
      argments->windowxy = atoi(arg);
      break;
    case 'z':
      argments->windowz = atoi(arg);
      break;
    case 'c':
      argments->saveAsCloud = false;
      break;
    case 'X':
      argments->somaX = atof(arg);
      argments->somaDefined = true;
      break;
    case 'Y':
      argments->somaY = atof(arg);
      argments->somaDefined = true;
      break;
    case 'Z':
      argments->somaZ = atof(arg);
      argments->somaDefined = true;
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 2)
      /* Too many arguments. */
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
  arguments.flag_min = 0;
  arguments.verbose = 0;
  arguments.threshold = 0;
  arguments.windowxy = 8;
  arguments.windowz = 10;
  arguments.flag_layer = 0;
  arguments.flag_log = 0;
  arguments.saveAsCloud = true;
  arguments.somaDefined = false;

  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  printf ("Volume = %s\nFile = %s\n"
          "bool_min = %s\n"
          "threshold = %f \n"
          "saveAsCloud = %i \n",
          arguments.args[0], arguments.args[1],
          arguments.flag_min ? "yes" : "no",
          arguments.threshold,
          arguments.saveAsCloud
          );

  if(arguments.flag_log && arguments.flag_layer){
    printf("Not yet implemented logarithmic decimation layer by layer\n");
  }

  Cube_P* cube = CubeFactory::load(arguments.args[0]);

  if(arguments.flag_log){
    cube->decimate_log(arguments.threshold, arguments.windowxy, arguments.windowz,
                       arguments.args[1], false);
  }

  if(!(arguments.flag_log || arguments.flag_layer)){
    cube->decimate(arguments.threshold, arguments.windowxy, arguments.windowz,
                       arguments.args[1], false);
  }

  if(arguments.flag_layer){
    // for(int z = 0; z < 
    // cube->decimate
    printf("Lazy enough for not doing this and I do not see the point now\n");
  }


  if(arguments.saveAsCloud){
    Cloud<Point3D >* cd = new Cloud<Point3D>();
    vector< vector< double > > idxs = loadMatrix(arguments.args[1]);
    // vector< int > indexes(3);
    // vector< float > micrometers(3);
    //The first point will be the soma
    if(arguments.somaDefined){
      cd->points.push_back
        (new Point3D(arguments.somaX, arguments.somaY, arguments.somaZ));
    }
    for(int i = 0; i < idxs.size(); i++){

      Point3D* pt = new Point3D( idxs[i][0],
                                 idxs[i][1],
                                 idxs[i][2]);

      cd->points.push_back( pt );
    }
    // cd->v_radius = cube->voxelWidth;
    cd->v_radius = 0.5;
    cd->saveToFile(arguments.args[1]);
  }

}
