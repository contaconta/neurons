
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
// Contact <german.gonzalez@epfl.ch> for comments & bug reports        //
/////////////////////////////////////////////////////////////////////////

#include <iostream>
#include <string>
#include "Neuron.h"
#include "Cube.h"
#include "Cloud.h"
#include <argp.h>
#include "CubeFactory.h"

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "neuronToCloud 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "converts an asc neuronlucida file into a neseg cloud. The points are in micrometers unless otherwise stated";

/* A description of the arguments we accept. */
static char args_doc[] = "neuron.asc cloud.cl";

/* The options we understand. */
static struct argp_option options[] = {
  {"verbose"  ,    'v', 0,           0,  "Produce verbose output" },
  {"min_width"  ,  'w', "min_width",           0,  "minimum width of the point" },
  {"max_width"  ,  'x', "max_width",           0,  "maximum width of the point" },
  {"save_width"  , 'W',  0,                    0,  "saves the width of the points" },
  {"onlyIfInCube",  'c', "cube",           0,  "Only takes the points that fall inside the cube" },
  {"orientation"  ,  'n', 0,           0,  "Saves the orientation of the points from the neuron" },
  {"theta"  ,  't', "theta_cube",           0,  "Takes the orientation from the theta of the cube" },
  {"phi"  ,  'p', "phi_cube",           0,  "Takes the orientation from the phi bbof the cube" },
  {"forcephi"  ,  'f', 0,           0,  "if defined forces the orientation in phi of the points to be Pi/2" },
  {"type"  ,         'y', 0,           0,  "Saves the type of the points" },
  { 0 }
};

struct arguments
{
  string name_neuron;
  string name_cloud;
  string name_cube;
  int verbose;
  double max_width;
  double min_width;
  bool saveWidth;
  bool saveOrientation;
  bool saveType;
  bool forcePhi;
  string name_cubeTheta;
  string name_cubePhi;
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
    case 'w':
      argments->max_width = atof(arg);
      argments->saveWidth = true;
      argments->saveOrientation = true;
      argments->saveType = true;
      break;
    case 'x':
      argments->min_width = atof(arg);
      argments->saveWidth = true;
      argments->saveOrientation = true;
      argments->saveType = true;
      break;
    case 'W':
      argments->saveWidth = true;
      argments->saveOrientation = true;
      argments->saveType = true;
      break;
    case 'n':
      argments->saveOrientation = true;
      break;
    case 'y':
      argments->saveType = true;
      break;
    case 'c':
      argments->name_cube = arg;
      break;
    case 't':
      argments->name_cubeTheta = arg;
      argments->saveOrientation = true;
      break;
    case 'p':
      argments->name_cubePhi = arg;
      argments->saveOrientation = true;
      break;
    case 'f':
      argments->forcePhi = true;
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 2)
      /* Too many arguments. */
        argp_usage (state);
      if(state->arg_num == 0)
        argments->name_neuron = arg;
      if(state->arg_num == 1)
        argments->name_cloud = arg;
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




int main(int argc, char** argv)
{

  struct arguments args;
  args.name_neuron = "";
  args.name_cloud = "";
  args.name_cube = "";
  args.saveOrientation = false;
  args.saveType = false;
  args.name_cubeTheta = "";
  args.name_cubePhi   = "";
  args.max_width = 1e100;
  args.min_width = 0;
  args.saveWidth = false;
  args.forcePhi = false;

  argp_parse (&argp, argc, argv, 0, 0, &args);

  Neuron* neuronita = new Neuron(args.name_neuron);
  double min_width = args.min_width;
  double max_width = args.max_width;

  Cube_P* cube = NULL;
  if(args.name_cube != ""){
    cube = CubeFactory::load(args.name_cube);
  }

  Cloud_P* cloud = neuronita->toCloud(args.name_cloud,
                                      args.saveOrientation, args.saveType,
                                      cube, args.saveWidth);

  if( (cube!= NULL) && (args.name_cubeTheta!="") && args.saveOrientation ){
    printf("Twisting the angles\n");
    vector< float > nmic(3);
    vector< int > idx(3);
    Cube<float, double>*  theta = new Cube<float, double>(args.name_cubeTheta);
    Cube<float, double>*  phi = NULL;
    if(args.name_cubePhi!= "")
      phi = new Cube<float, double>(args.name_cubePhi);
    for(int i = 0; i < cloud->points.size(); i++){
      Point3Do* pt = dynamic_cast<Point3Dot*>(cloud->points[i]);
      theta->micrometersToIndexes(pt->coords, idx);
      // printf("%i %i %i\n", idx[0], idx[1], idx[2]);
      pt->theta = theta->at(idx[0],idx[1],idx[2]);
      if(phi!= NULL)
        pt->phi   = phi->at(idx[0],idx[1],idx[2]);
      if (args.forcePhi)
        pt->phi = M_PI/2;
    }
  }

  printf("Removing whatever should be removed\n");
  if( ((args.min_width != 0) || (args.max_width!=1e100)) && args.saveWidth){
    for(int i = cloud->points.size()-1; i >= 0; i--){
      printf("%i\n", i);
      Point3Dotw* pt = dynamic_cast<Point3Dotw*>(cloud->points[i]);
      if( (pt->weight < args.min_width) || (pt->weight > args.max_width)){
         vector<Point*>::iterator itRemove = cloud->points.begin() + i;
         cloud->points.erase(itRemove);
      }
    }
  }


  cloud->saveToFile(args.name_cloud);

  return 0;
}
