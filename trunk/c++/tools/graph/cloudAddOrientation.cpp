
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
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <argp.h>
#include "Graph.h"
#include "Cloud.h"
#include "CubeFactory.h"

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "cloudAddOrientation 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "adds orientation informaton (theta, phi) to the points in the cloud given the extra volumes";

/* A description of the arguments we accept. */
static char args_doc[] = "cloud.cl volume_x volume_y volume_z outputfile";

/* The options we understand. */
static struct argp_option options[] = {
  {"verbose"  ,  'v', 0,           0,  "Produce verbose output" },
  { 0 }
};

struct arguments
{
  string cloudName, outputFile, volumeX, volumeY, volumeZ;
  int verbose;
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

    case ARGP_KEY_ARG:
      if (state->arg_num >= 5)
      /* Too many arguments. */
        argp_usage (state);
      if (state->arg_num == 0)
        argments->cloudName = arg;
      if (state->arg_num == 1)
        argments->volumeX = arg;
      if (state->arg_num == 2)
        argments->volumeY = arg;
      if (state->arg_num == 3)
        argments->volumeZ = arg;
      if (state->arg_num == 4)
        argments->outputFile = arg;
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
  arguments.verbose = 0;
  arguments.outputFile = "out.cl";
  arguments.cloudName = "";
  arguments.volumeX = "";
  arguments.volumeY = "";
  arguments.volumeZ = "";

  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  printf ("Cloud = %s\n OutputFile = %s\n",
          arguments.cloudName.c_str(), arguments.outputFile.c_str());

  //Code starts here
  Cloud<Point3D>* cl = new Cloud<Point3D>(arguments.cloudName);
  Cloud<Point3Do>* out = new Cloud<Point3Do>(arguments.outputFile);
  Cube< float, double>* vx = new Cube<float, double>(arguments.volumeX);
  Cube< float, double>* vy = new Cube<float, double>(arguments.volumeY);
  Cube< float, double>* vz = new Cube<float, double>(arguments.volumeZ);


  vector< int > idx(3);
  vector< float > micr(3);
  float theta, phi, r;

  for(int i = 0; i < cl->points.size(); i++){
    micr[0] = cl->points[i]->coords[0];
    micr[1] = cl->points[i]->coords[1];
    micr[2] = cl->points[i]->coords[2];
    vx->micrometersToIndexes(micr, idx);

    r = sqrt( vx->at(idx[0],idx[1],idx[2])*vx->at(idx[0],idx[1],idx[2]) +
              vy->at(idx[0],idx[1],idx[2])*vy->at(idx[0],idx[1],idx[2]) +
              vz->at(idx[0],idx[1],idx[2])*vz->at(idx[0],idx[1],idx[2]) );

    theta = atan2(vy->at(idx[0],idx[1],idx[2]), vx->at(idx[0],idx[1],idx[2]));
    phi   = acos(vz->at(idx[0],idx[1],idx[2])/r);

    Point3Do* pt = new Point3Do(micr[0], micr[1], micr[2],
                                theta, phi);

    out->points.push_back(pt);
  }
  out->saveToFile(arguments.outputFile);
}
