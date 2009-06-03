
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
#include "CloudFactory.h"
#include "GraphFactory.h"
#include "CubeFactory.h"

using namespace std;
/** Variables for the arguments.*/
const char *argp_program_version =
  "graphPrim 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "graphSampleLatticeArroundEdges samples a lattice of regularly sampled voxels of the cube arround the edges on the graph and saves those coefficients into a file";

/* A description of the arguments we accept. */
static char args_doc[] = "cube.nfo graph.gr outputfile";

/* The options we understand. */
static struct argp_option options[] = {
  {"verbose"  ,  'v', 0,           0,  "Produce verbose output" },
  { 0 }
};

struct arguments
{
  char *args[3];                /* arg1 & arg2 */
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
      if (state->arg_num >= 3)
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
  arguments.verbose = 0;

  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  printf ("Cube = %s\nGraph = %s\nOutputFile = %s\n",
          arguments.args[0], arguments.args[1], arguments.args[2]);

  Cube_P*  cube  = CubeFactory::load (arguments.args[0]);
  Graph<Point3D, Edge<Point3D> >* graph = new Graph<Point3D, Edge<Point3D> >(arguments.args[1]);


  vector< vector< double > > lattices = graph->sampleLatticeArroundEdges(cube, 10, 5, 3, 0.5, 3.5);
  saveMatrix(lattices, arguments.args[2]);

}
