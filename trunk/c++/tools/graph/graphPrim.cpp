
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

using namespace std;
/** Variables for the arguments.*/
const char *argp_program_version =
  "graphPrim 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "graphPrim computes the MST over a set of points using the euclidean distance";

/* A description of the arguments we accept. */
static char args_doc[] = "cloud.cl outputfile";

/* The options we understand. */
static struct argp_option options[] = {
  {"verbose"  ,  'v', 0,           0,  "Produce verbose output" },
  { 0 }
};

struct arguments
{
  char *args[2];                /* arg1 & arg2 */
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
  arguments.verbose = 0;

  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  printf ("Cloud = %s\n OutputFile = %s\n",
          arguments.args[0], arguments.args[1]);

  //Code starts here
//   Cloud_P* cl = CloudFactory::load(arguments.args[0]);
  Cloud<Point3Dt>* cl = new Cloud<Point3Dt>(arguments.args[0]);
//   Graph_P* gr = GraphFactory

  Graph<Point3Dt,Edge<Point3Dt> >* gr = new Graph<Point3Dt,Edge<Point3Dt> >(cl);
  gr->prim();
  gr->saveToFile(string(arguments.args[1]));
}
