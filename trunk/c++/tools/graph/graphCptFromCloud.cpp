
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
#include <fstream>
#include <cmath>
#include <stdio.h>
#include <stdlib.h>
#include "Graph.h"
#include "CloudFactory.h"
#include <argp.h>

using namespace std;

/* Parse a single option. */
const char *argp_program_version =
  "0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  " ";

/* A description of the arguments we accept. */
static char args_doc[] = " ";

/* The options we understand. */
static struct argp_option options[] = {
  {"radious",   'r',  "5.0", 0, "connects each point with the others within a radious r"},
  {"k-nn",      'k',  "5"  , 0, "connects each point to the knn, default"},
  { 0 }
};

struct arguments
{
  float radious;
  int   k;
  bool compute_radious;
  bool compute_k;
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
    case 'r':
      argments->radious = atof(arg);
      argments->compute_radious = true;
      break;
    case 'k':
      argments->k = atoi(arg);
      argments->compute_k = true;
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

  struct arguments args;
  /* Default values. */
  args.compute_radious = false;
  args.compute_k = true;
  args.radious = 5.0;
  args.k = 5;
  argp_parse (&argp, argc, argv, 0, 0, &args);

  Cloud_P* cloud   = CloudFactory::load(args.args[0]);
  Graph<Point3D, EdgeW<Point3D> >* gr = new  Graph<Point3D, EdgeW<Point3D> >(cloud);

  if(args.compute_radious){
    float    radious = args.radious;
    for(int i = 0; i < cloud->points.size(); i++){
      Point* pt1 = cloud->points[i];
      for(int j = i+1; j < cloud->points.size(); j++){
        Point* pt2 = cloud->points[j];
        float dist =  sqrt( (pt1->coords[0]-pt2->coords[0])*(pt1->coords[0]-pt2->coords[0]) +
                            (pt1->coords[1]-pt2->coords[1])*(pt1->coords[1]-pt2->coords[1]) +
                            (pt1->coords[2]-pt2->coords[2])*(pt1->coords[2]-pt2->coords[2]) );
        if(dist < radious)
          gr->eset.edges.push_back
            (new EdgeW<Point3D>(&gr->cloud->points,i, j, dist));
      }
    }
  }
  if(args.compute_k){
    int k = args.k;
    char visited[cloud->points.size()][cloud->points.size()];
    for(int i = 0; i < cloud->points.size(); i++){
      Point* pt1 = cloud->points[i];

      std::multimap<float, int> map;
      for(int j = 0; j < cloud->points.size(); j++){
        if(i==j) continue; //we do not want to include e(p1,p1)
        Point* pt2 = cloud->points[j];
        float distance =
          sqrt( (pt1->coords[0]-pt2->coords[0])*(pt1->coords[0]-pt2->coords[0]) +
                (pt1->coords[1]-pt2->coords[1])*(pt1->coords[1]-pt2->coords[1]) +
                (pt1->coords[2]-pt2->coords[2])*(pt1->coords[2]-pt2->coords[2]) );
        map.insert(pair<float, int>(distance, j));
      }

      multimap<float, int>::iterator ito = map.begin();
      for(int e = 0; e < k; e++) ++ito;

      for ( multimap<float, int>::iterator it = map.begin();
           it != ito;
            ++it){
        int j = (*it).second;
        if(visited[i][j]==0){
          gr->eset.edges.push_back
            (new EdgeW<Point3D>(&gr->cloud->points,i, j, it->first));
          visited[i][j] = 1;
          visited[j][i] = 1;
        }
      }

    } // loop for point i
  }//compute k

  gr->saveToFile(args.args[1]);
}
