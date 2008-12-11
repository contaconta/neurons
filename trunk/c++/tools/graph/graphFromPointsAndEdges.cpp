
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
#include "Cube.h"
#include "CubeFactory.h"
#include "Graph.h"
#include "Cloud.h"
#include "Edge2W.h"

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "graphPrim 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "graphFromPointsAndEdges converts the format points edges to a graph";

/* A description of the arguments we accept. */
static char args_doc[] = "point_file edges_file graph.gr";

/* The options we understand. */
static struct argp_option options[] = {
  {"volume"  ,    'v', "volume",           0,  "volume to change from indexes to microm" },
  {"doubleWeight"  , 'd', 0,           0,  "Double weighted edges" },
  {"ICM", 'i', "ICM file", 0, "the file where the ICM output is"},
  { 0 }
};

struct arguments
{
  char *args[3];                /* arg1 & arg2 */
  string volume;
  string ICM_name;
  bool flag_doubleWeights;
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
      argments->volume = arg;
      break;
    case 'd':
      argments->flag_doubleWeights = true;
      break;
    case 'i':
      argments->ICM_name = arg;
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
  arguments.volume = "";
  arguments.flag_doubleWeights = false;

  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  printf ("Points = %s\nEdges = %s\nOutputFile = %s\nVolumeFile = %s\n",
          arguments.args[0], arguments.args[1], arguments.args[2],
          arguments.volume.c_str());

  vector< vector< double > > points = loadMatrix(arguments.args[0]);
  vector< vector< double > > edges  = loadMatrix(arguments.args[1]);

  Cube_P* cube = NULL;
  if(arguments.volume != ""){
    cube = CubeFactory::load(arguments.volume);
  }

  vector< int > indexes(3);
  vector< float > micrometers(3);
  //Code starts here
  if(arguments.ICM_name == ""){
    Cloud< Point3D>*  cl = new Cloud<Point3D>();
    // cl = dynamic_cast< Cloud<Point3D> > (cl);
    for(int i = 0; i < points.size(); i++){
      if(cube == NULL)
        cl->points.push_back(new Point3D(points[i][0],points[i][1], points[i][2]));
      else{
        indexes[0] = points[i][0];
        indexes[1] = points[i][1];
        indexes[2] = points[i][2];
        cube->indexesToMicrometers(indexes, micrometers);
        cl->points.push_back(new Point3D(micrometers[0],micrometers[1], micrometers[2]));
      }
    }

    if(arguments.flag_doubleWeights){
      Graph<Point3D,Edge2W<Point3D> >* gr = new Graph<Point3D,Edge2W<Point3D> >(cl);
      for(int i = 0; i < edges.size(); i++){
        Edge2W<Point3D>* e =
          new Edge2W<Point3D>(&gr->cloud.points,
                              edges[i][0], edges[i][1], edges[i][3], edges[i][4]);
        gr->eset.edges.push_back(e) ;
      }
      gr->eset.v_radius = 2.0;
      gr->saveToFile(string(arguments.args[2]));
    }
    else{
      Graph<Point3D,Edge<Point3D> >* gr = new Graph<Point3D,Edge<Point3D> >(cl);
      for(int i = 0; i < edges.size(); i++)
        gr->eset.addEdge(edges[i][0], edges[i][1]);
      gr->saveToFile(string(arguments.args[2]));
    }
  }
  else{
    vector< vector< double > > ICM = loadMatrix(arguments.ICM_name);
    Cloud<Point3Dt>* cl = new Cloud<Point3Dt>();
    for(int i = 0; i < points.size(); i++){
      if(cube == NULL)
        cl->points.push_back(new Point3Dt(points[i][0],points[i][1],
                                          points[i][2], ICM[i][0]));
      else{
        indexes[0] = points[i][0];
        indexes[1] = points[i][1];
        indexes[2] = points[i][2];
        cube->indexesToMicrometers(indexes, micrometers);
        cl->points.push_back(new Point3Dt(micrometers[0],micrometers[1],
                                          micrometers[2], ICM[i][0]));
      }
    }
    if(arguments.flag_doubleWeights){
      Graph<Point3Dt,Edge2W<Point3Dt> >* gr = new Graph<Point3Dt,Edge2W<Point3Dt> >(cl);
      for(int i = 0; i < edges.size(); i++){
        Edge2W<Point3Dt>* e =
          new Edge2W<Point3Dt>(&gr->cloud.points,
                               edges[i][0], edges[i][1], edges[i][3], edges[i][4]);
        gr->eset.edges.push_back(e) ;
      }
      gr->eset.v_radius = 2.0;
      gr->saveToFile(string(arguments.args[2]));
    }
    else{
      Graph<Point3Dt,Edge<Point3Dt> >* gr = new Graph<Point3Dt,Edge<Point3Dt> >(cl);
      for(int i = 0; i < edges.size(); i++)
        gr->eset.addEdge(edges[i][0], edges[i][1]);
      gr->saveToFile(string(arguments.args[2]));
    }
  }

}
