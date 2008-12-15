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
#include "Cube.h"
#include "CubeFactory.h"
#include "Neuron.h"
#include "Cloud.h"
#include <argp.h>
#include <gsl/gsl_rng.h>


using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "imageToCloud 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "imageToCloud creates a cloud with the pixels that are not cero on the image";

/* A description of the arguments we accept. */
static char args_doc[] = "image cloud";

/* The options we understand. */
static struct argp_option options[] = {
  {"theta", 't',  "cube with theta", 0, "if defined, saves the theta orientation in the cloud"},
  {"phi",   'p',  "cube with phi",   0, "if defined, saves the phi orientation in the cloud"},
  {"mask",          'm',  "mask_image",        0, "if true, do not take any point whose mask is 0"},
  {"type",          'y',  0,                   0, "if true saves the type of the points"},
  {"numberPositive", 'N', "positive_points",-  0, "if defined, the number of positive points to get"},
  {"numberNegative", 'M', "negative_points",-  0, "if defined, the number of negative points to get"},
  { 0 }
};

struct arguments
{
  string name_theta;
  string name_phi;
  string name_cube;
  string name_cloud;
  string name_mask;
  int    number_points;
  int    number_negative_points;
  bool   save_type;
  bool   save_negative;
  bool   save_orientation;
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
      argments->name_theta = arg;
      argments->save_orientation = true;
      break;
    case 'p':
      argments->name_phi = arg;
      argments->save_orientation = true;;
      break;
    case 'm':
      argments->name_mask = arg;
      break;
    case 'y':
      argments->save_type = true;
      break;
    case 'N':
      argments->number_points = atoi(arg);
      break;
    case 'M':
      argments->number_negative_points = atoi(arg);
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 2)
      /* Too many arguments. */
        argp_usage (state);
      if (state->arg_num == 0)
        argments->name_cube = arg;
      if (state->arg_num == 1)
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

enum Mode {c3D, c3Do, c3Dt, c3Dot} mode;


// Given a cube, return a random point of it whose value is lower than 100
vector< int > samplePoint(Cube< uchar, ulong>* cube, gsl_rng* r)
{
  // Fix depth
  int x, y, z, nPointsLine, xRand, xCurr;
  int nLinesVisited = 0;
  int nLayersVisited = 0;
  vector< int > toRet(3);
  vector< bool > lines_visited(cube->cubeHeight);
  for(int i=0; i < cube->cubeHeight; i++)
    lines_visited[i] = false;
  vector< bool > layers_visited(cube->cubeDepth);
  for(int i=0; i < cube->cubeDepth; i++)
    layers_visited[i] = false;

  while( nLayersVisited <= cube->cubeDepth){
    //Change this for the search of the layer
    z = (int)(gsl_rng_uniform(r)*cube->cubeDepth);
    if(layers_visited[z] == true) {
      int zoffset = 0;
      while( (layers_visited[ (z+zoffset)%cube->cubeDepth]== true) &&
             (zoffset < cube->cubeDepth) ){
        zoffset ++;
      }
      z = (z + zoffset)%cube->cubeDepth;
    }
    layers_visited[z] = true;
    nLayersVisited++;
    nLinesVisited = 0;
    while( (nLinesVisited <= cube->cubeHeight) ){
      y = (int)(gsl_rng_uniform(r)*cube->cubeHeight);
      //Change this for the search of the line
      if(lines_visited[y] == true) {
        int yoffset = 0;
        while( (lines_visited[ (y+yoffset)%cube->cubeHeight]== true) &&
               (yoffset < cube->cubeHeight) ){
          yoffset ++;
        }
        y = (y + yoffset)%cube->cubeHeight;
      }
      lines_visited[y] = true;
      nLinesVisited++;
      nPointsLine == 0;
      for(int x = 0; x < cube->cubeWidth; x++){
        if(cube->at(x,y,z) < 100){
          nPointsLine++;
        }
      }
      if(nPointsLine == 0) continue;
      else {
        xCurr = 0;
        xRand = (int)(gsl_rng_uniform(r)*(nPointsLine+1));
        for(int x = 0; x < cube->cubeWidth; x++){
          if(cube->at(x,y,z) < 100){
            xCurr++;
            if(xCurr == xRand){
              toRet[0] = x; toRet[1] = y; toRet[2] = z;
              return toRet;
            }
          }
        }
      }
    }
  }

  //In case there is no point in the cube
  toRet[0] = cube->cubeWidth; toRet[1] = cube->cubeHeight;
  toRet[2] = cube->cubeDepth;
  return toRet;
}


int main(int argc, char **argv) {

  struct arguments args;
  /* Default values. */
  args.name_cube = "";
  args.name_theta = "";
  args.name_phi = "";
  args.name_mask = "";
  args.name_cloud = "cloud.cl";
  args.save_negative = false;
  args.save_type     = false;
  args.number_points   = 0;
  args.number_negative_points = 0;

  mode = c3D;

  argp_parse (&argp, argc, argv, 0, 0, &args);

  printf("Cube: %s\nTheta: %s\nPhi: %s\nOut: %s\nMsk: %s\nSave_negative: %i\nSave_type: %i\nSave_orientation: %i\nNumber_points: %i\nNumber_negative_points: %i\n",
         args.name_cube.c_str(),args.name_theta.c_str(),
         args.name_phi.c_str(), args.name_cloud.c_str(),
         args.name_mask.c_str(),
         args.save_negative, args.save_type,
         args.save_orientation, args.number_points,
         args.number_negative_points);

//   Cube_P* cborig = CubeFactory::load(args.name_cube);
//   if (cborig->type == "uchar")
//     cborig = dynamic_cast<Cube<uchar,ulong>*>(cborig);
//   else if (cborig->type == "float")
//     cborig = dynamic_cast<Cube<float,double>*>(cborig);
//   else{
//     printf("cubeToCloud: the cube is not of floats or uchars, exiting...\n");
//     exit(0);
//   }
  Cube<uchar, ulong>* cborig = new Cube<uchar, ulong>(args.name_cube);
  Cube<float, double>* theta;
  Cube<float, double>* phi;
  if( args.name_theta != "")
    theta = new Cube<float, double>(args.name_theta);
  if( args.name_phi != "")
    phi = new Cube<float, double>(args.name_phi);

  vector< float > micrometers(3);

//   Cloud_P* cloud;
//   if(args.save_orientation && args.save_type){
//     cloud = new Cloud< Point3Dot >();
//     mode = c3Dot;}
//   if(args.save_orientation && !args.save_type){
//     cloud = new Cloud< Point3Do >();
//     mode = c3Do;}
//   if(!args.save_orientation && args.save_type){
//     cloud = new Cloud< Point3Dt >();
//     mode = c3Dt;}
//   if(!args.save_orientation && !args.save_type){
  Cloud<Point3D>* cloud = new Cloud< Point3D >();
//     mode = c3D;}

  // Random number generation
  const gsl_rng_type * T2;
  gsl_rng * r;
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r = gsl_rng_alloc (T2);


  for(int i = 0; i < args.number_points; i++){
    vector< int > indexes = samplePoint(cborig, r);
    cborig->indexesToMicrometers(indexes, micrometers);
    cloud->points.push_back(new Point3D(micrometers[0],
                                        micrometers[1], micrometers[2]));

  }


  cloud->saveToFile(args.name_cloud);



}
