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
#include "CloudFactory.h"
#include "utils.h"
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
  "cubeToCloud creates a cloud with the pixels that are cero on the image as positives";

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
  {"negativeMask",   'Z', "negative_mask",     0, "if defined, the points of this mask that are 0 are the potential negative candidates"},
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
  string name_negativeMask;
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
    case 'Z':
      argments->name_negativeMask = arg;
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

vector< int > pointsBellowThreshold(Cube<uchar, ulong>* cube, float threshold)
{
  vector<int> toRet;
  for(int z = 0; z < cube->cubeDepth; z++)
    for(int y = 0; y < cube->cubeHeight; y++)
      for(int x = 0; x < cube->cubeWidth; x++)
        if(cube->at(x,y,z) < threshold)
          toRet.push_back(z*cube->cubeWidth*cube->cubeHeight +
                          y*cube->cubeWidth + x);
  return toRet;
}


// Given a cube, return a random point of it whose value is lower than 100
vector< int > samplePoint(Cube< uchar, ulong>* cube, gsl_rng* r, vector<int>& points)
{
  int idx = (int)(gsl_rng_uniform(r)*points.size());
  vector< int > toRet(3);
  toRet[2] = (int)( points[idx]/(cube->cubeWidth*cube->cubeHeight));
  toRet[1] = (int)( (points[idx] - toRet[2]*cube->cubeWidth*cube->cubeHeight)/
                    cube->cubeWidth);
  toRet[0] = (int)( points[idx] - toRet[2]*cube->cubeWidth*cube->cubeHeight -
                    toRet[1]*cube->cubeWidth);
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
  args.name_negativeMask = "";
  args.save_negative = false;
  args.save_type     = false;
  args.number_points   = 0;
  args.number_negative_points = 0;
  mode = c3D;

  argp_parse (&argp, argc, argv, 0, 0, &args);

  printf("Cube: %s\nTheta: %s\nPhi: %s\nOut: %s\nMsk: %s\nSave_negative: %i\nSave_type: %i\nSave_orientation: %i\nNumber_points: %i\nNumber_negative_points: %i\n name_negativeMask = %s\n",
         args.name_cube.c_str(),args.name_theta.c_str(),
         args.name_phi.c_str(), args.name_cloud.c_str(),
         args.name_mask.c_str(),
         args.save_negative, args.save_type,
         args.save_orientation, args.number_points,
         args.number_negative_points,
         args.name_negativeMask.c_str());

  Cube<uchar, ulong>* cborig = new Cube<uchar, ulong>(args.name_cube);
  Cube<float, double>* theta;
  Cube<float, double>* phi;
  if( args.name_theta != "")
    theta = new Cube<float, double>(args.name_theta);
  if( args.name_phi != "")
    phi = new Cube<float, double>(args.name_phi);

  vector< float > micr(3);
  vector< int > idx(3);

  Cloud_P* cloud;
  if(fileExists(args.name_cloud)){
    cloud = CloudFactory::load(args.name_cloud);
    if(args.save_orientation && args.save_type){
      mode = c3Dot;
      printf("Mode = 3Dot\n");
    }
    if(args.save_orientation && !args.save_type){
      mode = c3Do;
      printf("Mode = 3Do\n");
    }
    if(!args.save_orientation && args.save_type){
      mode = c3Dt;
      printf("Mode = 3Dt\n");
    }
    if(!args.save_orientation && !args.save_type){
      mode = c3D;
      printf("Mode = 3D\n");
    }
  }
  else{
    if(args.save_orientation && args.save_type){
      cloud = new Cloud< Point3Dot >();
      mode = c3Dot;
      printf("Mode = 3Dot\n");
    }
    if(args.save_orientation && !args.save_type){
      cloud = new Cloud< Point3Do >();
      mode = c3Do;
      printf("Mode = 3Do\n");
    }
    if(!args.save_orientation && args.save_type){
      cloud = new Cloud< Point3Dt >();
      mode = c3Dt;
      printf("Mode = 3Dt\n");
    }
    if(!args.save_orientation && !args.save_type){
      cloud = new Cloud< Point3D >();
      mode = c3D;
      printf("Mode = 3D\n");
    }
  }
  // Random number generation
  const gsl_rng_type * T2;
  gsl_rng * r;
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r = gsl_rng_alloc (T2);

  vector< int > points = pointsBellowThreshold(cborig, 100);

  // Positive points
  for(int i = 0; i < args.number_points; i++){
    vector< int > idx = samplePoint(cborig, r, points);
    // printf("%i %i %i\n", idx[0], idx[1], idx[2]);
    if(idx[0] == cborig->cubeWidth)
      continue;
    cborig->indexesToMicrometers(idx, micr);

    switch(mode)
      {
      case c3D:
        cloud->points.push_back(new Point3D(micr[0],
                                            micr[1], micr[2]));
        break;
      case c3Do:
        cloud->points.push_back(new Point3Do(micr[0],
                                             micr[1], micr[2],
                                             theta->at(idx[0], idx[1], idx[2]),
                                             phi->at(idx[0], idx[1], idx[2])));
        break;
      case c3Dt:
        cloud->points.push_back(new Point3Dt(micr[0],
                                             micr[1], micr[2],1));
        break;
      case c3Dot:
        cloud->points.push_back(new Point3Dot(micr[0],
                                              micr[1], micr[2],
                                              theta->at(idx[0], idx[1], idx[2]),
                                              phi->at(idx[0], idx[1], idx[2]),1
                                              )
                                );
        break;
      }
  }

  //Negative points
  Cube<uchar, ulong>* negativeMask;
  if(args.name_negativeMask != "")
    negativeMask = new Cube<uchar, ulong>(args.name_negativeMask);

  for(int i = 0; i < args.number_negative_points; i++){
    if(args.name_negativeMask == ""){
      //Take white points of the image
      do{
        idx[0] = (int)(gsl_rng_uniform(r)*cborig->cubeWidth);
        idx[1] = (int)(gsl_rng_uniform(r)*cborig->cubeHeight);
        idx[2] = (int)(gsl_rng_uniform(r)*cborig->cubeDepth);
      }while(cborig->at(idx[0],idx[1],idx[2]) < 100);
    } else{
      //Take black points of the negativeMask
      do{
        idx[0] = (int)(gsl_rng_uniform(r)*cborig->cubeWidth);
        idx[1] = (int)(gsl_rng_uniform(r)*cborig->cubeHeight);
        idx[2] = (int)(gsl_rng_uniform(r)*cborig->cubeDepth);
      }while(negativeMask->at(idx[0],idx[1],idx[2]) > 100);
    }
    cborig->indexesToMicrometers(idx, micr);
    switch(mode)
      {
      case c3D:
        cloud->points.push_back(new Point3D(micr[0],
                                            micr[1], micr[2]));
        break;
      case c3Do:
        cloud->points.push_back(new Point3Do(micr[0],
                                             micr[1], micr[2],
                                             theta->at(idx[0], idx[1], idx[2]),
                                             phi->at(idx[0], idx[1], idx[2])));
        break;
      case c3Dt:
        cloud->points.push_back(new Point3Dt(micr[0],
                                             micr[1], micr[2],-1));
        break;
      case c3Dot:
        cloud->points.push_back(new Point3Dot(micr[0],
                                              micr[1], micr[2],
                                              theta->at(idx[0], idx[1], idx[2]),
                                              phi->at(idx[0], idx[1], idx[2]),
                                              -1)
                                );
        break;
      }



  }

  cloud->saveToFile(args.name_cloud);



}
