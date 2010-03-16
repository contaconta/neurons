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
#include "Cloud.h"
#include <argp.h>
#include <gsl/gsl_rng.h>

const gsl_rng_type * T2;
gsl_rng * r;


using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "imageToCloudForFalsePositives 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "imageToCloud creates a cloud with the pixels that are not cero on the image";

/* A description of the arguments we accept. */
static char args_doc[] = "image cloud";

/* The options we understand. */
static struct argp_option options[] = {
  {"orientation",   'o',  "orientation_image", 0, "saves the per-pixel orientations present in the orientation_image, angles in radians in the image"},
  {"mask",          'm',  "mask_image",        0, "if true, do not take any point whose mask is 0"},
  {"symmetrize",    'z', 0,                    0, "if true generates also positive points rotated Pi degrees"},
  {"numberNegative", 'M', "negative_points",-  0, "if defined, the number of negative points to get"},
  { 0 }
};

struct arguments
{
  string name_orientation;
  string name_image;
  string name_cloud;
  string name_mask;
  bool   flag_symmetrize;
  int    number_negative_points;
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
    case 'o':
      argments->name_orientation = arg;
      break;
    case 'm':
      argments->name_mask = arg;
      break;
    case 'z':
      argments->flag_symmetrize = true;
      break;
    case 'M':
      argments->number_negative_points = atoi(arg);
      break;

    case ARGP_KEY_ARG:
      if (state->arg_num >= 2)
      /* Too many arguments. */
        argp_usage (state);
      if (state->arg_num == 0)
        argments->name_image = arg;
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

vector< double > getCumulativePdf(Image< float >* img, Image< float >* mask)
{

  //Only build the cumulative pdf for points that are false positives.
  vector< double > toRet(img->width*img->height);
  double cum = 0;
  for(int y = 0; y < img->height; y++){
    for(int x = 0; x < img->width; x++){
      if((img->at(x,y) > 0) && (mask!= NULL) && (mask->at(x,y) > 100))
        cum = cum + img->at(x,y) ;
      toRet[x + y*img->width] = cum;
    }
  }

  for(int i = 0; i < toRet.size(); i++)
    toRet[i] /= cum;

  return toRet;
}

int findClosestIdxFromOrderedVector(vector< double > &vct, double val, Image<float>* img)
{
  // printf("Trying to get the value %f\n", val);
  //Or fast ...
  int idx_min = 0;
  int idx_max = vct.size()-1;
  while( (idx_max - idx_min) > 1){
    if( vct[ floor((double)(idx_max + idx_min)/2)] > val)
      idx_max = floor((double)(idx_max + idx_min)/2);
    else
      idx_min = floor((double)(idx_max + idx_min)/2);
    // printf("idx_min = %i, idx_max = %i, val_min = %f, val_max = %f\n", idx_min, idx_max,
           // vct[idx_min], vct[idx_max]);
  }





  return idx_max;
}

int sampleCumulativePdf(vector< double > &cpdf, Image<float>* img)
{
  // double val = gsl_rng_uniform(r)*cpdf[cpdf.size()-1];
  double val = gsl_rng_uniform(r);
  // double val = 0.85;
  int idx = findClosestIdxFromOrderedVector(cpdf, val, img);
  return idx;
}



int main(int argc, char **argv) {

  struct arguments args;
  /* Default values. */
  args.name_image = "";
  args.name_orientation = "";
  args.name_mask = "";
  args.name_cloud = "cloud.cl";
  args.flag_symmetrize = false;
  args.number_negative_points = 0;

  argp_parse (&argp, argc, argv, 0, 0, &args);

  printf("Img: %s\nOrs: %s\nOut: %s\nMsk: %s\nSymmetrize: %i\nNumber_points: %i\n",
         args.name_image.c_str(),
         args.name_orientation.c_str(),
         args.name_cloud.c_str(),
         args.name_mask.c_str(),
         args.flag_symmetrize, args.number_negative_points);

  /** Random number generation.*/
  gsl_rng_env_setup();
  T2 = gsl_rng_default;
  r = gsl_rng_alloc (T2);

  Image< float >* img    = new Image< float >(args.name_image);
  Image< float >* orient = NULL;
  Image< float >* mask   = NULL;
  if(args.name_orientation != "")
    orient = new Image<float>(args.name_orientation);
  if(args.name_mask != "")
    mask = new Image<float>(args.name_mask);

  vector< int > indexes(3);
  vector< float > micrometers(3);
  indexes[2] = 0;

  vector< double > cumPDF = getCumulativePdf(img, mask);

  // saveVectorDouble(cumPDF, "cumPDF.txt");

  Cloud< Point2Dot >* cloud;
  if(fileExists(args.name_cloud))
    cloud = new Cloud< Point2Dot >(args.name_cloud);
  else
    cloud = new Cloud< Point2Dot >();

  double ort = 0;

  int nPoints = 0;
  while(nPoints < args.number_negative_points){

    int idx = sampleCumulativePdf(cumPDF, img);
    indexes[1] = (int)idx/img->width;
    indexes[0] = idx - img->width*indexes[1];

    //If the mask is 0, then loop again
    // if(mask != NULL)
      // if(mask->at(indexes[0], indexes[1]) < 100)
        // continue;

    img->indexesToMicrometers(indexes, micrometers);
    if(orient == NULL){
      ort = (gsl_rng_uniform(r)-0.5)*2*M_PI;
    } else {
      ort = orient->at(indexes[0], indexes[1]);
    }

    cloud->points.push_back(new Point2Dot(micrometers[0], micrometers[1],
                                          ort, -1));
    nPoints++;

    if(args.flag_symmetrize){
      cloud->points.push_back(new Point2Dot(micrometers[0], micrometers[1],
                                            ort + M_PI, -1));
      nPoints++;
    }
  }
  cloud->saveToFile(args.name_cloud);
}
