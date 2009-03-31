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
// Written and (C) by Aurelien Lucchi                                  //
// Contact <aurelien.lucchi@gmail.com> for comments & bug reports      //
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

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "colorImageToCloud 0.1";
const char *argp_program_bug_address =
  "<aurelien.lucchi@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "colorImageToCloud creates a cloud with the pixels whose green channel (negative samples) and blue channel (positive samples) have been tagged";

/* A description of the arguments we accept. */
static char args_doc[] = "image cloud";

/* The options we understand. */
static struct argp_option options[] = {
  {"orientation",   'o',  "orientation_image", 0, "saves the per-pixel orientations present in the orientation_image, angles in radians in the image"},
  {"scale",         's',  "scale_image" ,      0, "saves the per-pixel scale present in the scale_image"},
  {"ScaleNormalizetion",'S',"scale_image" ,    0, "samples the pixels so that every scale is evenly normalized"},
  {"type",          't',  0,                   0, "if true saves the type of the points"},
  {"negative",      'n',  0,                   0, "if true generates also negative points"},
  {"symmetrize",    'z', 0,                    0, "if true generates also positive points rotated Pi degrees"},
  {"numberPositive", 'N', "positive_points",-  0, "if defined, the number of positive points to get"},
  {"numberNegative", 'M', "negative_points",-  0, "if defined, the number of negative points to get"},
  {"numberBlueSample", 'B', "number_blue_sample_points",-  0, "if defined, the number of blue sample points to get"},
  {"classBlueSample", 'B', "class_blue_sample_points",-  0, "if defined, the number of blue sample points to get"},
  {"stepBlueSample", 'p', "step_blue_sample_points",-  0, "if defined, the space between the blue sample points"},
  { 0 }
};

struct arguments
{
  string name_orientation;
  string name_image;
  string name_cloud;
  string name_scale_normalization;
  bool   save_type;
  bool   save_negative;
  bool   save_blue_sample;
  bool   flag_symmetrize;
  int    number_points;
  int    number_negative_points;
  int    number_blue_sample_points;
  int    class_blue_sample_points;
  int    step_blue_sample_points;
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
    case 'S':
      argments->name_scale_normalization = arg;
      break;
    case 'z':
      argments->flag_symmetrize = true;
      break;
    case 't':
      argments->save_type = true;
      break;
    case 'n':
      argments->save_negative = true;
      break;
    case 'N':
      argments->number_points = atoi(arg);
      break;
    case 'M':
      argments->number_negative_points = atoi(arg);
      argments->save_negative = true;
      break;
    case 'B':
      argments->number_blue_sample_points = atoi(arg);
      argments->save_blue_sample = true;
      break;
    case 'c':
      argments->class_blue_sample_points = atoi(arg);
      break;
    case 'p':
      argments->step_blue_sample_points = atoi(arg);
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

int main(int argc, char **argv) {

  struct arguments args;
  /* Default values. */
  args.name_image = "";
  args.name_scale_normalization = "";
  args.name_orientation = "";
  args.name_cloud = "cloud.cl";
  args.save_blue_sample = false;
  args.save_negative = false;
  args.save_type     = false;
  args.flag_symmetrize = false;
  args.number_points   = -1;
  args.number_negative_points = -1;
  args.number_blue_sample_points = 0;
  args.class_blue_sample_points = 1;
  args.step_blue_sample_points = 10;

  argp_parse (&argp, argc, argv, 0, 0, &args);

  printf("Img: %s\nOrs: %s\nScl: %s\nSave_negative: %i\nSave_type: %i\nSymmetrize: %i\nNumber_points: %i\n"
         "Scale_norm: %s\nNumber_blue_points: %d\nClass_blue_points: %d\nStep_blue_points: %d\n",
         args.name_image.c_str(),args.name_orientation.c_str(),
         args.name_cloud.c_str(),
         args.save_negative, args.save_type,
         args.flag_symmetrize, args.number_points,
         args.name_scale_normalization.c_str(),
         args.number_blue_sample_points, args.class_blue_sample_points, args.step_blue_sample_points);

  Image< float >* imgf  = new Image< float >(args.name_image);
  IplImage* img = cvLoadImage(args.name_image.c_str(),CV_LOAD_IMAGE_COLOR);
  printf("width %d, height %d, nChannels: %d\n", img->width, img->height, img->nChannels);
  if(img->nChannels!=3)
    {
      printf("Error : input image is not a color image");
      exit(-1);
    }

  vector< int > indexes(3);
  vector< float > micrometers(3);
  indexes[2] = 0;

  if(!args.save_type)
    {
      if(args.name_orientation == "")
        {
          // No orientation
          Cloud< Point2D>* cloud = new Cloud< Point2D >();
          for(int x = 0; x < img->width; x++){
            for(int y = 0; y < img->height; y++){
              // pixel order = B,G,R
              uchar* ptrColImage = &((uchar*)(img->imageData + img->widthStep*y))[x*3];
              if(ptrColImage[2] != ptrColImage[1]){ // Green channel tagged
                indexes[0] = x;
                indexes[1] = y;
                imgf->indexesToMicrometers(indexes, micrometers);
                cloud->points.push_back(new Point2D(micrometers[0],micrometers[1]));
              }
            }
          }
          cloud->saveToFile(args.name_cloud);
        }
      else
        {
          //Case we have orientation
          Image< float >* ors = new Image< float >(args.name_orientation);
          Cloud< Point2Do >* cloud = new Cloud< Point2Do >();
          for(int x = 0; x < img->width; x++){
            for(int y = 0; y < img->height; y++){
              uchar* ptrColImage = &((uchar*)(img->imageData + img->widthStep*y))[x*3];
              if(ptrColImage[2] != ptrColImage[1]){ // Green channel tagged
                indexes[0] = x;
                indexes[1] = y;
                imgf->indexesToMicrometers(indexes, micrometers);
                cloud->points.push_back(
                                        new Point2Do(micrometers[0],micrometers[1]
                                                     , ors->at(x,y)));
                if(args.flag_symmetrize){
                  cloud->points.push_back(
                                          new Point2Do(micrometers[0],micrometers[1]
                                                       , ors->at(x,y)+M_PI ));
                }
              }
            }
          }
          cloud->saveToFile(args.name_cloud);
        }
    }
  else
    {
      // Save type
      const gsl_rng_type * T2;
      gsl_rng * r;
      gsl_rng_env_setup();
      T2 = gsl_rng_default;
      r = gsl_rng_alloc (T2);

      Image< float >* ors;
      float orientation;
      if(args.name_orientation != "")
	ors=new Image< float >(args.name_orientation);
      else
	ors=0;
      Cloud< Point2Dot >* cloud = new Cloud< Point2Dot >();

      // If we pass all the image
      if(args.number_points == 0){
        for(int x = 0; x < img->width; x++){
          for(int y = 0; y < img->height; y++){
            uchar* ptrColImage = &((uchar*)(img->imageData + img->widthStep*y))[x*3];
            if(ptrColImage[2] != ptrColImage[1]){ // Green channel tagged
              indexes[0] = x;
              indexes[1] = y;
              imgf->indexesToMicrometers(indexes, micrometers);
	      if(ors)
		orientation=ors->at(x,y);
	      else
		orientation=0;
              cloud->points.push_back(
                                      new Point2Dot(micrometers[0],micrometers[1]
                                                    , orientation, +1));
              if(args.flag_symmetrize){
                cloud->points.push_back(
                                        new Point2Dot(micrometers[0],micrometers[1]
                                                      , orientation+M_PI, 1 ));
              }
            }
          }
        }
      } // In case we want to limit the positive points
      else {
        int nPositivePoints = 0;
        int x, y;
        while(nPositivePoints < args.number_points)
          {
            x = (int)floor(gsl_rng_uniform(r)*img->width);
            y = (int)floor(gsl_rng_uniform(r)*img->height);
            uchar* ptrColImage = &((uchar*)(img->imageData + img->widthStep*y))[x*3];
            if(ptrColImage[2] != ptrColImage[1]){ // Green channel tagged
              indexes[0] = x;
              indexes[1] = y;
              imgf->indexesToMicrometers(indexes, micrometers);
	      if(ors)
		orientation=ors->at(x,y);
	      else
		orientation=0;
              cloud->points.push_back(
                                      new Point2Dot(micrometers[0],micrometers[1],
                                                    orientation, 1));
              nPositivePoints++;
              if(args.flag_symmetrize){
                cloud->points.push_back(
                                        new Point2Dot(micrometers[0],micrometers[1],
                                                      orientation+M_PI, 1 ));
                nPositivePoints ++;
              }
            }
          }
      }

      // In case we want to save the negative points
      // generate as many negative points as there are positive
      // they will be taken as random from the clear points
      if(args.save_negative)
        {
          //Creates the random number generator
          int nNegativePoints = 0;
          int nPositivePoints = cloud->points.size();
          int x, y;
          int limitNegative = 0;
          if(args.number_negative_points == -1)
            limitNegative = nPositivePoints;
          else
            limitNegative = args.number_negative_points;

	  printf("limitNegative:%d\n", limitNegative);

          int idx = 0;
          const int maxTry = limitNegative*100000;
          while(nNegativePoints < limitNegative){

            x = (int)floor(gsl_rng_uniform(r)*img->width);
            y = (int)floor(gsl_rng_uniform(r)*img->height);

            uchar* ptrColImage = &((uchar*)(img->imageData + img->widthStep*y))[x*3];
            if(ptrColImage[2] == ptrColImage[1] && ptrColImage[2] == ptrColImage[0]){ // No channel tagged
              indexes[0] = x;
              indexes[1] = y;
              imgf->indexesToMicrometers(indexes, micrometers);
	      if(ors)
		orientation=(gsl_rng_uniform(r)-0.5)*2*M_PI;
	      else
		orientation=0;
              cloud->points.push_back(
                                      new Point2Dot(micrometers[0],micrometers[1],
                                                    orientation,
                                                    -1));
              nNegativePoints++;
            }

            idx++;
            if(idx > maxTry)
              {
                printf("Number of max try achieved. Are you sure you have enough negative points in your image ?\n");
                exit(-1);
              }
          }
        }

      if(args.save_blue_sample)
        {
          //Creates the random number generator
          int nNegativePoints = 0;
          int nPositivePoints = cloud->points.size();
          int limitNegative = 0;
          int x= 0;
          int y = 0;
          if(args.number_blue_sample_points == 0)
            limitNegative = nPositivePoints;
          else
            limitNegative = args.number_blue_sample_points;

	  printf("limitBlueNegative:%d\n", limitNegative);          

          for(int x = 0; x < img->width && (nNegativePoints < limitNegative); x+=args.step_blue_sample_points){
            for(int y = 0; y < img->height; y+=args.step_blue_sample_points){

              if(nNegativePoints >= limitNegative)
                break;
              
              uchar* ptrColImage = &((uchar*)(img->imageData + img->widthStep*y))[x*3];
              if(ptrColImage[2] != ptrColImage[0]){ // Blue channel tagged

                //printf("add xy %d %d\n", x,y);
                indexes[0] = x;
                indexes[1] = y;
                imgf->indexesToMicrometers(indexes, micrometers);
                if(ors)
                  orientation=(gsl_rng_uniform(r)-0.5)*2*M_PI;
                else
                  orientation=0;
                cloud->points.push_back(
                                        new Point2Dot(micrometers[0],micrometers[1],
                                                      orientation,
                                                      args.class_blue_sample_points));
                nNegativePoints++;
              }
            }
          }
        }

      cloud->saveToFile(args.name_cloud);
    }


}
