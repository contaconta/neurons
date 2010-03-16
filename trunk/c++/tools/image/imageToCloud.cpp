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
  {"orientation",   'o',  "orientation_image", 0, "saves the per-pixel orientations present in the orientation_image, angles in radians in the image"},
  {"scale",         's',  "scale_image" ,      0, "saves the per-pixel scale present in the scale_image"},
  {"ScaleNormalizetion",'S',"scale_image" ,    0, "samples the pixels so that every scale is evenly normalized"},
  {"type",          't',  0,                   0, "if true saves the type of the points"},
  {"negative",      'n',  0,                   0, "if true generates also negative points"},
  {"mask",          'm',  "mask_image",        0, "if true, do not take any point whose mask is 0"},
  {"symmetrize",    'z', 0,                    0, "if true generates also positive points rotated Pi degrees"},
  {"numberPositive", 'N', "positive_points",-  0, "if defined, the number of positive points to get"},
  {"numberNegative", 'M', "negative_points",-  0, "if defined, the number of negative points to get"},
  {"save_index",     'i',  0,                   0, "if true output pixel coordinates instead of micrometers"},
  { 0 }
};

struct arguments
{
  string name_orientation;
  string name_scale;
  string name_image;
  string name_cloud;
  string name_mask;
  string name_scale_normalization;
  bool   save_type;
  bool   save_negative;
  bool   flag_symmetrize;
  int    number_points;
  int    number_negative_points;
  bool   save_index;
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
    case 's':
      argments->name_scale = arg;
      break;
    case 'S':
      argments->name_scale_normalization = arg;
      break;
    case 'm':
      argments->name_mask = arg;
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
    case 'i':
      argments->save_index = true;
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

void sampleWidthNormalization(arguments &args)
{
  printf("In theory I should be here: img %s\n", args.name_scale_normalization.c_str());

  Image< float >* img  = new Image< float >(args.name_image);

  vector< int > indexes(3);
  vector< float > micrometers(3);
  indexes[2] = 0;

  if( (args.name_orientation != "") &&
      args.save_type
      )
    {

      const gsl_rng_type * T2;
      gsl_rng * r;
      gsl_rng_env_setup();
      T2 = gsl_rng_default;
      r = gsl_rng_alloc (T2);

      Image< float >* ors = new Image< float >(args.name_orientation);
      Image< float >* scale = new Image< float>(args.name_scale_normalization);
      Cloud< Point2Dot >* cloud = new Cloud< Point2Dot >();

      // // If we pass all the image
      // if(args.number_points == 0){
        // for(int x = 0; x < img->width; x++){
          // for(int y = 0; y < img->height; y++){
            // if(img->at(x,y) < 100){
              // indexes[0] = x;
              // indexes[1] = y;
              // img->indexesToMicrometers(indexes, micrometers);
              // cloud->points.push_back(
                                      // new Point2Dot(micrometers[0],micrometers[1]
                                                    // , ors->at(x,y), +1));
              // if(args.flag_symmetrize){
                // cloud->points.push_back(
                                        // new Point2Dot(micrometers[0],micrometers[1]
                                                      // , ors->at(x,y)+M_PI, 1 ));
              // }
            // }
          // }
        // }
      // } // In case we want to limit the positive points
      // else {

      //Calculates the histogram ignoring the lower value (in this case, 0)
      vector< int   > hist;
      vector< float > values;
      scale->histogram(6, hist, values, true);

      for(int i = 0; i < hist.size(); i++){
        printf("%f : %i\n", values[i], hist[i]);
      }

      int nBin = 1; //ignore the first bin
      int nPositivePoints = 0;
      int x, y;
      while(nPositivePoints < args.number_points)
        {
          y = (int)floor(gsl_rng_uniform(r)*img->height);

          //Check for the existance of one of the points supposed to be on the bin in the image
          int nLinePointsInBin = 0;
          float min = values[nBin];
          float max = 0;
          if(nBin == values.size()-1)
            max = values[nBin]*values.size();
          else
            max = values[nBin+1];

          for(int x_i = 0; x_i < scale->width; x_i++)
            if( (scale->at(x_i,y) >= min) &&
                (scale->at(x_i,y) <= max) )
              nLinePointsInBin+=1;
          // printf("Row %i min %f max %f nLinePointsInBin %i\n", y, min, max, nLinePointsInBin);
          // printf("nBin = %i, max = %f and min = %f\n", nBin, max, min);
          // exit(0);


          if(nLinePointsInBin == 0)
            continue;
          else{
            int idx_p = (int)floor(gsl_rng_uniform(r)*(nLinePointsInBin+1));
            int idx_c = 0;
            for(int x_i = 0; x_i < img->width; x_i++)
              if( (scale->at(x_i,y) >= min) &&
                  (scale->at(x_i,y) < max) ){
                if(idx_c == idx_p){
                  x = x_i;
                  break;
                } else {
                  idx_c++;
                }
              }
          }

          //Just to check if the point is part of the mask
          if( img->at(x,y) > 100)
            continue;

          //If we have reached this point, we already have x and y
          indexes[0] = x;
          indexes[1] = y;
          img->indexesToMicrometers(indexes, micrometers);
          if(args.save_index)
            cloud->points.push_back(
                                  new Point2Dot(indexes[0],indexes[1],
                                                ors->at(x,y), 1));
          else
            cloud->points.push_back(
                                  new Point2Dot(micrometers[0],micrometers[1],
                                                ors->at(x,y), 1));
          nPositivePoints++;
          if(args.flag_symmetrize){
            if(args.save_index)
              cloud->points.push_back(
                                    new Point2Dot(indexes[0],indexes[1]
                                                  , ors->at(x,y)+M_PI, 1 ));
            else
              cloud->points.push_back(
                                    new Point2Dot(micrometers[0],micrometers[1]
                                                  , ors->at(x,y)+M_PI, 1 ));
            nPositivePoints ++;
          }
          //And now we advance in the bin
          nBin++;
          if(nBin == hist.size()-2) //ignore the last bin
            nBin = 0; // we skip the first bin
        }
      // }

      // In case we want to save the negative points
      // generate as many negative points as there are positive
      // they will be taken as random from the clear points
      if(args.save_negative)
        {
          printf("Should generate the negative points\n");
          Image<float>* mask = NULL;
          if(args.name_mask != ""){
            mask = new Image<float>(args.name_mask);
          }
          //Creates the random number generator
          int nNegativePoints = 0;
          int nPositivePoints = cloud->points.size();
          int x, y;
          int limitNegative = 0;
          if(args.number_negative_points == 0)
            limitNegative = nPositivePoints;
          else
            limitNegative = args.number_negative_points;

          while(nNegativePoints < limitNegative){
            x = (int)floor(gsl_rng_uniform(r)*img->width);
            y = (int)floor(gsl_rng_uniform(r)*img->height);
            if(mask!=NULL){
              if(mask->at(x,y) < 100)
                continue;
            }
            if(img->at(x,y) > 100){
              indexes[0] = x;
              indexes[1] = y;
              img->indexesToMicrometers(indexes, micrometers);
              if(args.save_index)
                cloud->points.push_back(
                                      new Point2Dot(indexes[0],indexes[1]
                                                    , (gsl_rng_uniform(r)-0.5)*2*M_PI,
                                                    -1));
              else
                cloud->points.push_back(
                                      new Point2Dot(micrometers[0],micrometers[1]
                                                    , (gsl_rng_uniform(r)-0.5)*2*M_PI,
                                                    -1));
              nNegativePoints++;
            }
          }
        }
      cloud->saveToFile(args.name_cloud);
    }
  else {
    printf("You called to construct the cloud with a histogram normalization, nevertheless, you did not indicated many other things, exiting ...\n");
  }


}



int main(int argc, char **argv) {

  struct arguments args;
  /* Default values. */
  args.name_image = "";
  args.name_scale = "";
  args.name_scale_normalization = "";
  args.name_orientation = "";
  args.name_mask = "";
  args.name_cloud = "cloud.cl";
  args.save_negative = false;
  args.save_type     = false;
  args.flag_symmetrize = false;
  args.number_points   = 0;
  args.number_negative_points = 0;
  args.save_index = false;

  argp_parse (&argp, argc, argv, 0, 0, &args);

  printf("Img: %s\nOrs: %s\nScl: %s\nOut: %s\nMsk: %s\nSave_negative: %i\nSave_type: %i\nSymmetrize: %i\nNumber_points: %i\n"
         "Scale_norm: %s\n",
         args.name_image.c_str(),args.name_orientation.c_str(),
         args.name_scale.c_str(), args.name_cloud.c_str(),
         args.name_mask.c_str(),
         args.save_negative, args.save_type,
         args.flag_symmetrize, args.number_points,
         args.name_scale_normalization.c_str() );



  //If we want to do equally sampling in width
  if(args.name_scale_normalization != ""){
    sampleWidthNormalization(args);
    exit(0);
  }


  Image< float >* img  = new Image< float >(args.name_image);

  vector< int > indexes(3);
  vector< float > micrometers(3);
  indexes[2] = 0;


  //Case that we have just the image without orientation information
  if( (args.name_scale == "") &&
      (args.name_orientation == "") &&
      !args.save_type
      )
    {
      Cloud< Point2D>* cloud = new Cloud< Point2D >();
      for(int x = 0; x < img->width; x++){
        for(int y = 0; y < img->height; y++){
          if(img->at(x,y) < 100){
            indexes[0] = x;
            indexes[1] = y;
            img->indexesToMicrometers(indexes, micrometers);
            if(args.save_index)
              cloud->points.push_back(new Point2D(indexes[0],indexes[1]));
            else
              cloud->points.push_back(new Point2D(micrometers[0],micrometers[1]));
          }
        }
      }
      cloud->saveToFile(args.name_cloud);
    }

  //Case we have orientation and not scale
  if( (args.name_scale == "") &&
      (args.name_orientation != "") &&
      !args.save_type
      )
    {
      Image< float >* ors = new Image< float >(args.name_orientation);
      Cloud< Point2Do >* cloud = new Cloud< Point2Do >();
      for(int x = 0; x < img->width; x++){
        for(int y = 0; y < img->height; y++){
          if(img->at(x,y) < 100){
            indexes[0] = x;
            indexes[1] = y;
            img->indexesToMicrometers(indexes, micrometers);
            if(args.save_index)
              cloud->points.push_back(
                        new Point2Do(indexes[0],indexes[1]
                                     , ors->at(x,y) ));
            else
              cloud->points.push_back(
                        new Point2Do(micrometers[0],micrometers[1]
                                     , ors->at(x,y)));
            if(args.flag_symmetrize){
              if(args.save_index)
                cloud->points.push_back(
                                        new Point2Do(indexes[0],indexes[1]
                                                     , ors->at(x,y)+M_PI ));
              else
                cloud->points.push_back(
                                        new Point2Do(micrometers[0],micrometers[1]
                                                     , ors->at(x,y)+M_PI ));
            }
          }
        }
      }
      cloud->v_radius = 0.1;
      cloud->saveToFile(args.name_cloud);
    }

  // In case we want to save the orientation and the type
  if( (args.name_scale == "") &&
      //      (args.name_orientation != "") &&
      args.save_type
      )
    {

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
            if(img->at(x,y) < 100){
              indexes[0] = x;
              indexes[1] = y;
              img->indexesToMicrometers(indexes, micrometers);
	      if(ors)
		orientation=ors->at(x,y);
	      else
		orientation=0;
              if(args.save_index)
                cloud->points.push_back(
                                      new Point2Dot(indexes[0],indexes[1]
                                                    , orientation, +1));
              else
                cloud->points.push_back(
                                      new Point2Dot(micrometers[0],micrometers[1]
                                                    , orientation, +1));
              if(args.flag_symmetrize){
                if(args.save_index)
                  cloud->points.push_back(
                                        new Point2Dot(indexes[0],indexes[1]
                                                      , orientation+M_PI, 1 ));
                else
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

        int idx = 0;
        const int maxTry = args.number_points*100000;
        while(nPositivePoints < args.number_points)
          {
            x = (int)floor(gsl_rng_uniform(r)*img->width);
            y = (int)floor(gsl_rng_uniform(r)*img->height);
            if(img->at(x,y) < 100){
              indexes[0] = x;
              indexes[1] = y;
              img->indexesToMicrometers(indexes, micrometers);
	      if(ors)
		orientation=ors->at(x,y);
	      else
		orientation=0;
              if(args.save_index)
                cloud->points.push_back(
                                      new Point2Dot(indexes[0],indexes[1],
                                                    orientation, 1));
              else
                cloud->points.push_back(
                                      new Point2Dot(micrometers[0],micrometers[1],
                                                    orientation, 1));
              nPositivePoints++;
              if(args.flag_symmetrize){
                if(args.save_index)
                  cloud->points.push_back(
                                       new Point2Dot(indexes[0],indexes[1],
                                                      orientation+M_PI, 1 ));
                else
                  cloud->points.push_back(
                                        new Point2Dot(micrometers[0],micrometers[1],
                                                      orientation+M_PI, 1 ));
                nPositivePoints ++;
              }
            }

            // Check if number of max try achieved
            idx++;
            if(idx > maxTry)
              {
                printf("Number of max try achieved. Are you sure you have enough negative points in your image ?\n");
                break;
              }
          }
      }

      // In case we want to save the negative points
      // generate as many negative points as there are positive
      // they will be taken as random from the clear points
      if(args.save_negative)
        {
          Image<float>* mask = NULL;
	  printf("args.name_mask:%s\n", args.name_mask.c_str());
          if(args.name_mask != ""){
            mask = new Image<float>(args.name_mask);
          }
          //Creates the random number generator
          int nNegativePoints = 0;
          int nPositivePoints = cloud->points.size();
          int x, y;
          int limitNegative = 0;
          if(args.number_negative_points == 0)
            limitNegative = nPositivePoints;
          else
            limitNegative = args.number_negative_points;

	  printf("limitNegative:%d\n", limitNegative);

          int idx = 0;
          const int maxTry = limitNegative*100000;
          while(nNegativePoints < limitNegative)
            {

              x = (int)floor(gsl_rng_uniform(r)*img->width);
              y = (int)floor(gsl_rng_uniform(r)*img->height);
              if(mask!=NULL){
                if(mask->at(x,y) < 100)
                  continue;
              }
              if(img->at(x,y) > 100){
                indexes[0] = x;
                indexes[1] = y;
                img->indexesToMicrometers(indexes, micrometers);
                if(ors)
                  orientation=(gsl_rng_uniform(r)-0.5)*2*M_PI;
                else
                  orientation=0;
                if(args.save_index)
                  cloud->points.push_back(
                                        new Point2Dot(indexes[0],indexes[1],
                                                      orientation,
                                                      -1));
                else
                  cloud->points.push_back(
                                        new Point2Dot(micrometers[0],micrometers[1],
                                                      orientation,
                                                      -1));
                nNegativePoints++;
              }

              // Check if number of max try achieved
              idx++;
              if(idx > maxTry)
                {
                  printf("Number of max try achieved. Are you sure you have enough negative points in your image ?\n");
                  break;
                }
            }
        }
      cloud->saveToFile(args.name_cloud);
    }

  // printf("Here we are: %s %s %i\n", args.name_scale, args.name_orientation, args.save_type);

  // In case we want to save the orientation and the type
  if( (args.name_scale != "") &&
      (args.name_orientation != "") &&
      args.save_type
      )
    {

      const gsl_rng_type * T2;
      gsl_rng * r;
      gsl_rng_env_setup();
      T2 = gsl_rng_default;
      r = gsl_rng_alloc (T2);

      printf("Creating the negative points for the Cloud<Point2Dotw>\n"); 
      Image<float>* scaleImg = new Image<float>(args.name_scale);
      Image<float>* thetaImg = new Image<float>(args.name_orientation);
      Cloud<Point2Dotw>* cl = new Cloud<Point2Dotw>();

      float orientation, scale;
      int nNegativePoints = 0;
      int x, y;
      int limitNegative = args.number_negative_points;
      while(nNegativePoints < limitNegative)
        {
          x = (int)floor(gsl_rng_uniform(r)*img->width);
          y = (int)floor(gsl_rng_uniform(r)*img->height);
          if(img->at(x,y) > 100){
            indexes[0] = x;
            indexes[1] = y;
            img->indexesToMicrometers(indexes, micrometers);
            orientation=thetaImg->at(x,y);
            scale = gsl_rng_uniform(r);
            cl->points.push_back
              ( new Point2Dotw(micrometers[0],micrometers[1],
                               orientation,-1, 5*pow(10,-scale)));
            nNegativePoints++;
          }
        }
      cl->saveToFile(args.name_cloud);
    }//Point2Dotw



}
