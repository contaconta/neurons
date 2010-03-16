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
#include <argp.h>

using namespace std;


/* Parse a single option. */
const char *argp_program_version =
  "imageFill 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "imageFill fills the points that are zero on a mask image with the most similar ones on the original one";

/* A description of the arguments we accept. */
static char args_doc[] = "image mask output";

/* The options we understand. */
static struct argp_option options[] = {
  { 0 }
};

struct arguments
{
  string image;
  string mask;
  string output;

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
    case ARGP_KEY_ARG:
      if (state->arg_num >= 3)
      /* Too many arguments. */
        argp_usage (state);
      if (state->arg_num == 0)
        argments->image = arg;
      if (state->arg_num == 1)
        argments->mask = arg;
      if (state->arg_num == 2)
        argments->output = arg;
      break;

    case ARGP_KEY_END:
      /* Not enough arguments. */
      if (state->arg_num < 3)
        argp_usage (state);
      break;

    default:
      return ARGP_ERR_UNKNOWN;
    }
  return 0;
}

/* Our argp parser. */
static struct argp argp = { options, parse_opt, args_doc, doc };

int getXIndex(Image< float > * img, int x){
  if (x < 0)
    return 0;
  else if (x >= img->width)
    return img->width -1;
  else
    return x;
}

int getYIndex(Image< float > * img, int y){
  if (y < 0)
    return 0;
  else if (y >= img->height)
    return img->height -1;
  else
    return y;
}


int main(int argc, char **argv) {

  struct arguments arguments;
  /* Default values. */
  arguments.image = "";
  arguments.mask = "";
  arguments.output = "";

  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  string dir = getDirectoryFromPath(arguments.output);

  Image< float >* img    = new Image< float >(arguments.image);
  Image< float >* mask   = new Image< float >(arguments.mask);
  Image< float >* output = img->copy(arguments.output);
  Image< float >* tmp    = img->copy(dir + "tmp.jpg");

  std::cout << arguments.image << std::endl;


  bool visited[img->height][img->width];
  bool visited_tmp[img->height][img->width];
 

  for(int y = 0; y < img->height; y++)
    for(int x = 0; x < img->width; x++)
      if(mask->at(x,y) < 100){
        visited[y][x] = 0;
        visited_tmp[y][x] = 0;
      }
      else{
        visited[y][x] = 1;
        visited_tmp[y][x] = 1;
      }

  bool there_is_change = true;
  float nNeighborsInImage = 0;
  float valueInNeighbors = 0;
  int nRounds = 0;

  while(there_is_change){
    there_is_change = false;

    // printf("Nrounds = %i\n", nRounds++);

    for(int y = 0; y < img->height; y++){
      for(int x = 0; x < img->width; x++){

    // for(int y = img->height-1; y >= 0; y--){
      // for(int x = img->width-1; x >=0; x--){

        if(visited[y][x] == 1)
          continue;

        //Check for the neighbors of (x,y), see if they have some value
        nNeighborsInImage = 0;
        valueInNeighbors = 0;

        //Closer 4 neighbors
        if(visited[getYIndex(img,y-1)][getXIndex(img,x)] == 1){
          nNeighborsInImage++;
          valueInNeighbors += output->at(getXIndex(img,x), getYIndex(img,y-1));
        }

        if(visited[getYIndex(img,y)][getXIndex(img,x-1)] == 1){
          nNeighborsInImage++;
          valueInNeighbors += output->at(getXIndex(img,x-1), getYIndex(img,y));
        }

        if(visited[getYIndex(img,y)][getXIndex(img,x+1)] == 1){
          nNeighborsInImage++;
          valueInNeighbors += output->at(getXIndex(img,x+1), getYIndex(img,y));
        }

        if(visited[getYIndex(img,y+1)][getXIndex(img,x)] == 1){
          nNeighborsInImage++;
          valueInNeighbors += output->at(getXIndex(img,x), getYIndex(img,y+1));
        }

        if(nNeighborsInImage != 0){

          //The four in the diagonal
          if(visited[getYIndex(img,y-1)][getXIndex(img,x+1)] == 1){
            nNeighborsInImage+= 0.75;
            valueInNeighbors += 0.75*output->at(getXIndex(img,x+1), getYIndex(img,y-1));
          }
          if(visited[getYIndex(img,y-1)][getXIndex(img,x-1)] == 1){
            nNeighborsInImage+= 0.75;
            valueInNeighbors += 0.75*output->at(getXIndex(img,x-1), getYIndex(img,y-1));
          }
          if(visited[getYIndex(img,y+1)][getXIndex(img,x+1)] == 1){
            nNeighborsInImage+= 0.75;
            valueInNeighbors += 0.75*output->at(getXIndex(img,x+1), getYIndex(img,y+1));
          }
          if(visited[getYIndex(img,y+1)][getXIndex(img,x-1)] == 1){
            nNeighborsInImage+= 0.75;
            valueInNeighbors += 0.75*output->at(getXIndex(img,x-1), getYIndex(img,y+1));
          }

          //The 4 far away in line
          if(visited[getYIndex(img,y-2)][getXIndex(img,x)] == 1){
            nNeighborsInImage+=0.5;
            valueInNeighbors += 0.5*output->at(getXIndex(img,x), getYIndex(img,y-2));
          }
          if(visited[getYIndex(img,y)][getXIndex(img,x-2)] == 1){
            nNeighborsInImage+=0.5;
            valueInNeighbors +=  0.5*output->at(getXIndex(img,x-2), getYIndex(img,y));
          }
          if(visited[getYIndex(img,y)][getXIndex(img,x+2)] == 1){
            nNeighborsInImage+=0.5;
            valueInNeighbors +=  0.5*output->at(getXIndex(img,x+2), getYIndex(img,y));
          }
          if(visited[getYIndex(img,y+2)][getXIndex(img,x)] == 1){
            nNeighborsInImage+=0.5;
            valueInNeighbors +=  0.5*output->at(getXIndex(img,x), getYIndex(img,y+2));
          }

          //The other 8 in diagonals
          if(visited[getYIndex(img,y-2)][getXIndex(img,x+1)] == 1){
            nNeighborsInImage+= 0.3;
            valueInNeighbors += 0.3*output->at(getXIndex(img,x+1), getYIndex(img,y-2));
          }
          if(visited[getYIndex(img,y-1)][getXIndex(img,x+2)] == 1){
            nNeighborsInImage+= 0.3;
            valueInNeighbors += 0.3*output->at(getXIndex(img,x+2), getYIndex(img,y-1));
          }
          if(visited[getYIndex(img,y-2)][getXIndex(img,x-1)] == 1){
            nNeighborsInImage+= 0.3;
            valueInNeighbors += 0.3*output->at(getXIndex(img,x-1), getYIndex(img,y-2));
          }
          if(visited[getYIndex(img,y-1)][getXIndex(img,x-2)] == 1){
            nNeighborsInImage+= 0.3;
            valueInNeighbors += 0.3*output->at(getXIndex(img,x-2), getYIndex(img,y-1));
          }
          if(visited[getYIndex(img,y+2)][getXIndex(img,x+1)] == 1){
            nNeighborsInImage+= 0.3;
            valueInNeighbors += 0.3*output->at(getXIndex(img,x+1), getYIndex(img,y+2));
          }
          if(visited[getYIndex(img,y+1)][getXIndex(img,x+2)] == 1){
            nNeighborsInImage+= 0.3;
            valueInNeighbors += 0.3*output->at(getXIndex(img,x+2), getYIndex(img,y+1));
          }
          if(visited[getYIndex(img,y+2)][getXIndex(img,x-1)] == 1){
            nNeighborsInImage+= 0.3;
            valueInNeighbors += 0.3*output->at(getXIndex(img,x-1), getYIndex(img,y+2));
          }
          if(visited[getYIndex(img,y+1)][getXIndex(img,x-2)] == 1){
            nNeighborsInImage+= 0.3;
            valueInNeighbors += 0.3*output->at(getXIndex(img,x-2), getYIndex(img,y+1));
          }

          there_is_change = true;
          output->put(x,y,valueInNeighbors/nNeighborsInImage);
          visited_tmp[y][x] = 1;
          // break;
        }
      }//x
    } //y

    for(int y = 0; y < img->height; y++)
      for(int x = 0; x < img->width; x++)
        visited[y][x] = visited_tmp[y][x];


    // output = tmp->copy(arguments.output);

  }//While
  output->save();
}//mean
