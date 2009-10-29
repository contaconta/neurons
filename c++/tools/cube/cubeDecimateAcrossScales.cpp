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
#include <argp.h>
#include "CubeFactory.h"
#include "Cube.h"
#include "Cloud_P.h"
#include "Cloud.h"
#include "Point3D.h"

using namespace std;

/** Variables for the arguments.*/
const char *argp_program_version =
  "cubeDecimate 0.1";
const char *argp_program_bug_address =
  "<german.gonzalez@epfl.ch>";
/* Program documentation. */
static char doc[] =
  "cubedecimate produces a list of local maxima of the volume";

/* A description of the arguments we accept. */
static char args_doc[] = "volume.nfo outputfile";

/* The options we understand. */
static struct argp_option options[] = {
  {"verbose"  ,  'v', 0,           0,  "Produce verbose output" },
  {"threshold",  't' , "float",      0,
   "Threshold above(bellow) the points" },
  {"min"      ,  'm',  0, 0,"The important points are the low ones"},
  {"layer"    ,  'l',  0, 0, "Do nonmax layer by layer"},
  {"windowxy" ,  'w',  "int", 0,"width and height of the window"},
  {"windowz" ,   'z',  "int", 0,"depth of the window"},
  {"logscale",   'o',  0, 0, "Do the decimation in a log scale"},
  {"not-cloud",  'c',  0, 0, "save the points as a list of indexes instead of a cloud"},
  {"somaX"   ,   'X',  "float", 0, "X coordinate of the soma"},
  {"somaY"   ,   'Y',  "float", 0, "Y coordinate of the soma"},
  {"somaZ"   ,   'Z',  "float", 0, "Z coordinate of the soma"},
  { 0 }
};

struct arguments
{
  char *args[2];                /* arg1 & arg2 */
  int flag_min, verbose, windowxy, windowz, flag_layer, flag_log;
  float threshold, somaX, somaY, somaZ;
  bool saveAsCloud;
  bool somaDefined;
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
    case 'm':
      argments->flag_min = 1;
      break;
    case 't':
      argments->threshold = atof(arg);
      break;
    case 'l':
      argments->flag_layer = 1;
      break;
    case 'o':
      argments->flag_log = 1;
      break;
    case 'w':
      argments->windowxy = atoi(arg);
      break;
    case 'z':
      argments->windowz = atoi(arg);
      break;
    case 'c':
      argments->saveAsCloud = false;
      break;
    case 'X':
      argments->somaX = atof(arg);
      argments->somaDefined = true;
      break;
    case 'Y':
      argments->somaY = atof(arg);
      argments->somaDefined = true;
      break;
    case 'Z':
      argments->somaZ = atof(arg);
      argments->somaDefined = true;
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
  arguments.flag_min = 0;
  arguments.verbose = 0;
  arguments.threshold = 0.02;
  arguments.windowxy = 8;
  arguments.windowz = 10;
  arguments.flag_layer = 0;
  arguments.flag_log = 0;
  arguments.saveAsCloud = true;
  arguments.somaDefined = false;

  argp_parse (&argp, argc, argv, 0, 0, &arguments);

  string directory(arguments.args[0]);
  float threshold =   arguments.threshold;

  printf ("Volume = %s\nFile = %s\n"
          "bool_min = %s\n"
          "threshold = %f \n"
          "saveAsCloud = %i \n",
          arguments.args[0], arguments.args[0],
          arguments.flag_min ? "yes" : "no",
          arguments.threshold,
          arguments.saveAsCloud
          );

  Cube<float, double>* c1 = new Cube<float, double>
    (directory + "/1/bf_2_3162.nfo");
  Cube<float, double>* c2 = new Cube<float, double>
    (directory + "/2/bf_2_3162.nfo");
  Cube<float, double>* c4 = new Cube<float, double>
    (directory + "/4/bf_2_3162.nfo");
  Cube<float, double>* c8 = new Cube<float, double>
    (directory + "/8/bf_2_3162.nfo");

  vector<Cube<float, double>*> thetas;
  vector<Cube<float, double>*> phis;
  thetas.push_back(new Cube<float, double>
                   (directory + "/1/aguet_2.00_2.00_theta.nfo"));
  thetas.push_back(new Cube<float, double>
                   (directory + "/2/aguet_2.00_2.00_theta.nfo"));
  thetas.push_back(new Cube<float, double>
                   (directory + "/4/aguet_2.00_2.00_theta.nfo"));
  thetas.push_back(new Cube<float, double>
                   (directory + "/8/aguet_2.00_2.00_theta.nfo"));
  phis.push_back(new Cube<float, double>
                   (directory + "/1/aguet_2.00_2.00_phi.nfo"));
  phis.push_back(new Cube<float, double>
                   (directory + "/2/aguet_2.00_2.00_phi.nfo"));
  phis.push_back(new Cube<float, double>
                   (directory + "/4/aguet_2.00_2.00_phi.nfo"));
  phis.push_back(new Cube<float, double>
                   (directory + "/8/aguet_2.00_2.00_phi.nfo"));



  vector< vector < float > > toReturn;
  vector< int > toReturnScales;
  vector< float > toReturnThetas;
  vector< float > toReturnPhis;
  int cubeCardinality = c1->cubeWidth*c1->cubeHeight*c1->cubeDepth;
  bool* visitedPoints = (bool*)malloc(cubeCardinality*sizeof(bool));
  for(int i = 0; i < cubeCardinality; i++)
    visitedPoints[i] = false;

  multimap< float, int > valueToCoordsC1;
  multimap< float, int > valueToCoordsC2;
  multimap< float, int > valueToCoordsC4;
  multimap< float, int > valueToCoordsC8;

  //Computes the min and the max
  float min_c1, max_c1, min_c2, max_c2,min_c4, max_c4,min_c8, max_c8, min_c, max_c;
  c1->min_max(&min_c1, &max_c1);
  c2->min_max(&min_c2, &max_c2);
  c4->min_max(&min_c4, &max_c4);
  c8->min_max(&min_c8, &max_c8);
  min_c = min(min_c1, min_c2);
  min_c = min(min_c , min_c4);
  min_c = min(min_c , min_c8);
  max_c = max(max_c1, max_c2);
  max_c = max(max_c , max_c4);
  max_c = max(max_c , max_c8);

  double step_size = (max_c - min_c) / 5;
  double current_threshold = max_c - step_size;

  int position = 0;
  int positionInC1 = 0;
  int ix, iy, iz;
  float mx,my,mz;

  printf("Cube<T,U>::decimate Creating the map[\n");
  // Loop for the decimation
  while(
        (current_threshold > min_c) &&
        (current_threshold > threshold - step_size)
        ){

    if( fabs(threshold - current_threshold) < step_size)
      current_threshold = threshold;

    valueToCoordsC1.erase(valueToCoordsC1.begin(), valueToCoordsC1.end() );
    valueToCoordsC2.erase(valueToCoordsC2.begin(), valueToCoordsC2.end() );
    valueToCoordsC4.erase(valueToCoordsC4.begin(), valueToCoordsC4.end() );
    valueToCoordsC8.erase(valueToCoordsC8.begin(), valueToCoordsC8.end() );

    // Find the non-visited points above the threshold for C1
    for(int z = 0; z < c1->cubeDepth; z++){
      for(int y = 0; y < c1->cubeHeight; y++){
        for(int x = 0; x < c1->cubeWidth; x++)
          {
            position = x + y*c1->cubeWidth + z*c1->cubeWidth*c1->cubeHeight;
            if( (c1->at(x,y,z) > current_threshold) &&
                (visitedPoints[position] == false))
              {
                valueToCoordsC1.insert(pair<float, int >(c1->at(x,y,z), position));
              }
          }
      }
      printf("Threshold: %f, Layer %02i and %07i points\r",
             current_threshold, z, (int)valueToCoordsC1.size()); fflush(stdout);
    }//z of c1
    printf("\n");
    // Find the non-visited points above the threshold for C2
    for(int z = 0; z < c2->cubeDepth; z++){
      for(int y = 0; y < c2->cubeHeight; y++){
        for(int x = 0; x < c2->cubeWidth; x++)
          {
            if(c2->at(x,y,z) > current_threshold){
                c2->indexesToMicrometers3(x,y,z,mx, my,mz);
                c1->micrometersToIndexes3(mx,my,mz,ix,iy,iz);
                position = ix + iy*c1->cubeWidth + iz*c1->cubeWidth*c1->cubeHeight;
                if(visitedPoints[position] == false)
                  {
                    valueToCoordsC2.insert(pair<float, int >(c2->at(x,y,z), position));
                  }
              }
          }
      }
      printf("Threshold: %f, Layer %02i and %07i points\r",
             current_threshold, z, (int)valueToCoordsC2.size()); fflush(stdout);
    }//z of c2
    printf("\n");
    // Find the non-visited points above the threshold for C2
    for(int z = 0; z < c4->cubeDepth; z++){
      for(int y = 0; y < c4->cubeHeight; y++){
        for(int x = 0; x < c4->cubeWidth; x++)
          {
            if(c4->at(x,y,z) > current_threshold){
                c4->indexesToMicrometers3(x,y,z,mx, my,mz);
                c1->micrometersToIndexes3(mx,my,mz,ix,iy,iz);
                position = ix + iy*c1->cubeWidth + iz*c1->cubeWidth*c1->cubeHeight;
                if(visitedPoints[position] == false)
                  {
                    valueToCoordsC4.insert(pair<float, int >(c4->at(x,y,z), position));
                  }
              }
          }
      }
      printf("Threshold: %f, Layer %02i and %07i points\r",
             current_threshold, z, (int)valueToCoordsC4.size()); fflush(stdout);
    }//z of c4
    printf("\n");
    // Find the non-visited points above the threshold for C2
    for(int z = 0; z < c8->cubeDepth; z++){
      for(int y = 0; y < c8->cubeHeight; y++){
        for(int x = 0; x < c8->cubeWidth; x++)
          {
            if(c8->at(x,y,z) > current_threshold){
                c8->indexesToMicrometers3(x,y,z,mx, my,mz);
                c1->micrometersToIndexes3(mx,my,mz,ix,iy,iz);
                position = ix + iy*c1->cubeWidth + iz*c1->cubeWidth*c1->cubeHeight;
                if(visitedPoints[position] == false)
                  {
                    valueToCoordsC8.insert(pair<float, int >(c8->at(x,y,z), position));
                  }
              }
          }
      }
      printf("Threshold: %f, Layer %02i and %07i points\r",
             current_threshold, z, (int)valueToCoordsC8.size()); fflush(stdout);
    }//z of c8
    printf("\n");

    printf("The maps have been creared, now it is time to see what happens\n");
    int nPointsToEvaluate =
      valueToCoordsC1.size() + valueToCoordsC2.size() +
      valueToCoordsC4.size() + valueToCoordsC8.size();
    int nPointsEvaluated = 0;
    int nPointsAdded     = 0;
    multimap< float, int >::reverse_iterator riterC1 = valueToCoordsC1.rbegin();
    multimap< float, int >::reverse_iterator riterC2 = valueToCoordsC2.rbegin();
    multimap< float, int >::reverse_iterator riterC4 = valueToCoordsC4.rbegin();
    multimap< float, int >::reverse_iterator riterC8 = valueToCoordsC8.rbegin();


    vector< float > values(4);
    float maxVal;
    int maxIdx, position, windowErase, z_p, x_p, y_p;
    while(nPointsEvaluated < nPointsToEvaluate){
      nPointsEvaluated++;
      for(int i = 0; i < 4; i++)
        values[i] = -1e8;
      //Puts the values in the vector
      if(riterC1!=valueToCoordsC1.rend()) values[0] = (*riterC1).first;
      if(riterC2!=valueToCoordsC2.rend()) values[1] = (*riterC2).first;
      if(riterC4!=valueToCoordsC4.rend()) values[2] = (*riterC4).first;
      if(riterC8!=valueToCoordsC8.rend()) values[3] = (*riterC8).first;
      maxIdx = 0; maxVal = values[0];
      for(int i = 1; i < 4; i++){
        if(values[i] > maxVal){
          maxVal = values[i];
          maxIdx = i;
        }
      }
      if(maxIdx==0){ position = (*riterC1).second; riterC1++; windowErase = 30; }
      if(maxIdx==1){ position = (*riterC2).second; riterC2++; windowErase = 35; }
      if(maxIdx==2){ position = (*riterC4).second; riterC4++; windowErase = 40; }
      if(maxIdx==3){ position = (*riterC8).second; riterC8++; windowErase = 40; }

      if(visitedPoints[position] == true)
        continue;

      z_p = position / (c1->cubeWidth*c1->cubeHeight);
      y_p = (position - z_p*c1->cubeWidth*c1->cubeHeight)/c1->cubeWidth;
      x_p =  position - z_p*c1->cubeWidth*c1->cubeHeight - y_p*c1->cubeWidth;

      //To prevent the bug in the images
      if(x_p<=5) continue;
      int counter = 0;
      for(int z = max(z_p-windowErase*2/3,0);
          z < min(z_p+windowErase*2/3, (int)c1->cubeDepth); z++)
        for(int y = max(y_p-windowErase,0); y < min(y_p+windowErase, (int)c1->cubeHeight); y++)
          for(int x = max(x_p-windowErase,0); x < min(x_p+windowErase, (int)c1->cubeWidth); x++){
            if(visitedPoints[x + y*c1->cubeWidth + z*c1->cubeWidth*c1->cubeHeight] == true)
              counter++;
            visitedPoints[x + y*c1->cubeWidth + z*c1->cubeWidth*c1->cubeHeight] = true;
          }
      //Only add the point if half of the points arround it have not been visited. Prevents small points to be added

      if(counter > windowErase*windowErase*windowErase*2*2*2/2)
        continue;
      vector< float > coords(3);
      c1->indexesToMicrometers3(x_p, y_p, z_p, coords[0], coords[1], coords[2]);
      toReturn.push_back(coords);
      toReturnScales.push_back(maxIdx);
      thetas[maxIdx]->micrometersToIndexes3(coords[0], coords[1], coords[2],
                                            x_p, y_p, z_p);
      toReturnThetas.push_back(thetas[maxIdx]->at(x_p,y_p,z_p));
      toReturnPhis.push_back(phis[maxIdx]->at(x_p,y_p,z_p));
      nPointsAdded++;
//       printf("nEval = %i of %i\n", nPointsEvaluated, nPointsToEvaluate);
    } // while goinf throught the nms
    printf("%i points have been added\n", nPointsAdded);
    current_threshold = current_threshold - step_size;
  }// While threshold


  vector<float> radius(4);
  radius[0] = 0.4;
  radius[1] = 0.8;
  radius[2] = 1.2;
  radius[3] = 1.6;

  Cloud<Point3D >* cd1 = new Cloud<Point3D>();
  Cloud<Point3D >* cd2 = new Cloud<Point3D>();
  Cloud<Point3D >* cd4 = new Cloud<Point3D>();
  Cloud<Point3D >* cd8 = new Cloud<Point3D>();
  Cloud<Point3D >* ct  = new Cloud<Point3D>();
  Cloud<Point3Dotw>* ctw = new Cloud<Point3Dotw>();

  if(arguments.somaDefined){
    ct->points.push_back
      (new Point3D(arguments.somaX, arguments.somaY, arguments.somaZ));
    ctw->points.push_back
      (new Point3Dotw(arguments.somaX, arguments.somaY, arguments.somaZ,
                      Point3Dot::TrainingPositive, 5.0));
  }

  vector< Cloud< Point3D>*> clouds;
  clouds.push_back(cd1);   clouds.push_back(cd2);
  clouds.push_back(cd4);   clouds.push_back(cd8);

  for(int i = 0; i < toReturn.size(); i++){
    Point3D* pt = new Point3D( toReturn[i][0],
                               toReturn[i][1],
                               toReturn[i][2]);
    clouds[toReturnScales[i]]->points.push_back( pt );


    ct->points.push_back( pt );
    ctw->points.push_back
      (new Point3Dotw(toReturn[i][0], toReturn[i][1], toReturn[i][2],
                      toReturnThetas[i], toReturnPhis[i],
                      Point3Dot::TrainingPositive,
                      radius[toReturnScales[i]]));
  }
  cd1->v_r = 1;   cd1->v_g = 0;   cd1->v_b = 0;   cd1->v_radius = radius[0];
  cd2->v_r = 0;   cd2->v_g = 1;   cd2->v_b = 0;   cd2->v_radius = radius[1];
  cd4->v_r = 0;   cd4->v_g = 0;   cd4->v_b = 1;   cd4->v_radius = radius[2];
  cd8->v_r = 1;   cd8->v_g = 1;   cd8->v_b = 0;   cd8->v_radius = radius[3];

  cd1->saveToFile(directory + "/decimation_1.cl");
  cd2->saveToFile(directory + "/decimation_2.cl");
  cd4->saveToFile(directory + "/decimation_4.cl");
  cd8->saveToFile(directory + "/decimation_8.cl");
  ct->saveToFile (directory + "/decimated.cl");
  ctw->saveToFile(directory + "/decimatedW.cl");

}
