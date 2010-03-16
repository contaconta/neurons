
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

using namespace std;

/** Each voxel is surrounded by 26 neighbors. The [x,y,z] coordinates
    of the cube follow the scheme of neseg. For the neighbors we
    follow the following schma:

View with Z in depth:

Layer 1:         Layer 2:         Layer 3:

0------>X
|  0 ----------3      ----------     1 ----------2
|  | 1 | 2 | 3 |    | 10| 11| 12|    | 18| 19| 20|
|  |--- --- ---|    |--- --- ---|    |--- --- ---|
|  | 4 | 5 | 6 |    | 13| X | 14|    | 21| 22| 23|
Y  |-----------|    |-----------|    |-----------|
   | 7 | 8 | 9 |    | 15| 16| 17|    | 24| 25| 26|
   4-----------7     -----------     5-----------6

Rotation of 90 degrees wuth respect to the Y axis:

Layer 1:         Layer 2:         Layer 3:

0------> -Z
|  1 ----------0      ----------     2 ----------3
|  | 18| 10| 1 |    | 19| 11| 2 |    | 20| 12| 3 |
|  |--- --- ---|    |--- --- ---|    |--- --- ---|
|  | 21| 13| 4 |    | 22| X | 5 |    | 23| 14| 6 |
Y  |-----------|    |-----------|    |-----------|
   | 24| 15| 7 |    | 25| 16| 8 |    | 26| 17| 9 |
   5-----------4     -----------     6-----------7

Rotation of 90 degrees wuth respect to the X axis:

Layer 1:         Layer 2:         Layer 3:

   1 ----------2      ----------     5 ----------6
   | 18| 19| 20|    | 21| 22| 23|    | 24| 25| 26|
   |--- --- ---|    |--- --- ---|    |--- --- ---|
Z  | 10| 11| 12|    | 13| X | 14|    | 15| 16| 17|
|  |-----------|    |-----------|    |-----------|
|  | 1 | 2 | 3 |    | 4 | 5 | 6 |    | 7 | 8 | 9 |
|  0-----------4     -----------     4-----------7
0------> X

The voxel X will be a local maxima in the direction of the other named
voxel, if it is greater than the values perpendicular to that voxel
direction. For it we should compute the voxels that are perpendicular
to the direction given. This is done by hand and the result is encoded
in the following neighboring table. The value 0 should be taken as
"there are no neighbors", as in some cases there are only 6
perpendicular voxels, instead of 8.
 */

int perpNeigh[26][8] = {
  {19, 12,  6,  8, 15, 21,  0, 0 }, //1
  {18, 13,  7, 19,  8, 20, 14, 9 },
  {10, 19, 23, 17,  8,  4,  0, 0 },
  {18, 11,  3, 21,  6, 24, 16, 9 },
  {10, 11, 12, 13, 14, 15, 16, 17}, //5
  { 1, 11, 20,  4, 23,  7, 14, 26},
  {17, 25, 21, 10,  2,  6,  0,  0},
  { 1, 13, 24,  2, 25,  3, 14, 26},
  { 2, 12, 23, 25, 15,  4,  0,  0},
  { 3,  5,  7, 12, 15, 20, 22, 24}, //10
  { 4,  5,  6, 13, 14, 21, 22, 23},
  { 1,  5,  9, 10, 17, 18, 22, 26},
  { 2,  5,  8, 11, 16, 19, 22, 25},
  { 2,  5,  8, 11, 16, 19, 22, 25},
  { 1,  5,  9, 10, 17, 18, 22, 26}, //15
  { 4,  5,  6, 13, 14, 21, 22, 23},
  { 3,  5,  7, 12, 15, 20, 22, 24},
  { 2, 12, 23, 25, 15,  4,  0,  0},
  { 1, 13, 24,  2, 25,  3, 14, 26},
  {17, 25, 21, 10,  2,  6,  0,  0}, //20
  { 1, 11, 20,  4, 23,  7, 16, 26},
  {10, 11, 12, 13, 14, 15, 16, 17},
  {18, 11,  3, 21,  6, 24, 16,  9},
  {10, 19, 23, 17,  8,  4,  0,  0},
  {18, 13,  7, 19,  8, 20, 14,  9}, //25
  {19, 12,  6,  8, 15, 21,  0,  0}
};

/** Table that codes the relationship between the indexes previously
    and the [x,y,z] indexes in the cube*/
int nbrToIdx[27][3] = {
  { 0, 0, 0}, //0
  {-1,-1,-1}, //Layer 1
  { 0,-1,-1},
  { 1,-1,-1},
  {-1, 0,-1},
  { 0, 0,-1}, //5
  { 1, 0,-1},
  {-1, 1,-1},
  { 0, 1,-1},
  { 1, 1,-1},
  {-1,-1, 0}, // 10 - Layer 2
  { 0,-1, 0},
  { 1,-1, 0},
  {-1, 0, 0},
  { 1, 0, 0},
  {-1, 1, 0}, //15
  { 0, 1, 0},
  { 1, 1, 0},
  {-1,-1, 1}, // Layer 3
  { 0,-1, 1},
  { 1,-1, 1}, //20
  {-1, 0, 1},
  { 0, 0, 1},
  { 1, 0, 1},
  {-1, 1, 1},
  { 0, 1, 1}, //25
  { 1, 1, 1}
};

/** Table for the module of the director vector for a given voxel index.*/
#define MOD_3 1.732
#define MOD_2 1.414
double modDirection[27] = {
  0,
  MOD_3,
  MOD_2,
  MOD_3,
  MOD_2,
  1, // 5
  MOD_2,
  MOD_3,
  MOD_2,
  MOD_3,
  MOD_2, //10
  1,
  MOD_2,
  1,
  1,
  MOD_2, // 15
  1,
  MOD_2,
  MOD_3,
  MOD_2,
  MOD_3, //20
  MOD_2,
  1,
  MOD_2,
  MOD_3,
  MOD_2, //25
  MOD_3
};


// Given two angles, compute which of the possible 26 orientaitons is the closest
int computeOrientation(float theta, float phi){
  int   idxMax = 0;
  float dotMax =  -1;
  //Unit vector in the direction theta, phi
  float x = cos(theta)*sin(phi);
  float y = sin(theta)*sin(phi);
  float z = cos(phi);
  float dot;

  for(int orIdx = 1; orIdx <= 26; orIdx++){
    //Computes the dot product between the unit vector and the unit vector in the orientation
    dot = (x*nbrToIdx[orIdx][0] + y*nbrToIdx[orIdx][1] + z*nbrToIdx[orIdx][2])
      / modDirection[orIdx];
    if(dot > dotMax){
      dotMax = dot;
      idxMax = orIdx;
    }
  }
  return idxMax;
}



int main(int argc, char **argv) {

  if(argc!=5){
    printf("Usage: cubeNonMaxSup cube.nfo theta.nfo phi.nfo output\n");
    exit(0);
  }

  Cube< float, double>* orig  = new Cube<float, double>(argv[1]);
  Cube< float, double>* theta = new Cube<float, double>(argv[2]);
  Cube< float, double>* phi   = new Cube<float, double>(argv[3]);
  Cube< float, double>* output;
  // Cube< float, double>* outputOrient;

  string outputName       = argv[4];
  // string outputOrientName = argv[5];
  if(fileExists(outputName)){
    output = new Cube<float, double>(outputName);
  } else {
    output = orig->duplicate_clean(outputName);
  }
  // if(fileExists(outputOrientName)){
    // outputOrient = new Cube<float, double>(outputOrientName);
  // } else {
    // outputOrient = orig->duplicate_clean(outputOrientName);
  // }


  // Loop for all the voxels of the cube. For now we will ignore the borders

  printf("Performing non-maxima-supression[");
#pragma omp parallel for
  for(int z = 1; z < orig->cubeDepth -1; z++){
    int  currOrient = 0;
    bool isMax = false;
    for(int y = 1; y < orig->cubeHeight -1; y++){
      for(int x = 1; x < orig->cubeWidth -1; x++){
        isMax = true;
        currOrient = computeOrientation(theta->at(x,y,z), phi->at(x,y,z));
        // outputOrient->put(x,y,z,currOrient);
        for(int neigh = 0; neigh < 8; neigh++){
          if(orig->at(x+nbrToIdx[perpNeigh[currOrient][neigh]][0],
                      y+nbrToIdx[perpNeigh[currOrient][neigh]][1],
                      z+nbrToIdx[perpNeigh[currOrient][neigh]][2])
             > orig->at(x,y,z)){
            isMax = false;
            break;
          }
        } //Neighbors
        if(isMax){
          output->put(x,y,z, orig->at(x,y,z));
        }
      }
    }
    printf("#"); fflush(stdout);
  }
  printf("]\n");

}
