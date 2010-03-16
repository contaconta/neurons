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
#include <vector>

#include "Cube.h"

using namespace std;

float colors[5][3]=
  { {1.0,0.0,0.0},
    {0.0,1.0,0.0},
    {0.0,0.0,1.0},
    {1.0,1.0,0.0},
    {1.0,1.0,1.0}
  };


int main(int argc, char **argv) {

  vector< Cube<float, double>* > cubes;

  for(int i = 1; i < argc-1; i++){
    cubes.push_back(new Cube<float, double>(argv[i]) );
  }

  string outR = string(argv[argc-1])+"R";
  string outG = string(argv[argc-1])+"G";
  string outB = string(argv[argc-1])+"B";

  Cube<uchar, ulong>* cubeR = cubes[0]->duplicate_uchar(outR);
  Cube<uchar, ulong>* cubeG = cubes[0]->duplicate_uchar(outG);
  Cube<uchar, ulong>* cubeB = cubes[0]->duplicate_uchar(outB);

  // Cube< float, double>* output = cubes[0]->create_blank_cube(argv[argc-1]);
  float maxval = DBL_MIN;
  int idxmaxval;
  for(int z = 0; z < cubes[0]->cubeDepth; z++)
    for(int y = 0; y < cubes[0]->cubeHeight; y++)
      for(int x = 0; x < cubes[0]->cubeWidth; x++)
        {
          maxval = DBL_MIN;
          for(int cb = 0; cb < cubes.size(); cb++)
            if(cubes[cb]->at(x,y,z)>maxval){
              maxval = cubes[cb]->at(x,y,z);
              idxmaxval = cb;
            }
          cubeR->put(x,y,z,uchar(255.0*maxval*colors[idxmaxval][0]) );
          cubeG->put(x,y,z,uchar(255.0*maxval*colors[idxmaxval][1]) );
          cubeB->put(x,y,z,uchar(255.0*maxval*colors[idxmaxval][2]) );
        }

  std::ofstream out(argv[argc-1]);
  out << "filenameVoxelDataR " << outR << ".nfo" << std::endl;
  out << "filenameVoxelDataG " << outG << ".nfo" << std::endl;
  out << "filenameVoxelDataB " << outB << ".nfo" << std::endl;
  out << "type color" << std::endl;
  out.close();
}
