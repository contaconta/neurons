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

int main(int argc, char **argv) {


  vector< Cube<float, double>* > cubes;
  for(int i = 1; i < argc-1; i++){
    cubes.push_back(new Cube<float, double>(argv[i]) );
  }

  Cube< float, double>* output = cubes[0]->create_blank_cube(argv[argc-1]);
  float maxval = DBL_MIN;
  for(int z = 0; z < cubes[0]->cubeDepth; z++)
    for(int y = 0; y < cubes[0]->cubeHeight; y++)
      for(int x = 0; x < cubes[0]->cubeWidth; x++)
        {
          maxval = DBL_MIN;
          for(int cb = 0; cb < cubes.size(); cb++)
            if(cubes[cb]->at(x,y,z)>maxval){
              maxval = cubes[cb]->at(x,y,z);
            }
          output->put(x,y,z,maxval);
        }

}
