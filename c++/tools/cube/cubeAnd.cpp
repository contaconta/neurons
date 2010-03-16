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

int main(int argc, char **argv) {

  if(argc!=4)
    {
      printf("Usage: cubeAnd volume1 volume2 result\n");
      exit(0);
    }

  Cube<uchar, ulong>* cube1 = new Cube<uchar, ulong>(argv[1]);
  Cube<uchar, ulong>* cube2 = new Cube<uchar, ulong>(argv[2]);
  Cube<uchar, ulong>* result = cube1->create_blank_cube_uchar(argv[3]);

  for(int x = 0; x < cube1->cubeWidth; x++)
    for(int y = 0; y < cube1->cubeHeight; y++)
      for(int z = 0; z < cube1->cubeDepth; z++)
        if(  ((cube1->at(x,y,z) > 100) & (cube2->at(x,y,z) > 100)) )
          result->put(x,y,z,255);
        else
          result->put(x,y,z,0);
}
