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

   if(argc<2)
    {
      printf("Usage: cubeEllipse volume_name\n");
      exit(0);
    }

   int rx = 200;
   int ry = 200;
   int rz = 100;

   string name(argv[1]);
   Cube<float, double>* output = new Cube<float,double>(rx, ry, rz, name);

   //Cube<uchar,ulong>* output = new Cube<uchar,ulong>();
   output->put_value_in_ellipsoid(100,100,100,50,500,10,10);

   delete output;
}
